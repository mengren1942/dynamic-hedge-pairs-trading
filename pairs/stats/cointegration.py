# pairs/stats/cointegration.py
"""
Fast parallel cointegration screening (EG or Johansen) and a dual-gate
combinator that requires both tests to pass. Expects a MultiIndex (ticker, datetime)
DataFrame with a 'close' column.
"""
from __future__ import annotations
from typing import Literal, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from functools import partial
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from threadpoolctl import threadpool_limits
from tqdm import tqdm

__all__ = [
    "find_cointegrated_pairs_executor",
    "find_cointegrated_pairs_dualgate",
]

# ---- worker globals (populated by _init_worker) ----
_SERIES_MAP = None
_KEYS = None
_FULL_START = None
_FULL_END = None

def _init_worker(series_map, keys, full_start, full_end):
    """Runs once per worker process. Store heavy objects in globals and cap BLAS threads."""
    global _SERIES_MAP, _KEYS, _FULL_START, _FULL_END
    _SERIES_MAP = series_map
    _KEYS = keys
    _FULL_START = full_start
    _FULL_END = full_end

    # Cap BLAS threads inside each process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Best-effort: expand CPU affinity to all CPUs (harmless if already full)
    try:
        import psutil
        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            p.cpu_affinity(list(range(os.cpu_count() or 1)))
    except Exception:
        pass

def _align_dropna(S1: pd.Series, S2: pd.Series) -> pd.DataFrame:
    return pd.concat([S1, S2], axis=1, keys=["S1", "S2"]).dropna()

def _alpha_to_col(alpha: float) -> int:
    if alpha <= 0.01: return 2
    if alpha <= 0.05: return 1
    if alpha <= 0.10: return 0
    return 1

def _task_eg(idx_pair, *, alpha: float, trend: str, maxlag: Optional[int], autolag: str):
    i, j = idx_pair
    k1, k2 = _KEYS[i], _KEYS[j]
    S1, S2 = _SERIES_MAP[k1], _SERIES_MAP[k2]

    # Require full-span coverage to avoid biased tests
    if not (S1.index[0] == _FULL_START and S1.index[-1] == _FULL_END and
            S2.index[0] == _FULL_START and S2.index[-1] == _FULL_END):
        return i, j, 0.0, np.nan, False

    df = _align_dropna(S1, S2)
    if len(df) < 50:
        return i, j, 0.0, np.nan, False

    with threadpool_limits(limits=1):
        tstat, pval, _ = coint(df["S1"], df["S2"], trend=trend, maxlag=maxlag, autolag=autolag)
    return i, j, float(tstat), float(pval), bool(pval < alpha)

def _task_johansen(idx_pair, *, alpha: float, det_order: int, k_ar_diff: int, stat: Literal["trace","maxeig"]):
    i, j = idx_pair
    k1, k2 = _KEYS[i], _KEYS[j]
    S1, S2 = _SERIES_MAP[k1], _SERIES_MAP[k2]

    if not (S1.index[0] == _FULL_START and S1.index[-1] == _FULL_END and
            S2.index[0] == _FULL_START and S2.index[-1] == _FULL_END):
        return i, j, 0.0, np.nan, False

    df = _align_dropna(S1, S2)
    if len(df) < 50:
        return i, j, 0.0, np.nan, False

    X = df[["S1", "S2"]].values
    with threadpool_limits(limits=1):
        res = coint_johansen(X, det_order=det_order, k_ar_diff=k_ar_diff)

    alpha_col = _alpha_to_col(alpha)
    if stat == "trace":
        test_stat, crit_val = float(res.lr1[0]), float(res.cvt[0, alpha_col])
    else:
        test_stat, crit_val = float(res.lr2[0]), float(res.cvm[0, alpha_col])
    return i, j, test_stat, np.nan, bool(test_stat > crit_val)

def find_cointegrated_pairs_executor(
    data: pd.DataFrame,
    *,
    alpha: float = 0.05,
    method: Literal["eg", "johansen"] = "eg",
    # EG
    eg_trend: str = "c",
    eg_maxlag: Optional[int] = None,
    eg_autolag: str = "aic",
    # Johansen
    joh_det_order: int = 0,
    joh_k_ar_diff: int = 1,
    joh_stat: Literal["trace", "maxeig"] = "trace",
    # parallel
    n_workers: Optional[int] = None,      # use physical cores on TR: 64
    chunksize: int = 4,                   # small chunks to keep all workers busy
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Expects `data` with a MultiIndex whose first level is ticker and second is datetime,
    and with a 'close' column.
    Returns:
        score_matrix (n x n)
        pvalue_matrix (n x n)  [np.nan for Johansen]
        pairs (List[Tuple[str, str]])  # native Python strings
    """
    # ---- Validate and normalize index ----
    if "close" not in data.columns:
        raise ValueError("data must have column 'close' and a MultiIndex (ticker, datetime).")
    if not isinstance(data.index, pd.MultiIndex) or data.index.nlevels < 2:
        raise ValueError("data must have a MultiIndex with (ticker, datetime) levels.")

    # Ensure level names = ("ticker", "datetime")
    names = list(data.index.names)
    if names[0] != "ticker" or names[1] != "datetime":
        new_names = names[:]
        new_names[0] = "ticker"
        new_names[1] = "datetime"
        data = data.copy()
        data.index = data.index.set_names(new_names)

    # Sort to guarantee monotonicity per series
    data = data.sort_index(level=["ticker", "datetime"])

    # ---- Build per-ticker close series with DatetimeIndex ----
    series_map = {}
    for k, g in data.groupby(level="ticker"):
        s = g["close"].copy()
        s.index = s.index.droplevel("ticker")          # leave only datetime in index
        if not isinstance(s.index, pd.DatetimeIndex):  # ensure dtype
            s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.sort_index()
        series_map[str(k)] = s                         # ensure keys are native str

    # Native Python str keys (avoid np.str_)
    keys: List[str] = [str(k) for k in series_map.keys()]
    n = len(keys)

    score_matrix  = np.zeros((n, n))
    pvalue_matrix = np.full((n, n), np.nan)
    pairs: List[Tuple[str, str]] = []

    # Global span from the datetime level
    dt_index = data.index.get_level_values("datetime")
    full_start, full_end = dt_index.min(), dt_index.max()

    combos = list(combinations(range(n), 2))

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) // 2)  # ~physical cores

    # Choose picklable task (partials are picklable; functions are top-level)
    if method == "eg":
        task = partial(_task_eg, alpha=alpha, trend=eg_trend, maxlag=eg_maxlag, autolag=eg_autolag)
        desc = "Engle–Granger cointegration tests"
    elif method == "johansen":
        task = partial(_task_johansen, alpha=alpha, det_order=joh_det_order,
                       k_ar_diff=joh_k_ar_diff, stat=joh_stat)
        desc = "Johansen cointegration tests"
    else:
        raise ValueError("method must be 'eg' or 'johansen'")

    # Run
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(series_map, keys, full_start, full_end),
    ) as ex:
        it = ex.map(task, combos, chunksize=chunksize)
        if show_progress:
            it = tqdm(it, total=len(combos), desc=desc, leave=False)

        for i, j, score, pval, is_coint in it:
            k1, k2 = keys[i], keys[j]           # plain str
            score_matrix[i, j]  = score
            pvalue_matrix[i, j] = pval
            if is_coint:
                pairs.append((k1, k2))          # tuples of str

    return score_matrix, pvalue_matrix, pairs

def find_cointegrated_pairs_dualgate(
    data: pd.DataFrame,
    *,
    # EG params
    alpha_eg: float = 0.05,
    eg_trend: str = "c",
    eg_maxlag: Optional[int] = None,
    eg_autolag: str = "aic",
    # Johansen params
    alpha_joh: float = 0.05,
    joh_det_order: int = 0,
    joh_k_ar_diff: int = 1,
    joh_stat: Literal["trace", "maxeig"] = "trace",
    # parallel
    n_workers: Optional[int] = None,
    chunksize: int = 4,
    show_progress: bool = True,
    # output options
    only_pass: bool = False,         # return only pairs that pass both tests
    sort_by: Optional[str] = "eg_p"  # e.g. "eg_p", "eg_t", "joh_stat" or None
) -> pd.DataFrame:
    """
    Run EG and Johansen, keep only pairs that pass BOTH (dual-gate), and
    return a stationarity-style summary DataFrame:

        index:  MultiIndex (ticker1, ticker2)
        cols :  ["eg_t", "eg_p", "eg_pass", "joh_stat", "joh_pass", "verdict"]
    """
    # ---- Run EG ----
    score_eg, pval_eg, pairs_eg = find_cointegrated_pairs_executor(
        data,
        alpha=alpha_eg,
        method="eg",
        eg_trend=eg_trend,
        eg_maxlag=eg_maxlag,
        eg_autolag=eg_autolag,
        n_workers=n_workers,
        chunksize=chunksize,
        show_progress=show_progress,
    )

    # ---- Run Johansen ----
    score_joh, pval_joh, pairs_joh = find_cointegrated_pairs_executor(
        data,
        alpha=alpha_joh,
        method="johansen",
        joh_det_order=joh_det_order,
        joh_k_ar_diff=joh_k_ar_diff,
        joh_stat=joh_stat,
        n_workers=n_workers,
        chunksize=chunksize,
        show_progress=show_progress,
    )

    # ticker order used for the matrices
    tickers: List[str] = [str(t) for t in data.index.get_level_values("ticker").unique().tolist()]
    n = len(tickers)

    # quick membership sets for pass/fail
    eg_set  = set(map(tuple, pairs_eg))
    joh_set = set(map(tuple, pairs_joh))

    # build rows from upper triangle
    rows = []
    for i in range(n):
        for j in range(i+1, n):
            k1, k2 = tickers[i], tickers[j]
            eg_t   = score_eg[i, j]
            eg_p   = pval_eg[i, j]
            joh_st = score_joh[i, j]  # pval_joh is nan by design

            eg_pass  = (k1, k2) in eg_set or (k2, k1) in eg_set
            joh_pass = (k1, k2) in joh_set or (k2, k1) in joh_set
            verdict  = "pass" if (eg_pass and joh_pass) else "fail"

            rows.append((k1, k2, eg_t, eg_p, eg_pass, joh_st, joh_pass, verdict))

    df_out = pd.DataFrame(
        rows,
        columns=["ticker1", "ticker2", "eg_t", "eg_p", "eg_pass", "joh_stat", "joh_pass", "verdict"],
    ).set_index(["ticker1", "ticker2"])

    if only_pass:
        df_out = df_out[df_out["verdict"] == "pass"]

    if sort_by is not None and sort_by in df_out.columns:
        if sort_by == "eg_p":
            df_out = df_out.sort_values(by=["verdict", sort_by], ascending=[False, True])
        elif sort_by in ("eg_t", "joh_stat"):
            df_out = df_out.sort_values(by=["verdict", sort_by], ascending=[False, False])
        else:
            df_out = df_out.sort_values(by=sort_by, ascending=True)

    return df_out
