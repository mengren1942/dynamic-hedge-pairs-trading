from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence
import json, pandas as pd

def load_tickers(path: str | Path, *, csv_col: str | None = None) -> pd.Index:
    p = Path(path); suf = p.suffix.lower()
    if suf in {".txt",".list"}:
        vals = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
        return _clean(vals)
    if suf == ".csv":
        df = pd.read_csv(p)
        col = csv_col or _guess_ticker_col(df.columns)
        return _clean(df[col].astype(str).tolist())
    if suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return _clean(obj)
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            for key in ("ticker","symbol","Ticker","SYMBOL"):
                if all(key in d for d in obj):
                    return _clean([d[key] for d in obj])
        raise ValueError(f"Unrecognized JSON format: {p}")
    raise ValueError(f"Unsupported file type: {suf}")

def combine_tickers(paths: Sequence[str | Path], op: str = "union") -> pd.Index:
    sets = [set(load_tickers(p).tolist()) for p in paths]
    if not sets: return pd.Index([])
    if op == "union":
        out = set().union(*sets)
    elif op == "intersection":
        out = sets[0].intersection(*sets[1:])
    elif op == "difference":
        it = iter(sets); out = next(it)
        for s in it: out -= s
    else:
        raise ValueError("op must be one of {'union','intersection','difference'}")
    return _clean(sorted(out))

def _guess_ticker_col(cols: Iterable[str]) -> str:
    cols = list(cols)
    for c in ("ticker","symbol","Ticker","Symbol","SYMBOL","TICKER"):
        if c in cols: return c
    return cols[0]

def _clean(vals: Iterable[str]) -> pd.Index:
    return pd.Index(
        pd.Series(list(vals), dtype="string")
          .str.strip().str.upper()
          .dropna().drop_duplicates().sort_values()
    )
