# pairs/strategies/evaluate.py
"""
Evaluate executed pair signals (next-bar safe) with flexible cost models.

Exports
-------
- evaluate_pair_signals(df_pair, signals, ...)

Inputs
------
df_pair : DataFrame indexed by datetime with columns:
    'P1', 'P2'                      # aligned leg prices
    (optionally) 'z'                # z-score at decision time (used in trade log)

signals : DataFrame indexed by same datetime with executed columns:
    'n1', 'n2'                      # executed share holdings (levels)
    (optionally) 'pos'              # +1/0/-1; enables position-based stats
    (optionally) 'z','entry','exit','stop'  # for logging only

Returns
-------
daily  : DataFrame of pnl/returns/exposure/turnover/drawdowns
trades : DataFrame of round-trips (reversals optional)
summary: dict with key performance metrics
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

__all__ = ["evaluate_pair_signals"]


def evaluate_pair_signals(
    df_pair: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    # cost model
    cost_bps: float = 0.0,            # commission+slippage, bps of traded notional (per leg)
    fee_per_share_1: float = 0.0,     # fixed fee per share for leg 1
    fee_per_share_2: float = 0.0,     # fixed fee per share for leg 2
    borrow_bps_per_year: float = 0.0, # short borrow (annual bps on short market value)
    days_per_year: int = 252,         # kept for backward-compat
    # return scaling
    capital_base: float | None = None,  # if None: percentile of in-position gross exposure
    capital_base_percentile: float = 0.5,  # 0.5 = median; set 0.6/0.7 to be more conservative
    min_capital_base: float = 1.0,
    # generalization & trade parsing
    bars_per_year: int | None = None,      # if None, defaults to days_per_year
    treat_reversals_as_round_trips: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate a single pair's executed signals (next-bar safe).

    Assumes `signals` has executed columns: ['pos','n1','n2'] (and optionally 'z','entry','exit','stop']).

    Returns:
        daily  : DataFrame with pnl/returns/exposure/turnover/drawdowns
        trades : DataFrame of round-trip trades (reversals optional)
        summary: dict of key metrics (backward-compatible keys preserved)
    """
    # --- align & sanity -------------------------------------------------------
    cols = ["P1", "P2"]
    if not set(cols).issubset(df_pair.columns):
        raise KeyError(f"df_pair must contain {cols}")

    df = pd.concat([df_pair[cols], signals], axis=1).sort_index().copy()

    # simple price sanity (optional): drop rows with non-finite or non-positive prices
    bad = ~np.isfinite(df["P1"]) | ~np.isfinite(df["P2"]) | (df["P1"] <= 0) | (df["P2"] <= 0)
    if bad.any():
        df = df.loc[~bad]

    n = len(df)
    if n == 0:
        empty = pd.DataFrame(columns=[
            "pnl_gross","cost","pnl_net","ret_gross","ret_net","equity","portfolio",
            "gross_exposure","net_exposure","turnover","in_pos","drawdown","drawdown_pct"
        ])
        return empty, pd.DataFrame(), {
            "start": None, "end": None, "bars": 0, "capital_base": float("nan"),
            "gross_pnl": 0.0, "net_pnl": 0.0, "ann_return": float("nan"),
            "sharpe": float("nan"), "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "avg_gross_exposure": 0.0, "avg_exposure_utilization": 0.0,
            "turnover_annualized": 0.0, "n_trades": 0, "hit_rate": float("nan"),
            "avg_win": float("nan"), "avg_loss": float("nan"),
            "profit_factor": float("inf"), "avg_hold_bars": float("nan"), "med_hold_bars": float("nan"),
            # extras
            "ann_vol": float("nan"), "sortino": float("nan"),
            "avg_daily_dollars_traded": 0.0
        }

    # executed holdings
    n1 = df["n1"].astype(float).fillna(0.0)
    n2 = df["n2"].astype(float).fillna(0.0)
    pos = (np.sign(n1) != 0) | (np.sign(n2) != 0)

    # convenience: side of spread (+1 long-spread, -1 short-spread, 0 flat)
    side = pd.Series(0, index=df.index, dtype=float)
    side.loc[n1 != 0] = np.sign(n1.loc[n1 != 0])
    side.loc[(side == 0) & (n2 != 0)] = np.sign(n2.loc[(side == 0) & (n2 != 0)])

    # --- price changes (close-to-close PnL with executed holdings) ------------
    dP1 = df["P1"].astype(float).diff().fillna(0.0)
    dP2 = df["P2"].astype(float).diff().fillna(0.0)
    pnl_gross = n1 * dP1 + n2 * dP2

    # --- trades and costs -----------------------------------------------------
    dn1 = n1.diff().fillna(n1)   # Δshares executed on this bar
    dn2 = n2.diff().fillna(n2)
    traded_notional = (dn1.abs() * df["P1"]) + (dn2.abs() * df["P2"])

    cost_comm_slip  = (cost_bps / 1e4) * traded_notional
    cost_per_share  = fee_per_share_1 * dn1.abs() + fee_per_share_2 * dn2.abs()

    short_notional = (np.where(n1 < 0, -n1 * df["P1"], 0.0)
                    + np.where(n2 < 0, -n2 * df["P2"], 0.0))
    # accrue borrow daily on short market value
    cost_borrow = short_notional * (borrow_bps_per_year / 1e4) / float(days_per_year)

    cost_total = cost_comm_slip + cost_per_share + cost_borrow
    pnl_net = pnl_gross - cost_total

    # --- exposures & returns --------------------------------------------------
    gross_exposure = n1.abs() * df["P1"] + n2.abs() * df["P2"]
    net_exposure   = n1 * df["P1"] + n2 * df["P2"]

    if capital_base is None:
        inpos_expo = gross_exposure.where(pos)
        if inpos_expo.notna().any():
            cb = float(np.nanquantile(inpos_expo, capital_base_percentile))
        else:
            cb = float(np.nanquantile(gross_exposure, capital_base_percentile))
        if not np.isfinite(cb) or cb <= 0:
            cb = float(min_capital_base)
        capital_base = cb

    # returns scaled to fixed capital base
    ret_gross = pnl_gross / capital_base
    ret_net   = pnl_net   / capital_base
    equity    = pnl_net.cumsum()

    # portfolio value & drawdowns (on portfolio, not equity)
    portfolio = capital_base + equity
    peak_port = portfolio.cummax()
    # Numerically stable drawdown% with safe first row
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = (portfolio / peak_port) - 1.0
    dd_pct = dd_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dd = portfolio - peak_port

    # --- turnover -------------------------------------------------------------
    daily_turnover = traded_notional / capital_base

    daily = pd.DataFrame({
        "pnl_gross": pnl_gross,
        "cost": cost_total,
        "pnl_net": pnl_net,
        "ret_gross": ret_gross,
        "ret_net": ret_net,
        "equity": equity,
        "portfolio": portfolio,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "turnover": daily_turnover,
        "in_pos": pos.astype(int),
        "drawdown": dd,
        "drawdown_pct": dd_pct,
    })

    # --- per-trade breakdown (round trips) -----------------------------------
    # default: entries on pos rising-edge; exits on pos falling-edge
    entries = pd.Index(daily.index[(pos) & (~pos.shift(1, fill_value=False))])
    exits   = pd.Index(daily.index[(~pos) & (pos.shift(1, fill_value=False))])

    # Optionally treat reversals as exit+entry on the same bar
    if treat_reversals_as_round_trips:
        side_change = (side != side.shift(1)).fillna(False)
        reversals = daily.index[pos & pos.shift(1, fill_value=False) & side_change]
        entries = entries.union(reversals).sort_values()
        exits   = exits.union(reversals).sort_values()

    trade_rows = []
    for start in entries:
        after = exits[exits >= start]  # include same-bar for reversal
        if len(after) == 0:
            break
        end = after[0]

        # slice the interval where position was held; exclude the bar when flat after exit
        sl = daily.loc[start:end]
        # if this is a normal flat exit, last bar is flat; for reversals, it’s still in-pos.
        if (end in exits) and (end not in entries) and (sl["in_pos"].iloc[-1] == 0):
            sl = sl.iloc[:-1]

        if sl.empty:
            continue

        sgn = int(np.sign(n1.loc[start]) or np.sign(n2.loc[start]))  # +1 long-spread, -1 short-spread
        pnl_g = float(sl["pnl_gross"].sum())
        cst   = float(sl["cost"].sum())
        pnl_n = float(sl["pnl_net"].sum())
        ret   = float(sl["ret_net"].sum())
        hold  = int(sl.shape[0])
        z_e   = float(df["z"].loc[start]) if "z" in df.columns and pd.notna(df["z"].loc[start]) else np.nan
        # decision z the day before the exit execution (next-bar safe)
        z_prev_exit = float(df["z"].shift(1).loc[end]) if "z" in df.columns and end in df.index else np.nan

        # peak adverse excursion (on portfolio) during the trade
        trade_port = sl["portfolio"]
        pae = float((trade_port - trade_port.cummax()).min())  # most negative drawdown $ during trade

        trade_rows.append({
            "entry": start, "exit": end, "side": sgn,
            "bars": hold, "pnl_gross": pnl_g, "cost": cst, "pnl_net": pnl_n, "ret_on_cap": ret,
            "z_at_entry": z_e, "z_prev_exit": z_prev_exit,
            "pae": pae,
            "turnover": float(sl["turnover"].sum())
        })

    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        wins = trades["pnl_net"] > 0
        profit_factor = (trades.loc[wins, "pnl_net"].sum() /
                         -trades.loc[~wins, "pnl_net"].sum()) if (~wins).any() else np.inf
        avg_hold = float(trades["bars"].mean())
        med_hold = float(trades["bars"].median())
        hitrate  = float(wins.mean())
        avg_win  = float(trades.loc[wins, "pnl_net"].mean()) if wins.any() else 0.0
        avg_loss = float(trades.loc[~wins, "pnl_net"].mean()) if (~wins).any() else 0.0
    else:
        profit_factor = np.inf; avg_hold = med_hold = hitrate = avg_win = avg_loss = np.nan

    # --- summary metrics ------------------------------------------------------
    periods_per_year = int(bars_per_year or days_per_year)
    n = daily.shape[0]
    ret_mean = daily["ret_net"].mean()
    ret_std  = daily["ret_net"].std(ddof=0)
    sharpe   = (ret_mean / ret_std) * np.sqrt(periods_per_year) if ret_std > 0 else np.nan

    # geometric annualized return on fixed-capital series
    gross_chain = (1.0 + daily["ret_gross"].fillna(0.0)).prod()
    net_chain   = (1.0 + daily["ret_net"].fillna(0.0)).prod()
    ann_ret     = (net_chain ** (periods_per_year / max(n, 1))) - 1.0

    # extras: annualized vol & Sortino (using 0 threshold)
    ann_vol  = (ret_std * np.sqrt(periods_per_year)) if ret_std > 0 else np.nan
    downside = daily["ret_net"].clip(upper=0.0)
    dstd = downside.std(ddof=0)
    sortino = (ret_mean / dstd) * np.sqrt(periods_per_year) if dstd > 0 else np.nan

    ann_turnover = daily["turnover"].sum() * (periods_per_year / max(n, 1))
    mdd = float(-daily["drawdown"].min())
    mdd_pct = float(-daily["drawdown_pct"].min())
    exposure_util = (daily.loc[pos, "gross_exposure"].mean() / capital_base) if pos.any() else 0.0

    summary = {
        "start": daily.index[0] if n else None,
        "end":   daily.index[-1] if n else None,
        "bars": n,
        "capital_base": float(capital_base),
        "gross_pnl": float(daily["pnl_gross"].sum()),
        "net_pnl": float(daily["pnl_net"].sum()),
        "ann_return": float(ann_ret),
        "sharpe": float(sharpe),
        "max_drawdown": mdd,
        "max_drawdown_pct": mdd_pct,
        "avg_gross_exposure": float(daily["gross_exposure"].mean()),
        "avg_exposure_utilization": float(exposure_util),
        "turnover_annualized": float(ann_turnover),
        "n_trades": int(len(trades)),
        "hit_rate": float(hitrate) if not np.isnan(hitrate) else np.nan,
        "avg_win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "avg_loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else np.inf,
        "avg_hold_bars": float(avg_hold) if not np.isnan(avg_hold) else np.nan,
        "med_hold_bars": float(med_hold) if not np.isnan(med_hold) else np.nan,
        # extras (non-breaking)
        "ann_vol": float(ann_vol) if not np.isnan(ann_vol) else np.nan,
        "sortino": float(sortino) if not np.isnan(sortino) else np.nan,
        "gross_return_chain": float(gross_chain),
        "net_return_chain": float(net_chain),
        "avg_daily_dollars_traded": float(traded_notional.mean()) if n else 0.0,
    }

    return daily, trades, summary
