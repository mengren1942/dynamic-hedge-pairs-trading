# pairs/utils/splits.py
from __future__ import annotations
import pandas as pd

__all__ = ["split_by_date", "split_train_val_test"]

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure MultiIndex names ('ticker','datetime') and sort by them."""
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
        raise ValueError("Expected MultiIndex with levels ('ticker','datetime').")
    names = list(df.index.names)
    if names[0] != "ticker" or names[1] != "datetime":
        df = df.copy()
        names[0], names[1] = "ticker", "datetime"
        df.index = df.index.set_names(names)
    return df.sort_index(level=["ticker", "datetime"])

def split_by_date(
    df_multi: pd.DataFrame, *, train_end: str | pd.Timestamp, val_end: str | pd.Timestamp
):
    """
    Split a MultiIndex (ticker, datetime) DataFrame with 'close' into TRAIN / VALIDATION.
    """
    df = _normalize_index(df_multi)
    dt = df.index.get_level_values("datetime")
    train_mask = dt <= pd.Timestamp(train_end)
    val_mask   = (dt > pd.Timestamp(train_end)) & (dt <= pd.Timestamp(val_end))
    return df.loc[train_mask], df.loc[val_mask]

def split_train_val_test(
    df_multi: pd.DataFrame, *, train_end: str | pd.Timestamp, val_end: str | pd.Timestamp
):
    """
    3-way split: TRAIN (≤ train_end), VALIDATION (between train_end and val_end),
    TEST (> val_end).
    """
    df = _normalize_index(df_multi)
    dt = df.index.get_level_values("datetime")
    train_mask = dt <= pd.Timestamp(train_end)
    val_mask   = (dt > pd.Timestamp(train_end)) & (dt <= pd.Timestamp(val_end))
    test_mask  = dt > pd.Timestamp(val_end)
    return df.loc[train_mask], df.loc[val_mask], df.loc[test_mask]
