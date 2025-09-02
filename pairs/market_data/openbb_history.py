from __future__ import annotations
import pandas as pd
from openbb import obb
obb.user.preferences.output_type = "dataframe"
import logging
from pandas.errors import EmptyDataError
from typing import Iterable, List, Tuple, Union, Optional

def download_history_openbb(
    tickers: Union[Iterable[str], pd.Series, pd.Index],
    start_date: Union[str, pd.Timestamp],
    end_date:   Union[str, pd.Timestamp],
    *,
    provider: str = "yfinance",
    show_progress: bool = True,
    silence_logs: bool = True,
    return_failed: bool = False,
):
    """
    Download historical OHLCV for multiple tickers via OpenBB and stitch into one DataFrame.

    Returns:
        df                      (pd.DataFrame): concatenated results with MultiIndex (ticker, datetime).
        (optionally) failed     (List[str])   : tickers that returned no data or errored.
    """
    from tqdm import tqdm
    try:
        from openbb import obb  # OpenBB v4
    except Exception as e:
        raise ImportError("OpenBB not available. `pip install openbb` (v4).") from e

    # Silence noisy logs if requested
    if silence_logs:
        for lg in ["openbb_core", "openbb_yfinance", "openbb", "yfinance"]:
            logging.getLogger(lg).setLevel(logging.CRITICAL)

    # Clean ticker list
    tickers = (
        pd.Index(pd.Series(list(tickers), dtype="object"))
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    frames: List[Tuple[str, pd.DataFrame]] = []
    failed: List[str] = []

    iterator = tqdm(tickers, desc="Download", disable=not show_progress)
    for tkr in iterator:
        if show_progress:
            iterator.set_description(tkr)
        try:
            df_new = obb.equity.price.historical(
                symbol=tkr,
                start_date=start_date,
                end_date=end_date,
                provider=provider,
            )
            if df_new is None or len(df_new) == 0:
                failed.append(tkr)
                continue

            # Ensure datetime-like index and name it consistently
            df_new = df_new.copy()
            if not isinstance(df_new.index, pd.DatetimeIndex):
                df_new.index = pd.to_datetime(df_new.index, errors="coerce")
            # Some providers name it 'date' — normalize to 'datetime'
            dt_name = df_new.index.name or "datetime"
            df_new.index = df_new.index.rename("datetime")

            frames.append((tkr, df_new))
        except (EmptyDataError, Exception):
            failed.append(tkr)

    if frames:
        # Concatenate using keys to form MultiIndex with ticker first, datetime second
        keys = [k for k, _ in frames]
        objs = [df for _, df in frames]
        df = pd.concat(objs, keys=keys, names=["ticker", "datetime"])
        # (Optional) sort by ticker then time
        df = df.sort_index(level=["ticker", "datetime"])
    else:
        # Return an empty frame with the correct MultiIndex shape and common OHLCV columns
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["ticker", "datetime"])
        df = pd.DataFrame(index=empty_index, columns=["open", "high", "low", "close", "adj_close", "volume"])

    if return_failed:
        return df, failed
    return df
