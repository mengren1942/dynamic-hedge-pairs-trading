# pairs/market_data/__init__.py
from __future__ import annotations
from typing import Any

__all__ = ["load_prices", "load_polygon_lake", "download_openbb"]

# --- polygon lake: ---
from .polygon_lake import load_polygonio_lake as load_polygon_lake

# --- openbb: ---
from .openbb_history import download_history_openbb as download_openbb


def load_prices(source: str, *args: Any, **kwargs: Any):
    src = source.lower()
    if src == "polygon":
        return load_polygon_lake(*args, **kwargs)
    if src == "openbb":
        return download_openbb(*args, **kwargs)
    raise ValueError(f"Unknown source: {source!r}. Expected 'polygon' or 'openbb'.")
