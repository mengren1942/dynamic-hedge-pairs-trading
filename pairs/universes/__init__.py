from __future__ import annotations
from importlib.resources import files
from pathlib import Path
from typing import Iterable

try:
    from ..utils.tickers import load_tickers as _load_tickers
except Exception:
    # minimal local fallback
    def _load_tickers(path: Path) -> pd.Index:
        suf = path.suffix.lower()
        if suf in {".txt",".list"}:
            vals = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")]
        elif suf == ".csv":
            df = pd.read_csv(path)
            col = next((c for c in ("ticker","symbol","Ticker","SYMBOL") if c in df.columns), df.columns[0])
            vals = df[col].astype(str).tolist()
        elif suf == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                vals = obj
            elif isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
                for key in ("ticker","symbol","Ticker","SYMBOL"):
                    if all(key in d for d in obj):
                        vals = [d[key] for d in obj]; break
                else:
                    raise ValueError("Unrecognized JSON format")
            else:
                raise ValueError("Unrecognized JSON format")
        else:
            raise ValueError(f"Unsupported file type: {suf}")
        s = pd.Series(vals, dtype="string").str.strip().str.upper().dropna().drop_duplicates().sort_values()
        return pd.Index(s)

__all__ = ["load_universe", "list_universes", "__UNIVERSES_VERSION__"]
__UNIVERSES_VERSION__ = "2025.08"

def _tickers_dir():
    return files(__package__) / "tickers"

def list_universes(extensions: Iterable[str] = (".txt", ".csv", ".json")) -> list[str]:
    """Names available under pairs/universes/tickers (without extension)."""
    base = _tickers_dir()
    names: set[str] = set()
    for entry in base.iterdir():
        if entry.is_file() and entry.suffix.lower() in extensions:
            names.add(entry.name[: -len(entry.suffix)])
    return sorted(names)

def load_universe(name: str):
    """
    Load a packaged ticker universe by base name (e.g., 'spx', 'ndx').
    Searches pairs/universes/tickers/<name>.(txt|csv|json).
    Returns a pandas.Index of clean, UPPERCASE tickers.
    """
    base = _tickers_dir()
    for ext in (".txt", ".csv", ".json"):
        p = base / f"{name}{ext}"
        if p.is_file():
            return _load_tickers(Path(p))
    raise FileNotFoundError(
        f"No packaged universe named {name!r}. "
        f"Available: {', '.join(list_universes()) or '(none)'}"
    )
