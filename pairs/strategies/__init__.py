# pairs/strategies/__init__.py
from .signals import estimate_halflife_window, zscore_from_spread, generate_pair_signals
from .evaluate import evaluate_pair_signals

__all__ = [
    "estimate_halflife_window",
    "zscore_from_spread",
    "generate_pair_signals",
    "evaluate_pair_signals",
]
