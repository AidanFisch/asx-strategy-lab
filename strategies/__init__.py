"""
Strategy registry. Each strategy module exposes a uniform interface:

    NAME            : str
    DEFAULT_PARAMS  : dict
    PARAM_GRID      : dict[str, list]   # for the Phase 3 grid search
    generate_signals(data, **params) -> (entries, exits)

STRATEGIES maps name -> module so the engine and scanner can iterate generically.
"""

from . import (
    ma_crossover,
    rsi_reversion,
    bollinger_breakout,
    macd_momentum,
    donchian_breakout,
)

STRATEGY_MODULES = [
    ma_crossover,
    rsi_reversion,
    bollinger_breakout,
    macd_momentum,
    donchian_breakout,
]

STRATEGIES = {m.NAME: m for m in STRATEGY_MODULES}


def get(name):
    return STRATEGIES[name]
