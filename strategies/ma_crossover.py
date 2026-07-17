"""
MA Crossover — go long when the fast moving average crosses above the slow MA,
exit when it crosses back below. A classic trend-following strategy.

Signal convention (shared by all strategies in this package):
    generate_signals(data, **params) -> (entries, exits)
where `data` is an OHLCV DataFrame (tz-aware DatetimeIndex) and entries/exits are
boolean Series aligned to data.index. Signals mark the bar on which the condition
becomes true; the engine fills at that bar's close (see backtest/engine.py for the
lookahead/friction assumptions).
"""

import vectorbt as vbt

NAME = "ma_crossover"
DEFAULT_PARAMS = {"fast": 20, "slow": 50}
PARAM_GRID = {
    "fast": [10, 20, 50],
    "slow": [50, 100, 200],
}


def generate_signals(data, fast=20, slow=50):
    close = data["Close"]
    if fast >= slow:
        # invalid combo; emit no trades so the grid can skip it cleanly
        empty = close.notna() & False
        return empty, empty
    fast_ma = vbt.MA.run(close, fast).ma
    slow_ma = vbt.MA.run(close, slow).ma
    entries = fast_ma.vbt.crossed_above(slow_ma)
    exits = fast_ma.vbt.crossed_below(slow_ma)
    return entries, exits
