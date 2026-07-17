"""
MACD Momentum — go long when the MACD line crosses above its signal line,
exit when it crosses back below. Trend/momentum following.
"""

import vectorbt as vbt

NAME = "macd_momentum"
DEFAULT_PARAMS = {"fast": 12, "slow": 26, "signal": 9}
PARAM_GRID = {
    "fast": [8, 12],
    "slow": [21, 26],
    "signal": [9],
}


def generate_signals(data, fast=12, slow=26, signal=9):
    close = data["Close"]
    if fast >= slow:
        empty = close.notna() & False
        return empty, empty
    macd = vbt.MACD.run(
        close,
        fast_window=fast,
        slow_window=slow,
        signal_window=signal,
    )
    entries = macd.macd.vbt.crossed_above(macd.signal)
    exits = macd.macd.vbt.crossed_below(macd.signal)
    return entries, exits
