"""
Bollinger Band Breakout — go long when price breaks out above the upper band
(momentum thrust), exit when it falls back through the middle band.

Entry: close crosses ABOVE the upper band (mean + mult*std).
Exit:  close crosses BELOW the middle band (the moving average).
"""

import vectorbt as vbt

NAME = "bollinger_breakout"
DEFAULT_PARAMS = {"window": 20, "mult": 2.0}
PARAM_GRID = {
    "window": [10, 20, 50],
    "mult": [1.5, 2.0, 2.5],
}


def generate_signals(data, window=20, mult=2.0):
    close = data["Close"]
    bb = vbt.BBANDS.run(close, window=window, alpha=mult)
    entries = close.vbt.crossed_above(bb.upper)
    exits = close.vbt.crossed_below(bb.middle)
    return entries, exits
