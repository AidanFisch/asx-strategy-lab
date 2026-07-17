"""
Donchian Channel Breakout — go long when price breaks above the highest high of
the prior N bars, exit when it breaks below the lowest low of the prior N bars.
The turtle-trading classic.

The channels use `.shift(1)` so the breakout is measured against *completed*
prior bars only — no lookahead onto the current bar's own high/low.
"""

NAME = "donchian_breakout"
DEFAULT_PARAMS = {"window": 20}
PARAM_GRID = {
    "window": [10, 20, 55],
}


def generate_signals(data, window=20):
    high, low, close = data["High"], data["Low"], data["Close"]
    upper = high.rolling(window).max().shift(1)   # prior-N-bar high
    lower = low.rolling(window).min().shift(1)     # prior-N-bar low
    entries = close > upper
    exits = close < lower
    # rolling window is NaN for the first `window` bars -> comparisons are False; good
    return entries.fillna(False), exits.fillna(False)
