"""
RSI Mean Reversion — buy when RSI drops into oversold territory, sell when it
climbs into overbought territory. Bets on price snapping back to the mean.

Entry: RSI crosses DOWN through the oversold threshold.
Exit:  RSI crosses UP through the overbought threshold.
"""

import vectorbt as vbt

NAME = "rsi_reversion"
DEFAULT_PARAMS = {"period": 14, "oversold": 30, "overbought": 70}
PARAM_GRID = {
    "period": [7, 14, 21],
    "oversold": [20, 30],
    "overbought": [70, 80],
}


def generate_signals(data, period=14, oversold=30, overbought=70):
    close = data["Close"]
    rsi = vbt.RSI.run(close, period).rsi
    entries = rsi.vbt.crossed_below(oversold)
    exits = rsi.vbt.crossed_above(overbought)
    return entries, exits
