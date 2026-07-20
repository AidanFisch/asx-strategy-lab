"""
Signal primitives — the building blocks strategies are composed from.

Two kinds of helper:
  * indicators: return a numeric Series (sma, rsi, atr, zscore, ...)
  * conditions: return a boolean Series (cross_up, above, oversold, breakout,
    volume_surge, uptrend, ...)

Strategies in registry.py combine these — e.g. an RSI oversold trigger AND a
Bollinger lower-band touch AND an "above the 200-day MA" price-check filter.

Everything is vectorised pandas so it works bar-by-bar without lookahead
(rolling windows use only past/current data; breakout levels use .shift(1)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator

# ---------------------------------------------------------------------------
# Indicators (numeric Series)
# ---------------------------------------------------------------------------
def sma(close, n):
    return close.rolling(n).mean()

def ema(close, n):
    return close.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    return vbt.RSI.run(close, n).rsi

def atr(data, n=14):
    """Average True Range (Wilder)."""
    h, l, c = data["High"], data["Low"], data["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def bbands(close, n=20, mult=2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=0)
    return mid - mult * sd, mid, mid + mult * sd   # lower, mid, upper

def zscore(close, n=20):
    m = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return (close - m) / sd

def roc(close, n=20):
    return close.pct_change(n) * 100.0

def adx(data, n=14):
    return ADXIndicator(data["High"], data["Low"], data["Close"], window=n, fillna=False).adx()

def stoch(data, n=14, smooth=3):
    return StochasticOscillator(data["High"], data["Low"], data["Close"],
                                window=n, smooth_window=smooth, fillna=False).stoch()

def rolling_high(data, n):
    """Highest HIGH of the prior n bars (excludes current bar -> no lookahead)."""
    return data["High"].rolling(n).max().shift(1)

def rolling_low(data, n):
    return data["Low"].rolling(n).min().shift(1)


# --- swing pivots -> real support/resistance levels (lookahead-safe) ----------
def swing_low_level(data, w=5):
    """
    Price of the most recent CONFIRMED swing-low support, usable without lookahead.

    A swing low at bar i (Low[i] is the min of the centred 2w+1 window) can only
    be *confirmed* w bars later, once the right shoulder exists. So we detect the
    pivot, then shift the level forward by w bars and forward-fill — meaning at any
    bar t you only ever see swing lows that were fully formed by bar t. This is the
    difference between real support (a level price returned to) and a rolling min.
    """
    low = data["Low"]
    win = 2 * w + 1
    is_piv = low == low.rolling(win, center=True).min()
    return low.where(is_piv).shift(w).ffill()

def swing_high_level(data, w=5):
    """Most recent CONFIRMED swing-high resistance (lookahead-safe, see swing_low_level)."""
    high = data["High"]
    win = 2 * w + 1
    is_piv = high == high.rolling(win, center=True).max()
    return high.where(is_piv).shift(w).ffill()


# ---------------------------------------------------------------------------
# Conditions (boolean Series)
# ---------------------------------------------------------------------------
def _prev(x, a):
    """Previous value of b, whether b is a Series or a scalar level."""
    return x.shift(1) if isinstance(x, pd.Series) else x

def cross_up(a, b):
    """`a` crosses above `b` (b may be a Series or a scalar threshold)."""
    return (a > b) & (a.shift(1) <= _prev(b, a))

def cross_down(a, b):
    return (a < b) & (a.shift(1) >= _prev(b, a))

def above(a, b):
    return a > b

def below(a, b):
    return a < b

def rising(series, k=1):
    return series > series.shift(k)

def falling(series, k=1):
    return series < series.shift(k)


# --- price-check / confirmation filters (the "consistency" guards) ---------
def uptrend(close, fast=50, slow=200):
    """Longer-term uptrend: fast SMA above slow SMA."""
    return sma(close, fast) > sma(close, slow)

def above_sma(close, n=200):
    return close > sma(close, n)

def volume_surge(data, mult=1.5, n=20):
    """Today's volume above `mult`x its n-day average — confirms conviction."""
    v = data["Volume"]
    return v > mult * v.rolling(n).mean()

def not_extended(close, n=20, max_pct=0.10):
    """Price not more than max_pct above its n-day SMA (avoid chasing blow-offs)."""
    return close <= sma(close, n) * (1 + max_pct)

def breakout_high(data, n=20):
    """Close breaks above the prior n-bar high."""
    return data["Close"] > rolling_high(data, n)

def breakdown_low(data, n=20):
    return data["Close"] < rolling_low(data, n)

def new_high(data, n=252):
    """Close makes a new n-bar high (incl. current) — e.g. 252 ~ 52-week high."""
    return data["Close"] >= data["Close"].rolling(n).max()

def gap_up(data, pct=0.02):
    """Open gaps up more than pct above prior close."""
    return data["Open"] > data["Close"].shift(1) * (1 + pct)


def all_true(*conds):
    """AND a set of boolean Series, aligning on index and treating NaN as False."""
    out = None
    for c in conds:
        c = c.fillna(False).astype(bool)
        out = c if out is None else (out & c)
    return out

def any_true(*conds):
    out = None
    for c in conds:
        c = c.fillna(False).astype(bool)
        out = c if out is None else (out | c)
    return out
