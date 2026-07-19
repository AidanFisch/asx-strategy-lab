"""
Brokerage cost model — CommSec (internet, CDIA-settled) tiered commissions.

Schedule (verify against CommSec's current rates before trading real money):
    <= $1,000            $5.00
    $1,000  - $3,000     $10.00
    $3,000  - $10,000    $19.95
    $10,000 - $25,000    $29.95
    >  $25,000           0.12% of trade value

Because the fee is a fixed dollar amount per tier, SMALL trades get eaten alive:
a $1,300 position pays $10 each way = 1.5% round trip, which is most of a typical
trade's edge. `min_value_for_drag()` finds the smallest position size whose
round-trip commission stays under a cap, so the monitor can bump sizes up to a
fee-efficient level (and tell you the resulting risk).
"""

from __future__ import annotations

TIERS = [  # (max_value_inclusive, flat_fee) — ASX internet trades
    (1_000.0, 5.00),
    (3_000.0, 10.00),
    (10_000.0, 19.95),
    (25_000.0, 29.95),
]
PCT_ABOVE = 0.0012      # 0.12% for ASX trades over $25,000

# CommSec International (non-ASX markets) is a different schedule entirely —
# approximated here as a % with a floor, in AUD terms. VERIFY against CommSec's
# current international rates; they are materially higher than ASX tiers.
INTL_PCT = 0.0031       # ~0.31% of trade value
INTL_MIN = 40.00        # minimum per trade (AUD-equivalent approximation)


def commission(value: float, market: str = "Australia") -> float:
    """One-way commission ($AUD) for a trade of this AUD value in a market."""
    if value <= 0:
        return 0.0
    if market != "Australia":
        return max(INTL_MIN, value * INTL_PCT)
    for max_v, fee in TIERS:
        if value <= max_v:
            return fee
    return value * PCT_ABOVE


def round_trip_drag(value: float, market: str = "Australia") -> float:
    """Round-trip commission as a fraction of position value (buy + sell)."""
    if value <= 0:
        return 0.0
    return 2.0 * commission(value, market) / value


def min_value_for_drag(cap: float, at_least: float = 0.0, upper: float = 60_000.0,
                       market: str = "Australia") -> float:
    """Smallest position value >= at_least whose round-trip drag <= cap."""
    v = max(at_least, 100.0)
    while v <= upper:
        if round_trip_drag(v, market) <= cap:
            return v
        v += 50.0
    return upper
