"""
Validate the v2 consumer logic that does NOT need the scan table:
  * plans.walk_forward  (real backtest across folds)
  * monitor.levels_for  (stop/target math per exit policy)
  * monitor.evaluate_open (stop / target / trailing / signal exit detection)
  * monitor.format_summary (alert rendering)
Runs against real BHP prices. Does not touch runs2 / the running scan.
"""

import numpy as np
import pandas as pd

import dataio
from strategies.registry import get
from strategies import primitives as P
from backtest import engine2
import plans as plans_mod
from live import monitor

df = engine2.clean_ohlcv(dataio.load("BHP.AX", "1d"))
close = float(df["Close"].iloc[-1])
atr = float(P.atr(df, 14).iloc[-1])
print(f"BHP last close={close:.2f}  ATR14={atr:.2f}\n")

# 1) levels_for for each exit policy
print("== levels_for (entry=100.00, ATR=%.2f) ==" % atr)
for pol in engine2.EXIT_CONFIGS:
    stop, target, trail = monitor.levels_for(pol, 100.0, atr)
    print(f"  {pol:12} stop={stop if stop is None else round(stop,2)}  "
          f"target={target if target is None else round(target,2)}  trail={trail}")
assert monitor.levels_for("sl_10", 100.0, atr)[0] == 90.0
assert monitor.levels_for("sl10_tp20", 100.0, atr)[1] == 120.0
assert abs(monitor.levels_for("atr_2x", 100.0, atr)[0] - (100.0 - 2 * atr)) < 1e-6
print("  OK\n")

# 2) walk_forward on a real plan
print("== walk_forward: rsi_reversion sl_10 on BHP ==")
wf = plans_mod.walk_forward("BHP.AX", "rsi_reversion", "period=14,oversold=30,overbought=70",
                            "sl_10", "1d", folds=4)
print(" ", wf)
assert wf["wf_folds"] >= 3 and 0 <= wf["wf_consistency"] <= 1
print("  OK\n")

# 3) evaluate_open — construct scenarios
strat = get("rsi_reversion"); params = dict(period=14, oversold=30, overbought=70)
def mkpos(**kw):
    base = dict(ticker="BHP.AX", strategy="rsi_reversion",
                params="period=14,oversold=30,overbought=70", exit_policy="sl_10",
                entry_date="2020-01-01", entry_price=close * 0.9, stop_level=None,
                target_level=None, trail_pct=None, hi_water=close * 0.9)
    base.update(kw); return base

print("== evaluate_open scenarios (close=%.2f) ==" % close)
# stop hit: stop just above close
sell, reason, *_ = monitor.evaluate_open(mkpos(stop_level=close + 1), strat, params, df)
print(f"  stop above close      -> sell={sell} reason={reason}"); assert sell and reason == "stop hit"
# target hit: target just below close
sell, reason, *_ = monitor.evaluate_open(mkpos(target_level=close - 1), strat, params, df)
print(f"  target below close    -> sell={sell} reason={reason}"); assert sell and reason == "target hit"
# trailing: hi_water high so stop = hi*(1-0.1); ensure stop recomputed upward from hi_water
sell, reason, new_stop, hw = monitor.evaluate_open(
    mkpos(exit_policy="trail_10", trail_pct=0.10, stop_level=close * 0.5, hi_water=close * 0.8),
    strat, params, df)
exp_stop = max(close * 0.8, close) * 0.9
print(f"  trailing recompute    -> stop={new_stop:.2f} (expect {exp_stop:.2f}) sell={sell}")
assert abs(new_stop - exp_stop) < 1e-6
# no trigger: stop far below, no target
sell, reason, *_ = monitor.evaluate_open(mkpos(stop_level=close * 0.5), strat, params, df)
print(f"  stop far below close  -> sell={sell} reason={reason}")
print("  OK\n")

# 4) format_summary rendering
summary = {
    "asof": df.index[-1].isoformat(),
    "buys": [{"ticker": "BHP.AX", "strategy": "rsi_bb_confluence", "exit_policy": "atr_2x",
              "entry_price": close, "stop_level": close - 2 * atr, "target_level": None,
              "entry_rule": "RSI<30 AND below lower BB AND above SMA200", "stop_rule": "2xATR stop"}],
    "sells": [{"ticker": "CBA.AX", "strategy": "ma_crossover", "reason": "stop hit",
               "entry_price": 100.0, "exit_price": 92.0, "pnl": -0.08}],
    "holds": [{"ticker": "CSL.AX", "strategy": "donchian_breakout_vol", "entry_price": 200.0,
               "close": 230.0, "stop_level": 207.0, "pnl": 0.15}],
}
print("== format_summary preview ==")
print(monitor.format_summary(summary))
print("\nVALIDATE v2 LOGIC: PASSED")
