"""Phase 2 validation: run all strategies on a couple of tickers and show a trade log."""

import pandas as pd

import dataio
import config
from strategies import STRATEGY_MODULES
from backtest import engine

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

INTERVAL = "1d"
TICKERS = ["BHP.AX", "CBA.AX"]

METRIC_COLS = ["ticker", "strategy", "params", "n_trades", "total_return",
               "buy_hold_return", "cagr", "sharpe", "max_drawdown",
               "win_rate", "profit_factor", "avg_pnl"]

rows = []
for ticker in TICKERS:
    data = dataio.load(ticker, INTERVAL)
    for strat in STRATEGY_MODULES:
        m = engine.run_backtest(data, strat, strat.DEFAULT_PARAMS,
                                ticker=ticker, interval=INTERVAL)
        rows.append(m)

df = pd.DataFrame(rows)[METRIC_COLS]
for c in ["total_return", "buy_hold_return", "cagr", "sharpe", "max_drawdown",
          "win_rate", "profit_factor", "avg_pnl"]:
    df[c] = df[c].round(3)
print("\n=== Phase 2 metrics: all strategies, default params, daily ===")
print(df.to_string(index=False))

# --- Trade log for manual sanity check: MA crossover on BHP ---
from strategies import ma_crossover
print("\n=== Trade log: ma_crossover (fast=20, slow=50) on BHP.AX — first & last 5 ===")
tl = engine.trade_log(dataio.load("BHP.AX", INTERVAL), ma_crossover,
                      ma_crossover.DEFAULT_PARAMS, INTERVAL)
cols = [c for c in ["Entry Timestamp", "Avg Entry Price", "Exit Timestamp",
                    "Avg Exit Price", "Size", "PnL", "Return", "Status"] if c in tl.columns]
print(tl[cols].head(5).to_string(index=False))
print("...")
print(tl[cols].tail(5).to_string(index=False))
print(f"\nTotal trades: {len(tl)} | init_cash={config.INIT_CASH} fees={config.FEES} slippage={config.SLIPPAGE}")
