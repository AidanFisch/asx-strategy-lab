# Strategy Lab — ASX + Asia

A free-tools-only system to research equity trading strategies: pull daily price
data, backtest 22 strategies (with risk-managed stops/targets) across the **ASX 200
plus liquid Asian large-caps** (Hong Kong, Japan, Korea, Taiwan, Singapore, India),
rank the results out-of-sample, and produce a daily BUY/SELL/HOLD signal summary.

**📊 Live dashboard:** https://aidanfisch.github.io/asx-strategy-lab/

> Research/education only — **not financial advice**. Backtested edges decay; you
> place any orders and stops yourself. No live broker execution.

## Status
- **Phase 1 — Data pipeline: ✅** (daily + intraday download, incremental Parquet cache)
- **Phase 2 — Backtest engine: ✅** (metrics, fees/slippage, daily & intraday)
- **Phase 3 — Scanner + leaderboard: ✅** (grid search, in/out-of-sample split, SQLite)
- **Phase 4 — Paper trading + Telegram: ✅** (signal scan, alerts, scheduling)
- **Phase 6 — Full trading system (v2): ✅** — 22 strategies incl. composites, risk-managed
  exits (stops/targets/trailing), walk-forward validation, per-ticker trade plans,
  position-aware daily BUY/SELL/HOLD monitor, 3-view dashboard.
- Phase 5 — Real broker execution: not started (deliberately deferred; you trade manually).

## v2 system at a glance
The v2 system (files below) supersedes the Phase 2–4 v1 modules but keeps them intact.

**Two-tier cadence:**
- **Weekly rescan** (heavy, ~1–2h): `run_rescan.cmd` → refresh data → `backtest.scanner2`
  (every ticker × strategy × params × exit-policy → `runs2`) → `plans.py` (best plan per
  ticker + walk-forward) → `results.dashboard2`.
- **Daily monitor** (fast): `run_scan.cmd` → `live.monitor --refresh` → refresh plan tickers,
  emit **BUY / SELL / HOLD** summary + Telegram. This is the "tell me what to do today" step.

```bash
# one-time / weekly:
py download_data.py --interval 1d --universe   # full ASX200 daily
py -m backtest.scanner2 --interval 1d          # full risk-managed grid -> runs2
py -m plans --interval 1d                       # best plan per ticker (+ walk-forward) -> plans
py -m results.dashboard2 --interval 1d          # dashboard.html + all_results.csv + strategy_summary.csv

# daily:
py -m live.monitor --interval 1d --refresh      # BUY / SELL / HOLD summary + Telegram
```

### Strategies (22, five families)
Momentum (`ma_crossover`, `macd_momentum`, `adx_trend`, `roc_momentum`), mean-reversion
(`rsi_reversion`, `rsi2_pullback`, `bb_reversion`, `zscore_reversion`, `pct_below_ma`),
breakout-with-confirmation (`donchian_breakout_vol`, `high_52w_breakout`, `bb_breakout_vol`,
`atr_channel_breakout`, `nbar_high_confirmed`), price-action (`turtle_breakout`,
`pullback_uptrend`, `gap_up_go`, `support_bounce`), and **composites** that stack conditions
for consistency (`rsi_bb_confluence`, `triple_confluence_breakout`, `rsi_macd_confluence`,
`bb_squeeze_breakout`). Built from `strategies/primitives.py`; defined in `strategies/registry.py`.

### Risk-managed exits (`backtest/engine2.py`)
Every strategy is tested with 6 exit policies: `signal_only`, `sl_10`, `sl_15`, `trail_10`,
`sl10_tp20`, `atr_2x` (2×ATR stop). Position closes on whichever fires first — sell signal,
stop, target, or trailing stop. Extra risk metrics: Sortino, Calmar, payoff, expectancy(R),
exposure, avg win/loss.

### Honest selection + walk-forward (`plans.py`)
Best plan per ticker is **chosen on in-sample** Sharpe and **reported out-of-sample**. Each
pick also gets a **walk-forward** check across contiguous folds; `wf_consistency` = share of
folds profitable. `recommended` = held up OOS **and** beat buy&hold **and** consistent.

### Validation & analysis modules
Run after `plans.py` (all included in `run_rescan.cmd`):
- `robustness.py` — Monte Carlo confidence intervals, probabilistic Sharpe (edge-is-real
  probability), per-year/regime breakdown, multiple-testing context → `robustness` table.
- `wfo.py` — **walk-forward optimization**: re-pick the best strategy each window, trade the
  next unseen one (the strictest test) → `wfo` table.
- `liquidity.py` — Average Daily Volume in AUD (via FX) + tradeability tiers → `liquidity` table.
- `portfolio.py` — treats plans as one **diversified book** (equal-risk sleeves); the real case
  for the system (individually plans lag buy&hold, but the book has far better Sharpe / smaller
  drawdown; avg sleeve correlation ≈ 0.06) → `portfolio` table.
- `regime.py` — market-regime overlay (sit in cash when a market's index is below its 200-day
  MA); improves book Sharpe and roughly halves drawdown → `regime` table.
- `tracker.py` — forward paper-trading scorecard (realized vs backtested), fills as the monitor runs.

Deliberately **not** built: ML meta-filter / Qlib — on a thin underlying edge they mostly
manufacture overfit confidence; the transparent regime rule delivers the risk-reduction instead.

### Daily monitor (`live/monitor.py`)
Position-aware, EOD, close-based (matches the backtest). Tracks paper positions in `positions`
/ `trades` tables. Each run: BUY (with entry, **stop level**, target), SELL (with reason:
stop/target/trailing/signal), HOLD (open positions + current stop + unrealised P&L). You place
the actual orders and stops — it never trades.

## Setup
```bash
py -m pip install -r requirements.txt   # Windows; use python3 elsewhere
```
Python 3.10+ (dev machine runs 3.10.4 via the `py` launcher).

## Layout
```
asx-strategy-lab/
├── config.py            # interval selection + yfinance lookback limits + paths
├── download_data.py     # Phase 1 data pipeline (this phase)
├── data/
│   ├── universe.csv     # ASX ticker list (starter ASX 200)
│   └── raw/<interval>/<TICKER>.parquet   # cached OHLCV
├── strategies/          # (Phase 2)
├── backtest/            # (Phase 2/3)
├── results/             # (Phase 3)
├── notify/              # (Phase 4)
└── live/                # (Phase 4)
```

## Data pipeline (Phase 1)

Downloads daily **or** intraday OHLCV from yfinance and caches each
ticker/interval as Parquet under `data/raw/<interval>/`. Re-runs only fetch bars
newer than the last cached one and append them.

```bash
# Daily, full universe
py download_data.py --interval 1d

# Intraday smoke test on the first 5 universe tickers
py download_data.py --interval 15m --limit 5

# Explicit tickers
py download_data.py --interval 1d --tickers BHP.AX CBA.AX CSL.AX
```

Interval is read from `config.INTERVAL` unless `--interval` is passed.

### ⚠️ yfinance free intraday data limits
Yahoo only serves a short **rolling window** of intraday history:

| Interval        | History available |
|-----------------|-------------------|
| `1m`            | ~7 days           |
| `2m/5m/15m/30m/90m` | ~60 days      |
| `60m` / `1h`    | ~730 days         |
| `1d` and coarser| full history      |

These limits are encoded in `config.INTERVAL_LIMITS`. **Implication:** for
intraday timeframes you cannot pull years of history in one go. The pipeline is
built to accumulate history incrementally — run it on a schedule so the local
Parquet cache grows past Yahoo's rolling window over time. Daily data has no such
limit (full history back to `config.DAILY_START`, default 2005).

### Notes / caveats
- OHLC is split/dividend-adjusted (`auto_adjust=True`).
- Index is timezone-aware (`Australia/Sydney`).
- Intraday data can include zero-volume bars (opening auction / illiquid periods)
  — worth filtering in Phase 2.
- Missing/delisted tickers are logged and skipped, never crash the run.

## Backtest engine (Phase 2)

Five parameterised strategies live in `strategies/` (each exposes `NAME`,
`DEFAULT_PARAMS`, `PARAM_GRID`, and `generate_signals(data, **params)`):

| Strategy | Idea | Params |
|---|---|---|
| `ma_crossover` | fast MA crosses slow MA | fast, slow |
| `rsi_reversion` | buy oversold / sell overbought | period, oversold, overbought |
| `bollinger_breakout` | break above upper band, exit at middle | window, mult |
| `macd_momentum` | MACD line crosses signal line | fast, slow, signal |
| `donchian_breakout` | break prior-N-bar high/low | window |

`backtest/engine.py` runs one strategy+params on one ticker and returns metrics:
total return, CAGR, Sharpe, max drawdown, win rate, number of trades, profit
factor, avg P&L per trade (plus a buy & hold benchmark).

```bash
py validate_phase2.py     # all strategies on BHP/CBA daily + a sample trade log
```

**Execution model:** long-only, all-in/all-out, fill at the signal bar's close,
with `config.FEES` (0.1%) + `config.SLIPPAGE` (5 bps) on every fill. See the
docstring in `backtest/engine.py` for the lookahead and annualisation caveats
(intraday Sharpe/CAGR are distorted by exchange-hours-only bars — compare within
the same interval).

## Scanner + leaderboard (Phase 3)

`backtest/scanner.py` backtests every **ticker × strategy × parameter combo** and
stores results in `results/leaderboard.db` (SQLite, table `runs`).

**Overfitting guard:** each ticker's history is split by time into an in-sample
(first 70%, `config.IS_FRACTION`) and an out-of-sample holdout (last 30%). Every
combo is scored on *both*; the DB stores `is_*` and `oos_*` metrics side by side.

```bash
py -m backtest.scanner --interval 1d                 # scan cached tickers
py -m results.query --interval 1d --top 20           # rank by in-sample Sharpe
py -m results.query --sort oos_sharpe --top 20       # rank by out-of-sample
```

`results/query.py` filters to combos with `>= MIN_TRADES` in both periods and
flags `oos_holds` (still profitable + positive Sharpe out-of-sample). In testing,
the top in-sample performers mostly *collapsed* out-of-sample — a live reminder
that the best backtest is usually overfit.

## Paper trading + Telegram (Phase 4)

`live/daily_scan.py` is the scheduled entrypoint. It (optionally) refreshes data,
regenerates signals for each watchlist combo, and if the **latest bar** fired a
fresh BUY/SELL it logs the signal (`signals` table, deduped) and sends a Telegram
alert. It never places orders — signals are recorded so you can track hypothetical
performance before risking real money.

```bash
py -m live.build_watchlist --interval 1d   # generate watchlist.json from OOS-robust combos
py -m live.daily_scan --interval 1d --refresh          # refresh + scan + alert
py -m live.daily_scan --interval 1d --dry-run          # scan + log, never send
```

### Telegram setup (free, ~5 min)
1. Message `@BotFather` → `/newbot` → copy the bot token.
2. Message your new bot once, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and read your `chat_id`.
3. Set env vars (open a fresh terminal afterwards):
   ```
   setx TELEGRAM_BOT_TOKEN "<token>"
   setx TELEGRAM_CHAT_ID   "<id>"
   ```
   Without these, alerts run in **dry-run** (printed, not sent) so everything else
   still works. Test with: `py -m notify.telegram_bot`.

### Scheduling
- **Local (Windows Task Scheduler)** — simplest, and best for intraday since the
  incremental cache persists on disk:
  ```
  schtasks /Create /TN "ASX daily scan" /TR "%CD%\run_scan.cmd" /SC DAILY /ST 17:00
  ```
  (`run_scan.cmd` refreshes + scans and logs to `logs/daily_scan.log`.)
- **GitHub Actions** — `.github/workflows/daily_scan.yml` runs on cron even with
  your PC off, and commits the refreshed cache back to the repo so it persists.
  Add `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` as repo secrets. Heavier for large
  universes; prefer local scheduling for frequent intraday runs.

> ⚠️ **Intraday history must be accumulated.** Yahoo only serves a short rolling
> window (see the table above), so schedule the scan to run regularly — the cache
> grows past that window only if you keep fetching. A single historical pull can't
> give you deep intraday history.

## Guardrails (built in from day one)
- Out-of-sample holdout on every scan — never just the best-looking backtest.
- Everything logged (data, params, IS+OOS metrics, signals) for reproducibility.
- Realistic frictions (brokerage + slippage) on every fill.
- Research/education tool — **not** financial advice, and past backtests can mislead.
