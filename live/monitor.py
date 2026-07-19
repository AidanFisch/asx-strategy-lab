"""
Position-aware daily monitor (the thing you run every day).

For each recommended trading plan it produces an END-OF-DAY summary:
  * BUY    — the plan's entry fired on the latest bar and you're not already in it.
             Reports entry (today's close), the STOP level, and the target (if any).
  * SELL   — you're in an open (paper) position and it hit its stop / target /
             trailing stop, or the strategy's own sell signal fired.
  * HOLD   — open positions with their current stop level and unrealised P&L.

Everything is EOD and close-based, matching the backtest: you act at the close.
It tracks hypothetical positions in the DB so it knows what to tell you to sell;
it never places orders. Stops shown are the levels YOU place with your broker.

Regime gate: new BUY signals are suppressed while the ticker's market index is
below its 200-day MA (bear regime) — the backtested overlay that halved the
book's drawdown at the same return. Suppressed setups are still listed in the
summary so you can see them; exits/stops are NEVER gated. --no-regime disables.

Usage
-----
    py -m live.monitor --refresh              # update data, then produce the summary
    py -m live.monitor --dry-run              # don't send Telegram
    py -m live.monitor --all-plans            # monitor every plan, not just 'recommended'
    py -m live.monitor --no-regime            # emit buys even in bear regimes
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import dataio
import download_data
from strategies.registry import STRATEGIES
from strategies import primitives as P
from backtest import engine2
from notify import telegram_bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("monitor")


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------
def _ensure_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS positions (
        ticker TEXT PRIMARY KEY, strategy TEXT, params TEXT, exit_policy TEXT,
        entry_date TEXT, entry_price REAL, stop_level REAL, target_level REAL,
        trail_pct REAL, hi_water REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS trades (
        ticker TEXT, strategy TEXT, params TEXT, exit_policy TEXT,
        entry_date TEXT, entry_price REAL, exit_date TEXT, exit_price REAL,
        pnl_pct REAL, reason TEXT, logged_at TEXT)""")
    # additive migrations (older DBs lack these columns)
    for tbl, col, typ in [("positions", "pos_value", "REAL"),
                          ("trades", "commission", "REAL"),
                          ("trades", "pnl_pct_net", "REAL")]:
        try:
            con.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # column already exists


def _load_plans(all_plans: bool) -> pd.DataFrame:
    """Plans come from the research leaderboard DB (not the live-state DB)."""
    try:
        pcon = sqlite3.connect(config.LEADERBOARD_DB)
        try:
            plans = pd.read_sql("SELECT * FROM plans", pcon)
        finally:
            pcon.close()
    except Exception:
        return pd.DataFrame()
    if plans.empty:
        return plans
    if not all_plans and "recommended" in plans.columns:
        rec = plans[plans["recommended"] == 1]
        return rec if not rec.empty else plans
    return plans


# ---------------------------------------------------------------------------
# Market regime (bull = index above its 200-day MA)
# ---------------------------------------------------------------------------
def current_regimes() -> dict:
    """{market: True/False} — latest regime per market; empty dict on failure."""
    try:
        import regime as regime_mod
        return {m: bool(s.iloc[-1]) for m, s in regime_mod.market_regimes().items() if len(s)}
    except Exception as e:
        log.warning("regime check unavailable (%s); buys not gated this run", e)
        return {}


def market_of(ticker: str) -> str:
    import regime as regime_mod
    return regime_mod.market_of(ticker)


def suggest_size(entry: float, stop) -> dict | None:
    """
    Equal-risk position size, commission-aware (CommSec tiers).

    1. Base size risks RISK_PER_TRADE of CAPITAL over the stop distance
       (no-stop plans: flat FALLBACK_POSITION_FRAC slice).
    2. If round-trip commission would exceed MAX_FEE_DRAG_RT of the position,
       bump the size UP to the cheapest fee-efficient value and report the
       true risk that implies. Never suggests more than CAPITAL. Advisory only.
    """
    import brokerage
    if not entry or entry <= 0:
        return None
    stop_frac = (entry - stop) / entry if (stop is not None and stop < entry) else None
    if stop_frac:
        risk_amt = config.CAPITAL * config.RISK_PER_TRADE
        value = risk_amt / stop_frac
        basis = f"risks ~{config.RISK_PER_TRADE:.0%} of ${config.CAPITAL:,.0f}"
    else:
        value = config.CAPITAL * config.FALLBACK_POSITION_FRAC
        basis = f"flat {config.FALLBACK_POSITION_FRAC:.0%} slice (no fixed stop)"

    bumped = False
    if brokerage.round_trip_drag(value) > config.MAX_FEE_DRAG_RT:
        value = brokerage.min_value_for_drag(config.MAX_FEE_DRAG_RT, at_least=value)
        bumped = True
    value = min(value, config.CAPITAL)

    shares = int(value / entry)
    if shares <= 0:
        return None
    value = shares * entry
    comm = brokerage.commission(value)
    out = {
        "shares": shares, "value": value, "basis": basis,
        "commission": comm, "rt_drag": brokerage.round_trip_drag(value),
        "bumped": bumped,
    }
    if stop_frac:
        out["risk_pct"] = value * stop_frac / config.CAPITAL   # true risk after any bump
    return out


# ---------------------------------------------------------------------------
# Stop / target levels from an exit policy
# ---------------------------------------------------------------------------
def levels_for(policy: str, entry_price: float, atr_at_entry: float):
    """Return (stop_level, target_level, trail_pct) for a plan's exit policy."""
    cfg = engine2.EXIT_CONFIGS.get(policy, {})
    stop = target = trail = None
    if "atr_mult" in cfg and atr_at_entry and not np.isnan(atr_at_entry):
        stop = entry_price - cfg["atr_mult"] * atr_at_entry
    elif "sl" in cfg:
        stop = entry_price * (1 - cfg["sl"])
        if cfg.get("trail"):
            trail = cfg["sl"]
    if "tp" in cfg:
        target = entry_price * (1 + cfg["tp"])
    return stop, target, trail


# ---------------------------------------------------------------------------
# Per-plan evaluation on the latest bar
# ---------------------------------------------------------------------------
def _latest(df):
    return df.index[-1], float(df["Close"].iloc[-1])


def check_entry(strat, params, df) -> bool:
    e = strat.entry(df, **params).reindex(df.index).fillna(False).astype(bool)
    return bool(e.iloc[-1])


def check_signal_exit(strat, params, df) -> bool:
    x = strat.exit(df, **params).reindex(df.index).fillna(False).astype(bool)
    return bool(x.iloc[-1])


def evaluate_open(pos, strat, params, df):
    """Return (should_sell, reason, updated_stop, hi_water) for an open position at today's close."""
    ts, close = _latest(df)
    hi_water = pos["hi_water"] if pos["hi_water"] is not None else pos["entry_price"]
    stop = pos["stop_level"]
    trail = pos["trail_pct"]
    target = pos["target_level"]

    if trail and not (stop is None):
        hi_water = max(hi_water, close)
        stop = hi_water * (1 - trail)

    reason = None
    if stop is not None and close <= stop:
        reason = "stop hit"
    elif target is not None and close >= target:
        reason = "target hit"
    elif check_signal_exit(strat, params, df):
        reason = "sell signal"
    return (reason is not None), reason, stop, hi_water


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(interval="1d", refresh=False, dry_run=False, all_plans=False, use_regime=True, out=None):
    con = sqlite3.connect(config.SIGNALS_DB)
    _ensure_tables(con)
    plans = _load_plans(all_plans)
    if plans.empty:
        log.warning("no plans found — run  py -m plans  after the scan completes.")
        con.close()
        return {"buys": [], "sells": [], "holds": [], "suppressed": []}

    plans = plans[plans.get("interval", interval) == interval] if "interval" in plans.columns else plans
    tickers = sorted(plans["ticker"].unique())
    log.info("monitoring %d plans (interval=%s, refresh=%s)", len(plans), interval, refresh)

    if refresh:
        download_data.run(list(tickers), interval)

    regimes = current_regimes() if use_regime else {}
    if regimes:
        log.info("market regimes: %s", ", ".join(f"{m}={'bull' if v else 'BEAR'}"
                                                 for m, v in sorted(regimes.items())))

    today = datetime.now(timezone.utc).isoformat()
    buys, sells, holds, suppressed = [], [], [], []

    open_rows = {r["ticker"]: dict(r) for _, r in
                 pd.read_sql("SELECT * FROM positions", con).iterrows()} if _has_rows(con, "positions") else {}

    for _, plan in plans.iterrows():
        ticker = plan["ticker"]
        strat = STRATEGIES.get(plan["strategy"])
        if strat is None:
            continue
        params = engine2.parse_params(plan["params"])
        data = dataio.load(ticker, interval)
        if data is None:
            continue
        df = engine2.clean_ohlcv(data)
        if df.shape[0] < 60:
            continue
        ts, close = _latest(df)

        if ticker in open_rows:
            pos = open_rows[ticker]
            sell, reason, new_stop, hi_water = evaluate_open(pos, strat, params, df)
            if sell:
                import brokerage
                pnl = close / pos["entry_price"] - 1.0
                pos_value = pos.get("pos_value")
                if pos_value:
                    comm = brokerage.commission(pos_value) + brokerage.commission(pos_value * (1 + pnl))
                    pnl_net = pnl - comm / pos_value
                else:
                    comm, pnl_net = None, pnl
                con.execute("""INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (ticker, pos["strategy"], pos["params"], pos["exit_policy"],
                             pos["entry_date"], pos["entry_price"], ts.isoformat(), close,
                             pnl, reason, today, comm, pnl_net))
                con.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
                sells.append({"ticker": ticker, "strategy": pos["strategy"], "reason": reason,
                              "entry_price": pos["entry_price"], "exit_price": close,
                              "pnl": pnl, "pnl_net": pnl_net, "commission": comm})
            else:
                # update trailing stop / hi-water
                con.execute("UPDATE positions SET stop_level=?, hi_water=? WHERE ticker=?",
                            (new_stop, hi_water, ticker))
                holds.append({"ticker": ticker, "strategy": pos["strategy"],
                              "entry_price": pos["entry_price"], "close": close,
                              "stop_level": new_stop, "pnl": close / pos["entry_price"] - 1.0})
        else:
            if check_entry(strat, params, df):
                mkt = market_of(ticker)
                if regimes and regimes.get(mkt) is False:
                    # bear regime: surface the setup but don't signal a buy
                    suppressed.append({"ticker": ticker, "strategy": plan["strategy"],
                                       "market": mkt, "entry_price": close})
                    log.info("  suppressed BUY %s (%s in bear regime)", ticker, mkt)
                    continue
                atr_at_entry = float(P.atr(df, 14).iloc[-1])
                stop, target, trail = levels_for(plan["exit_policy"], close, atr_at_entry)
                size = suggest_size(close, stop)
                con.execute("""INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                            (ticker, plan["strategy"], plan["params"], plan["exit_policy"],
                             ts.isoformat(), close, stop, target, trail, close,
                             size["value"] if size else None))
                buys.append({"ticker": ticker, "strategy": plan["strategy"],
                             "exit_policy": plan["exit_policy"], "entry_price": close,
                             "stop_level": stop, "target_level": target,
                             "entry_rule": plan.get("entry_rule", ""),
                             "stop_rule": plan.get("stop_rule", ""),
                             "size": size})
    con.commit()
    con.close()

    summary = {"buys": buys, "sells": sells, "holds": holds,
               "suppressed": suppressed, "regimes": regimes, "asof": ts.isoformat()}
    _report(summary, dry_run, out=out)
    return summary


def _has_rows(con, table):
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
    except Exception:
        return False


def format_summary(s) -> str:
    d = s["asof"][:10]
    lines = [f"<b>ASX Strategy Lab — daily summary {d}</b>"]
    if s["buys"]:
        lines.append(f"\n🟢 <b>BUY ({len(s['buys'])})</b>")
        for b in s["buys"]:
            tgt = f" · target {b['target_level']:.2f}" if b.get("target_level") else ""
            stop = f"{b['stop_level']:.2f}" if b.get("stop_level") is not None else "n/a"
            sz = b.get("size")
            szl = ""
            if sz:
                szl = (f"\n   size: ~{sz['shares']} shares (≈${sz['value']:,.0f}; {sz['basis']})"
                       f"\n   commission ≈ ${sz['commission']:.2f}/side (CommSec), "
                       f"{sz['rt_drag']:.2%} round-trip")
                if sz.get("bumped"):
                    szl += (f"\n   ⚠ size bumped up for fee efficiency — true risk "
                            f"≈ {sz.get('risk_pct', 0):.1%} of capital")
            lines.append(f"• <b>{b['ticker']}</b> {b['strategy']} @ {b['entry_price']:.2f} — "
                         f"STOP {stop}{tgt}{szl}\n   <i>{b.get('entry_rule','')}</i>")
    if s["sells"]:
        lines.append(f"\n🔴 <b>SELL ({len(s['sells'])})</b>")
        for x in s["sells"]:
            net = (f", {x['pnl_net']:+.1%} after ~${x['commission']:.0f} commission"
                   if x.get("commission") else "")
            lines.append(f"• <b>{x['ticker']}</b> {x['strategy']} @ {x['exit_price']:.2f} — "
                         f"{x['reason']} ({x['pnl']:+.1%}{net})")
    if s["holds"]:
        lines.append(f"\n⚪ <b>HOLDING ({len(s['holds'])})</b>")
        for h in s["holds"]:
            stop = f"{h['stop_level']:.2f}" if h.get("stop_level") is not None else "n/a"
            lines.append(f"• {h['ticker']} {h['strategy']} — now {h['close']:.2f} "
                         f"({h['pnl']:+.1%}), stop {stop}")
    if s.get("suppressed"):
        lines.append(f"\n🚫 <b>SUPPRESSED — bear regime ({len(s['suppressed'])})</b>")
        for x in s["suppressed"]:
            lines.append(f"• {x['ticker']} {x['strategy']} fired @ {x['entry_price']:.2f} — "
                         f"{x['market']} index below its 200-day MA, sitting out")
    if not (s["buys"] or s["sells"] or s["holds"] or s.get("suppressed")):
        lines.append("\nNo actions today — nothing triggered and no open positions.")
    reg = s.get("regimes") or {}
    if reg:
        bears = [m for m, v in sorted(reg.items()) if not v]
        lines.append(f"\nRegimes: {'all bull' if not bears else 'BEAR: ' + ', '.join(bears)}")
    lines.append("\n<i>Paper-trading decision support, not advice. You place the orders & stops.</i>")
    return "\n".join(lines)


def _report(summary, dry_run, out=None):
    text = format_summary(summary)
    plain = (text.replace("<b>", "**").replace("</b>", "**")
                 .replace("<i>", "_ ").replace("</i>", " _"))
    console = (text.replace("<b>", "").replace("</b>", "")
                   .replace("<i>", "").replace("</i>", ""))
    log.info("BUY=%d SELL=%d HOLD=%d SUPPRESSED=%d", len(summary["buys"]),
             len(summary["sells"]), len(summary["holds"]), len(summary.get("suppressed", [])))
    print("\n" + console)
    if out:
        # markdown summary for the notification step (GitHub issue body)
        from pathlib import Path
        Path(out).write_text(plain, encoding="utf-8")
        log.info("summary written to %s", out)
    if not dry_run:
        sent = telegram_bot.send_message(text)
        log.info("telegram: %s", "sent" if sent else "dry-run/not configured")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Position-aware daily EOD monitor.")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all-plans", action="store_true")
    ap.add_argument("--no-regime", action="store_true", help="emit buys even in bear regimes")
    ap.add_argument("--out", default=None, help="write the markdown summary to this file")
    args = ap.parse_args(argv)
    run(interval=args.interval, refresh=args.refresh, dry_run=args.dry_run,
        all_plans=args.all_plans, use_regime=not args.no_regime, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
