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
    py -m live.monitor --dry-run              # report only: does NOT touch the paper book
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


def _plans_db():
    """Prefer the small committed serving.db; fall back to the full leaderboard."""
    return config.SERVING_DB if config.SERVING_DB.exists() else config.LEADERBOARD_DB


def _load_plans(all_plans: bool) -> pd.DataFrame:
    """Plans come from the serving/research DB (not the live-state DB)."""
    try:
        pcon = sqlite3.connect(_plans_db())
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
        plans = rec if not rec.empty else plans
    return enrich_plans(plans)


def enrich_plans(plans: pd.DataFrame) -> pd.DataFrame:
    """Join robustness / walk-forward / liquidity / hold-time and assign a rating."""
    if plans.empty:
        return plans
    pcon = sqlite3.connect(_plans_db())
    try:
        def rd(sql):
            try:
                return pd.read_sql(sql, pcon)
            except Exception:
                return pd.DataFrame()
        rob = rd("SELECT ticker,strategy,exit_policy,psr,mc_p_profit,mc_p_dd_gt_30,mc_ret_p50 FROM robustness")
        liq = rd("SELECT ticker,liquidity_tier,fx_to_aud,market,currency FROM liquidity")
        wfo = rd("SELECT ticker,wfo_pct_windows_pos,wfo_total_return FROM wfo")
        pst = rd("SELECT ticker,strategy,exit_policy,avg_duration FROM plan_stats")
    finally:
        pcon.close()
    if not rob.empty:
        plans = plans.merge(rob, on=["ticker", "strategy", "exit_policy"], how="left")
    if not pst.empty:
        plans = plans.merge(pst, on=["ticker", "strategy", "exit_policy"], how="left")
    if not liq.empty:
        plans = plans.merge(liq, on="ticker", how="left")
    if not wfo.empty:
        plans = plans.merge(wfo, on="ticker", how="left")

    def rating(r):
        psr = r.get("psr"); pp = r.get("mc_p_profit"); dd = r.get("mc_p_dd_gt_30")
        liq_ok = str(r.get("liquidity_tier", "")) in ("liquid", "ok")
        wf = r.get("wfo_pct_windows_pos")
        hc = (bool(r.get("recommended")) and pd.notna(psr) and psr > 0.90
              and pd.notna(pp) and pp > 0.75 and pd.notna(dd) and dd < 0.25 and liq_ok)
        wfo_ok = pd.notna(wf) and wf >= 0.60
        if hc and wfo_ok:
            return "Strong Buy"
        if hc:
            return "Good Buy"
        return "Buy"
    plans["rating"] = plans.apply(rating, axis=1)
    return plans


# ---------------------------------------------------------------------------
# Market regime (bull = index above its 200-day MA)
# ---------------------------------------------------------------------------
def current_regimes() -> dict:
    """{market: True/False} — latest regime per market; empty dict on failure."""
    try:
        import regime as regime_mod
        # only the latest value matters here; a short window is ~10x faster
        return {m: bool(s.iloc[-1]) for m, s in regime_mod.market_regimes(period="2y").items() if len(s)}
    except Exception as e:
        log.warning("regime check unavailable (%s); buys not gated this run", e)
        return {}


def market_of(ticker: str) -> str:
    import regime as regime_mod
    return regime_mod.market_of(ticker)


def suggest_size(entry: float, stop, fx: float = 1.0, market: str = "Australia") -> dict | None:
    """
    Equal-risk position size, commission- and currency-aware.

    All budgeting happens in AUD: `entry`/`stop` are in the ticker's local
    currency and `fx` converts one unit to AUD (1.0 for ASX). Commissions use
    CommSec's ASX tiers for Australia and the (approximated) International
    schedule elsewhere — see brokerage.py.

    1. Base size risks RISK_PER_TRADE of CAPITAL over the stop distance
       (no-stop plans: flat FALLBACK_POSITION_FRAC slice).
    2. If round-trip commission would exceed MAX_FEE_DRAG_RT of the position,
       bump the size UP to the cheapest fee-efficient value and report the
       true risk that implies. Never suggests more than CAPITAL. Advisory only.
    """
    import brokerage
    if not entry or entry <= 0:
        return None
    if fx is None or (isinstance(fx, float) and (np.isnan(fx) or fx <= 0)):
        fx = 1.0
    price_aud = entry * fx
    if price_aud > config.CAPITAL:      # one share already exceeds the capital
        return None

    stop_frac = (entry - stop) / entry if (stop is not None and stop < entry) else None
    if stop_frac:
        risk_amt = config.CAPITAL * config.RISK_PER_TRADE
        value = risk_amt / stop_frac
        basis = f"risks ~{config.RISK_PER_TRADE:.0%} of ${config.CAPITAL:,.0f}"
    else:
        value = config.CAPITAL * config.FALLBACK_POSITION_FRAC
        basis = f"flat {config.FALLBACK_POSITION_FRAC:.0%} slice (no fixed stop)"

    bumped = fee_warning = False
    if brokerage.round_trip_drag(value, market) > config.MAX_FEE_DRAG_RT:
        eff = brokerage.min_value_for_drag(config.MAX_FEE_DRAG_RT, at_least=value, market=market)
        # only bump if it doesn't escalate risk beyond MAX_RISK_ESCALATION x target
        max_risk_value = (config.CAPITAL * config.RISK_PER_TRADE * config.MAX_RISK_ESCALATION
                          / stop_frac) if stop_frac else config.CAPITAL
        if eff <= max_risk_value:
            value, bumped = eff, True
        else:
            fee_warning = True           # stay risk-correct; warn about the drag
    value = min(value, config.CAPITAL)

    shares = int(value / price_aud)
    if shares <= 0:
        return None
    value = shares * price_aud           # actual AUD position value
    comm = brokerage.commission(value, market)
    out = {
        "shares": shares, "value": value, "basis": basis,
        "commission": comm, "rt_drag": brokerage.round_trip_drag(value, market),
        "bumped": bumped, "fee_warning": fee_warning, "market": market, "fx": fx,
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

    # Same-bar guard: if no NEW bar has closed since entry (e.g. two runs over a
    # weekend process the same Friday bar), don't act on the strategy's exit
    # signal — the backtest never exits on the entry bar either. Stops/targets
    # stay live regardless.
    same_bar = str(pos.get("entry_date", ""))[:10] == ts.isoformat()[:10]

    reason = None
    if stop is not None and close <= stop:
        reason = "stop hit"
    elif target is not None and close >= target:
        reason = "target hit"
    elif not same_bar and check_signal_exit(strat, params, df):
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

    if dry_run:
        log.info("DRY-RUN: paper positions/trades will NOT be modified")

    today = datetime.now(timezone.utc).isoformat()
    buys, sells, holds, suppressed = [], [], [], []
    asof = None

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
        asof = max(asof, ts) if asof is not None else ts
        mkt = market_of(ticker)

        if ticker in open_rows:
            pos = open_rows[ticker]
            sell, reason, new_stop, hi_water = evaluate_open(pos, strat, params, df)
            if sell:
                import brokerage
                pnl = close / pos["entry_price"] - 1.0
                pos_value = pos.get("pos_value")
                if pos_value:
                    comm = (brokerage.commission(pos_value, mkt)
                            + brokerage.commission(pos_value * (1 + pnl), mkt))
                    pnl_net = pnl - comm / pos_value
                else:
                    comm, pnl_net = None, pnl
                if not dry_run:
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
                if not dry_run:
                    con.execute("UPDATE positions SET stop_level=?, hi_water=? WHERE ticker=?",
                                (new_stop, hi_water, ticker))
                holds.append({"ticker": ticker, "strategy": pos["strategy"],
                              "entry_price": pos["entry_price"], "close": close,
                              "stop_level": new_stop, "pnl": close / pos["entry_price"] - 1.0})
        else:
            if check_entry(strat, params, df):
                if regimes and regimes.get(mkt) is False:
                    # bear regime: surface the setup but don't signal a buy
                    suppressed.append({"ticker": ticker, "strategy": plan["strategy"],
                                       "market": mkt, "entry_price": close})
                    log.info("  suppressed BUY %s (%s in bear regime)", ticker, mkt)
                    continue
                atr_at_entry = float(P.atr(df, 14).iloc[-1])
                stop, target, trail = levels_for(plan["exit_policy"], close, atr_at_entry)
                size = suggest_size(close, stop, fx=plan.get("fx_to_aud", 1.0), market=mkt)
                if not dry_run:
                    con.execute("""INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                                (ticker, plan["strategy"], plan["params"], plan["exit_policy"],
                                 ts.isoformat(), close, stop, target, trail, close,
                                 size["value"] if size else None))
                buys.append({"ticker": ticker, "strategy": plan["strategy"],
                             "exit_policy": plan["exit_policy"], "entry_price": close,
                             "stop_level": stop, "target_level": target,
                             "entry_rule": plan.get("entry_rule", ""),
                             "stop_rule": plan.get("stop_rule", ""),
                             "size": size, "market": mkt,
                             "currency": plan.get("currency", "AUD"),
                             "rating": plan.get("rating", "Buy"),
                             "win_rate": plan.get("oos_win_rate"),
                             "payoff": plan.get("oos_payoff"),
                             "avg_ret": plan.get("oos_avg_ret_pct"),
                             "cagr": plan.get("oos_cagr"),
                             "psr": plan.get("psr"),
                             "p_profit": plan.get("mc_p_profit"),
                             "avg_hold": plan.get("avg_duration")})
    if not dry_run:
        con.commit()
    con.close()

    asof_str = asof.isoformat() if asof is not None else today
    summary = {"buys": buys, "sells": sells, "holds": holds,
               "suppressed": suppressed, "regimes": regimes, "asof": asof_str}
    _report(summary, dry_run, out=out)
    return summary


def _has_rows(con, table):
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
    except Exception:
        return False


STARS = {"Strong Buy": "⭐⭐⭐", "Good Buy": "⭐⭐", "Buy": "⭐"}


def _pct(v, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:+.1f}%" if sign else f"{v*100:.1f}%"


def _buy_card(b) -> str:
    entry = b["entry_price"]
    stop = b.get("stop_level")
    tgt = b.get("target_level")
    ccy = b.get("currency") or "AUD"
    cur = "$" if ccy == "AUD" else f"{ccy} "   # foreign prices are in local currency
    stop_str = f"{cur}{stop:,.2f} ({_pct((stop-entry)/entry, True)})" if stop else "none"
    if tgt:
        tgt_str = f"{cur}{tgt:,.2f} ({_pct((tgt-entry)/entry, True)})"
        rr = (tgt - entry) / (entry - stop) if (stop and entry > stop) else None
        rr_basis = "target vs stop"
    else:
        tgt_str = "trailing / signal exit"
        rr = b.get("payoff")
        rr_basis = "historical avg win ÷ avg loss"
    if rr and not np.isnan(rr):
        rr_str = f"**{rr:.1f} : 1**"
        if rr < 1:
            rr_basis += "; small frequent wins profile"
    else:
        rr_str = "—"

    rating = b.get("rating", "Buy")
    lines = [f"**{STARS.get(rating,'⭐')} {rating.upper()} · {b['ticker']}** — {b['strategy']}"]
    lines.append(f"- 📈 Entry **~{cur}{entry:,.2f}** · 🛑 Stop **{stop_str}** · 🎯 Target **{tgt_str}**")
    exp = _pct(b.get("avg_ret"), True)
    yr = _pct(b.get("cagr"))
    lines.append(f"- 💰 Expected **{exp}** per trade (avg) · ~**{yr}/yr** on this name (out-of-sample)")
    conf = _pct(b.get("psr")); wr = _pct(b.get("win_rate")); mcp = _pct(b.get("p_profit"))
    lines.append(f"- ⚖️ Risk : reward ≈ {rr_str} _({rr_basis})_ · 🎲 Win rate **{wr}** · ✅ Edge-confidence **{conf}**")
    hold = b.get("avg_hold")
    hold_str = f"{round(hold)} trading days" if (hold and not np.isnan(hold)) else "—"
    lines.append(f"- ⏱️ Typical hold **{hold_str}** · 🧪 Monte-Carlo profit odds **{mcp}**")
    sz = b.get("size")
    if sz:
        note = ""
        if sz.get("bumped"):
            note = (" ⚠️ _sized up for fee efficiency; true risk "
                    + _pct(sz.get("risk_pct")) + " of capital_")
        elif sz.get("fee_warning"):
            note = (f" ⚠️ _fees eat {sz['rt_drag']:.1%} round-trip at this size — "
                    f"a fee-efficient size would exceed {_pct(config.RISK_PER_TRADE * config.MAX_RISK_ESCALATION)} "
                    f"risk, so consider skipping this one_")
        broker = "CommSec" if sz.get("market", "Australia") == "Australia" else "CommSec Intl (approx)"
        lines.append(f"- 📦 Size **~{sz['shares']} shares** (≈A${sz['value']:,.0f}) · "
                     f"{broker} ~A${sz['commission']:.0f}/side, {sz['rt_drag']:.2%} round-trip{note}")
    lines.append(f"- 💡 _Setup: {b.get('entry_rule','')}_")
    return "\n".join(lines)


def format_summary(s) -> str:
    d = s["asof"][:10]
    buys = s["buys"]
    n_strong = sum(1 for b in buys if b.get("rating") == "Strong Buy")
    md = [f"## 📊 Daily signals — {d}", ""]

    # one-line TL;DR
    tl = []
    if buys:
        tl.append(f"🟢 **{len(buys)} buy{'s' if len(buys)!=1 else ''}**" +
                  (f" ({n_strong} strong)" if n_strong else ""))
    if s["sells"]:
        tl.append(f"🔴 **{len(s['sells'])} sell{'s' if len(s['sells'])!=1 else ''}**")
    if s["holds"]:
        tl.append(f"⚪ {len(s['holds'])} holding")
    if s.get("suppressed"):
        tl.append(f"🚫 {len(s['suppressed'])} suppressed")
    md.append(" · ".join(tl) if tl else "_No actions today — nothing triggered and no open positions._")

    if buys:
        md += ["", f"### 🟢 Buy signals ({len(buys)})"]
        order = {"Strong Buy": 0, "Good Buy": 1, "Buy": 2}
        for b in sorted(buys, key=lambda x: order.get(x.get("rating"), 3)):
            md += ["", _buy_card(b)]

    if s["sells"]:
        md += ["", f"### 🔴 Sell signals ({len(s['sells'])})"]
        for x in s["sells"]:
            net = (f" ({_pct(x['pnl_net'], True)} after ~${x['commission']:.0f} commission)"
                   if x.get("commission") else "")
            md.append(f"- **{x['ticker']}** {x['strategy']} @ ${x['exit_price']:.2f} — "
                      f"**{x['reason']}**, {_pct(x['pnl'], True)}{net}")

    if s["holds"]:
        md += ["", f"### ⚪ Holding ({len(s['holds'])})"]
        for h in s["holds"]:
            stop = f"${h['stop_level']:.2f}" if h.get("stop_level") is not None else "n/a"
            md.append(f"- **{h['ticker']}** {h['strategy']} — now ${h['close']:.2f} "
                      f"(**{_pct(h['pnl'], True)}**), stop {stop}")

    if s.get("suppressed"):
        md += ["", f"### 🚫 Suppressed — bear regime ({len(s['suppressed'])})"]
        for x in s["suppressed"]:
            md.append(f"- **{x['ticker']}** {x['strategy']} fired @ ${x['entry_price']:.2f} — "
                      f"{x['market']} index below its 200-day MA, sitting out")

    reg = s.get("regimes") or {}
    if reg:
        bears = [m for m, v in sorted(reg.items()) if not v]
        md += ["", f"**Market regimes:** {'all bullish 🟢' if not bears else '🔴 bearish (buys paused): ' + ', '.join(bears)}"]

    md += ["", "---",
           "**Ratings:** ⭐⭐⭐ Strong Buy = robust + walk-forward-consistent + tradeable · "
           "⭐⭐ Good Buy = passed robustness stress-test · ⭐ Buy = out-of-sample validated.",
           "_Paper-trading decision support, not financial advice. You place the orders and stops._"]
    return "\n".join(md)


def _report(summary, dry_run, out=None):
    md = format_summary(summary)
    log.info("BUY=%d SELL=%d HOLD=%d SUPPRESSED=%d", len(summary["buys"]),
             len(summary["sells"]), len(summary["holds"]), len(summary.get("suppressed", [])))
    # console: keep structure, drop markdown emphasis noise
    console = md.replace("**", "").replace("### ", "").replace("## ", "").replace("- ", "• ")
    print("\n" + console)
    if out:
        from pathlib import Path
        Path(out).write_text(md, encoding="utf-8")
        log.info("summary written to %s", out)
    if not dry_run and telegram_bot.is_configured():
        telegram_bot.send_message(md)  # Telegram is optional/off by default


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
