"""
Forward self-tracking: how the SYSTEM is actually doing on the signals it has
generated live (not the backtest). Every BUY the daily monitor issues opens a
paper position; every close is a realised trade. This aggregates that live record
into a running P&L on your capital, sliceable by signal tier:

    all           — every recommended signal
    high_conf     — only ★ high-confidence signals
    strong_buy    — only ⭐⭐⭐ Strong Buy signals
    good_buy_plus — Strong Buy + Good Buy

For each slice it computes realised P&L ($ and %), win rate, an equity curve over
the trades' exit dates, best/worst, and current open positions with unrealised P&L.

Writes fwd_perf.json (repo root, committed daily → the dashboard's Live view reads
it) and a `fwd_perf` table in serving.db. This only becomes meaningful as the
monitor runs for weeks — that's the whole point: real forward evidence.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

import config
import dataio

FILTERS = {
    "all": lambda t: True,
    "high_conf": lambda t: t.get("high_conf") == 1,      # NB: bool(NaN) is True in Python — use ==1
    "strong_buy": lambda t: t.get("rating") == "Strong Buy",
    "good_buy_plus": lambda t: t.get("rating") in ("Strong Buy", "Good Buy"),
}
LABELS = {"all": "All recommended", "high_conf": "★ High-confidence only",
          "strong_buy": "⭐⭐⭐ Strong Buy only", "good_buy_plus": "Good Buy + Strong Buy"}


def _read(con, table):
    try:
        return pd.read_sql(f"SELECT * FROM {table}", con)
    except Exception:
        return pd.DataFrame()


def _expected_map():
    """{ticker: (expected_per_trade, cagr, stop_rule)} for recommended plans."""
    for db in (config.SERVING_DB, config.LEADERBOARD_DB):
        try:
            con = sqlite3.connect(db)
            p = pd.read_sql("SELECT ticker, oos_avg_ret_pct, oos_cagr, stop_rule FROM plans WHERE recommended=1", con)
            con.close()
            if not p.empty:
                return {r["ticker"]: (r.get("oos_avg_ret_pct"), r.get("oos_cagr"), r.get("stop_rule"))
                        for _, r in p.iterrows()}
        except Exception:
            continue
    return {}


def _slice(trades: pd.DataFrame, positions: pd.DataFrame, keep, expmap=None) -> dict:
    expmap = expmap or {}
    cap = config.CAPITAL
    tr = [r for _, r in trades.iterrows() if keep(r)] if not trades.empty else []
    tr = sorted(tr, key=lambda r: str(r.get("exit_date", "")))
    dates, eq, running = [], [], cap
    n = wins = 0
    rets, pnls, closed = [], [], []
    for r in tr:
        net = r.get("pnl_pct_net")
        if net is None or (isinstance(net, float) and np.isnan(net)):
            net = r.get("pnl_pct")
        if net is None or (isinstance(net, float) and np.isnan(net)):
            continue
        n += 1
        wins += 1 if net > 0 else 0
        rets.append(float(net))
        pv = r.get("pos_value")
        dollar = float(net) * float(pv) if (pv and not np.isnan(pv)) else float(net) * cap * config.RISK_PER_TRADE / 0.1
        pnls.append(dollar)
        running += dollar
        dates.append(str(r.get("exit_date", ""))[:10])
        eq.append(round(running, 2))
        closed.append({"ticker": r["ticker"], "strategy": r["strategy"],
                       "rating": r.get("rating"), "buy_date": str(r.get("entry_date", ""))[:10],
                       "sell_date": str(r.get("exit_date", ""))[:10],
                       "entry": r.get("entry_price"), "exit": r.get("exit_price"),
                       "ret": round(float(net), 4), "reason": r.get("reason")})
    closed = closed[-25:][::-1]   # most recent first

    open_rows = []
    if not positions.empty:
        for _, p in positions.iterrows():
            if not keep(p):
                continue
            cur = np.nan
            d = dataio.load(p["ticker"], "1d")
            if d is not None and not d.empty:
                cur = float(d["Close"].iloc[-1])
            unreal = (cur / p["entry_price"] - 1) if (cur and p.get("entry_price")) else np.nan
            exp, _cagr, stoprule = expmap.get(p["ticker"], (None, None, None))
            open_rows.append({"ticker": p["ticker"], "strategy": p["strategy"],
                              "rating": p.get("rating"), "buy_date": str(p.get("entry_date", ""))[:10],
                              "entry": p.get("entry_price"),
                              "current": None if np.isnan(cur) else round(cur, 3),
                              "unreal_pct": None if np.isnan(unreal) else round(float(unreal), 4),
                              "stop": p.get("stop_level"),
                              "expected": None if (exp is None or (isinstance(exp, float) and np.isnan(exp))) else round(float(exp), 4)})

    realized = running - cap
    return {
        "n_trades": n,
        "win_rate": (wins / n) if n else None,
        "avg_ret": float(np.mean(rets)) if rets else None,
        "realized_pnl": round(realized, 2),
        "total_return": round(realized / cap, 4),
        "best": max(rets) if rets else None,
        "worst": min(rets) if rets else None,
        "first": dates[0] if dates else None,
        "last": dates[-1] if dates else None,
        "equity_dates": dates,
        "equity_values": eq,
        "open": open_rows,
        "n_open": len(open_rows),
        "closed": closed,
    }


def build() -> dict:
    con = sqlite3.connect(config.SIGNALS_DB)
    trades = _read(con, "trades")
    positions = _read(con, "positions")
    con.close()
    expmap = _expected_map()
    out = {"capital": config.CAPITAL, "labels": LABELS, "slices": {}}
    for key, keep in FILTERS.items():
        out["slices"][key] = _slice(trades, positions, keep, expmap)
    return out


def _clean(o):
    """Recursively convert NaN/inf floats to None so the output is valid JSON
    (browser JSON.parse rejects NaN — this broke the runtime fetch otherwise)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


def save(perf: dict):
    perf = _clean(perf)
    js = json.dumps(perf, allow_nan=False)
    (config.PROJECT_ROOT / "fwd_perf.json").write_text(js, encoding="utf-8")
    con = sqlite3.connect(config.SERVING_DB)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS fwd_perf (k TEXT PRIMARY KEY, json TEXT)")
        con.execute("INSERT OR REPLACE INTO fwd_perf VALUES ('perf', ?)", (json.dumps(perf),))
        con.commit()
    finally:
        con.close()


def main(argv=None):
    perf = build()
    save(perf)
    print("Forward tracker updated (fwd_perf.json + serving.db):")
    for k, s in perf["slices"].items():
        print(f"  {LABELS[k]:26} closed={s['n_trades']:3}  "
              f"win={('%.0f%%'%(s['win_rate']*100)) if s['win_rate'] is not None else '—':>4}  "
              f"P&L=${s['realized_pnl']:>9,.0f} ({s['total_return']:+.1%})  open={s['n_open']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
