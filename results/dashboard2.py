"""
Dashboard v2 generator. Reads the v2 scan (`runs2`) + trading `plans` and emits:

  * results/dashboard.html   — interactive, 3 views:
       1. TRADE PLANS  — best full plan per ticker (entry, stop, target, OOS
          metrics, walk-forward consistency, recommended badge). Rows expand.
       2. STRATEGIES   — how each strategy performed ACROSS all tickers/markets.
       3. EXPLORER     — top combos per ticker, filterable.
  * results/all_results.csv        — the FULL grid (every strategy x params x
                                     exit policy x ticker) with IS + OOS metrics
  * results/strategy_summary.csv   — the per-strategy aggregate

Selection stays honest: plans are chosen on in-sample and reported out-of-sample.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import brokerage
import config
import dataio
from backtest import engine2

OUT_HTML = config.PROJECT_ROOT / "results" / "dashboard.html"
OUT_ALL = config.PROJECT_ROOT / "results" / "all_results.csv"
OUT_STRAT = config.PROJECT_ROOT / "results" / "strategy_summary.csv"


def _load(interval):
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        runs = pd.read_sql("SELECT * FROM runs2", con)
        try:
            plans = pd.read_sql("SELECT * FROM plans", con)
        except Exception:
            plans = pd.DataFrame()
        try:
            robust = pd.read_sql("SELECT * FROM robustness", con)
        except Exception:
            robust = pd.DataFrame()
        try:
            liq = pd.read_sql("SELECT ticker, liquidity_tier, adv_aud, max_pos_aud FROM liquidity", con)
        except Exception:
            liq = pd.DataFrame()
        try:
            wfo = pd.read_sql("SELECT ticker, wfo_total_return, wfo_bh_return, "
                              "wfo_pct_windows_pos, wfo_worst_window, wfo_beats_bh, "
                              "wfo_n_unique_picks FROM wfo", con)
        except Exception:
            wfo = pd.DataFrame()
        try:
            pstats = pd.read_sql("SELECT ticker, strategy, exit_policy, avg_duration, "
                                 "median_duration FROM plan_stats", con)
        except Exception:
            pstats = pd.DataFrame()
    finally:
        con.close()
    runs = runs[runs["interval"] == interval]
    # latest scan per ticker (so incrementally-added markets combine cleanly)
    latest = runs.groupby("ticker")["scan_id"].transform("max")
    runs = runs[runs["scan_id"] == latest]
    if not plans.empty and "interval" in plans.columns:
        plans = plans[plans["interval"] == interval]
    # merge robustness metrics onto plans
    if not plans.empty and not robust.empty:
        rcols = ["ticker", "strategy", "exit_policy", "psr", "mc_ret_p5", "mc_ret_p50",
                 "mc_ret_p95", "mc_dd_p95worst", "mc_p_profit", "mc_p_dd_gt_30",
                 "worst_year_ret", "years_positive", "covid_crash_ret", "n_trials",
                 "trials_oos_hit_rate"]
        rcols = [c for c in rcols if c in robust.columns]
        plans = plans.merge(robust[rcols], on=["ticker", "strategy", "exit_policy"], how="left")
    if not plans.empty and not liq.empty:
        plans = plans.merge(liq, on="ticker", how="left")
    if "liquidity_tier" not in plans.columns:
        plans["liquidity_tier"] = "unknown"
    if not plans.empty and not pstats.empty:
        plans = plans.merge(pstats, on=["ticker", "strategy", "exit_policy"], how="left")
    if not plans.empty and not wfo.empty:
        plans = plans.merge(wfo, on="ticker", how="left")
        # WFO survivor: consistently profitable under rolling re-optimization
        plans["wfo_survivor"] = ((plans["wfo_pct_windows_pos"] >= 0.6)
                                 & (plans["wfo_total_return"] > 0)).fillna(False)
    if not plans.empty and "psr" in plans.columns:
        # high-confidence = recommended AND edge likely real AND not fragile AND tradeable
        plans["high_conf"] = (plans.get("recommended", False).astype(bool)
                              & (plans["psr"] > 0.90) & (plans["mc_p_profit"] > 0.75)
                              & (plans["mc_p_dd_gt_30"] < 0.25)
                              & plans["liquidity_tier"].isin(["liquid", "ok"])).fillna(False)
    return runs, plans


def strategy_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per strategy across all tickers/params/exit policies."""
    g = runs.groupby(["strategy", "family"])
    out = pd.DataFrame({
        "combos": g.size(),
        "tickers": g["ticker"].nunique(),
        "median_oos_cagr": g["oos_cagr"].median(),
        "pct_positive_oos": g["oos_total_return"].apply(lambda s: float((s > 0).mean())),
        "median_oos_sharpe": g["oos_sharpe"].median(),
        "median_oos_win_rate": g["oos_win_rate"].median(),
        "median_oos_maxdd": g["oos_max_drawdown"].median(),
        "pct_beat_bh_oos": g.apply(lambda d: float((d["oos_total_return"] > d["oos_buy_hold_return"]).mean()),
                                   include_groups=False),
    }).reset_index()
    return out.sort_values("median_oos_sharpe", ascending=False)


def ticker_top(runs: pd.DataFrame, per=8) -> pd.DataFrame:
    """Top `per` combos per ticker by in-sample Sharpe (for the explorer)."""
    valid = runs[(runs["is_n_trades"].fillna(0) >= config.MIN_TRADES) &
                 (runs["oos_n_trades"].fillna(0) >= config.MIN_TRADES)].copy()
    valid = valid.sort_values("is_sharpe", ascending=False)
    return valid.groupby("ticker", group_keys=False).head(per)


def _round(df, cols, n=3):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(n)
    return df


def _records(df):
    return json.loads(df.to_json(orient="records"))


_MARKETS = {".AX": "Australia", ".HK": "Hong Kong", ".T": "Japan", ".KS": "Korea",
            ".KQ": "Korea", ".TW": "Taiwan", ".SI": "Singapore", ".NS": "India", ".BO": "India",
            ".SS": "China", ".SZ": "China"}


def market_of(ticker: str) -> str:
    for suf, name in _MARKETS.items():
        if str(ticker).endswith(suf):
            return name
    return "Other"


# Plain-English "what it does / when it works" per strategy, shown on click.
STRATEGY_BLURBS = {
 "ma_crossover": "Classic trend-following. Buys when a fast moving average crosses above a slow one and exits on the reverse cross. Works in sustained trends; whipsaws in choppy, sideways markets.",
 "macd_momentum": "Momentum via the MACD line crossing its signal line. Catches shifts in momentum earlier than a plain MA cross. Best in trending names; noisy in flat markets.",
 "adx_trend": "An MA cross that only fires when ADX confirms a genuinely strong trend (filters out weak, rangebound crosses). Fewer but higher-quality trend trades.",
 "roc_momentum": "Buys strength — when rate-of-change is positive and rising. Rides established up-moves; late to turn and prone to buying tops in blow-offs.",
 "rsi_reversion": "Mean reversion: buys oversold (RSI low), sells overbought. The single most consistent family here on the ASX. Best on range-bound, liquid names; dangerous on stocks in free-fall.",
 "rsi2_pullback": "Connors-style short-term dip buy — a very fast RSI(2) oversold reading while the stock is still above its 200-day MA. Quick 2-4 day holds; needs an uptrend to be safe.",
 "bb_reversion": "Buys when price stretches below the lower Bollinger band and exits back at the middle. A volatility-scaled version of dip-buying.",
 "zscore_reversion": "Statistical mean reversion: buys when price is 2+ standard deviations below its rolling mean. Same idea as Bollinger reversion, framed as a z-score.",
 "pct_below_ma": "Buys when price falls a fixed % below a long moving average — a simple, robust 'buy the deep dip' that scored well across markets.",
 "donchian_breakout_vol": "Turtle-style breakout above an N-day high, but only on a volume surge (conviction filter). Exits on a break of the N-day low.",
 "high_52w_breakout": "Buys new 52-week highs while in an uptrend — momentum's strongest signal. Great in bull runs; gives a lot back at tops.",
 "bb_breakout_vol": "Buys a thrust above the upper Bollinger band on volume (a volatility breakout), exits back at the middle band.",
 "atr_channel_breakout": "Buys when price pushes above an ATR-scaled channel around its mean — a volatility-adaptive breakout.",
 "nbar_high_confirmed": "Break of an N-day high confirmed by two consecutive rising closes (reduces false breakouts).",
 "turtle_breakout": "The original turtle system: buy the N-day high, exit the N/2-day low. Pure trend-following breakout.",
 "pullback_uptrend": "Buys a pullback (RSI dip) within an established uptrend and a rising close — 'buy the dip' in a trend.",
 "gap_up_go": "Buys a gap-up open on a volume surge, exits below a short MA. Momentum continuation of overnight news moves; noisy.",
 "support_bounce": "A crude support buy — near the rolling N-day low and turning up. (The pivot-based sr_* strategies are the more refined version.)",
 "rsi_bb_confluence": "Confluence: RSI oversold AND price below the lower Bollinger band AND above the 200-day MA. Stacks conditions so all must agree — fewer, cleaner mean-reversion entries.",
 "triple_confluence_breakout": "Breakout that must clear three hurdles at once: N-day high AND volume surge AND above the 200-day MA. High selectivity.",
 "rsi_macd_confluence": "Requires an RSI dip AND a MACD cross-up AND an uptrend — a momentum-plus-reversion combo.",
 "bb_squeeze_breakout": "Waits for a Bollinger 'squeeze' (low volatility) then buys the expansion break on volume — trades the transition from calm to trend.",
 "sr_support_bounce": "Buys a dip to a REAL swing-low support level (a price that was rejected before, not just a rolling low), still holding and turning up. Exits at swing-high resistance or if support breaks. Mean-reversion within structure.",
 "sr_breakout": "Buys a break above a real swing-high resistance level on volume, exits if price falls back below support. The best-performing S/R strategy — trades structure breaks.",
 "sr_support_uptrend": "Buys a swing-low support bounce ONLY in a confirmed uptrend, then RIDES it — holding through overhead resistance and exiting only when support finally breaks. Trend-following entry at structure.",
}


def whatif_stake(plans, pt, tier="high_conf"):
    """'What if I put $X into each <tier> plan?' — compound each ticker's actual
    OOS trades, gross and NET of CommSec commissions (buy+sell each trade), across
    a sweep of parcel sizes. Shows how badly small parcels get eaten by fixed fees.
    tier: 'high_conf' (high recommended) or 'recommended'."""
    if plans.empty or pt is None or pt.empty:
        return None
    if tier == "high_conf" and "high_conf" in plans.columns:
        sel = plans[plans["high_conf"] == True]          # noqa: E712
    else:
        sel = plans[plans.get("recommended", 0) == 1] if "recommended" in plans.columns else plans
    tickers = sorted(set(sel["ticker"]))
    if not tickers:
        return None
    c = pt[pt["ticker"].isin(tickers) & (pt["status"] == "Closed")].copy()
    if c.empty:
        return None
    c["entry_date"] = pd.to_datetime(c["entry_date"], errors="coerce")
    c["exit_date"] = pd.to_datetime(c["exit_date"], errors="coerce")
    c = c.sort_values("entry_date")

    def sim_one(stake, tk):
        g = n = float(stake)
        mk = market_of(tk)
        for _, tr in c[c["ticker"] == tk].iterrows():
            r = float(tr["ret"])
            g *= (1 + r)
            bc = brokerage.commission(n, mk)
            ev = (n - bc) * (1 + r)
            n = ev - brokerage.commission(ev, mk)
        return g, n

    per = []
    for tk in tickers:
        g, n = sim_one(1000.0, tk)
        tt = c[c["ticker"] == tk]
        row = sel[sel["ticker"] == tk].iloc[0]
        per.append({"ticker": tk, "market": market_of(tk), "strategy": row["strategy"],
                    "rating": row.get("rating", ""), "trades": int(len(tt)),
                    "first": str(tt["entry_date"].min())[:10], "last": str(tt["exit_date"].max())[:10],
                    "gross_1k": round(g), "net_1k": round(n)})
    sweep = []
    for s in (1000, 2500, 5000, 10000, 25000):
        gt = nt = 0.0
        for tk in tickers:
            g, n = sim_one(s, tk)
            gt += g; nt += n
        inv = s * len(tickers)
        sweep.append({"stake": s, "invested": round(inv), "gross": round(gt), "net": round(nt),
                      "gross_pct": round((gt - inv) / inv, 4), "net_pct": round((nt - inv) / inv, 4)})
    span_start = str(c["entry_date"].min())[:10]
    span_end = str(c["exit_date"].max())[:10]
    yrs = (c["exit_date"].max() - c["entry_date"].min()).days / 365.25
    return {"tier": tier, "n_tickers": len(tickers), "n_trades": int(len(c)),
            "span_start": span_start, "span_end": span_end, "years": round(yrs, 1),
            "sweep": sweep, "per": per}


def build_payload(interval="1d"):
    runs, plans = _load(interval)
    if runs.empty:
        return None

    runs.to_csv(OUT_ALL, index=False)
    runs.to_csv(str(OUT_ALL) + ".gz", index=False, compression="gzip")  # repo-friendly full dump
    ssum = strategy_summary(runs)
    ssum.to_csv(OUT_STRAT, index=False)
    # attach plain-English descriptions + entry/exit rules + params for the click-to-expand
    try:
        from strategies.registry import STRATEGIES as _REG
    except Exception:
        _REG = {}
    def _desc(row):
        s = _REG.get(row["strategy"])
        d = s.describe(s.param_grid and {k: v[0] for k, v in s.param_grid.items()} or {}) if s else {"entry": "", "exit": ""}
        grid = ", ".join(f"{k}={v}" for k, v in (s.param_grid or {}).items()) if s else ""
        return pd.Series({"blurb": STRATEGY_BLURBS.get(row["strategy"], ""),
                          "entry_desc": d["entry"], "exit_desc": d["exit"], "param_grid": grid})
    ssum = pd.concat([ssum, ssum.apply(_desc, axis=1)], axis=1)

    top = ticker_top(runs)
    for d in (plans, top):
        if not d.empty:
            d["market"] = d["ticker"].map(market_of)

    fam_cols = ["median_oos_cagr", "pct_positive_oos", "median_oos_sharpe",
                "median_oos_win_rate", "median_oos_maxdd", "pct_beat_bh_oos"]
    plan_num = ["oos_cagr", "oos_total_return", "oos_sharpe", "oos_max_drawdown",
                "oos_win_rate", "oos_payoff", "oos_expectancy_R", "oos_buy_hold_return",
                "is_cagr", "is_total_return", "is_sharpe", "is_max_drawdown",
                "is_win_rate", "is_payoff", "is_expectancy_R", "wf_consistency",
                "psr", "mc_ret_p5", "mc_ret_p50", "mc_ret_p95", "mc_dd_p95worst",
                "mc_p_profit", "mc_p_dd_gt_30", "worst_year_ret", "years_positive",
                "covid_crash_ret", "trials_oos_hit_rate",
                "wfo_total_return", "wfo_bh_return", "wfo_pct_windows_pos", "wfo_worst_window",
                "avg_duration", "median_duration"]
    top_num = ["is_sharpe", "oos_sharpe", "oos_cagr", "oos_total_return", "oos_max_drawdown"]

    summary = {
        "interval": interval,
        "scan_id": runs["scan_id"].max(),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_tickers": int(runs["ticker"].nunique()),
        "n_strategies": int(runs["strategy"].nunique()),
        "n_combos": int(len(runs)),
        "n_exit_policies": int(runs["exit_policy"].nunique()),
        "n_markets": int(runs["ticker"].map(market_of).nunique()),
    }
    if not plans.empty:
        summary["n_plans"] = int(len(plans))
        summary["n_recommended"] = int(plans["recommended"].sum()) if "recommended" in plans else 0
        summary["n_beat_bh"] = int(plans["beats_bh"].sum()) if "beats_bh" in plans else 0
        summary["median_plan_oos_cagr"] = float(np.nanmedian(plans["oos_cagr"])) if "oos_cagr" in plans else None
        summary["n_high_conf"] = int(plans["high_conf"].sum()) if "high_conf" in plans else 0

    portfolios = []
    con = sqlite3.connect(config.LEADERBOARD_DB)
    try:
        prows = con.execute("SELECT json FROM portfolio").fetchall()
        portfolios = [json.loads(r[0]) for r in prows]
    except Exception:
        pass
    regime_by_label = {}
    try:
        for (j,) in con.execute("SELECT json FROM regime").fetchall():
            d = json.loads(j)
            regime_by_label[d["label"]] = d
    except Exception:
        pass
    finally:
        con.close()
    for p in portfolios:
        rg = regime_by_label.get(p.get("label"))
        if rg:
            p["regime"] = rg.get("regime")
            p["regime_equity"] = rg.get("equity_values")

    # individual trade logs (compact arrays) for recommended plans -> shown on expand
    trades_map = {}
    if not plans.empty and "recommended" in plans.columns:
        rec_keys = {(r["ticker"], r["strategy"], r["exit_policy"])
                    for _, r in plans[plans["recommended"] == 1].iterrows()}
        con = sqlite3.connect(config.LEADERBOARD_DB)
        try:
            pt = pd.read_sql("SELECT * FROM plan_trades", con)
        except Exception:
            pt = pd.DataFrame()
        finally:
            con.close()
        if not pt.empty:
            for (tk, st, ep), g in pt.groupby(["ticker", "strategy", "exit_policy"]):
                if (tk, st, ep) in rec_keys:
                    trades_map[f"{tk}|{st}|{ep}"] = [
                        [r.entry_date, r.exit_date, float(r.entry_price) if pd.notna(r.entry_price) else None,
                         float(r.exit_price) if pd.notna(r.exit_price) else None,
                         float(r.ret) if pd.notna(r.ret) else None,
                         int(r.bars) if pd.notna(r.bars) else None, str(r.status)]
                        for r in g.itertuples()]

    # downsampled OOS price series per recommended ticker (for the Ticker chart)
    prices = {}
    if not plans.empty and "recommended" in plans.columns:
        for tk in plans[plans["recommended"] == 1]["ticker"].unique():
            raw = dataio.load(tk, interval)
            if raw is None or raw.empty:
                continue
            df = engine2.clean_ohlcv(raw)
            cut = int(len(df) * config.IS_FRACTION)
            oos = df.iloc[cut:]
            if len(oos) < 30:
                continue
            step = max(1, len(oos) // 300)
            s = oos.iloc[::step]
            prices[tk] = {"d": [d.strftime("%Y-%m-%d") for d in s.index],
                          "c": [round(float(v), 3) for v in s["Close"]],
                          "oos_start": oos.index[0].strftime("%Y-%m-%d")}

    # per-family aggregates for the comparison chart
    fg = runs.groupby("family")
    fam_df = pd.DataFrame({
        "family": fg.size().index,
        "combos": fg.size().values,
        "strategies": fg["strategy"].nunique().values,
        "median_oos_sharpe": fg["oos_sharpe"].median().values,
        "median_oos_cagr": fg["oos_cagr"].median().values,
        "pct_positive_oos": fg["oos_total_return"].apply(lambda s: float((s > 0).mean())).values,
        "pct_beat_bh_oos": fg.apply(lambda d: float((d["oos_total_return"] > d["oos_buy_hold_return"]).mean()),
                                    include_groups=False).values,
    }).sort_values("median_oos_sharpe", ascending=False)

    # forward-tracker snapshot (embedded fallback; Live view also fetches fresh at runtime)
    fwd = {}
    crossu = None
    aftertax = None
    try:
        scon = sqlite3.connect(config.SERVING_DB)
        row = scon.execute("SELECT json FROM fwd_perf WHERE k='perf'").fetchone()
        if row:
            fwd = json.loads(row[0])
        try:
            cvr = scon.execute("SELECT json FROM crossuniverse WHERE k='cv'").fetchone()
            if cvr:
                crossu = json.loads(cvr[0])
        except Exception:
            pass
        try:
            atr = scon.execute("SELECT json FROM aftertax WHERE k='at'").fetchone()
            if atr:
                aftertax = json.loads(atr[0])
        except Exception:
            pass
        scon.close()
    except Exception:
        pass

    # "What if I put $1k into each high-recommended plan?" — gross vs net of fees
    whatif = None
    try:
        con = sqlite3.connect(config.LEADERBOARD_DB)
        pt_all = pd.read_sql("SELECT * FROM plan_trades", con)
        con.close()
        whatif = whatif_stake(plans, pt_all, tier="high_conf")
    except Exception:
        whatif = None

    return {
        "summary": summary,
        "trades": trades_map,
        "prices": prices,
        "fwd": fwd,
        "whatif": whatif,
        "crossu": crossu,
        "aftertax": aftertax,
        "families": _records(_round(fam_df, ["median_oos_sharpe", "median_oos_cagr",
                                             "pct_positive_oos", "pct_beat_bh_oos"])),
        "plans": _records(_round(plans, plan_num)) if not plans.empty else [],
        "strategies": _records(_round(ssum, fam_cols)),
        "top": _records(_round(top, top_num)),
        "portfolios": portfolios,
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def render_html(payload) -> str:
    return _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))


_TEMPLATE = r"""
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#111827;--mut:#6b7280;--line:#e6e8ec;--soft:#f3f4f6;
 --accent:#2563eb;--pos:#059669;--neg:#dc2626;--amber:#b45309;
 --recbg:#ecfdf5;--recink:#047857;--heldbg:#fffbeb;--heldink:#b45309;}
@media (prefers-color-scheme:dark){:root{--bg:#0e1014;--card:#161922;--ink:#e8eaed;--mut:#9aa3b2;
 --line:#242835;--soft:#1c202a;--accent:#6ea8fe;--pos:#34d399;--neg:#f87171;--amber:#fbbf24;
 --recbg:#06281f;--recink:#34d399;--heldbg:#2a2410;--heldink:#fbbf24;}}
:root[data-theme=dark]{--bg:#0e1014;--card:#161922;--ink:#e8eaed;--mut:#9aa3b2;--line:#242835;
 --soft:#1c202a;--accent:#6ea8fe;--pos:#34d399;--neg:#f87171;--amber:#fbbf24;
 --recbg:#06281f;--recink:#34d399;--heldbg:#2a2410;--heldink:#fbbf24;}
:root[data-theme=light]{--bg:#f6f7f9;--card:#fff;--ink:#111827;--mut:#6b7280;--line:#e6e8ec;
 --soft:#f3f4f6;--accent:#2563eb;--pos:#059669;--neg:#dc2626;--amber:#b45309;
 --recbg:#ecfdf5;--recink:#047857;--heldbg:#fffbeb;--heldink:#b45309;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14.5px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:25px;margin:0 0 3px;letter-spacing:-.02em}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}
.sub b{color:var(--ink)}
details.help{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:2px 16px;margin-bottom:18px}
details.help summary{cursor:pointer;font-weight:600;padding:12px 0;list-style:none}
details.help summary::-webkit-details-marker{display:none}
details.help summary::before{content:'▸ ';color:var(--accent)}
details.help[open] summary::before{content:'▾ '}
.help-body{padding:4px 0 14px;color:var(--mut);font-size:13px}
.help-body b{color:var(--ink)} .help-body li{margin:4px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.card .k{color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:25px;font-weight:700;margin-top:4px;letter-spacing:-.02em}
.card .v small{font-size:13px;color:var(--mut);font-weight:500}
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.tab{padding:9px 15px;border:1px solid var(--line);background:var(--card);border-radius:10px;
 cursor:pointer;font-weight:600;color:var(--mut);font-size:13.5px}
.tab.on{color:#fff;background:var(--accent);border-color:var(--accent)}
.bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
input,select{padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);
 color:var(--ink);font-size:13.5px;outline:none}
input:focus,select:focus{border-color:var(--accent)}
input#q{min-width:240px;flex:1}
.chk{color:var(--mut);font-size:13px;display:flex;align-items:center;gap:7px;cursor:pointer;user-select:none}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:11px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--card);z-index:1}
th{cursor:pointer;color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;
 user-select:none;font-weight:700;background:var(--soft)}
th:first-child{background:var(--soft)}
th.txt,td.txt{text-align:left}
tbody tr.main{cursor:pointer}
tbody tr.main:hover td{background:rgba(110,168,254,.07)}
tbody tr.main:hover td:first-child{background:rgba(110,168,254,.07)}
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}
.tick{font-weight:700}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;background:var(--soft);
 font-size:11px;color:var(--mut);border:1px solid var(--line)}
.lq-ok{background:var(--recbg);color:var(--recink);border-color:transparent}
.lq-mid{background:var(--heldbg);color:var(--heldink);border-color:transparent}
.lq-bad{background:#fee2e2;color:#b91c1c;border-color:transparent}
:root[data-theme=dark] .lq-bad{background:#3a1414;color:#f87171}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}
.b-high{background:linear-gradient(90deg,#fde68a,#fbbf24);color:#7c2d12;box-shadow:0 0 0 1px #f59e0b33}
:root[data-theme=dark] .b-high{color:#1a1200}
.b-rec{background:var(--recbg);color:var(--recink)}
.b-held{background:var(--heldbg);color:var(--heldink)}
.b-marg{background:var(--soft);color:var(--mut)}
.minibar{display:inline-block;height:7px;background:var(--accent);border-radius:4px;vertical-align:middle;margin-right:6px}
tr.detail td{background:var(--soft);padding:0}
.dwrap{padding:16px 18px;white-space:normal}
.rules{display:flex;flex-direction:column;gap:5px;margin-bottom:12px;font-size:13px}
.rules .lab{color:var(--mut);display:inline-block;min-width:78px}
.iso{display:inline-grid;grid-template-columns:auto auto auto;gap:2px 20px;font-size:13px;
 background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px}
.iso .h{color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;font-weight:700}
.iso .r{text-align:right}
.foot{color:var(--mut);font-size:12.5px;margin-top:14px;padding:0 2px;line-height:1.6}
.chtip{position:absolute;pointer-events:none;display:none;background:var(--card);border:1px solid var(--line);
 border-radius:8px;padding:7px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.18);z-index:20;white-space:nowrap}
.chtip b{color:var(--ink)}
#tsvg{touch-action:none;user-select:none}
.fbar{fill:var(--accent)} .fbar.neg{fill:var(--neg)}
.zbtn{padding:4px 10px;border:1px solid var(--line);background:var(--card);color:var(--mut);border-radius:7px;
 cursor:pointer;font-size:12px;font-weight:600}
.zbtn.zon{background:var(--accent);color:#fff;border-color:var(--accent)}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;color:var(--mut);font-size:12px}
</style>

<div class="wrap">
  <h1>Strategy Lab — trading plans &amp; research</h1>
  <div class="sub" id="sub"></div>

  <details class="help">
    <summary>How to read this</summary>
    <div class="help-body">
      <ul>
        <li><b>Backtest window:</b> daily bars 2005→now (~21 yrs where available). Each ticker is split by time:
          the first <b>70% is in-sample (IS)</b> — used to <i>pick</i> the strategy — and the last
          <b>30% is out-of-sample (OOS)</b> — a holdout used to <i>judge</i> it on data it never saw.</li>
        <li><b>Sharpe ratio:</b> return per unit of risk. &gt;1 is strong, 0.5–1 decent, &lt;0 is losing on a
          risk-adjusted basis. We <b>choose on IS Sharpe but show OOS numbers</b>, so the headline figures are honest.</li>
        <li><b>Walk-forward:</b> the plan is re-tested on several separate time slices; the % shown is how many
          were profitable. High = consistent, not a one-off fluke.</li>
        <li><b>Verdict:</b> <span class="badge b-high">★ High confidence</span> = recommended <i>and</i> passed the
          robustness stress-test (PSR&gt;90%, profit-probability&gt;75%, low fragility).
          <span class="badge b-rec">✓ Recommended</span> = held up OOS <i>and</i> beat buy&amp;hold
          <i>and</i> walk-forward ≥75%. <span class="badge b-held">Held OOS</span> = profitable OOS but weaker.
          <span class="badge b-marg">Marginal</span> = didn't clear the bar.</li>
        <li><b>PSR &amp; Monte Carlo (click a row):</b> PSR = probability the edge is real given the track record
          (near 1.0 = strong, 0.5 = coin-flip). Monte Carlo reshuffles the plan's trades thousands of times to
          show a <i>range</i> of returns and the drawdown in a bad ordering — and flags plans with a high chance
          of a &gt;30% drawdown as <b>fragile</b>.</li>
        <li><b>WFO+ (walk-forward):</b> the strictest test — re-pick the best strategy on each window and trade
          the next unseen one (how the weekly rescan actually works). Green = consistently profitable forward.
          <b>Reality check:</b> most plans stay profitable but <i>don't beat buy&amp;hold</i> on strongly-trending
          stocks, because a long-only timing strategy sits out part of the run. Read these as
          <i>risk-managed, selective participation</i> (lower drawdown, defined stops) — not a promise of beating
          the index. Timing adds the most value on range-bound / mean-reverting names.</li>
        <li><b>Exit/stop</b> is the risk rule (e.g. <i>sl_15</i> = 15% stop, <i>trail_10</i> = 10% trailing,
          <i>atr_2x</i> = 2×ATR stop, <i>sl10_tp20</i> = 10% stop / 20% target). Click any plan row for full detail.</li>
      </ul>
    </div>
  </details>

  <div class="cards" id="cards"></div>

  <div class="tabs">
    <div class="tab on" data-v="plans">📋 Trade plans</div>
    <div class="tab" data-v="live">📡 Live tracker</div>
    <div class="tab" data-v="ticker">📈 Ticker</div>
    <div class="tab" data-v="portfolio">📦 Portfolio</div>
    <div class="tab" data-v="strategies">🧪 Strategy performance</div>
    <div class="tab" data-v="top">🔍 Explorer</div>
  </div>
  <div class="bar">
    <input id="q" placeholder="Search ticker / strategy / family…">
    <select id="mkt"><option value="">All markets</option></select>
    <select id="fam"><option value="">All families</option></select>
    <label class="chk"><input type="checkbox" id="recOnly" checked> recommended only</label>
    <label class="chk"><input type="checkbox" id="hcOnly"> ★ high-confidence only</label>
    <button class="tab" onclick="toggleTheme()">◐ Theme</button>
  </div>
  <div id="famChart" style="display:none"></div>
  <div class="panel scroll" id="tablePanel"><table id="tbl"><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
  <div id="portfolioPanel" style="display:none"></div>
  <div id="tickerPanel" style="display:none"></div>
  <div id="livePanel" style="display:none"></div>
  <div class="foot" id="foot"></div>
</div>

<script>
const P = __PAYLOAD__;
const pct=v=>v==null||v===''||isNaN(v)?'—':(v*100).toFixed(1)+'%';
const num=v=>v==null||v===''||isNaN(v)?'—':(+v).toFixed(2);
const sgn=v=>v==null||isNaN(v)?'':(+v>=0?'pos':'neg');
const esc=s=>(s==null?'':(''+s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

let view='plans', sortK='oos_cagr', sortDir=-1, openIdx=new Set();

function verdict(r){
  if(r.high_conf)return '<span class="badge b-high">★ High confidence</span>';
  if(r.recommended)return '<span class="badge b-rec">✓ Recommended</span>';
  if(r.oos_holds)return '<span class="badge b-held">Held OOS</span>';
  return '<span class="badge b-marg">Marginal</span>';
}

// column configs: [key,label,type]
const VIEWS = {
 plans:{data:P.plans, rec:true, sort:'oos_cagr', expand:true, cols:[
   ['ticker','Ticker','tick'],['market','Market','pill'],['strategy','Strategy','txt'],
   ['exit_policy','Exit / stop','pill'],['oos_cagr','OOS CAGR/yr','pct'],
   ['oos_total_return','OOS total','pct'],['oos_sharpe','OOS Sharpe','num'],
   ['oos_win_rate','Win','pct'],['avg_duration','Avg hold','days'],
   ['wf_consistency','Walk-fwd','pct'],['psr','PSR','num'],['wfo_pct_windows_pos','WFO+','wfo'],
   ['liquidity_tier','Liq','liq'],['verdict','Verdict','verdict']]},
 strategies:{data:P.strategies, rec:false, sort:'median_oos_sharpe', expand:true, cols:[
   ['strategy','Strategy','txt'],['family','Family','pill'],['tickers','Tickers','int'],
   ['median_oos_sharpe','Med OOS Sharpe','num'],['median_oos_cagr','Med OOS CAGR','pct'],
   ['pct_positive_oos','% profitable OOS','pctbar'],['pct_beat_bh_oos','% beat B&H','pct'],
   ['median_oos_win_rate','Med win','pct'],['median_oos_maxdd','Med maxDD','pct']]},
 top:{data:P.top, rec:false, sort:'oos_sharpe', expand:false, cols:[
   ['ticker','Ticker','tick'],['market','Market','pill'],['strategy','Strategy','txt'],
   ['params','Params','txt'],['exit_policy','Exit','pill'],['is_sharpe','IS Sharpe','num'],
   ['oos_sharpe','OOS Sharpe','num'],['oos_cagr','OOS CAGR','pct'],
   ['oos_total_return','OOS total','pct'],['oos_n_trades','Trades','int']]},
};

function fmt(v,type,r){
  if(type==='pct')return `<span class="${sgn(v)}">${pct(v)}</span>`;
  if(type==='num')return `<span class="${sgn(v)}">${num(v)}</span>`;
  if(type==='int')return v==null?'—':v;
  if(type==='days')return v==null||isNaN(v)?'—':Math.round(v)+'d';
  if(type==='txt')return esc(v==null?'—':v);
  if(type==='tick')return `<span class="tick">${esc(v)}</span>`;
  if(type==='pill')return v?`<span class="pill">${esc(v)}</span>`:'—';
  if(type==='liq'){const c={liquid:'lq-ok',ok:'lq-mid',thin:'lq-bad',unknown:'lq-un'}[v]||'lq-un';
    return `<span class="pill ${c}">${esc(v||'?')}</span>`;}
  if(type==='wfo'){if(v==null)return '<span class="pill">n/a</span>';
    const c=r.wfo_survivor?'lq-ok':'lq-bad';return `<span class="pill ${c}">${pct(v)}</span>`;}
  if(type==='verdict')return verdict(r);
  if(type==='pctbar'){const w=Math.round((v||0)*70);
    return `<span class="minibar" style="width:${w}px"></span>${pct(v)}`;}
  return esc(v);
}

function strategyDetail(r){
  const colspan = VIEWS.strategies.cols.length;
  return `<tr class="detail"><td colspan="${colspan}"><div class="dwrap">
    <div style="font-size:13.5px;max-width:760px;margin-bottom:12px">${esc(r.blurb||'—')}</div>
    <div class="rules">
      <div><span class="lab">Buy when</span>${esc(r.entry_desc||'—')}</div>
      <div><span class="lab">Sell when</span>${esc(r.exit_desc||'—')}</div>
      <div><span class="lab">Family</span><span class="pill">${esc(r.family||'—')}</span></div>
      <div><span class="lab">Params tested</span>${esc(r.param_grid||'—')}</div>
    </div>
    <div class="legend" style="margin-top:10px">
      <span>Tested on <b>${r.tickers}</b> tickers · <b>${r.combos}</b> combos</span>
      <span>Median OOS Sharpe: <b class="${sgn(r.median_oos_sharpe)}">${num(r.median_oos_sharpe)}</b></span>
      <span>Profitable OOS: <b>${pct(r.pct_positive_oos)}</b> of combos</span>
      <span>Beat buy&amp;hold: <b>${pct(r.pct_beat_bh_oos)}</b></span>
    </div>
  </div></td></tr>`;
}
function detail(r){
  const rule=(lab,val)=>`<div><span class="lab">${lab}</span>${esc(val||'—')}</div>`;
  const row=(lab,isk,ok,ty)=>`<div>${lab}</div><div class="r">${fmt(r['is_'+isk],ty)}</div><div class="r">${fmt(r['oos_'+ok],ty)}</div>`;
  return `<tr class="detail"><td colspan="10"><div class="dwrap">
    <div class="rules">
      ${rule('Buy when', r.entry_rule)}
      ${rule('Sell when', r.exit_rule)}
      ${rule('Stop / exit', r.stop_rule)}
    </div>
    <div class="iso">
      <div class="h"></div><div class="h r">In-sample</div><div class="h r">Out-of-sample</div>
      ${row('Total return','total_return','total_return','pct')}
      ${row('CAGR / yr','cagr','cagr','pct')}
      ${row('Sharpe','sharpe','sharpe','num')}
      ${row('Max drawdown','max_drawdown','max_drawdown','pct')}
      ${row('Win rate','win_rate','win_rate','pct')}
      ${row('Payoff (win/loss)','payoff','payoff','num')}
      ${row('Expectancy (R)','expectancy_R','expectancy_R','num')}
      ${row('# trades','n_trades','n_trades','int')}
    </div>
    <div class="legend">
      <span>Buy &amp; hold OOS: <b class="${sgn(r.oos_buy_hold_return)}">${pct(r.oos_buy_hold_return)}</b></span>
      <span>Beats buy&amp;hold: <b>${r.beats_bh?'yes':'no'}</b></span>
      <span>Walk-forward consistency: <b>${pct(r.wf_consistency)}</b></span>
    </div>
    ${r.psr!=null?`<div style="margin-top:12px;font-weight:700">Robustness (Monte Carlo, ${r.oos_n_trades||'—'} trades)</div>
    <div class="legend" style="margin-top:6px">
      <span>Likely-return range (5–95%): <b class="${sgn(r.mc_ret_p5)}">${pct(r.mc_ret_p5)}</b> … <b class="${sgn(r.mc_ret_p95)}">${pct(r.mc_ret_p95)}</b> (median <b>${pct(r.mc_ret_p50)}</b>)</span>
      <span>Prob. of profit: <b>${pct(r.mc_p_profit)}</b></span>
      <span>Prob. of &gt;30% drawdown: <b class="${(r.mc_p_dd_gt_30>0.25)?'neg':''}">${pct(r.mc_p_dd_gt_30)}</b>${(r.mc_p_dd_gt_30>0.25)?' ⚠ fragile':''}</span>
    </div>
    <div class="legend" style="margin-top:4px">
      <span>PSR (edge is real): <b>${pct(r.psr)}</b></span>
      <span>Worst holdout year: <b class="${sgn(r.worst_year_ret)}">${pct(r.worst_year_ret)}</b></span>
      <span>Years profitable: <b>${pct(r.years_positive)}</b></span>
      <span>Combos tried for this ticker: <b>${r.n_trials||'—'}</b> (${pct(r.trials_oos_hit_rate)} profitable)</span>
    </div>`:''}
    ${r.wfo_pct_windows_pos!=null?`<div style="margin-top:12px;font-weight:700">Walk-forward (re-optimized each window — strictest test)</div>
    <div class="legend" style="margin-top:6px">
      <span>Windows profitable: <b class="${r.wfo_survivor?'pos':'neg'}">${pct(r.wfo_pct_windows_pos)}</b></span>
      <span>Walk-forward return: <b class="${sgn(r.wfo_total_return)}">${pct(r.wfo_total_return)}</b> vs buy&amp;hold <b>${pct(r.wfo_bh_return)}</b></span>
      <span>Worst window: <b class="${sgn(r.wfo_worst_window)}">${pct(r.wfo_worst_window)}</b></span>
      <span>Beats buy&amp;hold: <b>${r.wfo_beats_bh?'yes':'no'}</b></span>
    </div>`:''}
    ${r.liquidity_tier?`<div style="margin-top:12px;font-weight:700">Tradeability</div>
    <div class="legend" style="margin-top:6px">
      <span>Liquidity: <b>${esc(r.liquidity_tier)}</b></span>
      <span>Avg daily volume: <b>${r.adv_aud?('$'+(r.adv_aud/1e6).toFixed(1)+'M'):'—'}</b></span>
      <span>Max position (~2% of ADV): <b>${r.max_pos_aud?('$'+Math.round(r.max_pos_aud/1e3)+'k'):'—'}</b></span>
      <span>Avg hold: <b>${r.avg_duration?Math.round(r.avg_duration)+' trading days':'—'}</b></span>
    </div>`:''}
    ${tradeLog(r)}
  </div></td></tr>`;
}

function tradeLog(r){
  const key=r.ticker+'|'+r.strategy+'|'+r.exit_policy, t=(P.trades||{})[key];
  if(!t||!t.length)return '';
  let cum=1;
  const rows=t.map((x,i)=>{const [ed,xd,ep,xp,ret,bars,st]=x;
    const sell=xd||((st&&st.toLowerCase()!=='closed')?'<i>open</i>':'—');
    if(ret!=null)cum*=(1+ret);
    const cumv=cum-1;
    return `<tr><td class="txt">${i+1}</td><td class="txt">${ed||'—'}</td><td class="txt">${sell}</td>`+
      `<td>${ep!=null?ep.toFixed(2):'—'}</td><td>${xp!=null?xp.toFixed(2):'—'}</td>`+
      `<td class="${sgn(ret)}">${ret!=null?(ret*100).toFixed(1)+'%':'—'}</td>`+
      `<td class="${sgn(cumv)}"><b>${ret!=null?(cumv*100).toFixed(1)+'%':'—'}</b></td>`+
      `<td>${bars!=null?bars+'d':'—'}</td></tr>`;}).join('');
  return `<div style="margin-top:14px;font-weight:700">All trades — out-of-sample (${t.length})</div>
    <div class="scroll" style="max-height:300px;overflow-y:auto;margin-top:6px;border:1px solid var(--line);border-radius:8px">
    <table style="font-size:12.5px"><thead><tr>
      <th class="txt">#</th><th class="txt">Buy date</th><th class="txt">Sell date</th>
      <th>Entry</th><th>Exit</th><th>Return</th><th>Cum P&amp;L</th><th>Held</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}
function options(id,vals){const sel=document.getElementById(id);
  [...new Set(vals.filter(Boolean))].sort().forEach(v=>{const o=document.createElement('option');
    o.value=o.textContent=v;sel.appendChild(o);});}

function cards(){
  const s=P.summary;
  const items=[['Tickers',s.n_tickers+(s.n_markets?` <small>/ ${s.n_markets} mkts</small>`:'')],
    ['Strategies',s.n_strategies],['Combos tested',s.n_combos.toLocaleString()],
    ['Recommended',(s.n_recommended??'—')+(s.n_plans?` <small>/ ${s.n_plans}</small>`:'')],
    ['★ High confidence',(s.n_high_conf??'—')+(s.n_recommended?` <small>/ ${s.n_recommended} rec</small>`:'')]];
  document.getElementById('cards').innerHTML=items.map(([k,v])=>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
  document.getElementById('sub').innerHTML=
    `Chosen on <b>in-sample</b> Sharpe, reported <b>out-of-sample</b> · ${s.n_exit_policies} exit/stop policies · `+
    `daily bars · scan ${s.scan_id} · ${s.generated}`;
}

function render(){
  const fc=document.getElementById('famChart');
  fc.style.display=(view==='strategies')?'block':'none';
  if(view==='strategies')fc.innerHTML=familyChart();
  const cfg=VIEWS[view], q=document.getElementById('q').value.toLowerCase();
  const mkt=document.getElementById('mkt').value, fam=document.getElementById('fam').value;
  const recOnly=document.getElementById('recOnly').checked;
  const hcOnly=document.getElementById('hcOnly').checked;
  let rows=cfg.data.slice();
  if(q)rows=rows.filter(r=>[r.ticker,r.strategy,r.family,r.params].some(x=>x&&(''+x).toLowerCase().includes(q)));
  if(mkt)rows=rows.filter(r=>r.market===mkt);
  if(fam)rows=rows.filter(r=>r.family===fam);
  if(hcOnly&&cfg.rec)rows=rows.filter(r=>r.high_conf);
  else if(recOnly&&cfg.rec)rows=rows.filter(r=>r.recommended);
  if(sortK)rows.sort((a,b)=>{let x=a[sortK],y=b[sortK];
    x=(x==null||x==='')?-1e18:(isNaN(x)?(''+x):+x);y=(y==null||y==='')?-1e18:(isNaN(y)?(''+y):+y);
    return (x>y?1:x<y?-1:0)*sortDir;});

  document.getElementById('head').innerHTML=cfg.cols.map(c=>
    `<th class="${['txt','tick','pill','verdict'].includes(c[2])?'txt':''}" onclick="sortBy('${c[0]}')">${c[1]}</th>`).join('');
  let html='';
  rows.forEach((r,i)=>{
    const idkey=(r.ticker||'')+r.strategy+(r.params||'')+(r.exit_policy||'');
    const cells=cfg.cols.map(c=>{const cls=['txt','tick','pill','verdict'].includes(c[2])?'txt':'';
      return `<td class="${cls}">${fmt(r[c[0]],c[2],r)}</td>`;}).join('');
    html+=`<tr class="main" data-k="${esc(idkey)}" ${cfg.expand?`onclick="toggle(this,${i})"`:''}>${cells}</tr>`;
    if(cfg.expand&&openIdx.has(idkey))html+=(view==='strategies'?strategyDetail(r):detail(r));
  });
  document.getElementById('body').innerHTML=html;

  const notes={plans:'One row per ticker: its best full plan, chosen on in-sample Sharpe and shown out-of-sample. '+
      'Click a row for the exact buy/sell rules and IS-vs-OOS breakdown. Turn off "recommended only" to see every plan.',
    strategies:'How each strategy did across ALL tickers & markets. <b>% profitable OOS</b> and <b>% beat B&amp;H</b> show consistency — not one lucky ticker.',
    top:'Top combos per ticker by in-sample Sharpe. Full 50k+ row grid is in results/all_results.csv.'};
  document.getElementById('foot').innerHTML=`Showing ${rows.length} rows. ${notes[view]}<br>`+
    `Returns net of 0.1% brokerage + 5bps slippage. <b>Research/education only — not financial advice.</b>`;
}

function toggle(tr,i){const k=tr.dataset.k; if(openIdx.has(k))openIdx.delete(k); else openIdx.add(k); render();}
function sortBy(k){if(sortK===k)sortDir*=-1;else{sortK=k;sortDir=-1;}render();}

function pchart(p){
  const W=820,H=260,pad=34,s=p.equity_values,b=p.bench_values,g=p.regime_equity,n=s.length;
  const all=s.concat(b).concat(g||[]),mn=Math.min(...all),mx=Math.max(...all);
  const x=i=>pad+(W-2*pad)*i/(n-1),y=v=>H-pad-(H-2*pad)*(v-mn)/((mx-mn)||1);
  const line=a=>a.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const gpath=(g&&g.length===n)?`<path d="${line(g)}" fill="none" stroke="var(--pos)" stroke-width="2.2"/>`:'';
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;max-height:280px">
    <path d="${line(b)}" fill="none" stroke="var(--mut)" stroke-width="1.5" opacity="0.65"/>
    <path d="${line(s)}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>${gpath}</svg>`;
}
function statRow(label,m){return `<tr><td class="txt">${label}</td>
  <td class="${sgn(m.cagr)}">${pct(m.cagr)}</td><td>${num(m.sharpe)}</td>
  <td class="neg">${pct(m.max_drawdown)}</td><td class="${sgn(m.total_return)}">${pct(m.total_return)}</td></tr>`;}
function renderPortfolio(){
  const el=document.getElementById('portfolioPanel');
  if(!P.portfolios||!P.portfolios.length){el.innerHTML='<div class="panel" style="padding:20px;color:var(--mut)">No portfolio data yet.</div>';return;}
  el.innerHTML=`<div class="sub" style="margin:-4px 0 14px">The real case for the system: each plan is one sleeve of a book. Individually they lag buy&amp;hold, but combined (near-zero correlation) the book has far better <b>risk-adjusted</b> returns and smaller drawdowns.</div>`+
   P.portfolios.map(p=>`<div class="panel" style="padding:18px;margin-bottom:16px">
    <div style="font-weight:700;font-size:16px">${p.label.replace(/_/g,' ')} book — ${p.n_sleeves} sleeves</div>
    <div class="sub" style="margin-bottom:12px">avg pairwise correlation <b>${num(p.avg_correlation)}</b> (low = well diversified)</div>
    <div class="scroll"><table style="width:auto;margin-bottom:14px"><thead><tr>
      <th class="txt"></th><th>CAGR/yr</th><th>Sharpe</th><th>Max DD</th><th>Total (OOS)</th></tr></thead>
      <tbody>${statRow('📦 Strategy book',p.strategy)}${p.regime?statRow('📦 + regime overlay',p.regime):''}${statRow('Equal-wt buy &amp; hold',p.buy_hold)}</tbody></table></div>
    <div style="display:flex;gap:16px;font-size:12px;color:var(--mut);margin-bottom:4px;flex-wrap:wrap">
      <span><span style="display:inline-block;width:16px;height:3px;background:var(--accent);vertical-align:middle"></span> strategy book</span>
      ${p.regime_equity?'<span><span style="display:inline-block;width:16px;height:3px;background:var(--pos);vertical-align:middle"></span> + regime overlay</span>':''}
      <span><span style="display:inline-block;width:16px;height:3px;background:var(--mut);vertical-align:middle"></span> buy &amp; hold</span>
      <span style="margin-left:auto">equity (× start), out-of-sample</span></div>
    ${pchart(p)}
  </div>`).join('');
}
function planForTicker(tk){
  const rec=P.plans.filter(p=>p.ticker===tk && p.recommended);
  const pool=(rec.length?rec:P.plans.filter(p=>p.ticker===tk)).slice();
  return pool.sort((a,b)=>(b.oos_cagr||0)-(a.oos_cagr||0))[0];
}
// --- interactive price chart (crosshair + zoom buttons/drag + trade P&L) ------
let CH={}, CHdrag=null;
const CW=900,CHH=340,PL=56,PRr=16,PT=18,PB=30;
function chIdx(ds){const d=CH.d;let lo=0,hi=d.length-1;while(lo<hi){const m=(lo+hi)>>1;if(d[m]<ds)lo=m+1;else hi=m;}return lo;}
function CX(i){return PL+(CW-PL-PRr)*(i-CH.lo)/Math.max(1,CH.hi-CH.lo);}
function CY(v){return CHH-PB-(CHH-PT-PB)*(v-CH.mn)/((CH.mx-CH.mn)||1);}
function setA(id,o){var e=document.getElementById(id); if(e){for(var k in o)e.setAttribute(k,o[k]);}}
function zoomTo(bars){if(!CH.c)return; const n=CH.c.length;
  if(!bars){CH.lo=0;CH.hi=n-1;} else {CH.hi=n-1; CH.lo=Math.max(0,n-1-bars);} drawChart();
  document.querySelectorAll('.zbtn').forEach(b=>b.classList.toggle('zon',(+b.dataset.b)===(bars||0)));}
function _pxi(clientX){const s=document.getElementById('tsvg'); const r=s.getBoundingClientRect();
  const x=(clientX-r.left)/r.width*CW; return {x:x, i:Math.round(CH.lo+(x-PL)/(CW-PL-PRr)*Math.max(1,CH.hi-CH.lo))};}
function drawChart(){
  const svg=document.getElementById('tsvg'); if(!svg)return;
  const c=CH.c,d=CH.d; let mn=Infinity,mx=-Infinity;
  for(let i=CH.lo;i<=CH.hi;i++){if(c[i]<mn)mn=c[i]; if(c[i]>mx)mx=c[i];}
  const pv=(mx-mn)*0.06||1; mn-=pv; mx+=pv; CH.mn=mn; CH.mx=mx;   // padding keeps the line off the edges
  let path=''; for(let i=CH.lo;i<=CH.hi;i++){path+=(i===CH.lo?'M':'L')+CX(i).toFixed(1)+','+CY(c[i]).toFixed(1)+' ';}
  CH.hits=[]; let marks='';
  (CH.trades||[]).forEach(t=>{const ei=chIdx(t[0]), xi=(t[1]?chIdx(t[1]):null);
    if(ei>=CH.lo&&ei<=CH.hi){marks+=`<path d="M${CX(ei).toFixed(1)},${(CY(t[2])+8).toFixed(1)} l-5,10 l10,0 z" fill="var(--pos)"/>`;CH.hits.push({x:CX(ei),kind:'buy',t});}
    if(xi!=null&&xi>=CH.lo&&xi<=CH.hi){marks+=`<path d="M${CX(xi).toFixed(1)},${(CY(t[3])-8).toFixed(1)} l-5,-10 l10,0 z" fill="var(--neg)"/>`;CH.hits.push({x:CX(xi),kind:'sell',t});}
  });
  let axis='';
  for(let k=0;k<=4;k++){const v=mn+(mx-mn)*k/4, yy=CY(v);   // 5 price gridlines
    axis+=`<line x1="${PL}" y1="${yy.toFixed(1)}" x2="${CW-PRr}" y2="${yy.toFixed(1)}" stroke="var(--line)" stroke-width="0.6"/>`;
    axis+=`<text x="${PL-6}" y="${(yy+3).toFixed(1)}" text-anchor="end" fill="var(--mut)" font-size="10">${v.toFixed(2)}</text>`;}
  const nd=Math.max(1,Math.min(7,CH.hi-CH.lo));                // up to 8 date gridlines
  for(let k=0;k<=nd;k++){const i=CH.lo+Math.round((CH.hi-CH.lo)*k/nd);
    axis+=`<line x1="${CX(i).toFixed(1)}" y1="${PT}" x2="${CX(i).toFixed(1)}" y2="${CHH-PB}" stroke="var(--line)" stroke-width="0.4" opacity="0.5"/>`;
    axis+=`<text x="${CX(i).toFixed(1)}" y="${CHH-9}" text-anchor="${k===0?'start':k===nd?'end':'middle'}" fill="var(--mut)" font-size="9.5">${d[i]}</text>`;}
  svg.innerHTML=axis
    +`<path d="${path}" fill="none" stroke="var(--accent)" stroke-width="1.6"/>`+marks
    +`<line id="cvx" y1="${PT}" y2="${CHH-PB}" x1="0" x2="0" stroke="var(--ink)" stroke-width="0.8" stroke-dasharray="3,3" opacity="0"/>`
    +`<line id="cvy" x1="${PL}" x2="${CW-PRr}" y1="0" y2="0" stroke="var(--ink)" stroke-width="0.8" stroke-dasharray="3,3" opacity="0"/>`
    +`<circle id="cdot" r="3.5" fill="var(--accent)" opacity="0"/>`
    +`<rect id="sel" y="${PT}" height="${CHH-PT-PB}" x="0" width="0" fill="var(--accent)" opacity="0.14"/>`
    +`<rect id="cov" x="${PL}" y="${PT}" width="${CW-PL-PRr}" height="${CHH-PT-PB}" fill="transparent" style="cursor:crosshair"/>`;
  bindChart();
}
function bindChart(){
  const ov=document.getElementById('cov'), svg=document.getElementById('tsvg'), tip=document.getElementById('chtip'); if(!ov)return;
  ov.onmousemove=e=>{const p=_pxi(e.clientX); const i=Math.max(CH.lo,Math.min(CH.hi,p.i)), cx=CX(i), cy=CY(CH.c[i]);
    setA('cvx',{x1:cx,x2:cx,opacity:0.4}); setA('cvy',{y1:cy,y2:cy,opacity:0.4}); setA('cdot',{cx:cx,cy:cy,opacity:1});
    let hit=null; for(const m of CH.hits){if(Math.abs(m.x-p.x)<7){hit=m;break;}}
    const r=svg.getBoundingClientRect(); tip.style.display='block';
    let tx=e.clientX-r.left+14; if(tx>r.width-175)tx=e.clientX-r.left-160;
    tip.style.left=tx+'px'; tip.style.top=(e.clientY-r.top-4)+'px';
    if(hit){const t=hit.t, ret=t[4]; tip.innerHTML=`<b>${hit.kind==='buy'?'▲ BUY':'▼ SELL'}</b> ${hit.kind==='buy'?t[0]:(t[1]||'open')} @ ${(hit.kind==='buy'?t[2]:t[3]).toFixed(2)}<br>trade gain: <b class="${ret>=0?'pos':'neg'}">${ret!=null?(ret>=0?'+':'')+(ret*100).toFixed(1)+'%':'—'}</b>${t[5]!=null?' · held '+t[5]+'d':''}`;}
    else tip.innerHTML=`<b>${CH.d[i]}</b> · price <b>${CH.c[i].toFixed(2)}</b>`;
    if(CHdrag!=null){const a=Math.min(CHdrag,p.x),b=Math.max(CHdrag,p.x); setA('sel',{x:a,width:b-a});}
  };
  ov.onmouseleave=()=>{tip.style.display='none'; ['cvx','cvy','cdot'].forEach(id=>setA(id,{opacity:0}));};
  ov.onmousedown=e=>{e.preventDefault(); CHdrag=_pxi(e.clientX).x;};
  ov.ondblclick=()=>zoomTo(0);
  if(!window._chUp){window._chUp=true; document.addEventListener('mouseup',e=>{ if(CHdrag==null||!CH.c)return;
    const x=_pxi(e.clientX).x; const toI=xx=>Math.round(CH.lo+(xx-PL)/(CW-PL-PRr)*Math.max(1,CH.hi-CH.lo));
    let a=toI(Math.min(CHdrag,x)), b=toI(Math.max(CHdrag,x)); CHdrag=null; setA('sel',{width:0});
    a=Math.max(0,a); b=Math.min(CH.c.length-1,b); if(b-a>=3){CH.lo=a;CH.hi=b;drawChart();} });}
}
function familyChart(){
  const fams=P.families||[]; if(!fams.length)return '';
  const mx=Math.max(...fams.map(f=>f.median_oos_sharpe))||1;
  const rows=fams.map(f=>{const w=Math.max(2,Math.round((f.median_oos_sharpe/mx)*100));
    return `<div style="display:grid;grid-template-columns:135px 1fr auto;gap:10px;align-items:center;margin:6px 0">
      <div style="font-size:13px">${esc(f.family.replace(/_/g,' '))}</div>
      <div style="background:var(--soft);border-radius:5px;height:18px"><div style="width:${w}%;height:100%;background:var(--accent);border-radius:5px"></div></div>
      <div style="font-size:12px;color:var(--mut)"><b class="${sgn(f.median_oos_sharpe)}">${num(f.median_oos_sharpe)}</b> Sharpe · ${pct(f.pct_positive_oos)} profit · ${pct(f.pct_beat_bh_oos)} beat B&amp;H</div>
    </div>`;}).join('');
  return `<div class="panel" style="padding:16px 18px;margin-bottom:14px">
    <div style="font-weight:700;margin-bottom:2px">Strategy families — median out-of-sample Sharpe</div>
    <div class="sub" style="margin-bottom:8px">Risk-adjusted performance across all tickers &amp; combos. Click any strategy row below for its description.</div>
    ${rows}</div>`;
}
function renderTicker(){
  const el=document.getElementById('tickerPanel');
  const tickers=[...new Set(P.plans.filter(p=>p.recommended).map(p=>p.ticker))].sort();
  if(!tickers.length){el.innerHTML='<div class="panel" style="padding:20px;color:var(--mut)">No recommended plans yet.</div>';return;}
  if(!window._tsel||!tickers.includes(window._tsel))window._tsel=tickers[0];
  const tk=window._tsel, plan=planForTicker(tk), pr=(P.prices||{})[tk];
  const key=tk+'|'+plan.strategy+'|'+plan.exit_policy, trades=(P.trades||{})[key];
  const opts=tickers.map(t=>`<option value="${t}" ${t===tk?'selected':''}>${t}</option>`).join('');
  const chart=pr?`<div style="display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;align-items:center">
      <span style="font-size:12px;color:var(--mut);margin-right:2px">zoom:</span>
      <button class="zbtn zon" data-b="0" onclick="zoomTo(0)">All</button>
      <button class="zbtn" data-b="756" onclick="zoomTo(756)">3Y</button>
      <button class="zbtn" data-b="252" onclick="zoomTo(252)">1Y</button>
      <button class="zbtn" data-b="126" onclick="zoomTo(126)">6M</button>
      <button class="zbtn" data-b="63" onclick="zoomTo(63)">3M</button></div>
     <div id="chartWrap" style="position:relative"><svg id="tsvg" viewBox="0 0 ${CW} ${CHH}" style="width:100%;height:auto;max-height:380px"></svg><div id="chtip" class="chtip"></div></div>
     <div class="sub" style="margin-top:4px">Hover for date/price · hover a ▲/▼ for that trade's gain · drag across the chart to zoom · double-click or "All" to reset</div>`:'<div style="color:var(--mut);padding:24px">No price series embedded for this ticker.</div>';
  const stat=(lab,v,cls)=>`<span>${lab} <b class="${cls||''}">${v}</b></span>`;
  el.innerHTML=`
   <div class="bar"><select id="tsel" onchange="window._tsel=this.value;renderTicker()" style="min-width:150px">${opts}</select>
     <span style="color:var(--mut);font-size:12.5px;align-self:center">best recommended plan for this ticker</span></div>
   <div class="panel" style="padding:16px 18px">
     <div style="font-weight:700;font-size:16px;margin-bottom:2px">${verdict(plan)} &nbsp; <span class="tick">${tk}</span> — ${plan.strategy} <span class="pill">${esc(plan.exit_policy)}</span></div>
     <div class="legend" style="margin:8px 0 12px">
       ${stat('OOS CAGR',pct(plan.oos_cagr),sgn(plan.oos_cagr))}
       ${stat('OOS total',pct(plan.oos_total_return),sgn(plan.oos_total_return))}
       ${stat('Sharpe',num(plan.oos_sharpe))}
       ${stat('Win rate',pct(plan.oos_win_rate))}
       ${stat('Max DD',pct(plan.oos_max_drawdown),'neg')}
       ${stat('PSR',pct(plan.psr))}
       ${stat('Walk-fwd',pct(plan.wf_consistency))}
       ${stat('Avg hold',plan.avg_duration?Math.round(plan.avg_duration)+'d':'—')}
     </div>
     <div style="display:flex;gap:16px;font-size:12px;color:var(--mut);margin-bottom:2px;flex-wrap:wrap">
       <span><span style="display:inline-block;width:16px;height:3px;background:var(--accent);vertical-align:middle"></span> price (out-of-sample)</span>
       <span style="color:var(--pos)">▲ buy</span><span style="color:var(--neg)">▼ sell</span></div>
     ${chart}
     <div class="rules" style="margin-top:12px">
       <div><span class="lab">Buy when</span>${esc(plan.entry_rule)}</div>
       <div><span class="lab">Sell when</span>${esc(plan.exit_rule)}</div>
       <div><span class="lab">Stop / exit</span>${esc(plan.stop_rule)}</div>
     </div>
     ${tradeLog(plan)}
   </div>`;
  if(pr){CH={d:pr.d,c:pr.c,trades:trades||[],lo:0,hi:pr.c.length-1}; drawChart();}
}
function leq(dates,vals,cap){
  const W=880,H=220,pad=48,n=vals.length;
  const all=vals.concat([cap]), mn=Math.min(...all), mx=Math.max(...all);
  const x=i=>pad+(W-2*pad)*i/Math.max(1,n-1), y=v=>H-28-(H-44)*(v-mn)/((mx-mn)||1);
  const line=vals.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const z=y(cap);
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;max-height:240px">
    <line x1="${pad}" x2="${W-pad}" y1="${z.toFixed(1)}" y2="${z.toFixed(1)}" stroke="var(--mut)" stroke-dasharray="3,3" opacity="0.5"/>
    <text x="${pad}" y="${(z-4).toFixed(1)}" fill="var(--mut)" font-size="9">start $${cap.toLocaleString()}</text>
    <path d="${line}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>
    <text x="${pad}" y="12" fill="var(--mut)" font-size="10">${dates[0]||''}</text>
    <text x="${W-pad}" y="12" fill="var(--mut)" font-size="10" text-anchor="end">${dates[n-1]||''}</text></svg>`;
}
function crossuPanel(){
  const c=P.crossu; if(!c||c.retention==null)return '';
  const pct=v=>(v*100).toFixed(0)+'%';
  const ret=c.retention, rc=c.rank_corr;
  // verdict colouring
  const good=ret>=0.7&&rc>=0.6, ok=ret>=0.5&&rc>=0.4;
  const vcol=good?'var(--pos)':(ok?'var(--amber)':'var(--neg)');
  const verdict=good?'✅ The edge transfers — real, not curve-fit'
    :(ok?'⚠️ Partial transfer — treat with caution':'❌ Edge does NOT transfer — likely overfit');
  const per=(c.per||[]).filter(p=>p.retention!=null);
  const top=per.slice(0,6), bot=per.slice(-4).reverse();
  const prow=p=>`<tr><td class="txt"><b>${esc(p.strategy)}</b></td>
     <td class="pos">+${(p.discovery*100).toFixed(2)}%</td>
     <td class="${p.heldout>=0?'pos':'neg'}"><b>${p.heldout>=0?'+':''}${(p.heldout*100).toFixed(2)}%</b></td>
     <td class="${p.retention>=0.7?'pos':(p.retention>=0.4?'':'neg')}">${p.retention!=null?(p.retention*100).toFixed(0)+'%':'—'}</td></tr>`;
  return `<div class="panel" style="padding:16px 18px;margin-bottom:14px;border-left:4px solid ${vcol}">
     <div style="font-weight:700;font-size:16px;margin-bottom:4px">🔬 Is the edge real? — out-of-universe test</div>
     <div class="sub" style="margin-bottom:10px">The overfitting acid test: pick each strategy's best parameters using a random <b>half</b> of the ${c.n_tickers} tickers, then measure that exact config on the <b>other half it never touched</b> — averaged over ${c.n_splits} random splits. If the edge were curve-fit luck, it would vanish on held-out tickers.</div>
     <div class="legend" style="margin-bottom:8px">
       <span>Edge where selected <b class="pos">+${(c.discovery_edge*100).toFixed(2)}%</b>/trade</span>
       <span>Edge on held-out <b class="${c.heldout_edge>=0?'pos':'neg'}">+${(c.heldout_edge*100).toFixed(2)}%</b>/trade</span>
       <span>Retention <b style="color:${vcol}">${pct(ret)}</b></span>
       <span>Still profitable <b class="pos">${pct(c.pct_heldout_positive)}</b></span>
       <span>Rank corr <b style="color:${vcol}">${rc.toFixed(2)}</b></span>
     </div>
     <div style="font-weight:700;color:${vcol};margin:6px 0 10px">${verdict}</div>
     <div class="sub" style="margin-bottom:10px">⚠️ This proves edges transfer <b>across tickers</b>, not across <b>time</b> — and it's a <b>gross</b> per-trade edge. A +${(c.heldout_edge*100).toFixed(2)}%/trade edge still has to clear commissions, which is why parcel size &amp; ASX-focus matter. Time-robustness is what the walk-forward and the live tracker prove.</div>
     <div class="grid2" style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
       <div><div style="font-weight:600;font-size:12.5px;margin-bottom:4px">Transfers cleanly (trust these)</div>
         <div class="scroll" style="border:1px solid var(--line);border-radius:8px"><table style="font-size:12px"><thead><tr><th class="txt">Strategy</th><th>Selected</th><th>Held-out</th><th>Keep</th></tr></thead><tbody>${top.map(prow).join('')}</tbody></table></div></div>
       <div><div style="font-weight:600;font-size:12.5px;margin-bottom:4px">Doesn't transfer (noise)</div>
         <div class="scroll" style="border:1px solid var(--line);border-radius:8px"><table style="font-size:12px"><thead><tr><th class="txt">Strategy</th><th>Selected</th><th>Held-out</th><th>Keep</th></tr></thead><tbody>${bot.map(prow).join('')}</tbody></table></div></div>
     </div>
   </div>`;
}
function whatifPanel(){
  const w=P.whatif; if(!w||!w.sweep)return '';
  const pc=v=>((v>=0?'+':'')+(v*100).toFixed(0)+'%');
  const base=w.sweep.find(x=>x.stake===1000)||w.sweep[0];
  const rowsSweep=w.sweep.map(x=>`<tr class="${x.stake===1000?'':''}">
     <td class="txt"><b>$${x.stake.toLocaleString()}</b> each</td>
     <td>$${x.invested.toLocaleString()}</td>
     <td>$${x.gross.toLocaleString()}</td>
     <td class="pos">${pc(x.gross_pct)}</td>
     <td><b>$${x.net.toLocaleString()}</b></td>
     <td class="${x.net_pct>=0?'pos':'neg'}"><b>${pc(x.net_pct)}</b></td></tr>`).join('');
  const per=(w.per||[]).slice().sort((a,b)=>b.net_1k-a.net_1k);
  const perRows=per.map(p=>`<tr>
     <td class="txt"><b>${esc(p.ticker)}</b></td><td class="txt">${esc(p.strategy)}</td>
     <td class="txt">${esc(p.rating||'—')}</td><td class="txt">${esc(p.market)}</td>
     <td>${p.trades}</td><td class="txt">${esc(p.first)}→${esc(p.last)}</td>
     <td>$${p.gross_1k.toLocaleString()}</td>
     <td class="${p.net_1k>=1000?'pos':'neg'}"><b>$${p.net_1k.toLocaleString()}</b></td></tr>`).join('');
  return `<div class="panel" style="padding:16px 18px;margin-bottom:14px">
     <div style="font-weight:700;font-size:16px;margin-bottom:4px">💡 What-if: $1k into each high-recommended plan (OOS backtest)</div>
     <div class="sub" style="margin-bottom:10px">If you'd put an equal parcel into all <b>${w.n_tickers}</b> high-confidence plans and taken every out-of-sample signal — <b>${w.n_trades}</b> trades across <b>${esc(w.span_start)} → ${esc(w.span_end)}</b> (~${w.years} yrs). Each ticker compounds its own trades; <b>Net</b> deducts CommSec commission on every buy &amp; sell.</div>
     <div class="scroll" style="border:1px solid var(--line);border-radius:8px"><table style="font-size:12.5px"><thead><tr>
       <th class="txt">Parcel</th><th>Invested</th><th>Gross value</th><th>Gross</th><th>Net value</th><th>Net (after fees)</th></tr></thead><tbody>${rowsSweep}</tbody></table></div>
     <div class="sub" style="margin-top:8px">⚠️ At <b>$1k parcels the fees eat ~80% of the gross profit</b> (+${pc(base.gross_pct)} gross → <b>+${pc(base.net_pct)}</b> net) — fixed CommSec fees crush small trades, and the International min (~$40) is brutal on $1k. Bigger parcels dilute the fee: this is a position-size problem, not a strategy problem. These are backtested OOS results, not live returns.</div>
     <details style="margin-top:10px"><summary style="cursor:pointer;font-weight:600">Per-ticker breakdown ($1k each)</summary>
       <div class="scroll" style="margin-top:6px;border:1px solid var(--line);border-radius:8px"><table style="font-size:12px"><thead><tr>
         <th class="txt">Ticker</th><th class="txt">Strategy</th><th class="txt">Rating</th><th class="txt">Market</th><th>Trades</th><th class="txt">OOS window</th><th>$1k gross</th><th>$1k net</th></tr></thead><tbody>${perRows}</tbody></table></div></details>
   </div>`;
}
function aftertaxPanel(){
  const a=P.aftertax; if(!a||!a.sweep)return '';
  const pc=v=>v==null?'—':((v>=0?'+':'')+(v*100).toFixed(0)+'%');
  const dollar=v=>'$'+Math.round(v).toLocaleString();
  const rows=a.sweep.map(s=>`<tr>
     <td class="txt"><b>$${s.stake.toLocaleString()}</b></td>
     <td>${dollar(s.invested)}</td>
     <td class="pos">${dollar(s.gross)}</td>
     <td>${dollar(s.after_slip)}</td>
     <td>${dollar(s.after_broker)}</td>
     <td class="neg">−${dollar(s.tax).slice(1)}</td>
     <td class="${s.after_tax>=s.invested?'pos':'neg'}"><b>${dollar(s.after_tax)}</b></td>
     <td class="${s.net_pct>=0?'pos':'neg'}"><b>${pc(s.net_pct)}</b></td>
     <td><b>${s.cagr!=null?(s.cagr*100).toFixed(0)+'%/yr':'—'}</b></td></tr>`).join('');
  return `<div class="panel" style="padding:16px 18px;margin-bottom:14px">
     <div style="font-weight:700;font-size:16px;margin-bottom:4px">🧾 Realistic take-home — after real fills &amp; the ATO</div>
     <div class="sub" style="margin-bottom:10px">The same high-conf book, but every haircut modelled: reconstructed <b>gross</b> price edge → minus realistic <b>${(a.assumptions.slippage_per_side*100).toFixed(2)}%/side slippage</b> → minus real <b>CommSec</b> brokerage → minus <b>Australian CGT</b>. <b>${(a.short_term_frac*100).toFixed(0)}% of trades are held under 12 months</b>, so they get NO 50% CGT discount — taxed at your full ${(a.assumptions.marginal_rate*100).toFixed(1)}% marginal rate. CAGR is on committed capital over ~${a.years}y.</div>
     <div class="scroll" style="border:1px solid var(--line);border-radius:8px"><table style="font-size:12.5px"><thead><tr>
       <th class="txt">Parcel each</th><th>Invested</th><th>Gross</th><th>−slippage</th><th>−brokerage</th><th>−tax</th><th>Take-home</th><th>Net%</th><th>CAGR</th></tr></thead><tbody>${rows}</tbody></table></div>
     <div class="sub" style="margin-top:8px">⚠️ At sensible parcel sizes the real take-home is roughly <b>8–9%/yr after everything</b> — a genuine edge, but close to index-like once tax &amp; fees bite. The two biggest levers to beat that: <b>bigger parcels</b> (dilute fixed fees) and <b>lower turnover</b> (hold winners &gt;12mo to unlock the 50% CGT discount). Not tax advice — edit MARGINAL_TAX_RATE in config to your rate.</div>
   </div>`;
}
function renderLive(){
  const el=document.getElementById('livePanel'); const data=window.LIVE||P.fwd||{};
  if(!data.slices){el.innerHTML='<div class="panel" style="padding:22px;color:var(--mut)">Forward tracker starts once the daily monitor has run. Check back after a few sessions.</div>';return;}
  if(!window._lf||!data.slices[window._lf])window._lf='all';
  const f=window._lf, s=data.slices[f], cap=data.capital||20000;
  const btn=k=>`<button class="zbtn ${k===f?'zon':''}" onclick="window._lf='${k}';renderLive()">${data.labels[k]}</button>`;
  const eq=(s.equity_values&&s.equity_values.length>=2)?leq(s.equity_dates,s.equity_values,cap)
    :`<div style="padding:20px;color:var(--mut)">No closed trades in this slice yet — the equity curve fills in as positions close (${s.n_open} open now).</div>`;
  const stat=(l,v,c)=>`<span>${l} <b class="${c||''}">${v}</b></span>`;
  const rp=s.realized_pnl||0, tr=s.total_return;
  const rp2=v=>v!=null?((v>=0?'+':'')+(v*100).toFixed(1)+'%'):'—';
  const openRows=(s.open||[]).map(o=>`<tr>
     <td class="txt"><b>${esc(o.ticker)}</b></td><td class="txt">${esc(o.strategy)}</td>
     <td class="txt">${esc(o.rating||'—')}</td><td class="txt">${esc(o.buy_date||'—')}</td>
     <td>${o.entry?o.entry.toFixed(2):'—'}</td>
     <td>${o.current?o.current.toFixed(2):'—'}${o.asof?`<div class="sub" style="font-size:10px">@ ${esc(o.asof)}</div>`:''}</td>
     <td class="${(o.unreal_pct||0)>=0?'pos':'neg'}">${rp2(o.unreal_pct)}</td>
     <td>${o.expected!=null?'+'+(o.expected*100).toFixed(1)+'%':'—'}</td>
     <td>${o.stop!=null?o.stop.toFixed(2):'—'}</td></tr>`).join('');
  const closedRows=(s.closed||[]).map(c=>`<tr>
     <td class="txt"><b>${esc(c.ticker)}</b></td><td class="txt">${esc(c.strategy)}</td>
     <td class="txt">${esc(c.rating||'—')}</td><td class="txt">${esc(c.buy_date)}</td><td class="txt">${esc(c.sell_date)}</td>
     <td>${c.entry?c.entry.toFixed(2):'—'}</td><td>${c.exit?c.exit.toFixed(2):'—'}</td>
     <td class="${c.ret>=0?'pos':'neg'}"><b>${rp2(c.ret)}</b></td><td class="txt">${esc(c.reason||'')}</td></tr>`).join('');
  el.innerHTML=`
   ${crossuPanel()}
   ${whatifPanel()}
   ${aftertaxPanel()}
   <div class="bar" style="flex-wrap:wrap">${Object.keys(data.slices).map(btn).join('')}</div>
   <div class="panel" style="padding:16px 18px">
     <div style="font-weight:700;font-size:16px;margin-bottom:8px">📡 Live forward performance — ${data.labels[f]}</div>
     <div class="legend" style="margin-bottom:12px">
       ${stat('Closed trades',s.n_trades)}
       ${stat('Win rate',s.win_rate!=null?(s.win_rate*100).toFixed(0)+'%':'—')}
       ${stat('Realized P&L','$'+rp.toLocaleString(),sgn(rp))}
       ${stat('Return on capital',rp2(tr),sgn(tr))}
       ${stat('Best trade',s.best!=null?'+'+(s.best*100).toFixed(1)+'%':'—')}
       ${stat('Worst trade',s.worst!=null?(s.worst*100).toFixed(1)+'%':'—')}
       ${stat('Open now',s.n_open)}
       ${s.first?stat('Since',s.first):''}
     </div>
     ${eq}
     ${openRows?`<div style="font-weight:700;margin-top:14px">Open positions (${s.n_open})</div>
       <div class="scroll" style="margin-top:6px;border:1px solid var(--line);border-radius:8px"><table style="font-size:12.5px"><thead><tr>
         <th class="txt">Ticker</th><th class="txt">Strategy</th><th class="txt">Rating</th><th class="txt">Buy date</th>
         <th>Entry</th><th>Now</th><th>Unrealised</th><th>Exp/trade</th><th>Stop</th></tr></thead><tbody>${openRows}</tbody></table></div>`:''}
     ${closedRows?`<div style="font-weight:700;margin-top:14px">Closed trades (${s.n_trades})</div>
       <div class="scroll" style="margin-top:6px;border:1px solid var(--line);border-radius:8px"><table style="font-size:12.5px"><thead><tr>
         <th class="txt">Ticker</th><th class="txt">Strategy</th><th class="txt">Rating</th><th class="txt">Buy date</th><th class="txt">Sell date</th>
         <th>Entry</th><th>Exit</th><th>Realised</th><th class="txt">Reason</th></tr></thead><tbody>${closedRows}</tbody></table></div>`:''}
     <div class="sub" style="margin-top:10px"><b>Rating</b> = the signal's grade at buy (⭐ Buy / ⭐⭐ Good Buy / ⭐⭐⭐ Strong Buy). <b>Exp/trade</b> = the plan's backtested average gain per trade (what the system "expects"). <b>Stop</b> = the level the system placed (blank = signal-exit plan, no fixed stop).</div>
   </div>`;
}
function loadLive(){
  fetch('fwd_perf.json',{cache:'no-store'}).then(r=>r.ok?r.json():null).then(j=>{if(j&&j.slices)window.LIVE=j; renderLive();}).catch(()=>renderLive());
}
function setView(v){view=v;openIdx.clear();
  document.getElementById('famChart').style.display='none';
  document.getElementById('livePanel').style.display=v==='live'?'':'none';
  document.querySelectorAll('.tab[data-v]').forEach(t=>t.classList.toggle('on',t.dataset.v===v));
  const special=(v==='portfolio'||v==='ticker'||v==='live');
  document.getElementById('tablePanel').style.display=special?'none':'';
  document.getElementById('portfolioPanel').style.display=v==='portfolio'?'':'none';
  document.getElementById('tickerPanel').style.display=v==='ticker'?'':'none';
  document.querySelector('.bar').style.display=special?'none':'flex';
  if(v==='live'){loadLive();
    document.getElementById('foot').innerHTML='How the SYSTEM is actually doing on the live signals it has issued (not the backtest). Filter by tier. Realized P&amp;L assumes the suggested position size on your capital. Accumulates as the daily monitor runs. <b>Not financial advice.</b>';
    return;}
  if(v==='portfolio'){renderPortfolio();
    document.getElementById('foot').innerHTML='Equal-weight book of the plans (each an independent sleeve), vs an equal-weight buy&amp;hold of the same tickers, over the out-of-sample period. Frictions included. <b>Research/education only — not financial advice.</b>';
    return;}
  if(v==='ticker'){renderTicker();
    document.getElementById('foot').innerHTML='Pick a ticker to see its out-of-sample price with the recommended strategy — ▲ buy / ▼ sell markers, its assessment, and every trade. <b>Research/education only — not financial advice.</b>';
    return;}
  sortK=VIEWS[v].sort;sortDir=-1;
  document.querySelectorAll('.chk').forEach(c=>c.style.display=VIEWS[v].rec?'flex':'none');
  render();}
function toggleTheme(){const r=document.documentElement;
  const cur=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  r.setAttribute('data-theme',cur==='dark'?'light':'dark');}

document.querySelectorAll('.tab[data-v]').forEach(t=>t.onclick=()=>setView(t.dataset.v));
['q','mkt','fam','recOnly','hcOnly'].forEach(id=>document.getElementById(id).addEventListener('input',render));
options('mkt',P.plans.concat(P.top).map(r=>r.market));
options('fam',P.plans.concat(P.top).concat(P.strategies).map(r=>r.family));
cards();render();
</script>
"""


def _validate_js(doc: str):
    """Syntax-check the embedded JS with `node --check` if available. A syntax
    error breaks the ENTIRE dashboard silently (Python can't see it), so fail
    loudly here rather than shipping a dead page."""
    import re, shutil, subprocess, tempfile, os
    node = shutil.which("node")
    if not node:
        print("  (node not found — skipping JS syntax check)")
        return
    m = re.search(r"<script>(.*)</script>", doc, re.S)
    if not m:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(m.group(1)); path = f.name
    try:
        r = subprocess.run([node, "--check", path], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"DASHBOARD JS SYNTAX ERROR — not shipping:\n{r.stderr}")
        print("  JS syntax OK")
    finally:
        os.unlink(path)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--pages", action="store_true", help="also write docs/index.html for GitHub Pages")
    args = ap.parse_args(argv)

    payload = build_payload(args.interval)
    if payload is None:
        print("No runs2 data — run the scanner first.")
        return 1
    doc = ("<!doctype html><html><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           "<title>Strategy Lab — trading plans</title></head><body>"
           + render_html(payload) + "</body></html>")
    OUT_HTML.write_text(doc, encoding="utf-8")
    _validate_js(doc)
    if args.pages:
        # Serve Pages from BOTH repo root and /docs, so it works whether the
        # Pages "folder" is set to / (root) or /docs. .nojekyll = serve raw HTML.
        root = config.PROJECT_ROOT
        docs = root / "docs"
        docs.mkdir(exist_ok=True)
        for base in (root, docs):
            (base / "index.html").write_text(doc, encoding="utf-8")
            (base / ".nojekyll").write_text("", encoding="utf-8")

    s = payload["summary"]
    print(f"Dashboard: {OUT_HTML}")
    print(f"Full results CSV: {OUT_ALL}  ({s['n_combos']:,} rows)")
    print(f"  {s['n_tickers']} tickers · {s.get('n_markets','?')} markets · {s['n_strategies']} strategies · "
          f"{s['n_combos']:,} combos · plans={s.get('n_plans','-')} · rec={s.get('n_recommended','-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
