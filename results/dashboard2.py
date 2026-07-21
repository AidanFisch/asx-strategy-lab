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

import config

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

    return {
        "summary": summary,
        "trades": trades_map,
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
  <div class="panel scroll" id="tablePanel"><table id="tbl"><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
  <div id="portfolioPanel" style="display:none"></div>
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
function setView(v){view=v;openIdx.clear();
  document.querySelectorAll('.tab[data-v]').forEach(t=>t.classList.toggle('on',t.dataset.v===v));
  const isPort=v==='portfolio';
  document.getElementById('tablePanel').style.display=isPort?'none':'';
  document.getElementById('portfolioPanel').style.display=isPort?'':'none';
  document.querySelector('.bar').style.display=isPort?'none':'flex';
  if(isPort){renderPortfolio();
    document.getElementById('foot').innerHTML='Equal-weight book of the plans (each an independent sleeve), vs an equal-weight buy&amp;hold of the same tickers, over the out-of-sample period. Frictions included. <b>Research/education only — not financial advice.</b>';
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
