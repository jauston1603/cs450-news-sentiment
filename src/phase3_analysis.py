"""
Phase 3 - Preliminary Data Analysis and Feasibility
Using Transformers to Extract Sentiment Signals from Financial News for Stock Prediction

CS 450-04, Advanced AI, Fall 2026. Dr. Benyamin Haghi, SDSU.

Reproduces every number and figure in the Phase 3 report from the derived parquet
files committed to the repository. Does NOT require the 800 MB raw FNSPID shards
and does NOT re-run FinBERT.

Run from the repo root or from src/:
    python src/phase3_analysis.py

All parquet I/O goes through DuckDB. pyarrow is broken in our Anaconda environment
(liborc.dylib expects a protobuf symbol the installed libprotobuf does not export),
so pandas.read_parquet / to_parquet are unavailable.
"""

from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------- setup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "src" else HERE
DATA, FIGS = ROOT / "data", ROOT / "figs"
FIGS.mkdir(exist_ok=True)

BLUE = "#3B6EA5"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

q = lambda s: duckdb.sql(s).df()           # noqa: E731
P = lambda name: (DATA / name).as_posix()  # noqa: E731


def head(n, title):
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


def save(name):
    plt.tight_layout()
    plt.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"    [figure] figs/{name}.png")


print(f"repo root: {ROOT}")

# ------------------------------------------------------------------- 1. dataset

head(1, "DATASET")

panel = q(f"SELECT * FROM read_parquet('{P('panel.parquet')}')")
panel["pub_date"] = pd.to_datetime(panel.pub_date)

sent = q(f"SELECT * FROM read_parquet('{P('sentiment.parquet')}')")
ret = q(f"SELECT * FROM read_parquet('{P('returns.parquet')}')")
agg = q(f"SELECT * FROM read_parquet('{P('ticker_day.parquet')}')")
agg["trade_date"] = pd.to_datetime(agg.trade_date)

print(f"articles (panel)      : {len(panel):,}")
print(f"unique headlines      : {panel.Article_title.nunique():,}")
print(f"tickers               : {panel.Stock_symbol.nunique()}")
print(f"window                : {panel.pub_date.min().date()} .. {panel.pub_date.max().date()}")
print(f"ticker-days w/ returns: {len(ret):,}")
print(f"ticker-days w/ news   : {len(agg):,}")

print("\nCleaning funnel (recorded from the raw subset):")
print("  raw FNSPID subset                                      358,154")
print("  top-25 tickers, 2019-2023, roundups dropped, deduped   119,527")
print("  after dropping delisted EA and WBA                     113,075")

print("\nTimestamp granularity (recorded from the raw subset):")
print("  hour == 00     357,192 / 358,154  (99.7%)")
print("  min:sec == 00  357,452 / 358,154  (99.8%)")
print("  -> 'Date' is a date-only field formatted as a timestamp")

# ----------------------------------------------------------------------- 2. EDA

head(2, "PRELIMINARY EDA")

# 2.1 coverage
yr = panel.groupby(panel.pub_date.dt.year).size()
print("\n2.1 Articles per year")
print(yr.to_string())
print(f"    2021 dip: {yr.get(2021, 0):,} vs {yr.max():,} peak")
plt.figure(figsize=(5, 2.6))
plt.bar(yr.index, yr.values, color=BLUE, width=.65)
plt.title("Articles per year (23 tickers, after cleaning)")
plt.ylabel("articles")
save("fig1_coverage_by_year")

# 2.2 ticker composition
tk = panel.groupby("Stock_symbol").size().sort_values()
print("\n2.2 Articles per ticker")
print(f"    max {tk.max():,} ({tk.idxmax()})   min {tk.min():,} ({tk.idxmin()})"
      f"   ratio {tk.max() / tk.min():.1f}x")
print("    " + ", ".join(sorted(tk.index)))
plt.figure(figsize=(4.5, 4.5))
plt.barh(tk.index, tk.values, color=BLUE, height=.7)
plt.title("Articles per ticker")
plt.xlabel("articles")
save("fig2_articles_per_ticker")

# 2.3 quality
print("\n2.3 Quality problems (recorded from the raw subset)")
print("    Publisher empty            225,187 / 358,154  (62.9%)")
print("    true duplicates             32,159            ( 8.98%)")
print("    extra rows from cross-filing 60,410           (16.87%)")
print("    one headline filed under up to 51 tickers")
print("    of 49,306 paren-ticker headlines, 24,769 (50.2%) omit the filed symbol")
sit = panel.symbol_in_title.mean()
print(f"    symbol_in_title in working set: {sit * 100:.1f}%  "
      f"(weak proxy - headlines name companies in words, not tickers)")

# 2.4 headline length
L = panel.Article_title.str.len()
print("\n2.4 Headline length (characters)")
print(f"    min {L.min()}  mean {L.mean():.0f}  p95 {L.quantile(.95):.0f}  max {L.max()}")
print("    -> ~15 tokens typical, <100 worst case, against BERT's 512 limit")

# 2.5 sparsity
print("\n2.5 Sparsity")
one = (agg.n_articles == 1).sum()
print(f"    single-article ticker-days: {one:,} ({one / len(agg) * 100:.1f}%)")
h = agg.n_articles.clip(upper=25).value_counts().sort_index()
plt.figure(figsize=(5, 2.6))
plt.bar(h.index, h.values, color=BLUE, width=.8)
plt.yscale("log")
plt.title("Articles per ticker-day (25+ pooled)")
plt.xlabel("articles that day")
plt.ylabel("ticker-days (log)")
save("fig4_articles_per_ticker_day")

# 2.6 sentiment classes
print("\n2.6 FinBERT class distribution")
vc = sent.fin_label.value_counts().reindex(["positive", "neutral", "negative"])
for k, v in vc.items():
    print(f"    {k:<9} {v:>7,}  {v / len(sent) * 100:5.1f}%")
print(f"    pos:neg = {vc['positive'] / vc['negative']:.2f}")
print("    -> accuracy is useless against a 62.6% majority class; use macro-F1")
plt.figure(figsize=(4, 2.6))
b = plt.bar(vc.index, vc.values, color=BLUE, width=.55)
plt.bar_label(b, fmt="%d", padding=2, fontsize=8)
plt.title(f"FinBERT labels, {len(sent):,} unique headlines")
plt.ylabel("headlines")
plt.margins(y=.15)
save("fig3_class_distribution")
print("\n    sent_score distribution:")
print("    " + sent.sent_score.describe().round(4).to_string().replace("\n", "\n    "))

# 2.7 returns
print("\n2.7 Next-day return distribution")
print("    " + ret.fwd_ret_1d.describe().round(5).to_string().replace("\n", "\n    "))
plt.figure(figsize=(5, 2.6))
plt.hist(ret.fwd_ret_1d.clip(-.15, .15), bins=80, color=BLUE)
plt.title("Next-day return, close to close")
plt.xlabel("return")
plt.ylabel("ticker-days")
save("fig5_return_distribution")

ext = pd.concat([ret.nlargest(5, "fwd_ret_1d"), ret.nsmallest(5, "fwd_ret_1d")]).copy()
ext["pct"] = (ext.fwd_ret_1d * 100).round(1)
print("\n    Ten extreme moves (all verified real events, no split artifacts):")
print("    " + ext[["trade_date", "Stock_symbol", "pct"]].to_string(index=False).replace("\n", "\n    "))
nb = (ext.Stock_symbol == "BIIB").sum()
print(f"    -> BIIB accounts for {nb} of 10; report per-ticker metrics, not only pooled")

print("\n    Per-ticker volatility (std of next-day return), top 5:")
print("    " + ret.groupby("Stock_symbol").fwd_ret_1d.std().nlargest(5).round(4)
      .to_string().replace("\n", "\n    "))

# --------------------------------------------------------------- 4. alignment

head(4, "ALIGNMENT RULE AND LEAKAGE CONTROL")

print("Article published on date D  ->  T0 = most recent trading day <= D  (ASOF JOIN)")
print("Label = return from close(T0) to the next trading day's close.")
print("  D a trading day -> claim only close(D)->close(D+1); an intraday article")
print("                     cannot be credited with that session's move.")
print("  D a weekend     -> T0 = Friday, label = Fri close -> Mon close, no leakage.")

nontrading = (panel.pub_date != pd.to_datetime(panel.trade_date)).sum()
lag = (panel.pub_date - pd.to_datetime(panel.trade_date)).dt.days.mean()
print(f"\njoin integrity      : {len(panel):,} rows in, {len(panel):,} out")
print(f"non-trading-day news: {nontrading:,} ({nontrading / len(panel) * 100:.1f}%)")
print(f"mean lag            : {lag:.3f} days")

# ----------------------------------------------------------------- 5. results

head(5, "PRELIMINARY RESULTS")

# 5.1 direction
print("\n5.1 DIRECTION - null, four independent ways")
r_p, p_p = pearsonr(agg.sent_mean, agg.fwd_ret_1d)
r_s = agg.sent_mean.corr(agg.fwd_ret_1d, method="spearman")
dir_ok = (np.sign(agg.sent_mean) == np.sign(agg.fwd_ret_1d)).mean()
base = (agg.fwd_ret_1d > 0).mean()
print(f"    ticker-days         : {len(agg):,}")
print(f"    Pearson             : {r_p:+.4f}  (p={p_p:.3f})")
print(f"    Spearman            : {r_s:+.4f}")
print(f"    direction agreement : {dir_ok * 100:.2f}%")
print(f"    always-up baseline  : {base * 100:.2f}%")
print(f"    -> sentiment sign does {'WORSE' if dir_ok < base else 'better'} than always guessing up")

agg["bucket"] = pd.cut(agg.n_articles, [0, 1, 4, 15, 10 ** 6],
                       labels=["1", "2-4", "5-15", "16+"])
print("\n    By news volume (Bonferroni threshold for 4 tests = 0.0125):")
for bkt, g in agg.groupby("bucket", observed=True):
    r, p = pearsonr(g.sent_mean, g.fwd_ret_1d)
    flag = "PASSES" if p < 0.0125 else "fails "
    sign = " <- WRONG SIGN" if (p < 0.0125 and r < 0) else ""
    print(f"      {str(bkt):>5}  n={len(g):>6,}  r={r:+.4f}  p={p:.4f}  {flag}{sign}")
print("    -> non-monotonic, and the only bucket clearing correction has the wrong sign")

pt = agg.groupby("Stock_symbol").apply(
    lambda g: g.sent_mean.corr(g.fwd_ret_1d), include_groups=False)
se = 1 / np.sqrt(len(agg) / agg.Stock_symbol.nunique() - 3)
print(f"\n    per-ticker Pearson : {pt.min():+.4f} .. {pt.max():+.4f}")
print(f"    SE(r) at ~{int(len(agg) / agg.Stock_symbol.nunique())} obs/ticker: {se:.4f}"
      f"  -> extremes ~{pt.abs().max() / se:.1f} SE, consistent with zero")

agg["dec"] = pd.qcut(agg.sent_mean, 10, labels=False, duplicates="drop")
gg = agg.groupby("dec").fwd_ret_1d
m2, s2 = gg.mean(), gg.std() / np.sqrt(gg.count())
plt.figure(figsize=(5, 2.8))
plt.errorbar(m2.index + 1, m2.values * 100, yerr=s2.values * 196,
             fmt="o-", color=BLUE, ms=5, lw=1.5, capsize=3)
plt.axhline(0, color="#888", lw=.8)
plt.title("Next-day return by sentiment decile (flat = no signal)")
plt.xlabel("sentiment decile, low to high")
plt.ylabel("mean return, %")
save("fig7_sentiment_decile_null")

# 5.2 magnitude
print("\n5.2 MAGNITUDE - strong, significant, monotonic")
agg["abs_ret"] = agg.fwd_ret_1d.abs()
agg["abs_sent"] = agg.sent_mean.abs()
agg["log_art"] = np.log1p(agg.n_articles)
for col, label in [("abs_sent", "|sentiment|"),
                   ("n_articles", "articles"),
                   ("log_art", "log1p(articles)")]:
    r, p = pearsonr(agg[col], agg.abs_ret)
    print(f"    {label:>16} vs |next-day return|   r={r:+.4f}   p={p:.3g}")

g = agg.groupby("bucket", observed=True).abs_ret
mu, se_ = g.mean(), g.std() / np.sqrt(g.count())
print("\n    Mean |next-day return| by news volume:")
for bkt in mu.index:
    print(f"      {str(bkt):>5}  {mu[bkt] * 100:.3f}%   n={g.count()[bkt]:,}")
print(f"    -> {mu.iloc[-1] / mu.iloc[0]:.2f}x spread, monotonic at every step")

plt.figure(figsize=(4, 2.8))
b = plt.bar(mu.index.astype(str), mu.values * 100, yerr=se_.values * 196,
            color=BLUE, width=.55, capsize=4)
plt.bar_label(b, fmt="%.2f%%", padding=6, fontsize=8)
plt.title("Mean |next-day return| by news volume")
plt.ylabel("percent")
plt.xlabel("articles that ticker-day")
plt.margins(y=.18)
save("fig6_magnitude_by_volume")

# 5.3 robustness
print("\n5.3 ROBUSTNESS - does it survive a volatility control?")
rr = ret.copy()
rr["trade_date"] = pd.to_datetime(rr.trade_date)
rr = rr.sort_values(["Stock_symbol", "trade_date"])
rr["daily_ret"] = rr.groupby("Stock_symbol").fwd_ret_1d.shift(1)
rr["trail_vol"] = rr.groupby("Stock_symbol").daily_ret.transform(
    lambda s: s.rolling(20).std())

m = (agg.merge(rr[["Stock_symbol", "trade_date", "trail_vol"]],
               on=["Stock_symbol", "trade_date"], how="inner")
     .dropna(subset=["trail_vol"]))
print(f"    merged ticker-days: {len(m):,}")
res = sm.OLS(m.abs_ret, sm.add_constant(m[["log_art", "trail_vol"]])).fit()
print("    " + str(res.summary().tables[1]).replace("\n", "\n    "))
eff = (np.log1p(16) - np.log1p(1)) * res.params["log_art"]
print(f"\n    1 -> 16 articles adds {eff * 100:.3f} pp to expected |move| "
      f"({eff / mu.iloc[0] * 100:.0f}% above the quiet-day baseline)")
print("    -> news VOLUME carries information beyond trailing volatility.")
print("       It is the amount of coverage that predicts, not the tone.")

head(6, "DONE")
print(f"figures written to {FIGS}")
