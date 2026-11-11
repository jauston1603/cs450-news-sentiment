# CS 450 Phase 3 — Findings Log

Analysis run Wed 11/11/26 on MacBook (slamdiego619). Every number below came out of the
pipeline described in section 7. This is the evidence record, not paper prose — numbers
organized by the rubric section they belong to.

---

## 1. Dataset plan (6 pts)

**Source.** FNSPID, filtered NASDAQ-100 subset, `benstaf/FNSPID-filtered-nasdaq-100` on
Hugging Face. Two parquet shards, 613 MB + 173 MB. License CC BY-NC-4.0 (non-commercial;
coursework use only — state this).

**Raw.** 358,154 rows, 12 columns, all typed VARCHAR. Columns: `Unnamed: 0`, `Date`,
`Article_title`, `Stock_symbol`, `Url`, `Publisher`, `Author`, `Article`, plus four
precomputed extractive summaries (Lsa, Luhn, Textrank, Lexrank).

**Actual coverage.** 2009-07-07 to 2024-01-09. NOT the 1999–2023 the FNSPID paper
describes — the subset is narrower. Per-year counts: 2009:324, 2010:3522, 2011:6384,
2012:9925, 2013:10882, 2014:15213, 2015:21870, 2016:23315, 2017:30313, 2018:38900,
2019:41343, 2020:32992, **2021:19795 (anomalous dip)**, 2022:39404, 2023:63902, 2024:70.

**Scope chosen.** Top 25 tickers by article volume, 2019-01-01 to 2023-12-31, headlines
only (not article bodies).

**Labels.** No sentiment labels exist in FNSPID. Sentiment labels are *generated* by
FinBERT; the supervised target is the next-day return computed from prices.

**Working file after cleaning:** 119,527 rows (33.4% of raw), 25 tickers.
**After dropping two delisted tickers:** 113,075 rows, 23 tickers.

**Prices.** yfinance, `auto_adjust=True`, 2018-12-01 to 2024-01-15, 1,286 trading days.

---

## 2. Preliminary EDA (4 pts)

### Timestamp granularity — the critical finding
`Date` is formatted as a timestamp (`2023-12-16 22:00:00 UTC`, length 23, consistent
across all 358,154 rows) but **357,452 rows (99.8%) have minutes/seconds = `00:00`** and
**357,192 (99.7%) sit at hour `00`**. It is a date-only field in timestamp clothing.
No usable intraday resolution.

### Date reliability
Recurring wire roundups carry timestamps that disagree with their own content:
- `2023-12-16 21:00:00` → "Company News for Dec 19, 2023" (3-day gap)
- `2023-12-12 00:00:00` → "After Hours Most Active for Dec 13, 2023"

### Junk / low-information headlines
| Pattern | Rows |
|---|---|
| `%Most Active for%` | 3,682 |
| `%Company News for%` | 622 |
| `%Stock Market News for%` | 812 |
| **Total** | **5,116 (1.43%)** |

These are ticker-list roundups ("...: MRK, CRH, CNHI, MTCH, AAPL, SCPH...") carrying no
company-specific sentiment.

### Missingness
`Publisher` is empty on **225,187 rows (62.9%)**. Rules out publisher-based filtering.

### Duplication and cross-attribution
- 358,154 rows / 265,585 distinct titles / 325,995 distinct (title, ticker) pairs
- True duplicates (same title + same ticker): **32,159 (8.98%)**
- Extra rows from cross-ticker filing: **60,410 (16.87%)**
- 39,659 distinct titles (14.9%) appear under >1 ticker; worst case **51 tickers**
- Of 49,306 headlines containing a parenthesized ticker, **24,769 (50.2%) do not contain
  the filed `Stock_symbol` anywhere in the headline**

Concrete example: "NVIDIA (NVDA) Up 243% YTD: Will It Carry Momentum in 2024?" filed under
`Stock_symbol = AAPL`.

*Caveat to state:* `symbol_in_title` is only 17.4% in the working file, but ticker-string
matching badly undercounts relevance ("Apple Unveils New iPhone" is relevant to AAPL and
contains no "AAPL"). The defensible claim is the 50.2% figure, restricted to headlines
that use ticker symbols.

### Headline length (justifies headlines-only + `max_length=64`)
min 10 chars, mean 61, p95 96, max 396. Roughly 15 tokens typical, ~24 at p95, <100 worst
case, against BERT's 512 limit.

### Sparsity
Articles per ticker-day: 9,275 ticker-days have exactly 1 article; 6,042 have 2; 3,751
have 3. Among ticker-days with ≤15 articles, **33.7% are single-article days**. Days with
>15 articles hold 31,015 articles (26% of the corpus).

### Final ticker set (23)
AAPL, ADBE, AMD, AMGN, BIIB, CMCSA, COST, CSX, EBAY, GILD, GOOG, INTC, MSFT, MU, NVDA,
PANW, PEP, PYPL, QCOM, SBUX, TSLA, TXN, VRTX

(The original top-25 also included EA and WBA; both dropped — see survivorship bias.)
Volume range TSLA 10,017 down to CSX 1,669 (~6x).

### FinBERT class distribution (83,470 unique headlines)
| Label | Count | Share |
|---|---|---|
| neutral | 52,238 | 62.6% |
| positive | 17,597 | 21.1% |
| negative | 13,635 | 16.3% |

pos:neg = 1.29. `sent_score` (p_pos − p_neg): mean 0.063, std 0.486, median 0.042,
IQR −0.023 to 0.249, range −0.969 to 0.945.

### Return distribution
29,555 ticker-days. mean 0.00095, std 0.0240, min −0.2923, max +0.4397, median 0.00078,
IQR −0.0098 to +0.0117. Per-ticker volatility: TSLA 4.07%, AMD 3.40%, NVDA 3.26%,
BIIB 3.12%, MU 2.88%.

**All 10 extreme moves verified as real events** (no split artifacts):

| Date (T0) | Ticker | Fwd return | Event |
|---|---|---|---|
| 2020-11-03 | BIIB | +44.0% | FDA adcomm briefing docs, aducanumab |
| 2022-09-27 | BIIB | +39.9% | lecanemab Clarity AD readout |
| 2021-06-04 | BIIB | +38.3% | Aduhelm approval |
| 2019-10-21 | BIIB | +26.1% | aducanumab filing revived |
| 2023-05-24 | NVDA | +24.4% | AI guidance blowout |
| 2019-03-20 | BIIB | −29.2% | aducanumab trials discontinued |
| 2020-11-06 | BIIB | −28.2% | adcomm votes against |
| 2022-02-01 | PYPL | −24.6% | guidance miss |
| 2020-09-04 | TSLA | −21.1% | S&P 500 exclusion |
| 2020-10-14 | VRTX | −20.7% | VX-814 discontinued |

**BIIB owns 6 of 10 extremes.** One ticker of 23 dominates both tails (biotech binary
events). Implication: report per-ticker results, not only pooled.

---

## 3. Preprocessing plan (4 pts) — each step tied to a finding above

| Step | Driven by |
|---|---|
| Drop `Unnamed: 0` | leftover pandas index |
| Drop 3 roundup patterns (5,116 rows, 1.43%) | ticker-list headlines, unreliable dates |
| `SELECT DISTINCT` on (date, ticker, title) | 8.98% true duplicates |
| Cast `Date` → DATE, discard time | 99.7% of timestamps are 00:00 |
| Carry `symbol_in_title` flag, do **not** filter on it yet | 50.2% paren mismatch, but ticker matching undercounts relevance |
| Restrict to top 25 tickers / 2019–2023 | coverage density; 2021 dip noted |
| Tokenize at `max_length=64` | p95 = 96 chars |
| Score distinct headlines only | saved 29,605 of 113,075 forward passes |
| Minimum-article threshold per ticker-day (proposed) | 33.7% single-article days make the strength score degenerate |

**Company-name matching as a relevance filter is Phase 4 work, not done here.**

---

## 4. Alignment rule (leakage) — for the Evaluation section

Each article's `pub_date` D maps to T0 = most recent trading day ≤ D (DuckDB `ASOF JOIN`).
Label = return from close(T0) to close(next trading day).

- D is a trading day → claim only close(D)→close(D+1), so an intraday article cannot
  "predict" the move it was published during.
- D is a weekend → T0 = Friday, label = Friday close → Monday close, which captures the
  weekend-news effect with no leakage.

Join result: 113,075 rows in → 113,075 out, 23 tickers, **10,574 articles (9.4%) on
non-trading days**, mean lag 0.153 days.

Trailing volatility feature uses a 20-day rolling std through T0 only.

---

## 5. Model plan (7 pts) — what the data supports

- Baseline: TF-IDF + logistic regression on headlines.
- Deep model: FinBERT fine-tuned on Financial PhraseBank, applied to headlines.
  (FNSPID has no sentiment labels — cannot fine-tune on it directly.)
- **Primary target revised:** next-day absolute return (magnitude), with next-day
  direction reported as a secondary, expected-null experiment. Evidence in section 6.
- Scoping decisions justified by measurement: headlines not bodies (p95 = 96 chars),
  23 tickers, 5 years, `max_length=64`.

---

## 6. Preliminary results

### Direction — null, four ways
21,895 ticker-days.

| Test | Result |
|---|---|
| Pearson, sent_mean vs fwd_ret | **−0.0042** (t ≈ −0.6) |
| Spearman | −0.0035 |
| Direction agreement | **50.97%** |
| Always-up baseline | **52.01%** |

Sentiment sign does *worse* than a naive always-up rule.

By news volume (Bonferroni threshold for 4 tests = 0.0125):

| Bucket | n | r | p |
|---|---|---|---|
| 1 | 5,056 | −0.0030 | 0.830 |
| 2–4 | 9,540 | −0.0261 | 0.011 ← significant but **wrong sign** |
| 5–15 | 5,807 | +0.0177 | 0.177 |
| 16+ | 1,492 | +0.0545 | 0.035 ← fails Bonferroni |

Non-monotonic, and the only result clearing correction has the wrong sign. Treat as noise.

Per-ticker Pearson ranges −0.0511 (CSX) to +0.0658 (INTC). With ~952 ticker-days each,
SE(r) ≈ 0.0325, so the extremes are ~1.6–2 SE. Consistent with zero everywhere plus noise.

### Magnitude — strong, significant, monotonic

| Predictor | r vs \|next-day return\| | p |
|---|---|---|
| \|sent_mean\| | +0.0124 | 0.066 (ns) |
| n_articles | +0.1190 | 6.6e−70 |
| log1p(n_articles) | **+0.1364** | **1.9e−91** |

Mean |next-day return| by bucket — monotonic at every step:

| Bucket | Mean \|return\| | n |
|---|---|---|
| 1 | 1.376% | 5,056 |
| 2–4 | 1.511% | 9,540 |
| 5–15 | 1.892% | 5,807 |
| 16+ | 2.176% | 1,492 |

**1.58× spread** from quietest to noisiest news days.

### Robustness: survives volatility control
OLS, |next-day return| ~ log1p(articles) + trailing 20-day vol, n = 21,872:

| Term | coef | t | p |
|---|---|---|---|
| const | 0.0030 | 9.25 | <0.001 |
| log_art | **+0.0020** | **11.99** | <0.001 |
| trail_vol | +0.4837 | 49.64 | <0.001 |

News volume carries information about tomorrow's move size beyond what recent volatility
already contains. 1 → 16 articles adds ≈ 0.43 pp to expected absolute move, ≈ 31% above
the 1.376% quiet-day baseline.

**Note: it is the *amount* of coverage that predicts, not the *tone*.**

---

## 7. Feasibility risks + mitigation (8 pts) — all evidence-backed

| Risk | Evidence from this analysis | Mitigation |
|---|---|---|
| Look-ahead / leakage | 99.7% of timestamps are 00:00 (date-only) | ASOF join to T0; label = close(T0)→close(T0+1); time-ordered split, never random |
| Unreliable dates | roundup headlines off by 1–3 days | drop the 3 roundup patterns (1.43%) |
| Label noise / mis-attribution | 50.2% of paren-ticker headlines don't name the filed symbol; one title filed under 51 tickers | carry `symbol_in_title`; company-name matching in Phase 4; report sensitivity |
| Class imbalance | 62.6% neutral | macro-F1, not accuracy; class weights |
| Direction is near-unpredictable | r = −0.004, 50.97% vs 52.01% baseline | pivot primary target to magnitude; report direction null as a finding |
| Sparse ticker-days | 33.7% single-article days | minimum-article threshold or rolling window; report article count with the signal |
| Single-ticker dominance | BIIB = 6 of 10 extreme moves | per-ticker metrics alongside pooled |
| Survivorship bias | **EA and WBA no longer downloadable** — both taken private after the news window (Sycamore/Walgreens closed Aug 2025; EA $55B PIF-led buyout, delisted). Cost: 6,452 articles, 5.4% | document explicitly; note free price sources silently drop dead companies |
| Coverage instability | 2021 dip to 19,795; ticker ranking shifts between full corpus and 2019–2023 window | state window; don't claim stable coverage |
| Compute | FinBERT over 83,470 headlines ran in minutes on MPS | headlines only, `max_length=64`, score distinct text |
| Tooling | pyarrow broken in Anaconda (`liborc.dylib` / protobuf symbol mismatch) — `pd.read_parquet`/`to_parquet` unusable | all parquet I/O through DuckDB |
| License | FNSPID is CC BY-NC-4.0 | coursework use only; state in report |

---

## 8. Figures written (`~/cs450/figs/`)

1. `fig1_coverage_by_year.png` — articles per year, shows 2021 dip
2. `fig2_articles_per_ticker.png` — 23 tickers, volume spread
3. `fig3_class_distribution.png` — FinBERT labels, 62.6% neutral
4. `fig4_articles_per_ticker_day.png` — sparsity, log scale
5. `fig5_return_distribution.png` — next-day returns, fat tails
6. `fig6_magnitude_by_volume.png` — **the result**, monotonic with 95% CI
7. `fig7_sentiment_decile_null.png` — **the null**, flat line with overlapping CIs

## 9. Pipeline artifacts (`~/cs450/data/`)

| File | Contents |
|---|---|
| `train-0000*.parquet` | raw FNSPID subset, 358,154 rows |
| `news_working.parquet` | cleaned, top-25, 2019–2023, 119,527 rows |
| `prices_raw.pkl` | yfinance OHLCV, 1,286 days |
| `returns.parquet` | 29,555 ticker-days, `fwd_ret_1d` |
| `panel.parquet` | article ↔ trading day, 113,075 rows |
| `sentiment.parquet` | FinBERT scores, 83,470 unique headlines |
| `ticker_day.parquet` | 21,895 ticker-days, aggregated signal + label |
