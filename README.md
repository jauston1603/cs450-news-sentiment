# Using Transformers to Extract Sentiment Signals from Financial News for Stock Prediction

CS 450-04, Advanced AI, Fall 2026. Dr. Benyamin Haghi, SDSU.

Jason Auston, Brianna Garcia, Anthony Nguyen, Maggi Savajiyani, Harish Raaj Sivakumar

<!-- TODO: one paragraph on what the project does and what we found -->

## Scope

23 NASDAQ-100 tickers, 2019-01-01 to 2023-12-31, headlines only.
113,075 articles, 83,470 unique headlines, 21,895 ticker-days.

## Setup

```bash
pip install -r requirements.txt
```

`pyarrow` is intentionally absent. All parquet I/O goes through DuckDB, which bundles
its own reader and writer. `pandas.read_parquet` and `to_parquet` are not used, because
pyarrow is broken in our Anaconda environment (`liborc.dylib` expects a protobuf symbol
the installed `libprotobuf` does not export).

## Reproducing

The derived data files are committed, so the full analysis runs immediately after a
clone. No 800 MB download, no GPU, no FinBERT pass.

```bash
python src/phase3_analysis.py    # prints every result, writes figs/
```

Or open `phase3.ipynb`, which runs the same analysis with the narrative alongside it.
Both locate the repo root automatically, so either can be run from anywhere.

## Committed data

| File | Rows | Contents |
|---|---|---|
| `data/news_working.parquet` | 119,527 | cleaned headlines, top-25 tickers, 2019-2023 |
| `data/panel.parquet` | 113,075 | article joined to its trading day and forward return |
| `data/sentiment.parquet` | 83,470 | FinBERT scores per unique headline |
| `data/returns.parquet` | 29,555 | next-day close-to-close return per ticker-day |
| `data/ticker_day.parquet` | 21,895 | aggregated signal and label, analysis-ready |

The raw FNSPID shards (800 MB) are not redistributed. See DATA.md.

## Alignment rule

Each article's publication date D maps to T0, the most recent trading day at or before D.
The label is the return from close(T0) to the next trading day's close. FNSPID timestamps
are effectively date-only (99.7% are 00:00 UTC), so this is the conservative choice: an
article published during a session cannot be credited with that session's move.

## Key results

- Direction is not predictable at a daily horizon. Pearson −0.0042; direction agreement
  50.97% against a 52.01% always-up baseline.
- News **volume** predicts move **magnitude**. log1p(articles) vs next-day absolute
  return, r = +0.1364, p = 1.9e−91, monotonic across four volume buckets.
- The effect survives controlling for trailing 20-day realised volatility (t = 12.0).

## Data license

See DATA.md. FNSPID is CC BY-NC-4.0, non-commercial use only.
