# Using Transformers to Extract Sentiment Signals from Financial News for Stock Prediction

CS 450-04, Advanced AI, Fall 2026. Dr. Benyamin Haghi, SDSU.

Jason Auston, Brianna Garcia, Anthony Nguyen, Maggi Savajiyani, Harish Raaj Sivakumar

<!-- TODO: one paragraph on what the project does and what we found -->

## Scope

23 NASDAQ-100 tickers, 2019-01-01 to 2023-12-31, headlines only.
113,075 articles, 83,470 unique headlines, 21,895 ticker-days.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Note: `pyarrow` is intentionally absent. All parquet I/O goes through DuckDB, which
bundles its own reader/writer. `pandas.read_parquet` / `to_parquet` are not used.

## Reproducing

Scripts run in order. Steps 1-4 are only needed to rebuild from raw data; the
derived files are committed, so `06` and `07` run immediately after a clone.

```bash
python src/01_download.py    # fetch FNSPID shards (800 MB, not in repo)
python src/02_clean.py       # filter, dedupe -> data/news_working.parquet
python src/03_prices.py      # yfinance -> data/returns.parquet
python src/04_sentiment.py   # FinBERT -> data/sentiment.parquet  (GPU/MPS recommended)
python src/05_join.py        # asof join -> data/panel.parquet, data/ticker_day.parquet
python src/06_analysis.py    # correlations, OLS
python src/07_figures.py     # -> figs/
```

## Committed data

| File | Rows | Contents |
|---|---|---|
| `data/news_working.parquet` | 119,527 | cleaned headlines, top-25 tickers, 2019-2023 |
| `data/panel.parquet` | 113,075 | article joined to its trading day and forward return |
| `data/sentiment.parquet` | 83,470 | FinBERT scores per unique headline |
| `data/returns.parquet` | 29,555 | next-day close-to-close return per ticker-day |
| `data/ticker_day.parquet` | 21,895 | aggregated signal + label, analysis-ready |

## Alignment rule

Each article's publication date D maps to T0, the most recent trading day at or
before D. The label is the return from close(T0) to the next trading day's close.
FNSPID timestamps are effectively date-only (99.7% are 00:00 UTC), so this is the
conservative choice: an article published during a session cannot be credited with
that session's move.

## Data license

See DATA.md. FNSPID is CC BY-NC-4.0, non-commercial use only.
