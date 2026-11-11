# Data sources

## News
FNSPID, filtered NASDAQ-100 subset.
Hugging Face: https://huggingface.co/datasets/benstaf/FNSPID-filtered-nasdaq-100

Z. Dong, X. Fan, and Z. Peng, "FNSPID: A comprehensive financial news dataset in
time series," in Proc. 30th ACM SIGKDD Conf. Knowledge Discovery and Data Mining,
Barcelona, Spain, 2024. doi: 10.1145/3637528.3671629

License: CC BY-NC-4.0. Non-commercial use only. The two raw shards (800 MB) are
NOT redistributed in this repository; `src/01_download.py` fetches them.
Derived files under `data/` are adaptations of this dataset, shared
non-commercially under the same terms.

## Prices
Yahoo Finance via the `yfinance` Python library. Adjusted for splits and dividends
(`auto_adjust=True`).

## Model
ProsusAI/finbert, https://huggingface.co/ProsusAI/finbert
