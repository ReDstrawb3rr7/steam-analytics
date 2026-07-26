# Steam Community Analytics

A configurable, end-to-end pipeline for analyzing player sentiment, engagement,
and community trends from Steam game reviews. Point it at any Steam appid:
it ingests reviews, scores sentiment, trains a recommendation-prediction
model, runs time-series forecasting, and surfaces everything through a
parameterized SQL layer (SQLite and BigQuery) and an interactive dashboard.

Built as a portfolio project to demonstrate applied data science across a
full pipeline: data ingestion from a real, quirky public API, NLP, supervised
ML with proper validation, time-series methodology, analytical SQL on two
backends, and an interactive front end.

Currently loaded with two contrasting case studies: **Palworld** (94%
recommended, "Overwhelmingly Positive") and **Marvel Rivals** (a genuinely
contested title with a large negative-review base), about 65,000 reviews
total.

## What it does

- **Ingests reviews for any Steam game** via its public appid, no API key or
  approval process required. Handles free-to-play titles, age-gated titles,
  and fails loudly (nonzero exit) if an ingest returns nothing, so chained
  pipelines stop instead of running on empty data (`ingestion/steam_ingest.py`)
- **Scores sentiment** on review text using a pretrained transformer suited
  to informal social text, incrementally: only unscored reviews are
  processed on each run (`analysis/sentiment.py`)
- **Predicts whether a review is "recommended"** from sentiment, engagement,
  and reviewer-history features, with a logistic regression baseline, a
  random forest, a feature-ablation flag, and saved evaluation artifacts
  (ROC curve, confusion matrix) for the dashboard
  (`analysis/features.py`, `analysis/recommendation_model.py`)
- **Decomposes and forecasts review volume and sentiment over time**
  (weekly seasonality, ARIMA), with log-transform handling for the extreme
  outlier spikes real launch events create (`analysis/time_series.py`)
- **Runs a parameterized SQL analytics layer** on SQLite
  (`sql/analytics_queries.sql` + `analysis/sql_report.py`) and the same
  queries rewritten for BigQuery's dialect with real query parameters
  (`sql/bigquery_analytics_queries.sql` + `analysis/bigquery_sql_report.py`)
- **Syncs to BigQuery** with native type conversion (TIMESTAMP/BOOLEAN),
  chunked uploads, and per-chunk retry so transient network drops don't
  kill the run (`migration/migrate_to_bigquery.py`)
- **One-command incremental updates**: `update_data.py` chains
  ingest -> sentiment -> features -> model retrain (-> optional BigQuery
  sync), doing only new work at each step
- **Interactive Streamlit dashboard** (`dashboard/app.py`): game selector,
  pivot-date comparison, sample reviews (most-helpful positive and
  negative), full model evaluation (ROC curve, confusion matrix, ablation
  comparison), a built-in SQL explorer, official game art and screenshots
  from Steam's CDN, a data-refresh button, and a SQLite/BigQuery backend
  toggle demonstrating backend portability through one data-access layer

## Architecture

```
Steam public API -> Python ingestion (steam_ingest.py) -> SQLite
                                                              |
                                          Sentiment scoring (sentiment.py)
                                                              |
                    +--------------------------+--------------------------+
          Feature engineering            Time-series               SQL analytics layer
          + recommendation model         decomposition/forecast    (window functions,
          (sklearn, ablation testing)    (statsmodels)             parameterized by appid)
                    +--------------------------+--------------------------+
                                                              |
                              Streamlit dashboard  <->  BigQuery (synced copy,
                              (SQLite or BigQuery)      same queries, BQ dialect)

                       update_data.py = one-command incremental refresh of all of the above
```

## Tech stack, and why each piece was chosen

| Tool | Why |
|---|---|
| Steam's public `appreviews` endpoint | No auth, no approval queue. Unlike Discord (requires server-admin permission to add a bot) or Reddit (closed self-service API access in Nov 2025 under its "Responsible Builder Policy") |
| SQLite | Simple, file-based, sufficient for local development. The schema is written to migrate cleanly to BigQuery |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | A sentiment model trained on short, informal social text, which closely matches the tone of Steam reviews. Domain-specific models (finance, product reviews, etc.) fit this kind of text poorly |
| scikit-learn | Logistic regression baseline plus a random forest, predicting `voted_up`, a real label already present in the data rather than one constructed with arbitrary heuristics |
| statsmodels | Proper time-series methodology: stationarity testing, log-transforming skewed count data before ARIMA, seasonal decomposition, not just eyeballing a line chart |
| SQL window functions | Rolling averages, `NTILE`/`PERCENT_RANK`, `LAG`: analytical SQL as a first-class part of the project, not just storage |
| BigQuery | Cloud-scale data warehouse. Its stricter standard SQL requires proper handling of native types (`TIMESTAMP`/`BOOLEAN`, strict `GROUP BY`) and real parameterized queries instead of ad-hoc string substitution |
| Streamlit + Plotly | Fast interactive dashboarding without building a custom frontend |

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

No API keys or credentials needed for the Steam ingestion pipeline. BigQuery
requires a one-time `gcloud init` and `gcloud auth application-default login`
(see "BigQuery setup" below).

## Running the pipeline (any game)

Find a game's appid from its Steam store URL: `store.steampowered.com/app/<appid>`

The easy way, after the first full run, is the one-command updater:

```bash
# First-time ingest of a new game
python update_data.py --appid <APPID> --max-reviews 30000

# Regular top-up (duplicates skipped automatically)
python update_data.py --appid <APPID>

# Including a BigQuery sync
python update_data.py --appid <APPID> --sync-bigquery --project <YOUR_PROJECT_ID>
```

Run `python update_data.py --help` for every argument explained with
examples. Individual steps can also be run separately:

```bash
python ingestion/steam_ingest.py --appid <APPID> --max-reviews 30000
python analysis/sentiment.py
python analysis/features.py
python analysis/recommendation_model.py
python analysis/recommendation_model.py --exclude weighted_vote_score   # ablation
python analysis/time_series.py --appid <APPID>
python analysis/sql_report.py --appid <APPID> --pivot-date YYYY-MM-DD
streamlit run dashboard/app.py
```

## BigQuery setup (for cloud-scale querying)

```bash
gcloud init
gcloud services enable bigquery.googleapis.com
gcloud auth application-default login
```

Then:

```bash
python migration/migrate_to_bigquery.py --project <YOUR_PROJECT_ID>
python analysis/bigquery_sql_report.py --project <YOUR_PROJECT_ID> --appid <APPID> --pivot-date YYYY-MM-DD
```

## Engineering challenges solved along the way

Real public APIs (and real cloud tooling) are a REAL WORK. These were genuine bugs and quirks discovered and fixed during the
build:

- **The free-to-play zero-reviews trap, and a corrected diagnosis.** Marvel
  Rivals returned zero reviews despite having 290K+. The first working
  theory blamed Steam's date-ordered filters (`recent`/`updated`), since
  only `filter=all` returned data in early tests. The real cause turned out
  to be Steam's `purchase_type` parameter, which defaults to `steam`
  (purchased on Steam) and therefore excludes essentially every reviewer of
  a free-to-play title. The early `filter=all` test had coincidentally also
  passed `purchase_type=all`, hiding which parameter actually mattered.
  Isolated with a controlled A/B request changing only `purchase_type`:
  0 reviews vs 291,812. The ingestion script now always sends
  `purchase_type=all`, and date-ordered filters work fine for F2P titles.
- **Silent empty ingests are a pipeline hazard.** The zero-review case
  above initially printed "Done. Ingested 0 reviews" and let downstream
  steps run on no data. The ingestion script now exits nonzero with an
  explanatory warning when nothing is ingested, so chained pipelines stop
  at the actual point of failure.
- **Token vs. character truncation.** The sentiment model has a hard 512-
  *token* limit. An early version truncated by *character* count, which
  crashed on reviews with heavy punctuation/caps that tokenize into more
  pieces than plain prose. Fixed by letting the tokenizer itself handle
  truncation.
- **Steam's age-gate cookie requirement.** Some titles carry mature-content
  descriptors (accurately or via tag-vandalism) that make the review API
  silently return zero results without an age-verification cookie. The
  ingestion script sets the cookie automatically.
- **Extreme outlier skew in time-series data.** Palworld's 1.0 launch
  produced a review-volume spike about 1000x the daily baseline, badly
  distorting a raw-count ARIMA fit (skew 10.45, kurtosis 189.57). A log1p
  transform brought skew down to 1.09 and made the AR/MA terms
  statistically significant.
- **Large uploads die on transient network drops.** A single-stream upload
  of the 65K-row reviews table (full review text) was killed by a
  connection reset mid-transfer. The migration now uploads in 20K-row
  chunks (first truncates, rest append, preserving idempotent full-replace
  semantics) with per-chunk retry and backoff.
- **SQLite and BigQuery don't speak identical SQL.** `AVG()` on a boolean
  needs an explicit `CAST`, week-bucketing needs `DATE_TRUNC`, and
  BigQuery's strict `GROUP BY` rejects a pattern SQLite tolerates.
  Validated by confirming BigQuery results matched the original SQLite
  output exactly.

## Case study: Palworld

The pipeline's first full run. Numbers from ~30,700 reviews spanning
Feb 2025 to Jul 2026 (since topped up via `update_data.py`).

**Recommendation model:**

| Model | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.950 |
| Random Forest | 0.968 |
| Random Forest, `weighted_vote_score` removed (ablation) | 0.942 |

Sentiment is the dominant feature by a wide margin. The base rate is ~94%
recommended, so precision/recall on the minority (not-recommended) class
matters more than the headline AUC.

**Time-series finding:** the 5 highest-volume days all fall between July 10
and 16, 2026, matching Palworld's full 1.0 release out of Early Access
(about 850K concurrent players) and its follow-up hotfix. Not random noise.

**SQL layer finding:** despite the massive volume spike, sentiment barely
moved (94.6% to 94.1% recommended, pre/post-launch). The update drove
attention, not a shift in opinion. Separately, reviews the sentiment model
scores as *negative* are still marked "recommended" 64.6% of the time:
players frequently vent real frustration in the text while still
recommending the game, which is exactly why sentiment alone doesn't hit a
perfect AUC on the recommendation label. Confirmed identically on both
SQLite and BigQuery.

## Second case study: Marvel Rivals

Added as a contrasting dataset: a free-to-play, "Mostly
Positive" (not "Overwhelmingly Positive") title with a much larger
negative-review base than Palworld, and the source of the free-to-play
ingestion bug documented above. With both games loaded, the recommendation
model trains across two very differently-shaped communities, and the
dashboard's game selector makes the contrast directly explorable.

## Known data quirks worth knowing about

- Steam's `purchase_type` defaults to purchased-on-Steam only, which
  silently zeroes out free-to-play titles (see "Engineering challenges").
  The ingestion script handles this, but be aware of it if querying the
  API directly.
- `filter=all` is relevance/helpfulness-ranked, not time-ordered. Don't use
  it for time-series work without accounting for the sampling bias it
  introduces. With the purchase-type fix, `filter=recent` (chronological)
  works for every title tested so far, including free-to-play ones.
- ADF/ARIMA diagnostics on a launch-driven dataset will likely never look
  fully "normal" (some residual kurtosis is expected). A launch that pulls
  850K concurrent players is a genuinely rare event, and that's a real
  finding, not a modeling failure to hide.

## Privacy note

Review text and Steam IDs are public data (visible to anyone on the Steam
store page), but the `.db` file and generated outputs are excluded from this
repo via `.gitignore`. The code here generates them on demand rather than
shipping a scraped dataset.

## Project structure

```
steam-analytics/
|-- update_data.py                 One-command incremental data update
|-- ingestion/
|   `-- steam_ingest.py            Pull reviews from Steam's public API
|-- analysis/
|   |-- sentiment.py               Score review sentiment (incremental)
|   |-- features.py                Build the ML feature matrix
|   |-- recommendation_model.py    Train/evaluate the recommendation model
|   |-- time_series.py             Decomposition + ARIMA forecasting
|   |-- sql_report.py              Run analytics queries (SQLite)
|   `-- bigquery_sql_report.py     Run analytics queries (BigQuery)
|-- sql/
|   |-- analytics_queries.sql      Analytics queries (SQLite dialect)
|   `-- bigquery_analytics_queries.sql   Same queries (BigQuery dialect)
|-- migration/
|   `-- migrate_to_bigquery.py     Chunked, retrying SQLite -> BigQuery sync
|-- dashboard/
|   `-- app.py                     Interactive Streamlit dashboard
|-- .streamlit/
|   `-- config.toml                Dashboard theme (dark)
|-- db/
|   `-- schema.sql                 SQLite schema
`-- requirements.txt
```

## Project status

- [x] Ingestion pipeline + schema (generalized to any appid, F2P-safe, fails loudly on empty)
- [x] Sentiment scoring (incremental)
- [x] Feature engineering
- [x] Recommendation prediction model + ablation testing + saved evaluation artifacts
- [x] Time-series decomposition + forecasting
- [x] Parameterized SQL analytics layer (SQLite)
- [x] BigQuery migration (chunked, retrying)
- [x] Parameterized SQL analytics layer (BigQuery)
- [x] Interactive Streamlit dashboard (dark theme, sample reviews, SQL explorer, model evaluation, backend toggle, game media)
- [x] One-command incremental update pipeline
- [x] Second case-study game (Marvel Rivals)
- [ ] Cloud-hosted dashboard (Looker Studio) - in progress
