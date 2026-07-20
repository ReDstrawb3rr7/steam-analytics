# Steam Community Analytics

A configurable, end-to-end pipeline for analyzing player sentiment, engagement,
and community trends from Steam game reviews. Point it at any Steam appid:
it ingests reviews, scores sentiment, trains a recommendation-prediction
model, runs time-series forecasting, and surfaces everything through a
parameterized SQL layer and an interactive dashboard.

Built as a portfolio project to demonstrate applied data science across a
full pipeline: data ingestion from a public API; NLP; supervised
ML with proper validation; time-series methodology; analytical SQL; and an
interactive front end.

## What it does

- **Ingests reviews for any Steam game** via its public appid, no API key or
  approval process required (`ingestion/steam_ingest.py`)
- **Scores sentiment** on review text using a pretrained transformer, tuned
  for informal social text rather than a domain-mismatched model
  (`analysis/sentiment.py`)
- **Predicts whether a review is "recommended"** from sentiment, engagement,
  and reviewer-history features, with a logistic regression baseline and a
  random forest, plus a built-in feature-ablation flag for testing which
  signals actually matter (`analysis/features.py`, `analysis/recommendation_model.py`)
- **Decomposes and forecasts review volume and sentiment over time**
  (weekly seasonality, ARIMA), with log-transform handling for the extreme
  outlier spikes that real launch/patch events create (`analysis/time_series.py`)
- **Runs a parameterized SQL analytics layer**: rolling averages, percentile
  ranking, day-over-day change, before/after comparisons around any date you
  choose, reusable across any ingested game, not hardcoded to one
  (`sql/analytics_queries.sql`, `analysis/sql_report.py`)
- **Surfaces all of the above in an interactive Streamlit dashboard**, with a
  game selector and adjustable pivot date (`dashboard/app.py`)
- **(In progress)** BigQuery migration for scale, with a Looker Studio /
  cloud-hosted dashboard option

## Architecture

```
Steam public API → Python ingestion (steam_ingest.py) → SQLite
                                                              |
                                          Sentiment scoring (sentiment.py)
                                                              |
                    +--------------------------+--------------------------+
          Feature engineering            Time-series               SQL analytics layer
          + recommendation model         decomposition/forecast    (window functions,
          (sklearn, ablation testing)    (statsmodels)             parameterized by appid)
                    +--------------------------+--------------------------+
                                                              |
                                    Streamlit dashboard (interactive, per-game)
```

## Tech stack, and why each piece was chosen

| Tool | Why |
|---|---|
| Steam's public `appreviews` endpoint | No auth, no approval queue. Unlike Discord (requires server-admin permission to add a bot) or Reddit (closed self-service API access in Nov 2025 under its "Responsible Builder Policy") |
| SQLite | Simple, file-based, sufficient at this scale; schema written to migrate cleanly to BigQuery |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | A general social-text sentiment model, deliberately not FinBERT, which is finance-domain-specific and a poor fit for informal review text |
| scikit-learn | Logistic regression baseline plus a random forest, chosen over an invented "churn" label (reviews aren't a repeat-engagement stream like chat messages) in favor of predicting `voted_up`, a real label already in the data |
| statsmodels | Proper time-series methodology: stationarity testing, log-transforming skewed count data before ARIMA, seasonal decomposition, not just eyeballing a line chart |
| SQL window functions | Rolling averages, `NTILE`/`PERCENT_RANK`, `LAG`: analytical SQL as a first-class part of the project, not just storage |
| Streamlit + Plotly | Fast interactive dashboarding without building a custom frontend |

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

No API keys or credentials needed for the Steam pipeline.

## Running the pipeline (any game)

Find a game's appid from its Steam store URL: `store.steampowered.com/app/<appid>`

```bash
# 1. Ingest reviews
python ingestion/steam_ingest.py --appid <APPID> --max-reviews 30000

# 2. Score sentiment
python analysis/sentiment.py

# 3. Build the feature matrix
python analysis/features.py

# 4. Train the recommendation model
python analysis/recommendation_model.py
python analysis/recommendation_model.py --exclude weighted_vote_score   # ablation test

# 5. Time-series decomposition and forecasting
python analysis/time_series.py --appid <APPID>

# 6. SQL analytics report
python analysis/sql_report.py --appid <APPID> --pivot-date YYYY-MM-DD   # pivot date optional

# 7. Interactive dashboard
streamlit run dashboard/app.py
```

## Engineering challenges solved along the way

Real public APIs are messier than tutorials suggest. These were genuine bugs
and quirks discovered and fixed during the build, not anticipated in advance:

- **Token vs. character truncation.** The sentiment model has a hard 512-
  *token* limit; an early version truncated by *character* count, which
  crashed on reviews with heavy punctuation/caps that tokenize into more
  pieces than plain prose. Fixed by letting the tokenizer itself handle
  truncation.
- **Steam's age-gate cookie requirement.** Some titles (accurately or via
  tag-vandalism) carry mature-content descriptors that make the review API
  silently return zero results without an age-verification cookie, instead
  of an error. Fixed by setting the cookie automatically in the ingestion
  script.
- **A high-volume title's date filters were broken.** Testing a second game
  (Marvel Rivals) revealed `filter=recent` and `filter=updated` both
  silently return zero reviews for that title, despite it having 290K+
  reviews. Isolated via direct, minimal API calls rather than assuming the
  pipeline code was at fault.
- **Extreme outlier skew in time-series data.** A single major game update
  produced a review-volume spike about 1000x the daily baseline, which badly
  distorted a raw-count ARIMA fit (skew 10.45, kurtosis 189.57). A log1p
  transform brought this down to a much more reasonable skew of 1.09.
- **Hardcoded queries don't scale.** The SQL layer originally hardcoded one
  game's appid and launch date; rebuilt with `{{APPID}}`/`{{PIVOT_DATE}}`
  placeholders and a Python runner so the same queries work for any game.

## Case study: Palworld

The pipeline's first full run, used throughout development to validate each
stage. All numbers below are from about 30,700 reviews spanning Feb 2025 to
Jul 2026.

**Recommendation model:**

| Model | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.949 |
| Random Forest | 0.967 |
| Random Forest, `weighted_vote_score` removed (ablation) | 0.940 |

Sentiment is the dominant feature by a wide margin. The base rate is 94.4%
recommended, so precision/recall on the minority (not-recommended) class
matters more than the headline AUC. See the full classification report from
`recommendation_model.py` for the breakdown.

**Time-series finding:** the 5 highest-volume days all fall between July 10
and 16, 2026, matching Palworld's full 1.0 release out of Early Access
(about 850K concurrent players) and its follow-up hotfix. Not random noise.

**SQL layer finding:** despite the massive volume spike, sentiment barely
moved (94.6% to 94.1% recommended, pre/post-launch). The update drove
attention, not a shift in opinion. Separately, reviews the sentiment model
scores as *negative* are still marked "recommended" 64.6% of the time:
players frequently vent real frustration in the text while still
recommending the game, which is exactly why sentiment alone doesn't hit a
perfect AUC on the recommendation label.

## Known data quirks worth knowing about

- Some high-volume/live-service titles break Steam's date-ordered filters
  (see "Engineering challenges" above). Check `filter=recent` returns data
  before assuming a large `--max-reviews` pull will give a clean
  chronological window for a new game.
- `filter=all` is relevance/helpfulness-ranked, not time-ordered. Don't use
  it for time-series work without accounting for the sampling bias it
  introduces.
- Steam's ADF/ARIMA diagnostics on a launch-driven dataset will likely never
  look fully "normal" (some residual kurtosis is expected). A launch this
  size is a genuinely rare event, and that's a real finding, not a modeling
  failure to hide.

## Privacy note

Review text and Steam IDs are public data (visible to anyone on the Steam
store page), but the `.db` file and generated outputs are excluded from this
repo via `.gitignore`. The code here generates them on demand rather than
shipping a scraped dataset.

## Project status

- [x] Ingestion pipeline + schema (generalized to any appid)
- [x] Sentiment scoring
- [x] Feature engineering
- [x] Recommendation prediction model + ablation testing
- [x] Time-series decomposition + forecasting
- [x] Parameterized SQL analytics layer
- [x] Interactive Streamlit dashboard
- [ ] BigQuery migration
- [ ] Cloud-hosted dashboard (Looker Studio)
