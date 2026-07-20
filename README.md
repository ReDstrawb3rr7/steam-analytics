# Steam Community Analytics

An end-to-end pipeline that ingests Steam game reviews and applies sentiment analysis and supervised ML to understand what drives player sentiment and recommendations.

Case study game: **Palworld** (appid 1623730), ~30,700 reviews spanning Feb 2025–Jul 2026.

## Why this project exists

This started as a Discord community analytics idea, pivoted to Reddit after Discord requires server-admin permission to add a bot, then pivoted again after Reddit closed self-service API access under its "Responsible Builder Policy" (Nov 2025). Steam's public review API requires no authentication, no approval process, and no admin permissions. It requires only just an appid.

## Architecture

```
Steam public API → Python ingestion (steam_ingest.py) →   SQLite
                                                              ↓
                                          Sentiment scoring (sentiment.py)
                                                              ↓
                                    Feature engineering (features.py)
                                                              ↓
                              Recommendation prediction (recommendation_model.py)
```

Currently working on: time-series forecasting of review volume/sentiment (statsmodels), and a SQL analytics layer (cohort/rolling-window queries).

## Why these tools

- **Steam's public `appreviews` endpoint** — no API key, no OAuth, no approval queue. Cursor-based pagination.
- **SQLite** — simple, file-based, sufficient for this scale. Schema is written to be portable to BigQuery later.
- **`cardiffnlp/twitter-roberta-base-sentiment-latest`** — a general social-text sentiment model.
- **scikit-learn** — logistic regression baseline + random forest to predict whether a review is recommended (`voted_up`), using sentiment, review length, playtime, and reviewer-history features.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

No API keys or credentials needed anywhere in this pipeline.

## Running the pipeline

```bash
# 1. Ingest reviews for a given Steam appid (find one at store.steampowered.com/app/<id>)
python ingestion/steam_ingest.py --appid 1623730 --max-reviews 30000

# 2. Score sentiment on all ingested reviews
python analysis/sentiment.py

# 3. Build the review-level feature matrix
python analysis/features.py

# 4. Train and evaluate the recommendation-prediction model
python analysis/recommendation_model.py

# Optional: ablation test, e.g. to check how much a feature is really contributing
python analysis/recommendation_model.py --exclude weighted_vote_score
```

## Known data quirks worth knowing about

- **Some high-volume/live-service titles break Steam's date-ordered filters.** `filter=recent` and `filter=updated` returned zero results for at least one title tested (Marvel Rivals, appid 2767030) despite having 290K+ reviews — `filter=all` (relevance-ranked, not chronological) was the only one that worked. If you point this pipeline at a new game, check `filter=recent` returns data before assuming a large `--max-reviews` pull will give you a clean chronological window.
- **`filter=all` is relevance/helpfulness-ranked, not time-ordered.** Don't use it for time-series work without accounting for the sampling bias it introduces (skews toward older, heavily-upvoted reviews).
- **Some titles require an age-verification cookie** to return any review data at all (`ingestion/steam_ingest.py` sets this automatically) This is separate from the filter issue above.
- **Steam's 512-token limit is not the same as 512 characters** — the sentiment script truncates via the tokenizer itself (`truncation=True, max_length=512`), not a naive character slice, after an earlier version of this crashed on reviews using lots of punctuation/caps that tokenize into more pieces than plain prose.

## Model results (Palworld, 30,732 scored reviews)

Base rate: 94.4% of reviews are recommended (matches Palworld's"Overwhelmingly Positive" rating on Steam).

| Model | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | 0.949 |
| Random Forest | 0.967 |
| Random Forest, `weighted_vote_score` removed | 0.940 |

Sentiment (`sentiment_signed` + label one-hots) is the dominant feature by a wide margin. Removing `weighted_vote_score` costs ~2.7 points of AUC and hurts the minority (not-recommended) class's precision/recall most — it's a genuine secondary signal, not leakage, but likely partly reflects "does this review agree with majority opinion" rather than pure sentiment.

**Caveat worth being upfront about:** the 94.4% base rate means the model's precision on the minority (not-recommended) class is substantially weaker than its headline AUC suggests — see the classification report from `recommendation_model.py` for the full breakdown, not just the AUC number.

## Privacy note

Review text and Steam IDs are public data (visible to anyone on the Steam store page), but the `.db` file and any generated CSVs are excluded from this repo via `.gitignore` — the code here generates them on demand rather than shipping the scraped dataset itself.

## Project status

- [x] Ingestion pipeline + schema
- [x] Sentiment scoring
- [x] Feature engineering
- [x] Recommendation prediction model (logistic regression + random forest)
- [ ] Time-series decomposition + forecasting
- [ ] SQL analytics layer
- [ ] BigQuery migration
- [ ] Dashboard
