# Findings: What 65,000 Steam Reviews Reveal

A narrative walkthrough of the main results from the Steam Community
Analytics pipeline, comparing two very different game communities:
Palworld (an "Overwhelmingly Positive" paid survival game) and Marvel
Rivals (a "Mostly Positive" free-to-play hero shooter with a much larger
negative-review base).

> **This is a snapshot, not a fixed result.** Every number below reflects
> the dataset at the time these runs were recorded. `update_data.py`
> retrains the model on the full corpus every time it runs, so adding
> more games or fresh reviews will shift these figures. That is expected
> and correct, since the whole point of the pipeline is to keep working
> on growing, real data. Re-run `python analysis/recommendation_model.py`
> at any time to see current numbers. This document is the single place
> current figures live. Other project files link here rather than
> repeating numbers that could drift out of sync.

All numbers come from the pipeline in this repo and are reproducible by
running it. Model metrics in section 4 reflect the combined two-game
dataset. Time-series and volume findings in sections 1 and 2 are
specific to Palworld's own history, as noted in each section.

## 1. A launch drives attention, not opinion

Palworld's five highest-volume review days all fall between July 10 and
16, 2026, the window of its full 1.0 release out of Early Access (which
drew roughly 850K concurrent players) and its follow-up hotfix. Daily
review volume jumped from a baseline of about 60 reviews per day to over
1,100.

The natural assumption is that a spike like that means controversy or a
shifted reception. The data says otherwise: the recommend rate barely
moved, 94.6% before the launch to 94.1% after, and average sentiment
confidence was effectively unchanged (0.808 to 0.813). The launch
multiplied the number of people talking without changing what they were
saying. Volume and opinion are separate signals, and conflating them is
an easy analytical mistake this dataset makes visible.

## 2. Negative words, positive verdict

The single most interesting number in the project: reviews that the
sentiment model scores as *negative* are still marked "recommended"
64.6% of the time.

This is not model failure. Reading the actual reviews shows why: players
routinely spend paragraphs venting real frustration (performance,
matchmaking, toxicity) and then conclude "I still recommend it." One of
the most-helpful Marvel Rivals reviews does exactly this, an extended
complaint about optimization that ends by recommending the game to
anyone with good hardware.

This is also exactly why text sentiment alone cannot perfectly predict
the recommendation label, and it quantifies the gap: tone and verdict
are correlated but genuinely distinct signals. For anyone building
review-analysis systems, this is the concrete reason to model them
separately.

## 3. Playtime predicts the verdict

Recommend rate climbs monotonically with hours played at review time
(Palworld):

| Playtime at review | Recommend rate |
|---|---|
| 0-10 hours | 88.9% |
| 10-50 hours | 94.7% |
| 50-200 hours | 95.6% |
| 200+ hours | 96.6% |

Partly self-selection: people who bounce off a game leave early,
negative reviews, while people with 200 hours are there because they
like it. But it is a clean, strong pattern either way, and the log-transformed playtime
feature carries real weight in the prediction model.

## 4. The model, and testing it honestly

Three models predict `voted_up` from sentiment, engagement, playtime,
and reviewer-history features, trained on the combined two-game dataset
(65,102 reviews, 81.1% recommended):

| Model | Random split ROC-AUC | Temporal split ROC-AUC | Gap |
|---|---|---|---|
| Logistic regression | 0.916 | 0.910 | +0.007 |
| Random forest | 0.943 | 0.938 | +0.004 |
| XGBoost | 0.949 | 0.941 | +0.008 |

XGBoost consistently edges out the random forest by a small margin, the
typical honest result for tabular data of this kind.

A note on comparing across dataset versions: on Palworld alone (94.4%
recommended), the random forest scored 0.968. The lower number here is
not a worse model but a harder problem. Adding Marvel Rivals made the
classes far more balanced, and minority-class detection actually
improved substantially (not-recommended F1 rose from about 0.56 to
0.71). Headline AUC and problem difficulty move together, which is
exactly why single-number comparisons across different datasets mislead.

Three honesty checks matter more than the headline numbers:

**Ablation.** `weighted_vote_score` (community helpfulness votes) looked
suspiciously strong as a feature on early runs. On the combined dataset,
removing it costs only about 0.9 AUC points (0.943 to 0.934 for the
random forest), confirming sentiment carries most of the signal and the
result does not depend on one possibly-circular feature.

**Temporal validation.** A random train/test split lets the model train
on some future reviews and test on past ones, which slightly overstates
real forecasting ability. Training on the earliest 75% of reviews and
testing on the most recent 25% answers the stricter question: could this
have predicted future reviews? The gaps above are tiny (under 0.01 AUC
for every model), meaning the learned relationships hold forward in time
rather than only across a shuffled sample.

**The base-rate caveat.** At an 81% base rate, a model that always
guesses "recommended" is 81% accurate, and on the earlier Palworld-only
data (94% base rate) that trivial baseline was even more flattering. AUC
and minority-class precision/recall are the honest metrics throughout.

One feature-importance shift worth noting: with both games in the data,
`written_during_early_access` jumped from near-zero importance to about
8%. It effectively distinguishes Palworld's Early Access era from
everything else, a reminder that feature importances describe the
dataset in front of the model, not universal truths.

## 5. Two communities, one pipeline

Marvel Rivals was added as a deliberate contrast: free-to-play, more
contested, with an order of magnitude more negative reviews
proportionally. The same pipeline handles both without modification,
which surfaced a genuinely instructive bug along the way: Steam's review
API defaults `purchase_type` to purchased-on-Steam, which excludes
essentially every reviewer of a free-to-play game and silently returns
zero reviews. The first diagnosis (blaming the API's date-ordering
filters) was wrong, and a controlled A/B request changing only that one
parameter found the real cause: 0 reviews versus 291,812.

The broader lesson from the whole build: real public APIs fail
silently and misleadingly, and the difference between a stuck project
and a finished one is isolating variables one at a time instead of
guessing.

## Where the data and code live

- Pipeline, models, dashboard: this repo (see README for architecture)
- Cloud copy of the dataset: BigQuery (`steam_analytics` dataset)
- Live hosted dashboard: linked in the README
