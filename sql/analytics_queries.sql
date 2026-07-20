-- =============================================================
-- Steam Community Analytics — analytical SQL layer
-- Run against db/steam_analytics.db (SQLite). Portable to BigQuery with minor date-function changes.
-- =============================================================
-- 1. Before vs. after a pivot date (e.g. a major update/launch)
-- Set --pivot-date when running via sql_report.py to whatever event one is investigating for a given game. Directly follows up on time-series findings: does sentiment or the recommendation rate actually shift around the event, or was it just a volume spike with the same underlying opinion mix?
SELECT
    CASE WHEN date(r.timestamp_created) < '{{PIVOT_DATE}}' THEN 'pre-pivot' ELSE 'post-pivot' END AS period,
    COUNT(*) AS review_count,
    ROUND(100.0 * AVG(r.voted_up), 1) AS recommend_pct,
    ROUND(AVG(s.sentiment_score), 3) AS avg_sentiment_confidence,
    SUM(CASE WHEN s.sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
    SUM(CASE WHEN s.sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count
FROM reviews r
LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
WHERE r.appid = {{APPID}}
GROUP BY period;

-- 2. 7-day rolling average of review volume and recommendation rate
WITH daily AS (
    SELECT
        date(timestamp_created) AS day,
        COUNT(*) AS review_count,
        AVG(voted_up) AS recommend_rate
    FROM reviews
    WHERE appid = {{APPID}}
    GROUP BY date(timestamp_created)
)
SELECT
    day,
    review_count,
    ROUND(AVG(review_count) OVER (
        ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 1) AS rolling_7d_avg_reviews,
    ROUND(100.0 * recommend_rate, 1) AS recommend_pct,
    ROUND(100.0 * AVG(recommend_rate) OVER (
        ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 1) AS rolling_7d_recommend_pct
FROM daily
ORDER BY day;

-- 3. Reviews by helpfulness percentile
-- Which reviews are the most/least helpful relative to the rest?
SELECT
    recommendation_id,
    votes_up,
    voted_up,
    NTILE(4) OVER (ORDER BY votes_up DESC) AS helpfulness_quartile,
    ROUND(PERCENT_RANK() OVER (ORDER BY votes_up), 3) AS percentile_rank
FROM reviews
WHERE appid = {{APPID}}
ORDER BY votes_up DESC;


-- 4. Day-over-day change in review volume (LAG window function)
WITH daily AS (
    SELECT date(timestamp_created) AS day, COUNT(*) AS review_count
    FROM reviews
    WHERE appid = {{APPID}}
    GROUP BY date(timestamp_created)
)
SELECT
    day,
    review_count,
    review_count - LAG(review_count) OVER (ORDER BY day) AS change_from_prev_day,
    ROUND(100.0 * (review_count - LAG(review_count) OVER (ORDER BY day))
        / NULLIF(LAG(review_count) OVER (ORDER BY day), 0), 1) AS pct_change
FROM daily
ORDER BY day;


-- 5. Does playtime relate to recommendation likelihood?
-- Buckets players by hours played at time of review, checks recommend rate per bucket.
SELECT
    CASE
        WHEN playtime_at_review < 600 THEN '0-10 hrs'
        WHEN playtime_at_review < 3000 THEN '10-50 hrs'
        WHEN playtime_at_review < 12000 THEN '50-200 hrs'
        ELSE '200+ hrs'
    END AS playtime_bucket,
    COUNT(*) AS review_count,
    ROUND(100.0 * AVG(voted_up), 1) AS recommend_pct,
    ROUND(AVG(votes_up), 1) AS avg_helpfulness_votes
FROM reviews
WHERE appid = {{APPID}}
GROUP BY playtime_bucket
ORDER BY MIN(playtime_at_review);


-- 6. Most prolific reviewers (useful once multiple games are ingested — shows cross-game reviewing behavior; currently scoped to one game)
SELECT
    rv.steamid,
    rv.num_games_owned,
    COUNT(r.recommendation_id) AS reviews_in_dataset,
    ROUND(100.0 * AVG(r.voted_up), 1) AS recommend_pct,
    ROUND(AVG(r.votes_up), 1) AS avg_helpfulness_votes
FROM reviews r
JOIN reviewers rv ON r.steamid = rv.steamid
GROUP BY rv.steamid
HAVING reviews_in_dataset >= 1
ORDER BY reviews_in_dataset DESC, avg_helpfulness_votes DESC
LIMIT 50;


-- 7. Sentiment label vs. actual recommendation — where do they disagree?
-- A review can be sentiment-negative but still marked recommended (or vice versa) — e.g. a review that's mostly complaints but ends with "I still recommend it". Useful sanity check on how well text sentiment tracks the explicit label.
SELECT
    s.sentiment_label,
    r.voted_up,
    COUNT(*) AS review_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY s.sentiment_label), 1) AS pct_within_sentiment
FROM reviews r
JOIN review_scores s ON r.recommendation_id = s.recommendation_id
WHERE r.appid = {{APPID}}
GROUP BY s.sentiment_label, r.voted_up
ORDER BY s.sentiment_label, r.voted_up;


-- 8. Weekly trend: volume, recommend rate, and sentiment together
WITH weekly AS (
    SELECT
        date(r.timestamp_created, 'weekday 0', '-6 days') AS week_start,
        COUNT(*) AS review_count,
        AVG(r.voted_up) AS recommend_rate,
        AVG(s.sentiment_score) AS avg_sentiment_confidence
    FROM reviews r
    LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
    WHERE r.appid = {{APPID}}
    GROUP BY week_start
)
SELECT
    week_start,
    review_count,
    ROUND(100.0 * recommend_rate, 1) AS recommend_pct,
    ROUND(avg_sentiment_confidence, 3) AS avg_sentiment_confidence
FROM weekly
ORDER BY week_start;