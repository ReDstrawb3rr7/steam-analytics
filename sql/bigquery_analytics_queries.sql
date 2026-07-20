-- =============================================================
-- Steam Community Analytics -- BigQuery dialect
-- Run via analysis/bigquery_sql_report.py, which sets the default
-- dataset and binds @appid / @pivot_date as real query parameters
-- =============================================================

-- 1. Before vs. after a pivot date (e.g. a major update/launch)
-- Skipped automatically by the runner if --pivot-date isn't given.
SELECT
    CASE WHEN DATE(r.timestamp_created) < @pivot_date THEN 'pre-pivot' ELSE 'post-pivot' END AS period,
    COUNT(*) AS review_count,
    ROUND(100.0 * AVG(CAST(r.voted_up AS INT64)), 1) AS recommend_pct,
    ROUND(AVG(s.sentiment_score), 3) AS avg_sentiment_confidence,
    COUNTIF(s.sentiment_label = 'positive') AS positive_count,
    COUNTIF(s.sentiment_label = 'negative') AS negative_count
FROM reviews r
LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
WHERE r.appid = @appid
GROUP BY period;


-- 2. 7-day rolling average of review volume and recommendation rate
WITH daily AS (
    SELECT
        DATE(timestamp_created) AS day,
        COUNT(*) AS review_count,
        AVG(CAST(voted_up AS INT64)) AS recommend_rate
    FROM reviews
    WHERE appid = @appid
    GROUP BY day
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
SELECT
    recommendation_id,
    votes_up,
    voted_up,
    NTILE(4) OVER (ORDER BY votes_up DESC) AS helpfulness_quartile,
    ROUND(PERCENT_RANK() OVER (ORDER BY votes_up), 3) AS percentile_rank
FROM reviews
WHERE appid = @appid
ORDER BY votes_up DESC;


-- 4. Day-over-day change in review volume (LAG window function)
WITH daily AS (
    SELECT DATE(timestamp_created) AS day, COUNT(*) AS review_count
    FROM reviews
    WHERE appid = @appid
    GROUP BY day
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
SELECT
    CASE
        WHEN playtime_at_review < 600 THEN '0-10 hrs'
        WHEN playtime_at_review < 3000 THEN '10-50 hrs'
        WHEN playtime_at_review < 12000 THEN '50-200 hrs'
        ELSE '200+ hrs'
    END AS playtime_bucket,
    COUNT(*) AS review_count,
    ROUND(100.0 * AVG(CAST(voted_up AS INT64)), 1) AS recommend_pct,
    ROUND(AVG(votes_up), 1) AS avg_helpfulness_votes
FROM reviews
WHERE appid = @appid
GROUP BY playtime_bucket
ORDER BY MIN(playtime_at_review);


-- 6. Most prolific reviewers
-- Note: BigQuery's strict GROUP BY requires num_games_owned in the GROUP
-- BY list even though steamid determines it uniquely -- SQLite doesn't
-- enforce this, BigQuery does.
SELECT
    rv.steamid,
    rv.num_games_owned,
    COUNT(r.recommendation_id) AS reviews_in_dataset,
    ROUND(100.0 * AVG(CAST(r.voted_up AS INT64)), 1) AS recommend_pct,
    ROUND(AVG(r.votes_up), 1) AS avg_helpfulness_votes
FROM reviews r
JOIN reviewers rv ON r.steamid = rv.steamid
GROUP BY rv.steamid, rv.num_games_owned
HAVING reviews_in_dataset >= 1
ORDER BY reviews_in_dataset DESC, avg_helpfulness_votes DESC
LIMIT 50;


-- 7. Sentiment label vs. actual recommendation -- where do they disagree?
SELECT
    s.sentiment_label,
    r.voted_up,
    COUNT(*) AS review_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY s.sentiment_label), 1) AS pct_within_sentiment
FROM reviews r
JOIN review_scores s ON r.recommendation_id = s.recommendation_id
WHERE r.appid = @appid
GROUP BY s.sentiment_label, r.voted_up
ORDER BY s.sentiment_label, r.voted_up;


-- 8. Weekly trend: volume, recommend rate, and sentiment together
WITH weekly AS (
    SELECT
        DATE_TRUNC(DATE(r.timestamp_created), WEEK) AS week_start,
        COUNT(*) AS review_count,
        AVG(CAST(r.voted_up AS INT64)) AS recommend_rate,
        AVG(s.sentiment_score) AS avg_sentiment_confidence
    FROM reviews r
    LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
    WHERE r.appid = @appid
    GROUP BY week_start
)
SELECT
    week_start,
    review_count,
    ROUND(100.0 * recommend_rate, 1) AS recommend_pct,
    ROUND(avg_sentiment_confidence, 3) AS avg_sentiment_confidence
FROM weekly
ORDER BY week_start;