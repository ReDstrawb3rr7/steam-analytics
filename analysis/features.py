import os
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "steam_analytics.db")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "review_features.csv")

def build_features(conn) -> pd.DataFrame:
    query = """
        SELECT
            r.recommendation_id,
            r.review_length,
            r.votes_up,
            r.votes_funny,
            r.weighted_vote_score,
            r.comment_count,
            r.playtime_forever,
            r.playtime_at_review,
            r.steam_purchase,
            r.received_for_free,
            r.written_during_early_access,
            r.timestamp_created,
            rv.num_games_owned,
            rv.num_reviews AS reviewer_num_reviews,
            s.sentiment_label,
            s.sentiment_score,
            r.voted_up
        FROM reviews r
        LEFT JOIN reviewers rv ON r.steamid = rv.steamid
        LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
        WHERE s.sentiment_label IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, parse_dates=["timestamp_created"])
 
    # One-hot encode sentiment label possible values (positive, neutral, negative) into separate columns
    sentiment_dummies = pd.get_dummies(df["sentiment_label"], prefix="sentiment")
    df = pd.concat([df, sentiment_dummies], axis=1)
    # sentiment_score usable as a single numeric feature instead of just a label + confidence pair.
    sign_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_signed"] = df["sentiment_label"].map(sign_map) * df["sentiment_score"]
 
    df["day_of_week"] = df["timestamp_created"].dt.dayofweek
    df["hour_of_day"] = df["timestamp_created"].dt.hour
 
    # Playtime is heavily right-skewed (a few users with thousands of hours), log-transform so the model isn't dominated by outliers
    df["log_playtime_at_review"] = np.log1p(df["playtime_at_review"].clip(lower=0).fillna(0))
 
    df = df.fillna({"num_games_owned": 0, "reviewer_num_reviews": 0, "votes_up": 0, "votes_funny": 0, "weighted_vote_score": 0, "comment_count": 0, "log_playtime_at_review": 0})
    return df

def main():
    conn = sqlite3.connect(DB_PATH)
    df = build_features(conn)
    conn.close()
 
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
 
    print(f"Wrote {len(df)} review feature rows to {OUT_PATH}")
    print(f"Recommended (voted_up=1) rate: {df['voted_up'].mean():.1%}")
    print(f"Sentiment label distribution:\n{df['sentiment_label'].value_counts()}")
 
 
if __name__ == "__main__":
    main()
