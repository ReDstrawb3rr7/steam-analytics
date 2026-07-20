import os
import sqlite3
from datetime import datetime, timezone
import pandas as pd
from transformers import pipeline

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "steam_analytics.db")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32

def load_unscored_reviews(conn) -> pd.DataFrame:
    query = """
        SELECT r.recommendation_id, r.review_text
        FROM reviews r
        LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
        WHERE s.recommendation_id IS NULL
          AND r.review_text IS NOT NULL
          AND length(trim(r.review_text)) > 0
    """
    return pd.read_sql_query(query, conn)

def score_reviews(df: pd.DataFrame, classifier) -> pd.DataFrame:
    results = []
    texts = df["review_text"].tolist()
 
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t[:2000] for t in texts[i : i + BATCH_SIZE]]
        results.extend(classifier(batch, truncation=True, max_length=512, padding=True))
 
    df = df.copy()
    df["sentiment_label"] = [r["label"].lower() for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    return df

def write_scores(conn, df: pd.DataFrame):
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (row.recommendation_id, row.sentiment_label, row.sentiment_score, now)
        for row in df.itertuples()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO review_scores
            (recommendation_id, sentiment_label, sentiment_score, scored_at)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()

def main():
    conn = sqlite3.connect(DB_PATH)
    df = load_unscored_reviews(conn)
 
    if df.empty:
        print("No unscored reviews found.")
        return
 
    print(f"Scoring {len(df)} reviews...")
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME, truncation=True)

    CHUNK = 500
    total_scored = 0
    for start in range(0, len(df), CHUNK):
        chunk_df = df.iloc[start : start + CHUNK]
        scored = score_reviews(chunk_df, classifier)
        write_scores(conn, scored)
        total_scored += len(scored)
        print(f"  Scored {total_scored}/{len(df)} reviews...")
 
    print(f"Done. Wrote {total_scored} sentiment scores.")
    conn.close()
 
 
if __name__ == "__main__":
    main()