"""
Migrates the local SQLite database to BigQuery.
Usage:
    python migration/migrate_to_bigquery.py
    python migration/migrate_to_bigquery.py --project steam-analytics-503010 --dataset steam_analytics
"""

import argparse
import os
import sqlite3
import pandas as pd
from google.cloud import bigquery

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "steam_analytics.db")

TABLE_SCHEMAS = {
    "games": [
        bigquery.SchemaField("appid", "INTEGER"),
        bigquery.SchemaField("name", "STRING"),
        bigquery.SchemaField("first_seen_at", "TIMESTAMP"),
    ],
    "reviewers": [
        bigquery.SchemaField("steamid", "STRING"),
        bigquery.SchemaField("num_games_owned", "INTEGER"),
        bigquery.SchemaField("num_reviews", "INTEGER"),
        bigquery.SchemaField("first_seen_at", "TIMESTAMP"),
    ],
    "reviews": [
        bigquery.SchemaField("recommendation_id", "STRING"),
        bigquery.SchemaField("appid", "INTEGER"),
        bigquery.SchemaField("steamid", "STRING"),
        bigquery.SchemaField("review_text", "STRING"),
        bigquery.SchemaField("review_length", "INTEGER"),
        bigquery.SchemaField("language", "STRING"),
        bigquery.SchemaField("voted_up", "BOOLEAN"),
        bigquery.SchemaField("votes_up", "INTEGER"),
        bigquery.SchemaField("votes_funny", "INTEGER"),
        bigquery.SchemaField("weighted_vote_score", "FLOAT64"),
        bigquery.SchemaField("comment_count", "INTEGER"),
        bigquery.SchemaField("playtime_forever", "INTEGER"),
        bigquery.SchemaField("playtime_at_review", "INTEGER"),
        bigquery.SchemaField("steam_purchase", "BOOLEAN"),
        bigquery.SchemaField("received_for_free", "BOOLEAN"),
        bigquery.SchemaField("written_during_early_access", "BOOLEAN"),
        bigquery.SchemaField("timestamp_created", "TIMESTAMP"),
        bigquery.SchemaField("timestamp_updated", "TIMESTAMP"),
    ],
    "review_scores": [
        bigquery.SchemaField("recommendation_id", "STRING"),
        bigquery.SchemaField("sentiment_label", "STRING"),
        bigquery.SchemaField("sentiment_score", "FLOAT64"),
        bigquery.SchemaField("scored_at", "TIMESTAMP"),
    ],
}
 
BOOL_COLUMNS = {
    "reviews": ["voted_up", "steam_purchase", "received_for_free", "written_during_early_access"],
}
 
TIMESTAMP_COLUMNS = {
    "games": ["first_seen_at"],
    "reviewers": ["first_seen_at"],
    "reviews": ["timestamp_created", "timestamp_updated"],
    "review_scores": ["scored_at"],
}

TABLE_ORDER = ["games", "reviewers", "reviews", "review_scores"]
 
 
def load_table_from_sqlite(sqlite_conn, table_name: str) -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
 
    for col in TIMESTAMP_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
 
    for col in BOOL_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = df[col].astype(bool)
 
    return df
 
 
def ensure_dataset(client: bigquery.Client, dataset_id: str, location: str):
    dataset_ref = bigquery.DatasetReference(client.project, dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset '{dataset_id}' already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)
        print(f"Created dataset '{dataset_id}' in {location}.")
    return dataset_ref
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None, help="GCP project ID; defaults to your gcloud default project")
    parser.add_argument("--dataset", default="steam_analytics")
    parser.add_argument("--location", default="US")
    args = parser.parse_args()
 
    client = bigquery.Client(project=args.project)
    print(f"Using GCP project: {client.project}")
 
    dataset_ref = ensure_dataset(client, args.dataset, args.location)
    sqlite_conn = sqlite3.connect(DB_PATH)
 
    for table_name in TABLE_ORDER:
        print(f"\nMigrating table: {table_name}")
        df = load_table_from_sqlite(sqlite_conn, table_name)
        print(f"  {len(df)} rows read from SQLite")
 
        if df.empty:
            print("  Skipping (no rows).")
            continue
 
        table_ref = dataset_ref.table(table_name)
        job_config = bigquery.LoadJobConfig(
            schema=TABLE_SCHEMAS[table_name],
            write_disposition="WRITE_TRUNCATE",
        )
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # blocks until the load finishes
        print(f"  Loaded {job.output_rows} rows into {args.dataset}.{table_name}")
 
    sqlite_conn.close()
    print("\nMigration complete.")
    print(f"View it at: https://console.cloud.google.com/bigquery?project={client.project}")
 
 
if __name__ == "__main__":
    main()