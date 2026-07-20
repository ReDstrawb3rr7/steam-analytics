import argparse
import datetime
import os
import re
import sys
 
import pandas as pd
from google.cloud import bigquery
 
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
 
SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "bigquery_analytics_queries.sql")
 
 
def load_query_blocks():
    text = open(SQL_PATH, encoding="utf-8").read()
    raw_chunks = re.split(r"\n\s*\n", text)
 
    blocks = []
    for raw in raw_chunks:
        raw = raw.strip()
        if not raw:
            continue
        lines = raw.split("\n")
        sql_lines = [l for l in lines if l.strip() and not l.strip().startswith("--")]
        if not sql_lines:
            continue
        header_lines = [l for l in lines if l.strip().startswith("--")]
        header = " ".join(l.lstrip("- ").strip() for l in header_lines) or "(untitled query)"
        query = raw.rstrip(";").strip()
        blocks.append((header, query))
    return blocks
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None, help="GCP project ID; defaults to your gcloud default project")
    parser.add_argument("--dataset", default="steam_analytics")
    parser.add_argument("--appid", type=int, required=True)
    parser.add_argument(
        "--pivot-date",
        default=None,
        help="YYYY-MM-DD. Skips query 1 if not given.",
    )
    args = parser.parse_args()
 
    client = bigquery.Client(project=args.project)
    print(f"Using GCP project: {client.project}, dataset: {args.dataset}")
 
    query_params = [bigquery.ScalarQueryParameter("appid", "INT64", args.appid)]
    if args.pivot_date:
        pivot_date = datetime.date.fromisoformat(args.pivot_date)
        query_params.append(bigquery.ScalarQueryParameter("pivot_date", "DATE", pivot_date))
 
    job_config = bigquery.QueryJobConfig(
        query_parameters=query_params,
        default_dataset=f"{client.project}.{args.dataset}",
    )
 
    blocks = load_query_blocks()
    for header, query in blocks:
        if "@pivot_date" in query and not args.pivot_date:
            print(f"\n--- {header} ---")
            print("Skipped: needs --pivot-date to run.")
            continue
 
        print(f"\n--- {header} ---")
        try:
            df = client.query(query, job_config=job_config).to_dataframe()
            with pd.option_context("display.max_rows", 20, "display.width", 120):
                print(df)
        except Exception as e:
            print(f"ERROR running this query: {e}")
 
 
if __name__ == "__main__":
    main()