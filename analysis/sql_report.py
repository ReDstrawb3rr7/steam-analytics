"""
Purpose: Runs the analytics SQL queries against the database, substituting the {{APPID}} (and optional {{PIVOT_DATE}}) placeholders so the same query
file works for any game that was ingested.
Usage:
eg:
    python analysis/sql_report.py --appid 1623730 --pivot-date 2026-07-10
    python analysis/sql_report.py --appid 2767030
        (query 1 is skipped automatically if --pivot-date isn't given)
"""
import argparse
import os
import re
import sqlite3
import pandas as pd
 
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "steam_analytics.db")
SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "analytics_queries.sql")
 
def load_query_blocks():
    """Intention: Split the .sql file into (comment_header, query_text) blocks, dropping any purely decorative comment-only chunks (like the file's top banner) that don't contain an actual SQL statement."""
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
    parser.add_argument("--appid", type=int, required=True)
    parser.add_argument(
        "--pivot-date",
        default=None,
        help="YYYY-MM-DD date to split pre/post comparison around (e.g. a patch date). "
             "Skips query 1 if not given.",
    )
    args = parser.parse_args()
 
    conn = sqlite3.connect(DB_PATH)
    blocks = load_query_blocks()
 
    for header, query in blocks:
        if "{{PIVOT_DATE}}" in query and not args.pivot_date:
            print(f"\n--- {header} ---")
            print("Skipped: needs --pivot-date to run.")
            continue
 
        rendered = query.replace("{{APPID}}", str(args.appid))
        if args.pivot_date:
            rendered = rendered.replace("{{PIVOT_DATE}}", args.pivot_date)
 
        print(f"\n--- {header} ---")
        try:
            df = pd.read_sql_query(rendered, conn)
            with pd.option_context("display.max_rows", 20, "display.width", 120):
                print(df)
        except Exception as e:
            print(f"ERROR running this query: {e}")
 
    conn.close()
 
if __name__ == "__main__":
    main()