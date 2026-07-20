import argparse
import os
import sqlite3
import time
from datetime import datetime, timezone
import requests

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "steam_analytics.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")

steam_reviews_url = "https://store.steampowered.com/appreviews/{appid}"
steam_app_url = "https://store.steampowered.com/api/appdetails"

PAGE_SIZE = 100
SLEEP_SECONDS = 1.0

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

def to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def fetch_game_name(appid: int) -> str:
    try:
        resp = requests.get(steam_app_url, params={"appids": appid}, timeout=10)
        data = resp.json()
        return data[str(appid)]["data"]["name"]
    except Exception:
        return f"App {appid}"

def upsert_game(conn, appid: int, name: str):
    conn.execute(
        "INSERT OR IGNORE INTO games (appid, name) VALUES (?, ?)", (appid, name)
    )
    conn.commit()

def upsert_reviewer(conn, author: dict):
    conn.execute(
        """
        INSERT INTO reviewers (steamid, num_games_owned, num_reviews)
        VALUES (?, ?, ?)
        ON CONFLICT(steamid) DO UPDATE SET
            num_games_owned = excluded.num_games_owned,
            num_reviews = excluded.num_reviews
        """,
        (author["steamid"], author.get("num_games_owned", 0), author.get("num_reviews", 0)),
    )

def insert_review(conn, appid: int, review: dict):
    author = review["author"]
    conn.execute(
        """
        INSERT OR IGNORE INTO reviews
            (recommendation_id, appid, steamid, review_text, review_length,
             language, voted_up, votes_up, votes_funny, weighted_vote_score,
             comment_count, playtime_forever, playtime_at_review,
             steam_purchase, received_for_free, written_during_early_access,
             timestamp_created, timestamp_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review["recommendationid"],
            appid,
            author["steamid"],
            review.get("review", ""),
            len(review.get("review", "")),
            review.get("language"),
            int(review.get("voted_up", False)),
            review.get("votes_up", 0),
            review.get("votes_funny", 0),
            review.get("weighted_vote_score", 0.0),
            review.get("comment_count", 0),
            author.get("playtime_forever", 0),
            author.get("playtime_at_review", 0),
            int(review.get("steam_purchase", False)),
            int(review.get("received_for_free", False)),
            int(review.get("written_during_early_access", False)),
            to_iso(review["timestamp_created"]),
            to_iso(review["timestamp_updated"]),
        ),
    )

def ingest(appid: int, max_reviews: int, review_filter: str, language: str):
    init_db()
    conn = get_conn()
 
    game_name = fetch_game_name(appid)
    upsert_game(conn, appid, game_name)
    print(f"Ingesting reviews for {game_name} (appid {appid})...")
 
    cursor = "*"
    total_pulled = 0
    earliest, latest = None, None
 
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    session.cookies.set("wants_mature_content", "1", domain="store.steampowered.com")
    session.cookies.set("birthtime", "631152000", domain="store.steampowered.com")  # 1990-01-01
    session.cookies.set("lastagecheckage", "1-January-1990", domain="store.steampowered.com")
 
    while total_pulled < max_reviews:
        params = {
            "json": 1,
            "filter": review_filter,
            "language": language,
            "num_per_page": min(PAGE_SIZE, max_reviews - total_pulled),
            "cursor": cursor,
        }
        resp = session.get(steam_reviews_url.format(appid=appid), params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
 
        if data.get("success") != 1:
            print("Steam API returned an error response, stopping.")
            break
 
        reviews = data.get("reviews", [])
        if not reviews:
            print("No more reviews returned, stopping.")
            break
 
        for review in reviews:
            upsert_reviewer(conn, review["author"])
            insert_review(conn, appid, review)
 
            created = to_iso(review["timestamp_created"])
            earliest = created if earliest is None or created < earliest else earliest
            latest = created if latest is None or created > latest else latest
 
        conn.commit()
        total_pulled += len(reviews)
        print(f"  Pulled {total_pulled} reviews so far...")
 
        next_cursor = data.get("cursor")
        if not next_cursor or next_cursor == cursor:
            print("Cursor stopped advancing, reached end of available reviews.")
            break
        cursor = next_cursor
 
        time.sleep(SLEEP_SECONDS)
 
    print(f"\nDone. Ingested {total_pulled} reviews for {game_name}.")
    print(f"Date range: {earliest} -> {latest}")
    conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--appid", type=int, required=True, help="Steam appid, e.g. 1623730 for Palworld")
    parser.add_argument("--max-reviews", type=int, default=2000)
    parser.add_argument("--filter", choices=["recent", "updated", "all"], default="recent")
    parser.add_argument("--language", default="english")
    args = parser.parse_args()

    ingest(args.appid, args.max_reviews, args.filter, args.language)

if __name__ == "__main__":
    main()
