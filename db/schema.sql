CREATE TABLE IF NOT EXISTS games (
    appid           INTEGER PRIMARY KEY,
    name            TEXT,
    first_seen_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reviewers (
    steamid             TEXT PRIMARY KEY,
    num_games_owned     INTEGER,
    num_reviews         INTEGER,
    first_seen_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS reviews (
    recommendation_id       TEXT PRIMARY KEY,
    appid                   INTEGER NOT NULL,
    steamid                 TEXT,
    review_text             TEXT,
    review_length           INTEGER,
    language                TEXT,
    voted_up                INTEGER,        -- 1 = recommended, 0 = not recommended
    votes_up                INTEGER,        -- helpfulness votes, Steam's engagement signal
    votes_funny             INTEGER,
    weighted_vote_score     REAL,
    comment_count           INTEGER,
    playtime_forever        INTEGER,        -- minutes, total playtime at time of pull
    playtime_at_review      INTEGER,        -- minutes, playtime when review was written
    steam_purchase          INTEGER,
    received_for_free       INTEGER,
    written_during_early_access INTEGER,
    timestamp_created       TEXT NOT NULL,  
    timestamp_updated       TEXT NOT NULL,
    FOREIGN KEY (appid) REFERENCES games(appid),
    FOREIGN KEY (steamid) REFERENCES reviewers(steamid)
);

CREATE INDEX IF NOT EXISTS idx_reviews_appid_time ON reviews(appid, timestamp_created);
CREATE INDEX IF NOT EXISTS idx_reviews_steamid ON reviews(steamid);
CREATE INDEX IF NOT EXISTS idx_reviews_voted_up ON reviews(voted_up);


CREATE TABLE IF NOT EXISTS review_scores (
    recommendation_id   TEXT PRIMARY KEY,
    sentiment_label      TEXT,
    sentiment_score      REAL,
    scored_at            TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (recommendation_id) REFERENCES reviews(recommendation_id)
);
