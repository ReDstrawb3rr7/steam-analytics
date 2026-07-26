"""
Usage:
    streamlit run dashboard/app.py
"""

import json
import os
import re
import sqlite3
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
 
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "steam_analytics.db")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
SQLITE_SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "analytics_queries.sql")
 
SENTIMENT_SIGN = {"positive": 1, "neutral": 0, "negative": -1}
 
pio.templates.default = "plotly_dark"
RED_PALETTE = ["#E63946", "#F1A208", "#6C757D", "#A4161A", "#D9BF77"]
CHART_LAYOUT = dict(
    plot_bgcolor="#0D0D0D",
    paper_bgcolor="#0D0D0D",
    font_color="#F5F5F5",
)
 
st.set_page_config(page_title="Steam Community Analytics", page_icon="🎮", layout="wide")
 
# ---------------------------------------------------------------------------
# Data access layer: SQLite by default, BigQuery opt-in.
# All read functions route through run_query() so the rest of the app doesn't care which backend is active.
# ---------------------------------------------------------------------------
 
def run_query(query: str, source: str, bq_project: str | None, params: tuple = ()) -> pd.DataFrame:
    if source == "BigQuery":
        from google.cloud import bigquery
 
        client = bigquery.Client(project=bq_project or None)
        job_config = bigquery.QueryJobConfig(
            default_dataset=f"{client.project}.steam_analytics",
        )
        # BigQuery uses @param placeholders; SQLite uses ?. The queries in
        # this app are written with ? and converted here, with positional
        # parameters mapped in order.
        bq_query = query
        bq_params = []
        for i, value in enumerate(params):
            bq_query = bq_query.replace("?", f"@p{i}", 1)
            ptype = "INT64" if isinstance(value, int) else "STRING"
            bq_params.append(bigquery.ScalarQueryParameter(f"p{i}", ptype, value))
        job_config.query_parameters = bq_params
        return client.query(bq_query, job_config=job_config).to_dataframe()
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
 
 
@st.cache_data
def get_games(source: str, bq_project: str | None):
    return run_query("SELECT appid, name FROM games ORDER BY name", source, bq_project)
 
 
@st.cache_data
def get_review_counts(source: str, bq_project: str | None):
    return run_query(
        "SELECT appid, COUNT(*) AS review_count FROM reviews GROUP BY appid",
        source, bq_project,
    )
 
 
@st.cache_data
def load_reviews(appid: int, source: str, bq_project: str | None) -> pd.DataFrame:
    query = """
        SELECT
            r.recommendation_id, r.voted_up, r.votes_up, r.review_length,
            r.playtime_at_review, r.timestamp_created,
            s.sentiment_label, s.sentiment_score
        FROM reviews r
        LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
        WHERE r.appid = ?
    """
    df = run_query(query, source, bq_project, params=(appid,))
    df["timestamp_created"] = pd.to_datetime(df["timestamp_created"], utc=True)
    # BigQuery returns real booleans; SQLite returns 0/1. Normalize to int
    # so aggregation code works identically on both.
    df["voted_up"] = df["voted_up"].astype(int)
    df["sentiment_signed"] = df["sentiment_label"].map(SENTIMENT_SIGN) * df["sentiment_score"]
    df["day"] = df["timestamp_created"].dt.date
    return df
 
 
@st.cache_data
def load_sample_reviews(appid: int, source: str, bq_project: str | None) -> pd.DataFrame:
    """Most-helpful positive and negative reviews, for the samples section."""
    query = """
        SELECT r.review_text, r.voted_up, r.votes_up, r.playtime_at_review,
               r.timestamp_created, s.sentiment_label
        FROM reviews r
        LEFT JOIN review_scores s ON r.recommendation_id = s.recommendation_id
        WHERE r.appid = ?
          AND r.review_text IS NOT NULL
          AND LENGTH(r.review_text) > 100
        ORDER BY r.votes_up DESC
        LIMIT 200
    """
    df = run_query(query, source, bq_project, params=(appid,))
    df["voted_up"] = df["voted_up"].astype(int)
    return df
 
 
@st.cache_data(ttl=86400)
def _fetch_game_media(appid: int) -> dict:
    """Fetch the current header image URL and official screenshot URLs
    from Steam's public appdetails endpoint, in one call.
 
    The header_image URL from appdetails is cache-busted (it embeds a
    content hash), unlike the bare CDN path .../apps/{appid}/header.jpg,
    which can serve stale art from years-old updates.
 
    Raises on failure rather than returning empty: st.cache_data would
    otherwise cache a transient failure as [] for the full TTL. The
    caller catches and degrades gracefully without caching the failure.
    """
    import requests
    resp = requests.get(
        "https://store.steampowered.com/api/appdetails",
        params={"appids": appid},
        timeout=10,
    )
    data = resp.json()[str(appid)]["data"]
    return {
        "header": data.get("header_image"),
        "screenshots": [s["path_thumbnail"] for s in data.get("screenshots", [])[:4]],
    }
 
 
def load_game_media(appid: int) -> dict:
    """Wrapper that degrades to empty media on any failure, without the
    failure being cached."""
    try:
        return _fetch_game_media(appid)
    except Exception:
        return {"header": None, "screenshots": []}
 
 
def load_model_eval(suffix: str = ""):
    path = os.path.join(OUTPUTS_DIR, f"recommendation_model{suffix}_eval.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
 
 
def load_model_importances(suffix: str = ""):
    filename = f"recommendation_model{suffix}.joblib"
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    importances = pd.Series(
        bundle["model"].feature_importances_, index=bundle["features"]
    ).sort_values(ascending=False)
    return importances
 
 
def load_sql_query_blocks():
    """Parse the SQLite analytics queries file into (title, sql) pairs for
    the SQL explorer. Same parsing approach as analysis/sql_report.py,
    plus: strips decorative banner lines (=====) that would otherwise
    pollute the first query's title if the file header sits adjacent to it."""
    if not os.path.exists(SQLITE_SQL_PATH):
        return []
    text = open(SQLITE_SQL_PATH, encoding="utf-8").read()
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
        # Use only the contiguous comment block immediately above the first
        # SQL line as the title. This drops any file-level banner/header
        # comments that happen to sit adjacent to query 1 with no blank
        # line between them.
        first_sql_idx = next(
            i for i, l in enumerate(lines)
            if l.strip() and not l.strip().startswith("--")
        )
        header_lines = []
        for l in reversed(lines[:first_sql_idx]):
            if not l.strip().startswith("--"):
                break
            stripped = l.lstrip("- ").strip()
            if stripped and set(stripped) <= {"="}:
                break  # hit the banner; everything above is file header
            header_lines.append(stripped)
        header_lines.reverse()
        header = " ".join(h for h in header_lines if h)
        header = re.sub(r"=+", "", header).strip()[:80] or "(untitled)"
        blocks.append((header, raw.rstrip(";").strip()))
    return blocks
 
 
def style_fig(fig):
    fig.update_layout(**CHART_LAYOUT)
    return fig
 
 
def main():
    st.title("🎮 Steam Community Analytics")
    st.caption("Player sentiment, engagement, and recommendation trends from Steam reviews.")
 
    # --- Sidebar: data source, refresh, game, pivot ---
    with st.sidebar:
        st.header("Data source")
        source = st.radio("Backend", ["Local (SQLite)", "BigQuery"], index=0)
        source = "BigQuery" if source == "BigQuery" else "SQLite"
        st.caption(
            "Both backends serve the same data (BigQuery is synced from the "
            "local database). The toggle demonstrates backend portability: "
            "the whole dashboard runs through one data-access layer that "
            "handles both engines' differences behind a single interface."
        )
        bq_project = None
        if source == "BigQuery":
            bq_project = st.text_input("GCP project ID", value="steam-analytics-503010")
            st.caption(
                "Requires `gcloud auth application-default login` to have been "
                "run on this machine, and the dataset migrated via "
                "`migration/migrate_to_bigquery.py`."
            )
 
        if st.button("🔄 Refresh data", help="Clears cached query results and re-reads from the database"):
            st.cache_data.clear()
            st.rerun()
 
        st.divider()
        st.header("Filters")
 
    # Everything below can fail if BigQuery isn't set up properly!! fail visibly, not cryptically
    try:
        games = get_games(source, bq_project)
    except Exception as e:
        st.error(
            f"Could not connect to the **{source}** backend.\n\n"
            f"Error: `{e}`\n\n"
            + ("If using BigQuery: check you've run `gcloud auth application-default login`, "
               "the project ID is correct, and the dataset has been migrated."
               if source == "BigQuery" else
               "Check that `db/steam_analytics.db` exists (run the ingestion pipeline first).")
        )
        return
 
    if games.empty:
        st.warning(
            "No games in the database yet. Ingest one first, for example:\n\n"
            "```\npython ingestion/steam_ingest.py --appid 1623730 --max-reviews 30000\n```"
        )
        return
 
    review_counts = get_review_counts(source, bq_project)
    games = games.merge(review_counts, on="appid", how="left")
    games["review_count"] = games["review_count"].fillna(0).astype(int)
 
    with st.sidebar:
        game_options = [
            f"{row['name']} ({row['appid']}) - {row['review_count']:,} reviews"
            for _, row in games.iterrows()
        ]
        # Track the chosen appid (not the label string) across reruns: the
        # label embeds the review count, so after a data refresh the label
        # text changes and a string-keyed selectbox would fall back to the
        # first option. Restoring by appid keeps the same game selected.
        appids_in_order = games["appid"].tolist()
        default_index = 0
        if "selected_appid" in st.session_state and st.session_state.selected_appid in appids_in_order:
            default_index = appids_in_order.index(st.session_state.selected_appid)
        game_label = st.selectbox("Game", game_options, index=default_index)
        appid = int(game_label.split("(")[1].split(")")[0])
        st.session_state.selected_appid = appid
        game_name = game_label.split(" (")[0]
        pivot_date = st.date_input("Pivot date (optional, e.g. a patch date)", value=None, key="pivot_select")
 
    df = load_reviews(appid, source, bq_project)
    if df.empty:
        st.warning(
            f"**{game_name}** is in the database, but has no reviews loaded yet. "
            "This game hasn't been ingested (or the ingestion returned zero results, "
            "which happens for some titles due to Steam API quirks, see the README). "
            "Run this to pull its reviews:\n\n"
            f"```\npython ingestion/steam_ingest.py --appid {appid} --max-reviews 30000\n"
            f"python analysis/sentiment.py\n```"
        )
        return
 
    # --- Overview ---
    st.subheader("📊 Overview")
 
    media = load_game_media(appid)
    # Prefer the cache-busted header from appdetails (always current art);
    # fall back to the bare CDN path, which works but can serve stale art.
    header_url = media["header"] or f"https://cdn.cloudflare.steamstatic.com/steam/apps/{appid}/header.jpg"
    header_col, text_col = st.columns([1, 2])
    with header_col:
        st.image(header_url, use_container_width=True)
 
    avg_sent = df["sentiment_signed"].mean()
    sentiment_word = "positive" if avg_sent > 0.15 else ("negative" if avg_sent < -0.15 else "mixed")
    recommend_pct = df["voted_up"].mean()
    start_date = df["timestamp_created"].min().date()
    end_date = df["timestamp_created"].max().date()
 
    with text_col:
        st.markdown(
            f"**{game_name}** has **{len(df):,} reviews** collected from **{start_date}** to **{end_date}**. "
            f"**{recommend_pct:.0%}** of reviewers recommend it, and overall review sentiment reads as **{sentiment_word}**. "
            f"Data source: **{source}**."
        )
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total reviews", f"{len(df):,}")
    col2.metric("Recommended", f"{recommend_pct:.1%}")
    col3.metric("Avg. sentiment", f"{avg_sent:+.2f}")
    col4.metric("Date range", f"{start_date} → {end_date}")
 
    screenshots = media["screenshots"]
    if screenshots:
        with st.expander("📷 Game screenshots (from Steam)"):
            shot_cols = st.columns(len(screenshots))
            for c, url in zip(shot_cols, screenshots):
                c.image(url, use_container_width=True)
 
    st.divider()
 
    # --- Daily activity ---
    st.subheader("Daily activity")
    daily = df.groupby("day").agg(
        review_count=("recommendation_id", "count"),
        avg_sentiment=("sentiment_signed", "mean"),
        recommend_rate=("voted_up", "mean"),
    ).reset_index()
    daily["day"] = pd.to_datetime(daily["day"])
    daily["rolling_7d_volume"] = daily["review_count"].rolling(7, min_periods=1).mean()
 
    tab1, tab2 = st.tabs(["Volume", "Sentiment"])
    with tab1:
        fig = px.line(daily, x="day", y=["review_count", "rolling_7d_volume"],
                       labels={"value": "Reviews", "day": "Date", "variable": ""},
                       title="Daily review volume (raw + 7-day rolling average)",
                       color_discrete_sequence=RED_PALETTE)
        if pivot_date:
            fig.add_vline(x=pd.Timestamp(pivot_date), line_dash="dash", line_color="#F1A208",
                           annotation_text="Pivot date")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with tab2:
        fig = px.line(daily, x="day", y="avg_sentiment",
                       labels={"avg_sentiment": "Avg. signed sentiment", "day": "Date"},
                       title="Daily average sentiment (-1 to +1)",
                       color_discrete_sequence=RED_PALETTE)
        fig.add_hline(y=0, line_dash="dot", line_color="#6C757D")
        if pivot_date:
            fig.add_vline(x=pd.Timestamp(pivot_date), line_dash="dash", line_color="#F1A208",
                           annotation_text="Pivot date")
        st.plotly_chart(style_fig(fig), use_container_width=True)
 
    if pivot_date:
        st.subheader(f"Before vs. after {pivot_date}")
        df["period"] = df["timestamp_created"].dt.date.apply(
            lambda d: "pre-pivot" if d < pivot_date else "post-pivot"
        )
        summary = df.groupby("period").agg(
            review_count=("recommendation_id", "count"),
            recommend_pct=("voted_up", "mean"),
            avg_sentiment=("sentiment_signed", "mean"),
        )
        summary["recommend_pct"] = (summary["recommend_pct"] * 100).round(1)
        summary["avg_sentiment"] = summary["avg_sentiment"].round(3)
        st.dataframe(summary, use_container_width=True)
 
    st.divider()
 
    # --- Sample reviews ---
    st.subheader("🗣️ What players are actually saying")
    st.caption("The most-helpful (most upvoted) reviews on each side.")
    samples = load_sample_reviews(appid, source, bq_project)
    pos_samples = samples[samples["voted_up"] == 1].head(3)
    neg_samples = samples[samples["voted_up"] == 0].head(3)
 
    scol1, scol2 = st.columns(2)
    with scol1:
        st.markdown("**👍 Recommended**")
        for _, row in pos_samples.iterrows():
            hours = row["playtime_at_review"] / 60
            with st.expander(f"{row['votes_up']:,} helpful votes · {hours:,.0f} hrs played"):
                st.write(row["review_text"][:1500] + ("..." if len(row["review_text"]) > 1500 else ""))
    with scol2:
        st.markdown("**👎 Not recommended**")
        for _, row in neg_samples.iterrows():
            hours = row["playtime_at_review"] / 60
            with st.expander(f"{row['votes_up']:,} helpful votes · {hours:,.0f} hrs played"):
                st.write(row["review_text"][:1500] + ("..." if len(row["review_text"]) > 1500 else ""))
 
    st.divider()
 
    # --- Playtime vs recommendation ---
    st.subheader("Recommendation rate by playtime")
    bins = [0, 600, 3000, 12000, float("inf")]
    labels = ["0-10 hrs", "10-50 hrs", "50-200 hrs", "200+ hrs"]
    df["playtime_bucket"] = pd.cut(df["playtime_at_review"], bins=bins, labels=labels)
    playtime_summary = df.groupby("playtime_bucket", observed=True).agg(
        review_count=("recommendation_id", "count"),
        recommend_pct=("voted_up", "mean"),
    ).reset_index()
    playtime_summary["recommend_pct"] = playtime_summary["recommend_pct"] * 100
    fig = px.bar(playtime_summary, x="playtime_bucket", y="recommend_pct",
                 text_auto=".1f", labels={"recommend_pct": "Recommend %", "playtime_bucket": "Playtime"},
                 title="Does playtime predict recommendation?",
                 color_discrete_sequence=RED_PALETTE)
    st.plotly_chart(style_fig(fig), use_container_width=True)
 
    # --- Sentiment vs recommendation ---
    st.subheader("Sentiment vs. actual recommendation")
    st.caption("Where does review text tone disagree with the explicit thumbs-up/down?")
    disagreement = df.groupby(["sentiment_label", "voted_up"]).size().reset_index(name="count")
    fig = px.bar(disagreement, x="sentiment_label", y="count", color="voted_up", barmode="group",
                 labels={"sentiment_label": "Sentiment", "count": "Reviews", "voted_up": "Recommended"},
                 title="Recommendation outcome by sentiment label",
                 color_discrete_sequence=RED_PALETTE)
    st.plotly_chart(style_fig(fig), use_container_width=True)
 
    st.divider()
 
    # --- Model evaluation ---
    st.subheader("🤖 Recommendation model evaluation")
    eval_data = load_model_eval()
    if eval_data is None:
        st.info(
            "No saved evaluation found yet. Train the model (after this update, "
            "it saves its evaluation results too):\n\n"
            "```\npython analysis/recommendation_model.py\n```"
        )
    else:
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Random Forest ROC-AUC", f"{eval_data['random_forest']['roc_auc']:.3f}")
        mcol2.metric("Logistic Regression ROC-AUC", f"{eval_data['logreg']['roc_auc']:.3f}")
        if "xgboost" in eval_data:
            mcol3.metric("XGBoost ROC-AUC", f"{eval_data['xgboost']['roc_auc']:.3f}")
        mcol4.metric("Base rate (recommended)", f"{eval_data['base_rate']:.1%}")
 
        if "temporal_split" in eval_data:
            temporal_bits = ", ".join(
                f"{name.replace('_', ' ')} {res['roc_auc']:.3f}"
                for name, res in eval_data["temporal_split"].items()
            )
            st.caption(
                f"Temporal validation (train on earlier reviews, test on the most "
                f"recent 25%): {temporal_bits}. This answers the stricter question "
                f"of whether the model generalizes forward in time, not just "
                f"across a shuffled sample."
            )
 
        ablation = load_model_eval("_minus_weighted_vote_score")
        if ablation:
            st.caption(
                f"Ablation check: removing `weighted_vote_score` drops RF ROC-AUC to "
                f"**{ablation['random_forest']['roc_auc']:.3f}**, confirming sentiment "
                f"carries most of the signal."
            )
 
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            cm = eval_data["random_forest"]["confusion_matrix"]
            fig = px.imshow(
                cm, text_auto=True,
                x=["Predicted: not rec.", "Predicted: rec."],
                y=["Actual: not rec.", "Actual: rec."],
                color_continuous_scale=["#1A1A1A", "#E63946"],
                title="Confusion matrix (Random Forest, test set)",
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with ecol2:
            roc = eval_data["random_forest"]["roc_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                     name="Random Forest", line=dict(color="#E63946")))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Chance", line=dict(color="#6C757D", dash="dot")))
            fig.update_layout(title="ROC curve (Random Forest, test set)",
                              xaxis_title="False positive rate", yaxis_title="True positive rate")
            st.plotly_chart(style_fig(fig), use_container_width=True)
 
        importances = load_model_importances()
        if importances is not None:
            fig = px.bar(importances.head(10)[::-1], orientation="h",
                         labels={"value": "Importance", "index": ""},
                         title="Top 10 features (Random Forest)",
                         color_discrete_sequence=RED_PALETTE)
            st.plotly_chart(style_fig(fig), use_container_width=True)
 
    st.divider()
 
    # --- SQL explorer ---
    st.subheader("🔎 SQL explorer")
    st.caption(
        "Run the project's analytics queries directly. Queries run against the "
        "local SQLite database (the BigQuery versions live in "
        "`sql/bigquery_analytics_queries.sql` and use a different dialect)."
    )
    if source == "BigQuery":
        st.info(
            "The SQL explorer currently runs against the local SQLite database "
            "only, since the two backends use different SQL dialects. Switch the "
            "backend to Local (SQLite) to use it, or run the BigQuery versions "
            "via `python analysis/bigquery_sql_report.py`."
        )
    else:
        blocks = load_sql_query_blocks()
        if not blocks:
            st.info("Could not find `sql/analytics_queries.sql`.")
        else:
            titles = [title for title, _ in blocks]
            chosen = st.selectbox("Pick a query", titles)
            chosen_sql = dict(blocks)[chosen]
            rendered = chosen_sql.replace("{{APPID}}", str(appid))
            if "{{PIVOT_DATE}}" in rendered:
                if pivot_date:
                    rendered = rendered.replace("{{PIVOT_DATE}}", str(pivot_date))
                else:
                    st.warning("This query needs a pivot date. Set one in the sidebar.")
                    rendered = None
 
            if rendered:
                with st.expander("View SQL"):
                    st.code(rendered, language="sql")
                if st.button("▶ Run query"):
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        result = pd.read_sql_query(rendered, conn)
                        conn.close()
                        st.dataframe(result, use_container_width=True)
                        st.caption(f"{len(result):,} rows returned.")
                    except Exception as e:
                        st.error(f"Query failed: {e}")
 
 
if __name__ == "__main__":
    main()
 