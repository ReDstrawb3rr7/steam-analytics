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
from google.cloud import bigquery

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "steam_analytics.db")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
SQLITE_SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "analytics_queries.sql")
BIGQUERY_SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "bigquery_analytics_queries.sql")

SENTIMENT_SIGN = {"positive": 1, "neutral": 0, "negative": -1}

pio.templates.default = "plotly_dark"
RED_PALETTE = ["#E63946", "#F1A208", "#6C757D", "#A4161A", "#D9BF77"]
CHART_LAYOUT = dict(
    plot_bgcolor="#0D0D0D",
    paper_bgcolor="#0D0D0D",
    font_color="#F5F5F5",
)

st.set_page_config(page_title="Steam Community Analytics", page_icon="🎮", layout="wide")

# Card styling for metrics and bordered containers: a subtle background,
# rounded corners, and a red accent border matching the dashboard's
# color palette, so KPI numbers read as distinct cards rather than bare
# text sitting on the page background.
st.markdown("""
<style>
[data-testid="stMetric"] {
    background-color: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-left: 4px solid #E63946;
    border-radius: 8px;
    padding: 14px 16px 10px 16px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
    opacity: 0.85;
}
[data-testid="stExpander"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

def get_bigquery_client(bq_project: str | None):
    from google.cloud import bigquery

    try:
        has_secret = "gcp_service_account" in st.secrets
    except Exception:
        has_secret = False  # no secrets.toml at all, e.g. plain local dev

    if has_secret:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"])
        )
        return bigquery.Client(project=bq_project or credentials.project_id, credentials=credentials)

    return bigquery.Client(project=bq_project or None)


def run_query(query: str, source: str, bq_project: str | None, params: tuple = ()) -> pd.DataFrame:
    if source == "BigQuery":
        client = get_bigquery_client(bq_project)
        job_config = bigquery.QueryJobConfig(
            default_dataset=f"{client.project}.steam_analytics",
        )
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


def load_sql_query_blocks(path: str = SQLITE_SQL_PATH):
    """Parse a .sql file into (title, sql) pairs for the SQL explorer.
    Same parsing approach as analysis/sql_report.py, plus: strips
    decorative banner lines (=====) that would otherwise pollute the
    first query's title if the file header sits adjacent to it. Works
    for either dialect's file since both use the same comment/query
    layout."""
    if not os.path.exists(path):
        return []
    text = open(path, encoding="utf-8").read()
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
                break
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

    with st.sidebar:
        st.header("Data source")
        default_source_index = 0 if os.path.exists(DB_PATH) else 1
        source = st.radio("Backend", ["Local (SQLite)", "BigQuery"], index=default_source_index)
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
        appids_in_order = games["appid"].tolist()
        if "selected_appid" in st.session_state and st.session_state.selected_appid not in appids_in_order:
            del st.session_state["selected_appid"]

        labels_by_appid = {
            row["appid"]: f"{row['name']} ({row['appid']}) - {row['review_count']:,} reviews"
            for _, row in games.iterrows()
        }
        appid = st.selectbox(
            "Game",
            options=appids_in_order,
            format_func=lambda a: labels_by_appid[a],
            key="selected_appid",
        )
        game_name = games.loc[games["appid"] == appid, "name"].iloc[0]
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

    st.subheader("📊 Overview")

    media = load_game_media(appid)
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

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total reviews", f"{len(df):,}")
        col2.metric("Recommended", f"{recommend_pct:.1%}")
        col3.metric("Avg. sentiment", f"{avg_sent:+.2f}")
        st.caption(f"📅 Date range: {start_date} → {end_date}")

    screenshots = media["screenshots"]
    if screenshots:
        with st.expander("📷 Game screenshots (from Steam)", expanded=True):
            shot_cols = st.columns(len(screenshots))
            for c, url in zip(shot_cols, screenshots):
                c.image(url, use_container_width=True)

    st.divider()

    st.subheader("📈 Daily activity")
    daily = df.groupby("day").agg(
        review_count=("recommendation_id", "count"),
        avg_sentiment=("sentiment_signed", "mean"),
        recommend_rate=("voted_up", "mean"),
    ).reset_index()
    daily["day"] = pd.to_datetime(daily["day"])
    daily["rolling_7d_volume"] = daily["review_count"].rolling(7, min_periods=1).mean()

    with st.container(border=True):
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
        st.subheader(f"📅 Before vs. after {pivot_date}")
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

        pcol1, pcol2 = st.columns(2)
        for col, period, label in [(pcol1, "pre-pivot", "⬅️ Pre-pivot"), (pcol2, "post-pivot", "➡️ Post-pivot")]:
            with col:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    if period in summary.index:
                        row = summary.loc[period]
                        st.metric("Reviews", f"{int(row['review_count']):,}")
                        st.metric("Recommend %", f"{row['recommend_pct']:.1f}%")
                        st.metric("Avg. sentiment", f"{row['avg_sentiment']:+.3f}")
                    else:
                        st.caption("No reviews in this period.")

    st.divider()

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

    st.subheader("⏱️ Recommendation rate by playtime")
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

    st.subheader("⚖️ Sentiment vs. actual recommendation")
    st.caption("Where does review text tone disagree with the explicit thumbs-up/down?")
    disagreement = df.groupby(["sentiment_label", "voted_up"]).size().reset_index(name="count")
    fig = px.bar(disagreement, x="sentiment_label", y="count", color="voted_up", barmode="group",
                 labels={"sentiment_label": "Sentiment", "count": "Reviews", "voted_up": "Recommended"},
                 title="Recommendation outcome by sentiment label",
                 color_discrete_sequence=RED_PALETTE)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.divider()

    st.subheader("🤖 Recommendation model evaluation")
    eval_data = load_model_eval()
    if eval_data is None:
        st.info(
            "No saved evaluation found yet. Train the model (after this update, "
            "it saves its evaluation results too):\n\n"
            "```\npython analysis/recommendation_model.py\n```"
        )
    else:
        with st.container(border=True):
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

    st.subheader("🔎 SQL explorer")
    st.caption(
        "Run the project's analytics queries directly against whichever "
        "backend is selected above."
    )

    if source == "BigQuery":
        blocks = load_sql_query_blocks(BIGQUERY_SQL_PATH)
        if not blocks:
            st.info("Could not find `sql/bigquery_analytics_queries.sql`.")
        else:
            titles = [title for title, _ in blocks]
            chosen = st.selectbox("Pick a query", titles, key="bq_sql_choice")
            chosen_sql = dict(blocks)[chosen]
            needs_pivot = "@pivot_date" in chosen_sql

            if needs_pivot and not pivot_date:
                st.warning("This query needs a pivot date. Set one in the sidebar.")
            else:
                with st.expander("View SQL"):
                    st.code(chosen_sql, language="sql")
                if st.button("▶ Run query", key="bq_sql_run"):
                    try:
                        client = get_bigquery_client(bq_project)
                        query_params = [bigquery.ScalarQueryParameter("appid", "INT64", appid)]
                        if needs_pivot:
                            query_params.append(
                                bigquery.ScalarQueryParameter("pivot_date", "DATE", pivot_date)
                            )
                        job_config = bigquery.QueryJobConfig(
                            default_dataset=f"{client.project}.steam_analytics",
                            query_parameters=query_params,
                        )
                        result = client.query(chosen_sql, job_config=job_config).to_dataframe()
                        st.dataframe(result, use_container_width=True)
                        st.caption(f"{len(result):,} rows returned.")
                    except Exception as e:
                        st.error(f"Query failed: {e}")
    else:
        blocks = load_sql_query_blocks(SQLITE_SQL_PATH)
        if not blocks:
            st.info("Could not find `sql/analytics_queries.sql`.")
        else:
            titles = [title for title, _ in blocks]
            chosen = st.selectbox("Pick a query", titles, key="sqlite_sql_choice")
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
                if st.button("▶ Run query", key="sqlite_sql_run"):
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