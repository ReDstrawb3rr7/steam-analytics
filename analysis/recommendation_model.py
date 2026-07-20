import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "review_features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "recommendation_model.joblib")

FEATURE_COLS = [
    "review_length",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "log_playtime_at_review",
    "num_games_owned",
    "reviewer_num_reviews",
    "sentiment_signed",
    "sentiment_negative",
    "sentiment_neutral",
    "sentiment_positive",
    "day_of_week",
    "hour_of_day",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated feature names to drop, e.g. --exclude weighted_vote_score,votes_up",
    )
    args = parser.parse_args()
    excluded = {c.strip() for c in args.exclude.split(",") if c.strip()}

    df = pd.read_csv(FEATURES_PATH)
    available_cols = [c for c in FEATURE_COLS if c in df.columns and c not in excluded]
    missing = set(FEATURE_COLS) - set(available_cols) - excluded
    if missing:
        print(f"Note: skipping columns not present in this run: {missing}")
    if excluded:
        print(f"Excluding features for this run: {excluded}")
 
    X = df[available_cols]
    y = df["voted_up"]
 
    print(f"Dataset: {len(df)} reviews, {y.mean():.1%} recommended")
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    logreg_probs = logreg.predict_proba(X_test_scaled)[:, 1]
 
    print("\nLogistic Regression (baseline)")
    print(classification_report(y_test, logreg.predict(X_test_scaled)))
    print(f"ROC-AUC: {roc_auc_score(y_test, logreg_probs):.3f}")
 
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
 
    print("\nRandom Forest")
    print(classification_report(y_test, rf.predict(X_test)))
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.3f}")
 
    importances = pd.Series(rf.feature_importances_, index=available_cols).sort_values(ascending=False)
    print("\nFeature importances (Random Forest):")
    print(importances.to_string())

    out_path = MODEL_PATH
    if excluded:
        suffix = "_minus_" + "_".join(sorted(excluded))
        out_path = out_path.replace(".joblib", f"{suffix}.joblib")
 
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": rf, "scaler": scaler, "features": available_cols}, out_path)
    print(f"\nSaved model to {out_path}")
 
if __name__ == "__main__":
    main()