import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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

def build_models(y_train):
    """The model lineup. XGBoost's scale_pos_weight mirrors the
    class_weight='balanced' handling of the sklearn models for the
    imbalanced recommended/not-recommended label."""
    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            True,   # needs scaling
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            False,
        ),
    }
    if HAS_XGBOOST:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        models["xgboost"] = (
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                scale_pos_weight=neg / max(pos, 1),
                eval_metric="logloss", random_state=42, n_jobs=-1,
            ),
            False,
        )
    return models
 
 
def evaluate_split(X_train, X_test, y_train, y_test, split_name: str):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    results = {}
    for name, (model, needs_scaling) in build_models(y_train).items():
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test
        model.fit(Xtr, y_train)
        probs = model.predict_proba(Xte)[:, 1]
        preds = model.predict(Xte)
        auc = roc_auc_score(y_test, probs)
 
        print(f"\n=== {name} ({split_name} split) ===")
        print(classification_report(y_test, preds))
        print(f"ROC-AUC: {auc:.3f}")
 
        results[name] = {
            "model": model,
            "auc": auc,
            "probs": probs,
            "preds": preds,
        }
 
    results["_scaler"] = scaler
    return results
 
 
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
    if not HAS_XGBOOST:
        print("Note: xgboost not installed, skipping that model")
 
    X = df[available_cols]
    y = df["voted_up"]
    print(f"Dataset: {len(df)} reviews, {y.mean():.1%} recommended")
 
    # ---- Random split: "do these features predict the label?" ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    random_results = evaluate_split(X_train, X_test, y_train, y_test, "random")
 
    # ---- Temporal split: "could this predict FUTURE reviews?" ----
    # Sort by review creation time, train on the earliest 75%, test on
    # the most recent 25%. No shuffling, no stratification: the future
    # arrives with whatever class balance it has.
    temporal_results = None
    if "timestamp_created" in df.columns:
        df_sorted = df.sort_values("timestamp_created")
        cut = int(len(df_sorted) * 0.75)
        X_sorted = df_sorted[available_cols]
        y_sorted = df_sorted["voted_up"]
        temporal_results = evaluate_split(
            X_sorted.iloc[:cut], X_sorted.iloc[cut:],
            y_sorted.iloc[:cut], y_sorted.iloc[cut:],
            "temporal",
        )
 
        print("\n=== Random vs. temporal split (ROC-AUC) ===")
        for name in random_results:
            if name.startswith("_"):
                continue
            r = random_results[name]["auc"]
            t = temporal_results[name]["auc"]
            print(f"  {name:>22}: random {r:.3f} | temporal {t:.3f} | gap {r - t:+.3f}")
        print("A small gap means the model generalizes forward in time, "
              "not just across a shuffled sample.")
    else:
        print("\nNote: timestamp_created not in the feature file, skipping "
              "the temporal split. Re-run analysis/features.py to include it.")
 
    # ---- Persist the best random-split model + evaluation artifacts ----
    # Random forest stays the saved/deployed model for dashboard
    # consistency (feature importances etc.); XGBoost's numbers are
    # reported alongside for comparison.
    rf = random_results["random_forest"]["model"]
    rf_probs = random_results["random_forest"]["probs"]
    rf_preds = random_results["random_forest"]["preds"]
    scaler = random_results["_scaler"]
 
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
 
    cm = confusion_matrix(y_test, rf_preds)
    fpr, tpr, _ = roc_curve(y_test, rf_probs)
    eval_artifacts = {
        "n_test": int(len(y_test)),
        "base_rate": float(y.mean()),
        "excluded_features": sorted(excluded),
        "logreg": {
            "roc_auc": float(random_results["logistic_regression"]["auc"]),
        },
        "random_forest": {
            "roc_auc": float(random_results["random_forest"]["auc"]),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_test, rf_preds, output_dict=True),
            "roc_curve": {
                "fpr": [float(x) for x in fpr[:: max(1, len(fpr) // 200)]],
                "tpr": [float(x) for x in tpr[:: max(1, len(tpr) // 200)]],
            },
        },
        "feature_importances": {k: float(v) for k, v in importances.items()},
    }
    if HAS_XGBOOST:
        eval_artifacts["xgboost"] = {
            "roc_auc": float(random_results["xgboost"]["auc"]),
        }
    if temporal_results:
        eval_artifacts["temporal_split"] = {
            name: {"roc_auc": float(res["auc"])}
            for name, res in temporal_results.items()
            if not name.startswith("_")
        }
 
    eval_path = out_path.replace(".joblib", "_eval.json")
    with open(eval_path, "w") as f:
        json.dump(eval_artifacts, f, indent=2)
    print(f"Saved evaluation artifacts to {eval_path}")
 
 
if __name__ == "__main__":
    main()