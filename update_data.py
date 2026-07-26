"""
One-command incremental data update.
Chains the pipeline steps needed to bring the dataset up to date for a game, doing only new work at each stage:

    1. Ingest    -- pulls latest reviews; already-stored ones are skipped
                    (INSERT OR IGNORE on recommendation_id)
    2. Sentiment -- scores only reviews that don't have a score yet
    3. Features  -- rebuilds the feature matrix (fast, full rebuild)
    4. Model     -- retrains on the updated features (optional, --skip-model)
    5. BigQuery  -- re-syncs to BigQuery (optional, --sync-bigquery)

After running this, hit "Refresh data" in the dashboard sidebar to see the new data.

ARGUMENTS
    --appid <number>        REQUIRED. The game's Steam app ID, found in its
                            store URL: store.steampowered.com/app/<appid>
                            e.g. 1623730 = Palworld, 2767030 = Marvel Rivals

    --max-reviews <number>  How many reviews to pull this run (default 5000).
                            Already-stored reviews are skipped automatically,
                            so for a regular top-up a few thousand is enough.
                            For a first-time ingest of a new game, use a
                            bigger number (e.g. 30000).

    --filter <mode>         Which Steam review ordering to pull (default:
                            recent).
                              recent  = newest first (best for time-series)
                              updated = by last-edit date
                              all     = Steam's relevance ranking. NEEDED for
                                        some games (e.g. Marvel Rivals) where
                                        the date-ordered filters are broken
                                        and silently return zero reviews.
                                        Caveat: not chronological, so the
                                        sample skews toward popular reviews.

    --skip-model            Skip retraining the recommendation model. Useful
                            when you just want fresh data in the dashboard
                            quickly.

    --sync-bigquery         Also re-sync the database to BigQuery at the end.

    --project <id>          Your GCP project ID (only used with
                            --sync-bigquery), e.g. steam-analytics-503010

EXAMPLES
    # Regular top-up for Palworld, retrain model
    python update_data.py --appid 1623730

    # First-time ingest of Marvel Rivals (note: needs --filter all)
    python update_data.py --appid 2767030 --max-reviews 30000 --filter all

    # Quick data refresh only, no model retrain
    python update_data.py --appid 1623730 --skip-model

    # Full update including BigQuery sync
    python update_data.py --appid 1623730 --sync-bigquery --project steam-analytics-503010
"""

import argparse
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nStep '{name}' failed (exit code {result.returncode}). Stopping here.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="One-command incremental data update. See the docstring at "
                    "the top of this file for full argument explanations and examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python update_data.py --appid 1623730\n"
               "  python update_data.py --appid 2767030 --max-reviews 30000 --filter all\n"
               "  python update_data.py --appid 1623730 --sync-bigquery --project steam-analytics-503010",
    )
    parser.add_argument("--appid", type=int, required=True,
                        help="Steam app ID from the store URL (e.g. 1623730 for Palworld)")
    parser.add_argument("--max-reviews", type=int, default=5000,
                        help="Reviews to pull this run (default 5000; duplicates are "
                             "skipped, so use ~30000 for a first-time ingest of a new game)")
    parser.add_argument("--filter", choices=["recent", "updated", "all"], default="recent",
                        help="Steam review ordering. Use 'all' for games where the "
                             "date-ordered filters silently return zero (e.g. Marvel Rivals)")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip retraining the recommendation model")
    parser.add_argument("--sync-bigquery", action="store_true",
                        help="Also re-sync the database to BigQuery afterwards")
    parser.add_argument("--project", default=None,
                        help="GCP project ID (only used with --sync-bigquery)")
    args = parser.parse_args()

    py = sys.executable

    steps = [
        ("Ingest new reviews",
         [py, "ingestion/steam_ingest.py", "--appid", str(args.appid),
          "--max-reviews", str(args.max_reviews), "--filter", args.filter]),
        ("Score new sentiment",
         [py, "analysis/sentiment.py"]),
        ("Rebuild features",
         [py, "analysis/features.py"]),
    ]

    if not args.skip_model:
        steps.append(("Retrain recommendation model",
                      [py, "analysis/recommendation_model.py"]))
        steps.append(("Retrain ablation model (for the dashboard's comparison)",
                      [py, "analysis/recommendation_model.py", "--exclude", "weighted_vote_score"]))

    if args.sync_bigquery:
        bq_cmd = [py, "migration/migrate_to_bigquery.py"]
        if args.project:
            bq_cmd += ["--project", args.project]
        steps.append(("Sync to BigQuery", bq_cmd))

    for name, cmd in steps:
        if not run_step(name, cmd):
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("UPDATE COMPLETE")
    print("Open the dashboard and hit 'Refresh data' in the sidebar to see the new data.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()