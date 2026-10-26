from __future__ import annotations

import argparse

import pandas as pd

from src.evaluation.data_health import run_data_health, save_data_health
from src.evaluation.price_backtest import run_price_backtest, save_price_backtest
from src.evaluation.runs import (
    compare_runs,
    create_run_dir,
    load_evaluation_config,
    snapshot_pipeline_outputs,
    write_run_metadata,
)


DEFAULT_CONFIG_PATH = "config/universe_config.yaml"


def print_table(title: str, table: pd.DataFrame) -> None:
    """
    Print a compact table to the console.
    """
    print(f"\n{title}")

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 160,
        "display.float_format", "{:.3f}".format,
    ):
        print(table.to_string(index=False))


def run_evaluation(
    config_path: str,
    label: str,
    include_holdout: bool = False,
    skip_backtest: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Create a labelled evaluation run: snapshot outputs, check data health,
    and backtest the price score.
    """
    evaluation_config = load_evaluation_config(config_path)
    run_dir = create_run_dir(evaluation_config, label, overwrite=overwrite)

    print(f"Evaluation run '{label}' -> {run_dir}")

    captured = snapshot_pipeline_outputs(run_dir, evaluation_config)

    write_run_metadata(
        run_dir,
        label,
        extra={
            "config_path": config_path,
            "snapshotted_outputs": captured,
            "include_holdout": include_holdout,
            "backtest_run": not skip_backtest,
        },
    )

    health_summary, health_issues = run_data_health(config_path)
    save_data_health(health_summary, health_issues, run_dir)
    print_table("Data health", health_summary)

    if skip_backtest:
        return

    backtest_summary, backtest_by_date = run_price_backtest(
        config_path,
        include_holdout=include_holdout,
    )
    save_price_backtest(backtest_summary, backtest_by_date, run_dir)

    headline_columns = [
        "period",
        "horizon",
        "dates",
        "mean_ic",
        "ic_t_stat",
        "pct_dates_ic_positive",
        "mean_top_minus_weak",
        "mean_top_hit_rate",
    ]
    print_table(
        "Price score backtest (returns are vs equal-weighted universe)",
        backtest_summary[
            [c for c in headline_columns if c in backtest_summary.columns]
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track whether model changes improve results.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Create a labelled evaluation run from current outputs.",
    )
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Also evaluate the sealed holdout period. Use sparingly.",
    )
    run_parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Only snapshot outputs and check data health.",
    )
    run_parser.add_argument("--overwrite", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two labelled evaluation runs.",
    )
    compare_parser.add_argument("before")
    compare_parser.add_argument("after")
    compare_parser.add_argument("--top-n", type=int, default=25)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "run":
        run_evaluation(
            config_path=args.config,
            label=args.label,
            include_holdout=args.include_holdout,
            skip_backtest=args.skip_backtest,
            overwrite=args.overwrite,
        )
    elif args.command == "compare":
        summary = compare_runs(
            before_label=args.before,
            after_label=args.after,
            config_path=args.config,
            top_n=args.top_n,
        )
        print_table(f"Changes from {args.before} to {args.after}", summary)


if __name__ == "__main__":
    main()
