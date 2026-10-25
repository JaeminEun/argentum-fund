from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src import __version__
from src.universe.config import load_config


DEFAULT_RUNS_DIR = "data/evaluation/runs"

# Pipeline outputs worth keeping a copy of for each evaluation run.
# Each entry maps a short name to the file path and the columns used
# when comparing runs.
DEFAULT_SNAPSHOT_FILES = {
    "price_scores": {
        "path": "data/scores/latest_price_scores.csv",
        "score_column": "risk_adjusted_timing_score",
        "label_column": "primary_signal",
        "bucket_column": "score_bucket",
    },
    "fundamental_scores": {
        "path": "data/scores/latest_fundamental_scores.csv",
        "score_column": "risk_adjusted_fundamental_score",
        "label_column": "fundamental_signal",
        "bucket_column": "fundamental_bucket",
    },
    "composite_scores": {
        "path": "data/scores/latest_composite_scores.csv",
        "score_column": "composite_research_score",
        "label_column": "composite_signal",
        "bucket_column": "composite_bucket",
    },
    "deployment_plan": {
        "path": "data/deployment/latest_deployment_plan.csv",
        "score_column": "suggested_amount",
        "label_column": "deployment_action",
        "bucket_column": None,
    },
}


def load_evaluation_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load the evaluation section from the project configuration file.

    The evaluation section is optional; defaults are used when missing.
    """
    config = load_config(config_path)

    return config.get("evaluation", {})


def get_runs_dir(evaluation_config: Dict[str, Any]) -> Path:
    """
    Return the directory where evaluation runs are stored.
    """
    return Path(evaluation_config.get("output_dir", DEFAULT_RUNS_DIR))


def get_git_state() -> Dict[str, Any]:
    """
    Capture the current git commit, branch, and whether tracked source
    files have uncommitted changes.

    This ties every evaluation run to the exact code that produced it.
    """

    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = run_git("status", "--porcelain", "--", "src", "config")

    return {
        "git_commit": run_git("rev-parse", "--short", "HEAD"),
        "git_branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_code_changes": bool(status),
    }


def create_run_dir(
    evaluation_config: Dict[str, Any],
    label: str,
    overwrite: bool = False,
) -> Path:
    """
    Create the folder for a labelled evaluation run.
    """
    run_dir = get_runs_dir(evaluation_config) / label

    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Evaluation run '{label}' already exists at {run_dir}. "
            "Choose a new label or pass --overwrite."
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def write_run_metadata(
    run_dir: Path,
    label: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Write run metadata so results can be traced back to code and settings.
    """
    metadata = {
        "label": label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_version": __version__,
        **get_git_state(),
    }

    if extra:
        metadata.update(extra)

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    return metadata


def get_snapshot_files(evaluation_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Return snapshot file settings, allowing config to override paths.
    """
    snapshot_files = {
        name: dict(settings) for name, settings in DEFAULT_SNAPSHOT_FILES.items()
    }

    for name, settings in evaluation_config.get("snapshot_files", {}).items():
        snapshot_files.setdefault(name, {}).update(settings)

    return snapshot_files


def snapshot_pipeline_outputs(
    run_dir: Path,
    evaluation_config: Dict[str, Any],
) -> list[str]:
    """
    Copy current pipeline outputs into the run folder.

    Returns the names of files that were captured.
    """
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    captured = []

    for name, settings in get_snapshot_files(evaluation_config).items():
        source = Path(settings["path"])

        if not source.exists():
            print(f"  Warning: {source} not found; skipping {name} snapshot.")
            continue

        shutil.copy2(source, outputs_dir / f"{name}.csv")
        captured.append(name)

    return captured


def load_run_output(run_dir: Path, name: str) -> pd.DataFrame | None:
    """
    Load a snapshotted output file from a run, if present.
    """
    path = run_dir / "outputs" / f"{name}.csv"

    if not path.exists():
        return None

    frame = pd.read_csv(path)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()

    return frame


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    """
    Spearman rank correlation between two aligned series.
    """
    valid = left.notna() & right.notna()

    if valid.sum() < 3:
        return np.nan

    return left[valid].rank().corr(right[valid].rank())


def compare_label_distribution(
    before: pd.DataFrame,
    after: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """
    Count how many tickers fall into each label in two runs.
    """
    before_counts = before[label_column].value_counts(dropna=False).rename("before")
    after_counts = after[label_column].value_counts(dropna=False).rename("after")

    distribution = pd.concat([before_counts, after_counts], axis=1).fillna(0)
    distribution = distribution.astype(int)
    distribution["change"] = distribution["after"] - distribution["before"]
    distribution.index.name = label_column

    return distribution.reset_index()


def compare_top_membership(
    before: pd.DataFrame,
    after: pd.DataFrame,
    score_column: str,
    top_n: int,
) -> pd.DataFrame:
    """
    List tickers that entered or left the top N by score.
    """
    before_top = set(before.nlargest(top_n, score_column)["ticker"])
    after_top = set(after.nlargest(top_n, score_column)["ticker"])

    records = [
        {"ticker": ticker, "change": "entered_top"}
        for ticker in sorted(after_top - before_top)
    ] + [
        {"ticker": ticker, "change": "left_top"}
        for ticker in sorted(before_top - after_top)
    ]

    return pd.DataFrame(records, columns=["ticker", "change"])


def compare_output(
    before: pd.DataFrame,
    after: pd.DataFrame,
    settings: Dict[str, Any],
    top_n: int,
) -> Dict[str, Any]:
    """
    Compare one pipeline output between two runs.
    """
    score_column = settings.get("score_column")
    label_column = settings.get("label_column")
    bucket_column = settings.get("bucket_column")

    merged = before.merge(
        after,
        on="ticker",
        how="outer",
        suffixes=("_before", "_after"),
        indicator=True,
    )

    result: Dict[str, Any] = {
        "tickers_before": len(before),
        "tickers_after": len(after),
        "tickers_added": int((merged["_merge"] == "right_only").sum()),
        "tickers_removed": int((merged["_merge"] == "left_only").sum()),
    }

    if score_column and score_column in before.columns and score_column in after.columns:
        result["score_rank_correlation"] = rank_correlation(
            pd.to_numeric(merged[f"{score_column}_before"], errors="coerce"),
            pd.to_numeric(merged[f"{score_column}_after"], errors="coerce"),
        )

    if bucket_column and bucket_column in before.columns and bucket_column in after.columns:
        both = merged[merged["_merge"] == "both"]
        result["bucket_changes"] = int(
            (
                both[f"{bucket_column}_before"].astype(str)
                != both[f"{bucket_column}_after"].astype(str)
            ).sum()
        )

    tables: Dict[str, pd.DataFrame] = {}

    if label_column and label_column in before.columns and label_column in after.columns:
        tables["label_distribution"] = compare_label_distribution(
            before,
            after,
            label_column,
        )

    if score_column and score_column in before.columns and score_column in after.columns:
        before_scored = before.assign(
            **{score_column: pd.to_numeric(before[score_column], errors="coerce")}
        )
        after_scored = after.assign(
            **{score_column: pd.to_numeric(after[score_column], errors="coerce")}
        )
        tables["top_membership"] = compare_top_membership(
            before_scored,
            after_scored,
            score_column,
            top_n,
        )

    result["tables"] = tables

    return result


def compare_runs(
    before_label: str,
    after_label: str,
    config_path: str | Path,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Compare snapshotted outputs and evaluation summaries between two runs.

    Writes comparison tables to the 'after' run folder and returns
    a one-row-per-output summary.
    """
    evaluation_config = load_evaluation_config(config_path)
    runs_dir = get_runs_dir(evaluation_config)

    before_dir = runs_dir / before_label
    after_dir = runs_dir / after_label

    for run_dir in (before_dir, after_dir):
        if not run_dir.exists():
            raise FileNotFoundError(f"Evaluation run not found: {run_dir}")

    comparison_dir = after_dir / f"comparison_vs_{before_label}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for name, settings in get_snapshot_files(evaluation_config).items():
        before = load_run_output(before_dir, name)
        after = load_run_output(after_dir, name)

        if before is None or after is None:
            continue

        result = compare_output(before, after, settings, top_n=top_n)

        for table_name, table in result.pop("tables").items():
            table.to_csv(comparison_dir / f"{name}_{table_name}.csv", index=False)

        summary_records.append({"output": name, **result})

    summary = pd.DataFrame(summary_records)
    summary.to_csv(comparison_dir / "output_changes.csv", index=False)

    # Side-by-side backtest and data health summaries, when both runs have them.
    for file_name, key_columns in (
        ("price_backtest_summary.csv", ["period", "horizon"]),
        ("data_health_summary.csv", ["area", "check"]),
    ):
        before_path = before_dir / file_name
        after_path = after_dir / file_name

        if not (before_path.exists() and after_path.exists()):
            continue

        side_by_side = pd.read_csv(before_path).merge(
            pd.read_csv(after_path),
            on=key_columns,
            how="outer",
            suffixes=(f"_{before_label}", f"_{after_label}"),
        )
        side_by_side.to_csv(comparison_dir / file_name, index=False)

    print(f"Saved run comparison to {comparison_dir}")

    return summary
