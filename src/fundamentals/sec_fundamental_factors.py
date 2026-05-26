from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.universe.config import load_config


def load_sec_fundamental_factors_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load project config and validate the sec_fundamental_factors block.
    """
    config = load_config(config_path)

    if "sec_fundamental_factors" not in config:
        raise ValueError("Missing 'sec_fundamental_factors' section in config file.")

    if not config["sec_fundamental_factors"].get("enabled", False):
        raise ValueError("sec_fundamental_factors is disabled in config file.")

    return config


def load_accounting_concepts(input_path: str | Path) -> pd.DataFrame:
    """
    Load extracted SEC accounting concepts.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Accounting concepts file not found: {input_path}. "
            "Run sec_accounting_concepts first."
        )

    concepts = pd.read_csv(input_path, dtype={"cik_padded": str})

    return concepts


def validate_accounting_concepts(concepts: pd.DataFrame) -> None:
    """
    Validate required columns for fundamental factor calculation.
    """
    required_columns = {
        "ticker",
        "metric_name",
        "val",
        "filed",
        "end",
        "form",
    }

    missing_columns = required_columns - set(concepts.columns)

    if missing_columns:
        raise ValueError(
            f"Accounting concepts file missing required columns: {missing_columns}"
        )

    if concepts.empty:
        raise ValueError("Accounting concepts file is empty.")


def prepare_accounting_concepts(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Clean SEC accounting concept records before pivoting.
    """
    concepts = concepts.copy()

    concepts["ticker"] = concepts["ticker"].astype(str).str.upper().str.strip()
    concepts["metric_name"] = concepts["metric_name"].astype(str).str.lower().str.strip()
    concepts["val"] = pd.to_numeric(concepts["val"], errors="coerce")
    concepts["filed"] = pd.to_datetime(concepts["filed"], errors="coerce")
    concepts["end"] = pd.to_datetime(concepts["end"], errors="coerce")

    concepts = concepts.dropna(subset=["ticker", "metric_name", "val"])

    return concepts


def pivot_latest_accounting_values(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot accounting concepts from long format to one row per ticker.

    Assumes sec_accounting_concepts already selected the latest record
    for each ticker and metric.
    """
    values = concepts.pivot_table(
        index="ticker",
        columns="metric_name",
        values="val",
        aggfunc="last",
    )

    values = values.reset_index()
    values.columns.name = None

    return values


def build_metric_metadata(concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Build metadata columns showing concept names, filing dates, and forms.

    This makes the factor output easier to audit.
    """
    metadata_records = []

    for _, row in concepts.iterrows():
        ticker = row["ticker"]
        metric = row["metric_name"]

        metadata_records.append(
            {
                "ticker": ticker,
                f"{metric}_concept_name": row.get("concept_name", pd.NA),
                f"{metric}_form": row.get("form", pd.NA),
                f"{metric}_fy": row.get("fy", pd.NA),
                f"{metric}_fp": row.get("fp", pd.NA),
                f"{metric}_filed": row.get("filed", pd.NA),
                f"{metric}_period_end": row.get("end", pd.NA),
            }
        )

    if not metadata_records:
        return pd.DataFrame()

    metadata_long = pd.DataFrame(metadata_records)

    # Collapse one row per ticker by taking first non-null value per column.
    metadata = metadata_long.groupby("ticker", as_index=False).first()

    return metadata


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divide while avoiding infinite values from zero denominators.
    """
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def add_accounting_factor_columns(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate accounting-based fundamental factors.
    """
    factors = factors.copy()

    # Ensure columns exist so formulas do not fail.
    expected_numeric_columns = [
        "revenue",
        "net_income",
        "assets",
        "liabilities",
        "stockholders_equity",
        "operating_income",
        "operating_cash_flow",
        "capex",
    ]

    for column in expected_numeric_columns:
        if column not in factors.columns:
            factors[column] = np.nan

        factors[column] = pd.to_numeric(factors[column], errors="coerce")

    # SEC capex concepts are generally positive cash outflows.
    factors["free_cash_flow"] = (
        factors["operating_cash_flow"] - factors["capex"]
    )

    factors["return_on_equity"] = safe_divide(
        factors["net_income"],
        factors["stockholders_equity"],
    )

    factors["net_margin"] = safe_divide(
        factors["net_income"],
        factors["revenue"],
    )

    factors["operating_margin"] = safe_divide(
        factors["operating_income"],
        factors["revenue"],
    )

    factors["fcf_margin"] = safe_divide(
        factors["free_cash_flow"],
        factors["revenue"],
    )

    factors["asset_turnover"] = safe_divide(
        factors["revenue"],
        factors["assets"],
    )

    factors["liabilities_to_assets"] = safe_divide(
        factors["liabilities"],
        factors["assets"],
    )

    factors["equity_to_assets"] = safe_divide(
        factors["stockholders_equity"],
        factors["assets"],
    )

    factors["operating_cash_flow_to_net_income"] = safe_divide(
        factors["operating_cash_flow"],
        factors["net_income"],
    )

    return factors


def add_data_quality_columns(
    factors: pd.DataFrame,
    required_metrics: list[str],
    optional_metrics: list[str],
    min_required_metrics_present: int,
) -> pd.DataFrame:
    """
    Add data coverage and quality flags.
    """
    factors = factors.copy()

    all_metrics = required_metrics + optional_metrics

    for metric in all_metrics:
        if metric not in factors.columns:
            factors[metric] = np.nan

    required_present_columns = []

    for metric in required_metrics:
        present_column = f"has_{metric}"
        factors[present_column] = factors[metric].notna()
        required_present_columns.append(present_column)

    optional_present_columns = []

    for metric in optional_metrics:
        present_column = f"has_{metric}"
        factors[present_column] = factors[metric].notna()
        optional_present_columns.append(present_column)

    factors["required_metrics_present"] = factors[required_present_columns].sum(axis=1)

    if optional_present_columns:
        factors["optional_metrics_present"] = factors[optional_present_columns].sum(axis=1)
    else:
        factors["optional_metrics_present"] = 0

    factors["total_metrics_present"] = (
        factors["required_metrics_present"] + factors["optional_metrics_present"]
    )

    factors["fundamental_data_status"] = np.where(
        factors["required_metrics_present"] >= min_required_metrics_present,
        "sufficient",
        "limited",
    )

    return factors


def add_basic_fundamental_flags(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple warning flags for potentially weak fundamentals.

    These are not final investment judgments. They are diagnostic labels.
    """
    factors = factors.copy()

    factors["negative_net_income_flag"] = factors["net_income"] < 0
    factors["negative_free_cash_flow_flag"] = factors["free_cash_flow"] < 0

    factors["high_liabilities_flag"] = factors["liabilities_to_assets"] > 0.80

    factors["negative_equity_flag"] = factors["stockholders_equity"] < 0

    factors["weak_margin_flag"] = factors["net_margin"] < 0

    return factors


def build_fundamental_factor_table(
    concepts: pd.DataFrame,
    factor_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build latest SEC fundamental factor table from extracted accounting concepts.
    """
    validate_accounting_concepts(concepts)

    concepts = prepare_accounting_concepts(concepts)

    values = pivot_latest_accounting_values(concepts)

    metadata = build_metric_metadata(concepts)

    if not metadata.empty:
        factors = values.merge(metadata, on="ticker", how="left")
    else:
        factors = values

    factors = add_accounting_factor_columns(factors)

    required_metrics = factor_config.get(
        "required_metrics",
        [
            "revenue",
            "net_income",
            "assets",
            "stockholders_equity",
            "operating_cash_flow",
            "capex",
        ],
    )

    optional_metrics = factor_config.get(
        "optional_metrics",
        [
            "liabilities",
            "operating_income",
        ],
    )

    min_required_metrics_present = int(
        factor_config.get("min_required_metrics_present", 4)
    )

    factors = add_data_quality_columns(
        factors=factors,
        required_metrics=required_metrics,
        optional_metrics=optional_metrics,
        min_required_metrics_present=min_required_metrics_present,
    )

    factors = add_basic_fundamental_flags(factors)

    factors = factors.sort_values("ticker").reset_index(drop=True)

    return factors


def save_fundamental_factors(
    factors: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save latest fundamental factors.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    factors.to_csv(output_path, index=False)

    print(f"Saved SEC fundamental factors to {output_path}")
    print(f"Rows written: {len(factors)}")


def build_sec_fundamental_factors(
    config_path: str | Path,
) -> pd.DataFrame:
    """
    Build SEC-derived fundamental factor table.
    """
    config = load_sec_fundamental_factors_config(config_path)
    factor_config = config["sec_fundamental_factors"]

    input_path = factor_config["input_accounting_concepts_path"]
    output_path = factor_config["output_path"]

    concepts = load_accounting_concepts(input_path)

    factors = build_fundamental_factor_table(
        concepts=concepts,
        factor_config=factor_config,
    )

    save_fundamental_factors(
        factors=factors,
        output_path=output_path,
    )

    return factors


if __name__ == "__main__":
    factors = build_sec_fundamental_factors("config/universe_config.yaml")

    print("\nSEC fundamental factors preview:")
    preview_columns = [
        "ticker",
        "revenue",
        "net_income",
        "assets",
        "stockholders_equity",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "return_on_equity",
        "net_margin",
        "operating_margin",
        "fcf_margin",
        "liabilities_to_assets",
        "fundamental_data_status",
        "negative_net_income_flag",
        "negative_free_cash_flow_flag",
        "high_liabilities_flag",
        "negative_equity_flag",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in factors.columns
    ]

    print(factors[available_preview_columns].head(30))

    print("\nFundamental data status counts:")
    print(factors["fundamental_data_status"].value_counts(dropna=False))

    print("\nWarning flag counts:")
    warning_columns = [
        "negative_net_income_flag",
        "negative_free_cash_flow_flag",
        "high_liabilities_flag",
        "negative_equity_flag",
        "weak_margin_flag",
    ]

    available_warning_columns = [
        column for column in warning_columns if column in factors.columns
    ]

    print(factors[available_warning_columns].sum().sort_values(ascending=False))
