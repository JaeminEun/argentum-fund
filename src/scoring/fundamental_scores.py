from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.universe.config import load_config
from src.utils.io import load_export_settings, save_csv_outputs


REQUIRED_FUNDAMENTAL_COLUMNS = [
    "ticker",
    "fundamental_data_status",
    "negative_net_income_flag",
    "negative_free_cash_flow_flag",
    "high_liabilities_flag",
    "negative_equity_flag",
]


def load_fundamental_score_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load full project config and validate the fundamental_scores block.
    """
    config = load_config(config_path)

    if "fundamental_scores" not in config:
        raise ValueError("Missing 'fundamental_scores' section in config file.")

    if not config["fundamental_scores"].get("enabled", False):
        raise ValueError("fundamental_scores is disabled in config file.")

    return config


def load_fundamental_factors(input_path: str | Path) -> pd.DataFrame:
    """
    Load SEC-derived fundamental factors.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Fundamental factors file not found: {input_path}. "
            "Run sec_fundamental_factors first."
        )

    return pd.read_csv(input_path)


def validate_fundamental_factors(factors: pd.DataFrame) -> None:
    """
    Validate required columns for fundamental scoring.
    """
    missing_columns = set(REQUIRED_FUNDAMENTAL_COLUMNS) - set(factors.columns)

    if missing_columns:
        raise ValueError(
            f"Fundamental factors are missing required columns: {missing_columns}"
        )

    if factors.empty:
        raise ValueError("Fundamental factors file is empty.")


def prepare_fundamental_factors(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Clean fundamental factors for scoring.
    """
    factors = factors.copy()

    factors["ticker"] = factors["ticker"].astype(str).str.upper().str.strip()

    numeric_candidates = [
        "revenue",
        "net_income",
        "assets",
        "liabilities",
        "stockholders_equity",
        "operating_income",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "return_on_equity",
        "net_margin",
        "operating_margin",
        "fcf_margin",
        "asset_turnover",
        "liabilities_to_assets",
        "equity_to_assets",
        "operating_cash_flow_to_net_income",
        "required_metrics_present",
        "optional_metrics_present",
        "total_metrics_present",
    ]

    for column in numeric_candidates:
        if column in factors.columns:
            factors[column] = pd.to_numeric(factors[column], errors="coerce")

    bool_candidates = [
        "negative_net_income_flag",
        "negative_free_cash_flow_flag",
        "high_liabilities_flag",
        "negative_equity_flag",
        "weak_margin_flag",
    ]

    for column in bool_candidates:
        if column in factors.columns:
            factors[column] = factors[column].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            )

    return factors


def percentile_score(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Convert a numeric series into a 0-100 percentile score.

    Missing values remain missing.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)

    ranks = numeric.rank(pct=True)

    if not higher_is_better:
        ranks = 1 - ranks

    return ranks * 100


def build_metric_score(
    factors: pd.DataFrame,
    metric_name: str,
    direction: str,
) -> pd.Series:
    """
    Build a percentile score for a single metric.
    """
    if metric_name not in factors.columns:
        return pd.Series(np.nan, index=factors.index)

    direction = str(direction).lower().strip()

    if direction not in {"higher", "lower"}:
        raise ValueError(
            f"Invalid direction for metric '{metric_name}': {direction}. "
            "Use 'higher' or 'lower'."
        )

    return percentile_score(
        factors[metric_name],
        higher_is_better=(direction == "higher"),
    )


def average_component_scores(
    score_frame: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """
    Average score columns while ignoring missing values.
    """
    available_columns = [column for column in columns if column in score_frame.columns]

    if not available_columns:
        return pd.Series(np.nan, index=score_frame.index)

    return score_frame[available_columns].mean(axis=1, skipna=True)


def calculate_component_score(
    factors: pd.DataFrame,
    metric_config: Dict[str, str],
    component_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Calculate a component score from configured metrics.

    Returns:
        score_details: individual metric scores
        component_score: average component score
    """
    score_details = pd.DataFrame(index=factors.index)
    metric_score_columns = []

    for metric_name, direction in metric_config.items():
        score_column = f"{metric_name}_score"

        score_details[score_column] = build_metric_score(
            factors=factors,
            metric_name=metric_name,
            direction=direction,
        )

        metric_score_columns.append(score_column)

    component_score = average_component_scores(
        score_frame=score_details,
        columns=metric_score_columns,
    )

    score_details[component_name] = component_score

    return score_details, component_score


def calculate_weighted_fundamental_score(
    scores: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Calculate weighted fundamental score while ignoring missing components.
    """
    component_columns = [
        "quality_score",
        "cash_flow_score",
        "balance_sheet_score",
    ]

    weighted_sum = pd.Series(0.0, index=scores.index)
    active_weight_sum = pd.Series(0.0, index=scores.index)

    for column in component_columns:
        weight = float(weights.get(column, 0.0))

        if column not in scores.columns:
            continue

        valid = scores[column].notna()

        weighted_sum.loc[valid] += scores.loc[valid, column] * weight
        active_weight_sum.loc[valid] += weight

    result = weighted_sum / active_weight_sum
    result.loc[active_weight_sum == 0] = np.nan

    return result


def apply_fundamental_penalties(
    scores: pd.DataFrame,
    score_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply accounting warning penalties to the fundamental score.
    """
    scores = scores.copy()

    penalty_settings = score_config.get("penalty_settings", {})

    negative_net_income_penalty = float(
        penalty_settings.get("negative_net_income_penalty", 15.0)
    )
    negative_free_cash_flow_penalty = float(
        penalty_settings.get("negative_free_cash_flow_penalty", 12.5)
    )
    high_liabilities_penalty = float(
        penalty_settings.get("high_liabilities_penalty", 10.0)
    )
    negative_equity_penalty = float(
        penalty_settings.get("negative_equity_penalty", 25.0)
    )
    limited_data_penalty = float(
        penalty_settings.get("limited_data_penalty", 10.0)
    )

    scores["fundamental_penalty"] = 0.0

    scores.loc[
        scores["negative_net_income_flag"] == True,
        "fundamental_penalty",
    ] += negative_net_income_penalty

    scores.loc[
        scores["negative_free_cash_flow_flag"] == True,
        "fundamental_penalty",
    ] += negative_free_cash_flow_penalty

    scores.loc[
        scores["high_liabilities_flag"] == True,
        "fundamental_penalty",
    ] += high_liabilities_penalty

    scores.loc[
        scores["negative_equity_flag"] == True,
        "fundamental_penalty",
    ] += negative_equity_penalty

    scores.loc[
        scores["fundamental_data_status"].astype(str).str.lower() == "limited",
        "fundamental_penalty",
    ] += limited_data_penalty

    scores["risk_adjusted_fundamental_score"] = (
        scores["fundamental_score"] - scores["fundamental_penalty"]
    )

    scores["risk_adjusted_fundamental_score"] = scores[
        "risk_adjusted_fundamental_score"
    ].clip(lower=0, upper=100)

    return scores


def add_fundamental_rank_and_buckets(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add ranks and percentile-based buckets for fundamental scores.
    """
    scores = scores.copy()

    scores["fundamental_rank"] = scores["risk_adjusted_fundamental_score"].rank(
        ascending=False,
        method="min",
    )

    scores["fundamental_percentile"] = scores[
        "risk_adjusted_fundamental_score"
    ].rank(
        pct=True,
        ascending=True,
    )

    scores["fundamental_bucket"] = pd.cut(
        scores["fundamental_percentile"],
        bins=[0, 0.30, 0.70, 0.85, 0.95, 1.00],
        labels=[
            "weak",
            "neutral",
            "watchlist",
            "strong_fundamental",
            "top_fundamental",
        ],
        include_lowest=True,
    )

    return scores


def add_fundamental_signal(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple plain-language signal labels.
    """
    scores = scores.copy()

    conditions = [
        (
            scores["risk_adjusted_fundamental_score"] >= 75
        ),
        (
            (scores["quality_score"] >= 65)
            & (scores["cash_flow_score"] >= 65)
            & (scores["fundamental_penalty"] == 0)
        ),
        (
            scores["negative_equity_flag"] == True
        ),
        (
            (scores["negative_net_income_flag"] == True)
            & (scores["negative_free_cash_flow_flag"] == True)
        ),
        (
            scores["high_liabilities_flag"] == True
        ),
        (
            scores["fundamental_data_status"].astype(str).str.lower() == "limited"
        ),
    ]

    choices = [
        "strong_accounting_profile",
        "profitable_cash_generative",
        "negative_equity_risk",
        "profit_and_cash_flow_warning",
        "high_liability_burden",
        "limited_fundamental_data",
    ]

    scores["fundamental_signal"] = np.select(
        conditions,
        choices,
        default="mixed_fundamental_profile",
    )

    return scores


def add_fundamental_interpretation(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add a compact interpretation string.
    """
    scores = scores.copy()

    def pct(value: float | int | None) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value * 100:.1f}%"

    def interpret_row(row: pd.Series) -> str:
        return (
            f"{row['fundamental_signal']}: "
            f"ROE {pct(row.get('return_on_equity'))}, "
            f"net margin {pct(row.get('net_margin'))}, "
            f"FCF margin {pct(row.get('fcf_margin'))}, "
            f"liabilities/assets {pct(row.get('liabilities_to_assets'))}."
        )

    scores["fundamental_interpretation"] = scores.apply(interpret_row, axis=1)

    return scores


def calculate_fundamental_scores(
    factors: pd.DataFrame,
    score_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate fundamental scores from SEC-derived fundamental factors.
    """
    validate_fundamental_factors(factors)

    factors = prepare_fundamental_factors(factors)

    weights = score_config.get(
        "score_weights",
        {
            "quality_score": 0.45,
            "cash_flow_score": 0.30,
            "balance_sheet_score": 0.25,
        },
    )

    quality_metrics = score_config.get(
        "quality_metrics",
        {
            "return_on_equity": "higher",
            "net_margin": "higher",
            "operating_margin": "higher",
            "asset_turnover": "higher",
        },
    )

    cash_flow_metrics = score_config.get(
        "cash_flow_metrics",
        {
            "fcf_margin": "higher",
            "operating_cash_flow_to_net_income": "higher",
        },
    )

    balance_sheet_metrics = score_config.get(
        "balance_sheet_metrics",
        {
            "liabilities_to_assets": "lower",
            "equity_to_assets": "higher",
        },
    )

    quality_details, quality_score = calculate_component_score(
        factors=factors,
        metric_config=quality_metrics,
        component_name="quality_score",
    )

    cash_flow_details, cash_flow_score = calculate_component_score(
        factors=factors,
        metric_config=cash_flow_metrics,
        component_name="cash_flow_score",
    )

    balance_sheet_details, balance_sheet_score = calculate_component_score(
        factors=factors,
        metric_config=balance_sheet_metrics,
        component_name="balance_sheet_score",
    )

    scores = pd.concat(
        [
            factors.reset_index(drop=True),
            quality_details.reset_index(drop=True),
            cash_flow_details.reset_index(drop=True),
            balance_sheet_details.reset_index(drop=True),
        ],
        axis=1,
    )

    # Remove duplicate component columns if they appeared during concat.
    scores = scores.loc[:, ~scores.columns.duplicated()].copy()

    scores["quality_score"] = quality_score
    scores["cash_flow_score"] = cash_flow_score
    scores["balance_sheet_score"] = balance_sheet_score

    scores["fundamental_score"] = calculate_weighted_fundamental_score(
        scores=scores,
        weights=weights,
    )

    scores = apply_fundamental_penalties(
        scores=scores,
        score_config=score_config,
    )

    scores = add_fundamental_rank_and_buckets(scores)
    scores = add_fundamental_signal(scores)
    scores = add_fundamental_interpretation(scores)

    output_columns = [
        "fundamental_rank",
        "ticker",
        "risk_adjusted_fundamental_score",
        "fundamental_score",
        "fundamental_bucket",
        "fundamental_signal",
        "fundamental_penalty",
        "quality_score",
        "cash_flow_score",
        "balance_sheet_score",
        "return_on_equity",
        "net_margin",
        "operating_margin",
        "fcf_margin",
        "asset_turnover",
        "liabilities_to_assets",
        "equity_to_assets",
        "operating_cash_flow_to_net_income",
        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "assets",
        "liabilities",
        "stockholders_equity",
        "fundamental_data_status",
        "required_metrics_present",
        "optional_metrics_present",
        "negative_net_income_flag",
        "negative_free_cash_flow_flag",
        "high_liabilities_flag",
        "negative_equity_flag",
        "weak_margin_flag",
        "fundamental_interpretation",
    ]

    available_output_columns = [
        column for column in output_columns if column in scores.columns
    ]

    scores = scores[available_output_columns].sort_values(
        ["fundamental_rank", "ticker"]
    )

    return scores


def save_fundamental_scores(
    scores: pd.DataFrame,
    output_path: str | Path,
    export_settings: Dict[str, Any] | None = None,
) -> None:
    """
    Save fundamental scores.
    """
    save_csv_outputs(
        df=scores,
        output_path=output_path,
        export_settings=export_settings,
    )

    print(f"Rows written: {len(scores)}")


def build_fundamental_scores(config_path: str | Path) -> pd.DataFrame:
    """
    Build fundamental scores from SEC-derived fundamental factors.
    """
    full_config = load_fundamental_score_config(config_path)

    score_config = full_config["fundamental_scores"]
    export_settings = load_export_settings(full_config)

    input_path = score_config["input_fundamental_factors_path"]
    output_path = score_config["output_scores_path"]

    factors = load_fundamental_factors(input_path)

    scores = calculate_fundamental_scores(
        factors=factors,
        score_config=score_config,
    )

    save_fundamental_scores(
        scores=scores,
        output_path=output_path,
        export_settings=export_settings,
    )

    return scores


if __name__ == "__main__":
    scores = build_fundamental_scores("config/universe_config.yaml")

    print("\nTop fundamental scores:")
    preview_columns = [
        "fundamental_rank",
        "ticker",
        "risk_adjusted_fundamental_score",
        "fundamental_bucket",
        "fundamental_signal",
        "quality_score",
        "cash_flow_score",
        "balance_sheet_score",
        "fundamental_penalty",
        "fundamental_interpretation",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in scores.columns
    ]

    print(scores[available_preview_columns].head(30))

    print("\nFundamental buckets:")
    print(scores["fundamental_bucket"].value_counts(dropna=False))

    print("\nFundamental signals:")
    print(scores["fundamental_signal"].value_counts(dropna=False))
