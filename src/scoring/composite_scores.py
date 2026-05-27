from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.universe.config import load_config
from src.utils.io import load_export_settings, save_csv_outputs


REQUIRED_PRICE_COLUMNS = [
    "ticker",
    "risk_adjusted_timing_score",
    "risk_adjusted_rank",
    "primary_signal",
    "score_bucket",
    "risk_flag",
    "interpretation",
]


REQUIRED_FUNDAMENTAL_COLUMNS = [
    "ticker",
    "risk_adjusted_fundamental_score",
    "fundamental_rank",
    "fundamental_signal",
    "fundamental_bucket",
    "fundamental_interpretation",
]


def load_composite_score_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load full project config and validate the composite_scores block.
    """
    config = load_config(config_path)

    if "composite_scores" not in config:
        raise ValueError("Missing 'composite_scores' section in config file.")

    if not config["composite_scores"].get("enabled", False):
        raise ValueError("composite_scores is disabled in config file.")

    return config


def load_score_file(input_path: str | Path, score_type: str) -> pd.DataFrame:
    """
    Load a score CSV file.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"{score_type} score file not found: {input_path}"
        )

    return pd.read_csv(input_path)


def validate_score_inputs(
    price_scores: pd.DataFrame,
    fundamental_scores: pd.DataFrame,
) -> None:
    """
    Validate required columns for composite scoring.
    """
    missing_price_columns = set(REQUIRED_PRICE_COLUMNS) - set(price_scores.columns)

    if missing_price_columns:
        raise ValueError(
            f"Price scores are missing required columns: {missing_price_columns}"
        )

    missing_fundamental_columns = (
        set(REQUIRED_FUNDAMENTAL_COLUMNS) - set(fundamental_scores.columns)
    )

    if missing_fundamental_columns:
        raise ValueError(
            "Fundamental scores are missing required columns: "
            f"{missing_fundamental_columns}"
        )

    if price_scores.empty:
        raise ValueError("Price scores file is empty.")

    if fundamental_scores.empty:
        raise ValueError("Fundamental scores file is empty.")


def prepare_price_scores(price_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and rename price score columns for composite merge.
    """
    price_scores = price_scores.copy()

    price_scores["ticker"] = price_scores["ticker"].astype(str).str.upper().str.strip()

    rename_map = {
        "risk_adjusted_timing_score": "price_score",
        "risk_adjusted_rank": "price_rank",
        "score_bucket": "price_bucket",
        "primary_signal": "price_signal",
        "interpretation": "price_interpretation",
    }

    price_scores = price_scores.rename(columns=rename_map)

    keep_columns = [
        "ticker",
        "company_name",
        "asset_type",
        "sector",
        "industry",
        "universe_name",
        "source_type",
        "source_name",
        "strategy_role",
        "account_target",
        "target_weight",
        "latest_date",
        "adjusted_close",
        "price_score",
        "timing_score",
        "price_rank",
        "score_rank",
        "price_bucket",
        "price_signal",
        "dip_flag",
        "trend_flag",
        "momentum_flag",
        "risk_flag",
        "risk_penalty",
        "below_13w_high",
        "below_52w_high",
        "distance_from_ma_200d",
        "return_13w",
        "volatility_60d",
        "price_interpretation",
    ]

    available_columns = [
        column for column in keep_columns if column in price_scores.columns
    ]

    return price_scores[available_columns]


def prepare_fundamental_scores(fundamental_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and rename fundamental score columns for composite merge.

    The composite model uses the risk-adjusted fundamental score as the
    primary fundamental score, while preserving the unadjusted score under
    a separate name.
    """
    fundamental_scores = fundamental_scores.copy()

    fundamental_scores["ticker"] = (
        fundamental_scores["ticker"].astype(str).str.upper().str.strip()
    )

    # Avoid duplicate column names:
    # latest_fundamental_scores.csv has both:
    # - risk_adjusted_fundamental_score
    # - fundamental_score
    #
    # We want:
    # - risk_adjusted_fundamental_score -> fundamental_score
    # - fundamental_score -> raw_fundamental_score
    if "fundamental_score" in fundamental_scores.columns:
        fundamental_scores = fundamental_scores.rename(
            columns={"fundamental_score": "raw_fundamental_score"}
        )

    if "risk_adjusted_fundamental_score" in fundamental_scores.columns:
        fundamental_scores = fundamental_scores.rename(
            columns={"risk_adjusted_fundamental_score": "fundamental_score"}
        )

    keep_columns = [
        "ticker",
        "fundamental_score",
        "raw_fundamental_score",
        "fundamental_rank",
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
        "fundamental_data_status",
        "negative_net_income_flag",
        "negative_free_cash_flow_flag",
        "high_liabilities_flag",
        "negative_equity_flag",
        "weak_margin_flag",
        "sanity_filter_flag",
        "sanity_filter_notes",
        "fundamental_interpretation",
    ]

    available_columns = [
        column for column in keep_columns if column in fundamental_scores.columns
    ]

    fundamental_scores = fundamental_scores[available_columns]

    # Defensive cleanup in case duplicate columns still slipped through.
    fundamental_scores = fundamental_scores.loc[
        :, ~fundamental_scores.columns.duplicated()
    ].copy()

    return fundamental_scores

def merge_scores(
    price_scores: pd.DataFrame,
    fundamental_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge price and fundamental scores by ticker.

    Uses an outer join so partial-score rows remain visible.
    """
    price = prepare_price_scores(price_scores)
    fundamental = prepare_fundamental_scores(fundamental_scores)

    merged = price.merge(
        fundamental,
        on="ticker",
        how="outer",
        indicator=True,
    )

    merged["composite_data_status"] = merged["_merge"].map(
        {
            "both": "complete",
            "left_only": "price_only",
            "right_only": "fundamentals_only",
        }
    )

    merged = merged.drop(columns=["_merge"])

    return merged


def calculate_weighted_composite_score(
    scores: pd.DataFrame,
    score_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate weighted composite research score.

    Missing component scores are ignored and remaining weights are rescaled,
    but data status remains visible.
    """
    scores = scores.copy()

    weights = score_config.get(
        "score_weights",
        {
            "price_score": 0.60,
            "fundamental_score": 0.40,
        },
    )

    price_weight = float(weights.get("price_score", 0.60))
    fundamental_weight = float(weights.get("fundamental_score", 0.40))

    scores["price_score"] = pd.to_numeric(scores["price_score"], errors="coerce")
    scores["fundamental_score"] = pd.to_numeric(
        scores["fundamental_score"],
        errors="coerce",
    )

    weighted_sum = pd.Series(0.0, index=scores.index)
    active_weight_sum = pd.Series(0.0, index=scores.index)

    valid_price = scores["price_score"].notna()
    valid_fundamental = scores["fundamental_score"].notna()

    weighted_sum.loc[valid_price] += scores.loc[valid_price, "price_score"] * price_weight
    active_weight_sum.loc[valid_price] += price_weight

    weighted_sum.loc[valid_fundamental] += (
        scores.loc[valid_fundamental, "fundamental_score"] * fundamental_weight
    )
    active_weight_sum.loc[valid_fundamental] += fundamental_weight

    scores["composite_research_score"] = weighted_sum / active_weight_sum
    scores.loc[active_weight_sum == 0, "composite_research_score"] = np.nan

    scores["price_score_available"] = valid_price
    scores["fundamental_score_available"] = valid_fundamental

    return scores


def add_composite_ranks_and_buckets(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite rank and percentile-based buckets.
    """
    scores = scores.copy()

    scores["composite_rank"] = scores["composite_research_score"].rank(
        ascending=False,
        method="min",
    )

    scores["composite_percentile"] = scores[
        "composite_research_score"
    ].rank(
        pct=True,
        ascending=True,
    )

    scores["composite_bucket"] = pd.cut(
        scores["composite_percentile"],
        bins=[0, 0.30, 0.70, 0.85, 0.95, 1.00],
        labels=[
            "low_priority",
            "neutral",
            "watchlist",
            "strong_candidate",
            "top_candidate",
        ],
        include_lowest=True,
    )

    return scores


def add_composite_signal(
    scores: pd.DataFrame,
    score_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add interpretable composite signal labels.
    """
    scores = scores.copy()

    thresholds = score_config.get("thresholds", {})

    strong_price = float(thresholds.get("strong_price_score", 65))
    weak_price = float(thresholds.get("weak_price_score", 40))
    strong_fundamental = float(thresholds.get("strong_fundamental_score", 65))
    weak_fundamental = float(thresholds.get("weak_fundamental_score", 40))

    price = scores["price_score"]
    fundamental = scores["fundamental_score"]

    conditions = [
        (
            scores["composite_data_status"].ne("complete")
        ),
        (
            price.ge(strong_price)
            & fundamental.ge(strong_fundamental)
        ),
        (
            price.ge(strong_price)
            & fundamental.between(weak_fundamental, strong_fundamental, inclusive="left")
        ),
        (
            price.ge(strong_price)
            & fundamental.lt(weak_fundamental)
        ),
        (
            price.lt(weak_price)
            & fundamental.ge(strong_fundamental)
        ),
        (
            price.lt(weak_price)
            & fundamental.lt(weak_fundamental)
        ),
    ]

    choices = [
        "partial_data_review_required",
        "aligned_candidate",
        "timing_candidate_fundamentals_neutral",
        "possible_value_trap",
        "quality_watchlist_wait_for_timing",
        "low_priority",
    ]

    scores["composite_signal"] = np.select(
        conditions,
        choices,
        default="mixed_signal_review_required",
    )

    return scores


def add_composite_interpretation(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add a compact plain-language interpretation.
    """
    scores = scores.copy()

    def fmt_score(value: float | int | None) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value:.1f}"

    def interpret_row(row: pd.Series) -> str:
        return (
            f"{row['composite_signal']}: "
            f"price score {fmt_score(row.get('price_score'))} "
            f"({row.get('price_signal', 'n/a')}), "
            f"fundamental score {fmt_score(row.get('fundamental_score'))} "
            f"({row.get('fundamental_signal', 'n/a')})."
        )

    scores["composite_interpretation"] = scores.apply(interpret_row, axis=1)

    return scores


def calculate_composite_scores(
    price_scores: pd.DataFrame,
    fundamental_scores: pd.DataFrame,
    score_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build composite research scores from price and fundamental scores.
    """
    validate_score_inputs(price_scores, fundamental_scores)

    scores = merge_scores(
        price_scores=price_scores,
        fundamental_scores=fundamental_scores,
    )

    scores = calculate_weighted_composite_score(
        scores=scores,
        score_config=score_config,
    )

    scores = add_composite_ranks_and_buckets(scores)
    scores = add_composite_signal(scores, score_config=score_config)
    scores = add_composite_interpretation(scores)

    output_columns = [
        "composite_rank",
        "ticker",
        "company_name",
        "asset_type",
        "sector",
        "industry",
        "universe_name",
        "strategy_role",
        "account_target",
        "target_weight",
        "latest_date",
        "adjusted_close",
        "composite_research_score",
        "composite_bucket",
        "composite_signal",
        "composite_data_status",
        "price_score",
        "price_rank",
        "price_bucket",
        "price_signal",
        "fundamental_score",
        "raw_fundamental_score",
        "fundamental_rank",
        "fundamental_bucket",
        "fundamental_signal",
        "quality_score",
        "cash_flow_score",
        "balance_sheet_score",
        "below_13w_high",
        "below_52w_high",
        "distance_from_ma_200d",
        "return_13w",
        "volatility_60d",
        "risk_flag",
        "risk_penalty",
        "return_on_equity",
        "net_margin",
        "operating_margin",
        "fcf_margin",
        "liabilities_to_assets",
        "fundamental_penalty",
        "negative_net_income_flag",
        "negative_free_cash_flow_flag",
        "high_liabilities_flag",
        "negative_equity_flag",
        "sanity_filter_flag",
        "sanity_filter_notes",
        "price_interpretation",
        "fundamental_interpretation",
        "composite_interpretation",
    ]

    available_output_columns = [
        column for column in output_columns if column in scores.columns
    ]

    scores = scores[available_output_columns].sort_values(
        ["composite_rank", "ticker"]
    )

    return scores


def save_composite_scores(
    scores: pd.DataFrame,
    output_path: str | Path,
    export_settings: Dict[str, Any] | None = None,
) -> None:
    """
    Save composite scores.
    """
    save_csv_outputs(
        df=scores,
        output_path=output_path,
        export_settings=export_settings,
    )

    print(f"Rows written: {len(scores)}")


def build_composite_scores(config_path: str | Path) -> pd.DataFrame:
    """
    Build composite scores from latest price and fundamental score files.
    """
    full_config = load_composite_score_config(config_path)

    score_config = full_config["composite_scores"]
    export_settings = load_export_settings(full_config)

    price_scores_path = score_config["input_price_scores_path"]
    fundamental_scores_path = score_config["input_fundamental_scores_path"]
    output_path = score_config["output_scores_path"]

    price_scores = load_score_file(
        input_path=price_scores_path,
        score_type="Price",
    )

    fundamental_scores = load_score_file(
        input_path=fundamental_scores_path,
        score_type="Fundamental",
    )

    scores = calculate_composite_scores(
        price_scores=price_scores,
        fundamental_scores=fundamental_scores,
        score_config=score_config,
    )

    save_composite_scores(
        scores=scores,
        output_path=output_path,
        export_settings=export_settings,
    )

    return scores


if __name__ == "__main__":
    scores = build_composite_scores("config/universe_config.yaml")

    print("\nTop composite research scores:")
    preview_columns = [
        "composite_rank",
        "ticker",
        "sector",
        "composite_research_score",
        "composite_bucket",
        "composite_signal",
        "price_score",
        "fundamental_score",
        "price_signal",
        "fundamental_signal",
        "composite_interpretation",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in scores.columns
    ]

    print(scores[available_preview_columns].head(30))

    print("\nComposite buckets:")
    print(scores["composite_bucket"].value_counts(dropna=False))

    print("\nComposite signals:")
    print(scores["composite_signal"].value_counts(dropna=False))
