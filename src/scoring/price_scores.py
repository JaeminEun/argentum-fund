from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.universe.config import load_config
from src.utils.io import load_export_settings, save_csv_outputs


REQUIRED_SCORE_INPUT_COLUMNS = [
    "ticker",
    "latest_date",
    "adjusted_close",
    "below_13w_high",
    "below_52w_high",
    "distance_from_ma_200d",
    "return_13w",
    "volatility_60d",
]


def load_price_score_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load the price_scores section from the project configuration file.
    """
    config = load_config(config_path)

    if "price_scores" not in config:
        raise ValueError("Missing 'price_scores' section in config file.")

    score_config = config["price_scores"]

    if not score_config.get("enabled", False):
        raise ValueError("price_scores is disabled in the config file.")

    return score_config


def load_latest_price_factors(input_path: str | Path) -> pd.DataFrame:
    """
    Load the latest price factor snapshot.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Latest price factors file not found: {input_path}. "
            "Run the price factor calculator first."
        )

    return pd.read_csv(input_path)

def load_current_universe(input_path: str | Path) -> pd.DataFrame:
    """
    Load the standardized current universe file.

    This provides context such as universe_name, sector, industry,
    strategy_role, account_target, and target_weight.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Current universe file not found: {input_path}. "
            "Run the universe builder first."
        )

    return pd.read_csv(input_path)


def prepare_universe_metadata(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare universe metadata for merging into the score output.

    If a ticker appears in multiple universes, this collapses metadata
    into semicolon-separated labels so the scoring output keeps one row
    per ticker.
    """
    universe = universe.copy()

    if "ticker" not in universe.columns:
        raise ValueError("Universe file is missing required column: ticker")

    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()

    metadata_columns = [
        "ticker",
        "company_name",
        "asset_type",
        "universe_name",
        "source_type",
        "source_name",
        "strategy_role",
        "account_target",
        "target_weight",
        "sector",
        "industry",
    ]

    available_columns = [
        column for column in metadata_columns if column in universe.columns
    ]

    metadata = universe[available_columns].copy()

    def combine_unique_values(series: pd.Series) -> str | float:
        values = (
            series.dropna()
            .astype(str)
            .str.strip()
        )

        values = values[values != ""]

        if values.empty:
            return np.nan

        unique_values = sorted(values.unique())

        return "; ".join(unique_values)

    aggregation = {
        column: combine_unique_values
        for column in available_columns
        if column != "ticker"
    }

    metadata = (
        metadata.groupby("ticker", as_index=False)
        .agg(aggregation)
    )

    return metadata


def merge_universe_metadata(
    scores: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge prepared universe metadata into price scores.
    """
    scores = scores.copy()
    scores["ticker"] = scores["ticker"].astype(str).str.upper().str.strip()

    metadata = prepare_universe_metadata(universe)

    scores = scores.merge(
        metadata,
        on="ticker",
        how="left",
    )

    return scores


def validate_score_inputs(factors: pd.DataFrame) -> None:
    """
    Validate that required score input columns exist.
    """
    missing_columns = set(REQUIRED_SCORE_INPUT_COLUMNS) - set(factors.columns)

    if missing_columns:
        raise ValueError(
            f"Latest price factors are missing required columns: {missing_columns}"
        )

    if factors.empty:
        raise ValueError("Latest price factors file is empty.")


def percentile_score(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Convert a numeric series into 0-100 percentile scores.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)

    ranks = numeric.rank(pct=True)

    if not higher_is_better:
        ranks = 1 - ranks

    return ranks * 100


def calculate_dip_score(factors: pd.DataFrame) -> pd.Series:
    """
    Score securities based on how far they are below recent highs.

    More negative below-high values indicate larger dips.
    Larger dips receive higher dip scores.
    """
    below_13w_score = percentile_score(
        factors["below_13w_high"],
        higher_is_better=False,
    )

    below_52w_score = percentile_score(
        factors["below_52w_high"],
        higher_is_better=False,
    )

    return 0.70 * below_13w_score + 0.30 * below_52w_score


def calculate_trend_score(factors: pd.DataFrame) -> pd.Series:
    """
    Score securities based on relationship to the 200-day moving average.
    """
    return percentile_score(
        factors["distance_from_ma_200d"],
        higher_is_better=True,
    )


def calculate_momentum_score(factors: pd.DataFrame) -> pd.Series:
    """
    Score securities based on 13-week return.
    """
    return percentile_score(
        factors["return_13w"],
        higher_is_better=True,
    )


def calculate_risk_score(factors: pd.DataFrame) -> pd.Series:
    """
    Score securities based on 60-day volatility.

    Lower volatility receives a higher score.
    """
    return percentile_score(
        factors["volatility_60d"],
        higher_is_better=False,
    )


def calculate_weighted_timing_score(
    factors: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Calculate weighted timing score while ignoring missing component scores.
    """
    score_columns = [
        "dip_score",
        "trend_score",
        "momentum_score",
        "risk_score",
    ]

    weighted_sum = pd.Series(0.0, index=factors.index)
    active_weight_sum = pd.Series(0.0, index=factors.index)

    for column in score_columns:
        weight = float(weights.get(column, 0))
        valid = factors[column].notna()

        weighted_sum.loc[valid] += factors.loc[valid, column] * weight
        active_weight_sum.loc[valid] += weight

    timing_score = weighted_sum / active_weight_sum
    timing_score.loc[active_weight_sum == 0] = np.nan

    return timing_score


def add_signal_flags(
    scores: pd.DataFrame,
    score_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add interpretable signal flags based on price factors.
    """
    scores = scores.copy()

    thresholds = score_config.get("signal_thresholds", {})

    moderate_dip = thresholds.get("moderate_dip", -0.05)
    deep_dip = thresholds.get("deep_dip", -0.10)
    severe_dip = thresholds.get("severe_dip", -0.20)
    near_high = thresholds.get("near_high", -0.03)

    above_200d_ma = thresholds.get("above_200d_ma", 0.00)
    slightly_below_200d_ma = thresholds.get("slightly_below_200d_ma", -0.05)
    broken_trend = thresholds.get("broken_trend", -0.10)

    positive_13w_return = thresholds.get("positive_13w_return", 0.00)
    strong_13w_return = thresholds.get("strong_13w_return", 0.10)

    high_volatility = thresholds.get("high_volatility", 0.50)
    severe_volatility = thresholds.get("severe_volatility", 0.75)

    scores["dip_flag"] = np.select(
        [
            scores["below_13w_high"] <= severe_dip,
            scores["below_13w_high"] <= deep_dip,
            scores["below_13w_high"] <= moderate_dip,
            scores["below_13w_high"] >= near_high,
        ],
        [
            "severe_dip",
            "deep_dip",
            "moderate_dip",
            "near_high",
        ],
        default="small_dip",
    )

    scores["trend_flag"] = np.select(
        [
            scores["distance_from_ma_200d"] >= above_200d_ma,
            scores["distance_from_ma_200d"] >= slightly_below_200d_ma,
            scores["distance_from_ma_200d"] <= broken_trend,
        ],
        [
            "above_200d_ma",
            "slightly_below_200d_ma",
            "broken_trend",
        ],
        default="below_200d_ma",
    )

    scores["momentum_flag"] = np.select(
        [
            scores["return_13w"] >= strong_13w_return,
            scores["return_13w"] >= positive_13w_return,
        ],
        [
            "strong_13w_momentum",
            "positive_13w_momentum",
        ],
        default="negative_13w_momentum",
    )

    scores["risk_flag"] = np.select(
        [
            scores["volatility_60d"] >= severe_volatility,
            scores["volatility_60d"] >= high_volatility,
        ],
        [
            "severe_volatility",
            "high_volatility",
        ],
        default="normal_volatility",
    )

    return scores


def add_primary_signal(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add a primary signal label summarizing the security's price setup.
    """
    scores = scores.copy()

    conditions = [
        (
            scores["dip_flag"].isin(["moderate_dip", "deep_dip"])
            & scores["trend_flag"].isin(["above_200d_ma", "slightly_below_200d_ma"])
            & scores["momentum_flag"].isin(
                ["positive_13w_momentum", "strong_13w_momentum"]
            )
        ),
        (
            scores["dip_flag"].isin(["deep_dip", "severe_dip"])
            & scores["trend_flag"].eq("broken_trend")
        ),
        (
            scores["dip_flag"].eq("near_high")
            & scores["trend_flag"].eq("above_200d_ma")
            & scores["momentum_flag"].isin(
                ["positive_13w_momentum", "strong_13w_momentum"]
            )
        ),
        (
            scores["trend_flag"].eq("above_200d_ma")
            & scores["momentum_flag"].eq("strong_13w_momentum")
            & scores["risk_flag"].eq("normal_volatility")
        ),
    ]

    choices = [
        "pullback_in_uptrend",
        "falling_knife_risk",
        "strong_but_not_discounted",
        "stable_strength",
    ]

    scores["primary_signal"] = np.select(
        conditions,
        choices,
        default="mixed_signal",
    )

    return scores


def calculate_risk_adjusted_score(
    scores: pd.DataFrame,
    score_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add a risk-adjusted timing score with penalties for high volatility
    and broken long-term trend.
    """
    scores = scores.copy()

    risk_config = score_config.get("risk_adjustment", {})

    if not risk_config.get("enabled", True):
        scores["risk_adjusted_timing_score"] = scores["timing_score"]
        scores["risk_penalty"] = 0.0
        return scores

    high_volatility_threshold = risk_config.get("high_volatility_threshold", 0.50)
    severe_volatility_threshold = risk_config.get("severe_volatility_threshold", 0.75)

    high_volatility_penalty = risk_config.get("high_volatility_penalty", 7.5)
    severe_volatility_penalty = risk_config.get("severe_volatility_penalty", 15.0)
    broken_trend_penalty = risk_config.get("broken_trend_penalty", 10.0)
    severe_broken_trend_penalty = risk_config.get("severe_broken_trend_penalty", 20.0)

    scores["risk_penalty"] = 0.0

    scores.loc[
        scores["volatility_60d"] >= high_volatility_threshold,
        "risk_penalty",
    ] += high_volatility_penalty

    scores.loc[
        scores["volatility_60d"] >= severe_volatility_threshold,
        "risk_penalty",
    ] += severe_volatility_penalty

    scores.loc[
        scores["distance_from_ma_200d"] <= -0.05,
        "risk_penalty",
    ] += broken_trend_penalty

    scores.loc[
        scores["distance_from_ma_200d"] <= -0.10,
        "risk_penalty",
    ] += severe_broken_trend_penalty

    scores["risk_adjusted_timing_score"] = (
        scores["timing_score"] - scores["risk_penalty"]
    )

    scores["risk_adjusted_timing_score"] = scores[
        "risk_adjusted_timing_score"
    ].clip(lower=0, upper=100)

    return scores


def add_rank_and_buckets(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add rank columns and percentile-based score buckets.
    """
    scores = scores.copy()

    scores["score_rank"] = scores["timing_score"].rank(
        ascending=False,
        method="min",
    )

    scores["risk_adjusted_rank"] = scores["risk_adjusted_timing_score"].rank(
        ascending=False,
        method="min",
    )

    scores["risk_adjusted_percentile"] = scores[
        "risk_adjusted_timing_score"
    ].rank(
        pct=True,
        ascending=True,
    )

    scores["score_bucket"] = pd.cut(
        scores["risk_adjusted_percentile"],
        bins=[0, 0.30, 0.70, 0.85, 0.95, 1.00],
        labels=[
            "weak",
            "neutral",
            "watchlist",
            "strong_candidate",
            "top_candidate",
        ],
        include_lowest=True,
    )

    return scores


def add_interpretation(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Add a compact plain-English interpretation for each security.
    """
    scores = scores.copy()

    def format_percent(value: float | int | None) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{value * 100:.1f}%"

    def interpret_row(row: pd.Series) -> str:
        return (
            f"{row['primary_signal']}: "
            f"{format_percent(row['below_13w_high'])} below 13w high, "
            f"{format_percent(row['distance_from_ma_200d'])} vs 200d MA, "
            f"{format_percent(row['return_13w'])} 13w return, "
            f"{format_percent(row['volatility_60d'])} 60d vol."
        )

    scores["interpretation"] = scores.apply(interpret_row, axis=1)

    return scores


def calculate_price_scores(
    factors: pd.DataFrame,
    score_config: Dict[str, Any],
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calculate interpretable price-based scores.
    """
    scores = factors.copy()

    validate_score_inputs(scores)

    weights = score_config.get(
        "score_weights",
        {
            "dip_score": 0.35,
            "trend_score": 0.30,
            "momentum_score": 0.20,
            "risk_score": 0.15,
        },
    )

    scores["dip_score"] = calculate_dip_score(scores)
    scores["trend_score"] = calculate_trend_score(scores)
    scores["momentum_score"] = calculate_momentum_score(scores)
    scores["risk_score"] = calculate_risk_score(scores)

    scores["timing_score"] = calculate_weighted_timing_score(
        factors=scores,
        weights=weights,
    )

    scores = add_signal_flags(scores, score_config=score_config)
    scores = add_primary_signal(scores)
    scores = calculate_risk_adjusted_score(scores, score_config=score_config)
    scores = add_rank_and_buckets(scores)
    scores = add_interpretation(scores)

    if universe is not None:
        scores = merge_universe_metadata(
            scores=scores,
            universe=universe,
        )

    output_columns = [
    "risk_adjusted_rank",
    "score_rank",
    "ticker",
    "company_name",
    "asset_type",
    "universe_name",
    "source_type",
    "source_name",
    "strategy_role",
    "account_target",
    "target_weight",
    "sector",
    "industry",
    "latest_date",
    "adjusted_close",
    "risk_adjusted_timing_score",
    "timing_score",
    "score_bucket",
    "primary_signal",
    "dip_flag",
    "trend_flag",
    "momentum_flag",
    "risk_flag",
    "risk_penalty",
    "dip_score",
    "trend_score",
    "momentum_score",
    "risk_score",
    "below_13w_high",
    "below_52w_high",
    "distance_from_ma_200d",
    "return_13w",
    "volatility_60d",
    "interpretation",
]

    available_output_columns = [
        column for column in output_columns if column in scores.columns
    ]

    scores = scores[available_output_columns].sort_values(
        ["risk_adjusted_rank", "ticker"]
    )

    return scores


def save_price_scores(
    scores: pd.DataFrame,
    output_path: str | Path,
    export_settings: Dict[str, Any] | None = None,
) -> None:
    """
    Save latest price scores.

    Saves the canonical standard CSV and optionally an Excel-friendly CSV.
    """
    save_csv_outputs(
        df=scores,
        output_path=output_path,
        export_settings=export_settings,
    )

    print(f"Rows written: {len(scores)}")


def build_price_scores(config_path: str | Path) -> pd.DataFrame:
    """
    Build latest price-based scores from latest price factors
    and enrich them with universe metadata.
    """
    full_config = load_config(config_path)

    if "price_scores" not in full_config:
        raise ValueError("Missing 'price_scores' section in config file.")

    score_config = full_config["price_scores"]

    if not score_config.get("enabled", False):
        raise ValueError("price_scores is disabled in the config file.")

    export_settings = load_export_settings(full_config)

    input_path = score_config["input_latest_factors_path"]
    universe_path = score_config.get("input_universe_path")
    output_path = score_config["output_scores_path"]

    factors = load_latest_price_factors(input_path)

    universe = None

    if universe_path is not None:
        universe = load_current_universe(universe_path)

    scores = calculate_price_scores(
        factors=factors,
        score_config=score_config,
        universe=universe,
    )

    save_price_scores(
        scores=scores,
        output_path=output_path,
        export_settings=export_settings,
    )

    return scores


if __name__ == "__main__":
    latest_scores = build_price_scores("config/universe_config.yaml")

    print("\nTop risk-adjusted scoring securities:")
    preview_columns = [
    "risk_adjusted_rank",
    "ticker",
    "company_name",
    "sector",
    "industry",
    "universe_name",
    "latest_date",
    "risk_adjusted_timing_score",
    "timing_score",
    "score_bucket",
    "primary_signal",
    "risk_flag",
    "risk_penalty",
    "interpretation",
]

    available_preview_columns = [
        column for column in preview_columns if column in latest_scores.columns
    ]

    print(latest_scores[available_preview_columns].head(25))