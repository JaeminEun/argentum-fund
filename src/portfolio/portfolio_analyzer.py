from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.universe.config import load_config
from src.utils.io import load_export_settings, save_csv_outputs


REQUIRED_HOLDINGS_COLUMNS = [
    "account",
    "ticker",
    "shares",
    "average_cost",
    "strategy",
    "active",
]


def load_portfolio_analysis_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load full project config and validate portfolio_analysis block.
    """
    config = load_config(config_path)

    if "portfolio_analysis" not in config:
        raise ValueError("Missing 'portfolio_analysis' section in config file.")

    if not config["portfolio_analysis"].get("enabled", False):
        raise ValueError("portfolio_analysis is disabled in config file.")

    return config


def load_csv(
    path: str | Path,
    label: str,
    delimiter: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """
    Load a CSV file with a useful error message.

    Parameters
    ----------
    path:
        CSV file path.
    label:
        Human-readable label for error messages.
    delimiter:
        Column delimiter. Use "," for standard CSV or ";" for many
        international Excel exports.
    decimal:
        Decimal marker. Use "." for standard CSV or "," for many
        international Excel exports.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    try:
        return pd.read_csv(path, sep=delimiter, decimal=decimal)
    except Exception as error:
        raise ValueError(
            f"Failed to read {label} CSV: {path}\n"
            f"delimiter={delimiter!r}, decimal={decimal!r}\n"
            f"Original error: {error}"
        ) from error


def validate_holdings(holdings: pd.DataFrame) -> None:
    """
    Validate required columns in holdings file.
    """
    missing = set(REQUIRED_HOLDINGS_COLUMNS) - set(holdings.columns)

    if missing:
        raise ValueError(
            f"Portfolio holdings file missing columns: {missing}. "
            "If the columns look merged into one column, check "
            "portfolio_analysis.holdings_delimiter in the YAML config."
    )

    if holdings.empty:
        raise ValueError("Portfolio holdings file is empty.")


def normalize_active(value: object) -> bool:
    """
    Normalize active flag values.
    """
    if pd.isna(value):
        return True

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def prepare_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Clean holdings file.
    """
    holdings = holdings.copy()

    validate_holdings(holdings)

    holdings["account"] = holdings["account"].astype(str).str.strip()
    holdings["ticker"] = holdings["ticker"].astype(str).str.upper().str.strip()
    holdings["strategy"] = holdings["strategy"].astype(str).str.strip()
    holdings["active"] = holdings["active"].apply(normalize_active)

    holdings["shares"] = pd.to_numeric(holdings["shares"], errors="coerce")
    holdings["average_cost"] = pd.to_numeric(
        holdings["average_cost"],
        errors="coerce",
    )

    if "current_value_override" not in holdings.columns:
        holdings["current_value_override"] = np.nan

    holdings["current_value_override"] = pd.to_numeric(
        holdings["current_value_override"],
        errors="coerce",
    )

    if "notes" not in holdings.columns:
        holdings["notes"] = ""

    holdings = holdings[holdings["active"] == True].copy()

    holdings = holdings.dropna(subset=["ticker", "shares", "average_cost"])

    return holdings.reset_index(drop=True)


def get_latest_prices(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Get latest adjusted close for each ticker.
    """
    prices = price_history.copy()

    required = {"ticker", "date", "adjusted_close"}

    missing = required - set(prices.columns)

    if missing:
        raise ValueError(f"Price history missing columns: {missing}")

    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["adjusted_close"] = pd.to_numeric(
        prices["adjusted_close"],
        errors="coerce",
    )

    latest = (
        prices.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    latest = latest.rename(
        columns={
            "date": "latest_price_date",
            "adjusted_close": "latest_price",
        }
    )

    return latest[["ticker", "latest_price_date", "latest_price"]]


def prepare_composite_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare composite scores for merging into holdings.
    """
    scores = scores.copy()

    scores["ticker"] = scores["ticker"].astype(str).str.upper().str.strip()

    keep_columns = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "universe_name",
        "composite_rank",
        "composite_research_score",
        "composite_bucket",
        "composite_signal",
        "price_score",
        "price_rank",
        "price_signal",
        "fundamental_score",
        "fundamental_rank",
        "fundamental_signal",
        "composite_interpretation",
    ]

    available = [column for column in keep_columns if column in scores.columns]

    return scores[available]


def calculate_position_values(
    holdings: pd.DataFrame,
    latest_prices: pd.DataFrame,
    composite_scores: pd.DataFrame,
    portfolio_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Merge holdings with latest prices and scores, then calculate value.
    """
    positions = holdings.copy()

    positions = positions.merge(
        latest_prices,
        on="ticker",
        how="left",
    )

    positions = positions.merge(
        prepare_composite_scores(composite_scores),
        on="ticker",
        how="left",
    )

    cash_tickers = {
        str(ticker).upper().strip()
        for ticker in portfolio_config.get("cash_tickers", ["CASH_USD"])
    }

    synthetic_tickers = {
        str(ticker).upper().strip()
        for ticker in portfolio_config.get("synthetic_strategy_tickers", [])
    }

    positions["is_cash"] = positions["ticker"].isin(cash_tickers)
    positions["is_synthetic_strategy"] = positions["ticker"].isin(synthetic_tickers)

    positions["latest_price_used"] = positions["latest_price"]

    positions.loc[positions["is_cash"], "latest_price_used"] = 1.0

    positions["market_value"] = positions["shares"] * positions["latest_price_used"]

    override_mask = positions["current_value_override"].notna()

    positions.loc[override_mask, "market_value"] = positions.loc[
        override_mask,
        "current_value_override",
    ]

    positions["cost_basis"] = positions["shares"] * positions["average_cost"]

    positions["unrealized_gain"] = (
        positions["market_value"] - positions["cost_basis"]
    )

    positions["unrealized_return"] = positions["unrealized_gain"] / positions[
        "cost_basis"
    ]

    positions["unrealized_return"] = positions["unrealized_return"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    total_value = positions["market_value"].sum()

    positions["portfolio_weight"] = positions["market_value"] / total_value

    account_totals = positions.groupby("account")["market_value"].transform("sum")
    positions["account_weight"] = positions["market_value"] / account_totals

    return positions


def add_holding_flags(
    positions: pd.DataFrame,
    portfolio_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Add simple analysis flags for current holdings.
    """
    positions = positions.copy()

    thresholds = portfolio_config.get("score_thresholds", {})

    strong_score = float(thresholds.get("strong_composite_score", 65))
    weak_score = float(thresholds.get("weak_composite_score", 40))

    positions["holding_analysis_flag"] = "review_holding"

    positions.loc[
        positions["is_cash"] == True,
        "holding_analysis_flag",
    ] = "cash_position"

    positions.loc[
        positions["is_synthetic_strategy"] == True,
        "holding_analysis_flag",
    ] = "synthetic_strategy_sleeve"

    positions.loc[
        positions["composite_research_score"].isna()
        & (positions["is_cash"] == False)
        & (positions["is_synthetic_strategy"] == False),
        "holding_analysis_flag",
    ] = "missing_model_score"

    positions.loc[
        positions["composite_research_score"] >= strong_score,
        "holding_analysis_flag",
    ] = "strong_holding"

    positions.loc[
        positions["composite_research_score"] < weak_score,
        "holding_analysis_flag",
    ] = "weak_holding"

    return positions


def summarize_portfolio(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple portfolio summary table.
    """
    total_market_value = positions["market_value"].sum()
    total_cost_basis = positions["cost_basis"].sum()
    total_unrealized_gain = positions["unrealized_gain"].sum()

    cash_value = positions.loc[positions["is_cash"], "market_value"].sum()
    invested_value = total_market_value - cash_value

    summary = pd.DataFrame(
        [
            {
                "metric": "total_market_value",
                "value": total_market_value,
            },
            {
                "metric": "total_cost_basis",
                "value": total_cost_basis,
            },
            {
                "metric": "total_unrealized_gain",
                "value": total_unrealized_gain,
            },
            {
                "metric": "total_unrealized_return",
                "value": (
                    total_unrealized_gain / total_cost_basis
                    if total_cost_basis != 0
                    else np.nan
                ),
            },
            {
                "metric": "cash_value",
                "value": cash_value,
            },
            {
                "metric": "invested_value",
                "value": invested_value,
            },
            {
                "metric": "cash_weight",
                "value": (
                    cash_value / total_market_value
                    if total_market_value != 0
                    else np.nan
                ),
            },
        ]
    )

    return summary


def summarize_sector_exposure(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize market value by sector.
    """
    positions = positions.copy()

    positions["sector"] = positions["sector"].fillna("Unknown")

    positions.loc[positions["is_cash"], "sector"] = "Cash"
    positions.loc[positions["is_synthetic_strategy"], "sector"] = "Strategy Sleeve"

    sector = (
        positions.groupby("sector", as_index=False)
        .agg(
            market_value=("market_value", "sum"),
            cost_basis=("cost_basis", "sum"),
            unrealized_gain=("unrealized_gain", "sum"),
            holding_count=("ticker", "nunique"),
        )
    )

    total_value = sector["market_value"].sum()
    sector["portfolio_weight"] = sector["market_value"] / total_value

    return sector.sort_values("market_value", ascending=False).reset_index(drop=True)


def summarize_strategy_exposure(positions: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize market value and performance by strategy label.
    """
    strategy = (
        positions.groupby("strategy", as_index=False)
        .agg(
            market_value=("market_value", "sum"),
            cost_basis=("cost_basis", "sum"),
            unrealized_gain=("unrealized_gain", "sum"),
            holding_count=("ticker", "nunique"),
            avg_composite_score=("composite_research_score", "mean"),
            avg_price_score=("price_score", "mean"),
            avg_fundamental_score=("fundamental_score", "mean"),
        )
    )

    strategy["unrealized_return"] = strategy["unrealized_gain"] / strategy[
        "cost_basis"
    ]

    strategy["unrealized_return"] = strategy["unrealized_return"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    total_value = strategy["market_value"].sum()
    strategy["portfolio_weight"] = strategy["market_value"] / total_value

    return strategy.sort_values("market_value", ascending=False).reset_index(drop=True)


def build_autopilot_lookthrough(
    current_universe: pd.DataFrame,
    composite_scores: pd.DataFrame,
    portfolio_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Analyze individual securities inside configured autopilot universes.

    This is the level-2 analysis:
    it evaluates the holdings inside Burry/Simons style universes against
    the broader composite scoring system.
    """
    universe = current_universe.copy()
    scores = prepare_composite_scores(composite_scores)

    if "universe_name" not in universe.columns:
        return pd.DataFrame()

    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    universe["universe_name"] = universe["universe_name"].astype(str).str.strip()

    autopilot_universes = {
        str(name).strip()
        for name in portfolio_config.get(
            "autopilot_universes",
            ["burry_autopilot", "simons_autopilot"],
        )
    }

    lookthrough = universe[universe["universe_name"].isin(autopilot_universes)].copy()

    if lookthrough.empty:
        return pd.DataFrame()

    metadata_columns = [
        "ticker",
        "universe_name",
        "company_name",
        "asset_type",
        "sector",
        "industry",
        "target_weight",
        "strategy_role",
        "account_target",
    ]

    available_metadata = [
        column for column in metadata_columns if column in lookthrough.columns
    ]

    lookthrough = lookthrough[available_metadata].copy()

    lookthrough = lookthrough.merge(
        scores,
        on="ticker",
        how="left",
        suffixes=("", "_score"),
    )

    lookthrough["lookthrough_flag"] = "review"

    lookthrough.loc[
        lookthrough["composite_signal"].eq("aligned_candidate"),
        "lookthrough_flag",
    ] = "strong_internal_candidate"

    lookthrough.loc[
        lookthrough["composite_signal"].eq("possible_value_trap"),
        "lookthrough_flag",
    ] = "fundamental_caution"

    lookthrough.loc[
        lookthrough["composite_signal"].eq("quality_watchlist_wait_for_timing"),
        "lookthrough_flag",
    ] = "quality_wait_for_timing"

    lookthrough.loc[
        lookthrough["composite_research_score"].isna(),
        "lookthrough_flag",
    ] = "missing_composite_score"

    output_columns = [
        "universe_name",
        "ticker",
        "company_name",
        "sector",
        "industry",
        "target_weight",
        "composite_rank",
        "composite_research_score",
        "composite_signal",
        "price_score",
        "price_signal",
        "fundamental_score",
        "fundamental_signal",
        "lookthrough_flag",
        "composite_interpretation",
    ]

    available_output = [
        column for column in output_columns if column in lookthrough.columns
    ]

    return lookthrough[available_output].sort_values(
        ["universe_name", "composite_rank", "ticker"],
        ascending=[True, True, True],
    )


def save_portfolio_outputs(
    positions: pd.DataFrame,
    summary: pd.DataFrame,
    sector_exposure: pd.DataFrame,
    strategy_exposure: pd.DataFrame,
    autopilot_lookthrough: pd.DataFrame,
    portfolio_config: Dict[str, Any],
    export_settings: Dict[str, Any] | None,
) -> None:
    """
    Save all portfolio analysis outputs.
    """
    save_csv_outputs(
        positions,
        portfolio_config["output_positions_path"],
        export_settings,
    )

    save_csv_outputs(
        summary,
        portfolio_config["output_summary_path"],
        export_settings,
    )

    save_csv_outputs(
        sector_exposure,
        portfolio_config["output_sector_exposure_path"],
        export_settings,
    )

    save_csv_outputs(
        strategy_exposure,
        portfolio_config["output_strategy_exposure_path"],
        export_settings,
    )

    if not autopilot_lookthrough.empty:
        save_csv_outputs(
            autopilot_lookthrough,
            portfolio_config["output_autopilot_lookthrough_path"],
            export_settings,
        )


def run_portfolio_analysis(config_path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Run portfolio analyzer from project config.
    """
    full_config = load_portfolio_analysis_config(config_path)

    portfolio_config = full_config["portfolio_analysis"]
    export_settings = load_export_settings(full_config)

    holdings = load_csv(
        portfolio_config["input_holdings_path"],
        "Portfolio holdings",
        delimiter=portfolio_config.get("holdings_delimiter", ","),
        decimal=portfolio_config.get("holdings_decimal", "."),
)

    price_history = load_csv(
        portfolio_config["input_price_history_path"],
        "Price history",
    )

    composite_scores = load_csv(
        portfolio_config["input_composite_scores_path"],
        "Composite scores",
    )

    current_universe = load_csv(
        portfolio_config["input_current_universe_path"],
        "Current universe",
    )

    holdings = prepare_holdings(holdings)
    latest_prices = get_latest_prices(price_history)

    positions = calculate_position_values(
        holdings=holdings,
        latest_prices=latest_prices,
        composite_scores=composite_scores,
        portfolio_config=portfolio_config,
    )

    positions = add_holding_flags(
        positions=positions,
        portfolio_config=portfolio_config,
    )

    summary = summarize_portfolio(positions)
    sector_exposure = summarize_sector_exposure(positions)
    strategy_exposure = summarize_strategy_exposure(positions)

    autopilot_lookthrough = build_autopilot_lookthrough(
        current_universe=current_universe,
        composite_scores=composite_scores,
        portfolio_config=portfolio_config,
    )

    save_portfolio_outputs(
        positions=positions,
        summary=summary,
        sector_exposure=sector_exposure,
        strategy_exposure=strategy_exposure,
        autopilot_lookthrough=autopilot_lookthrough,
        portfolio_config=portfolio_config,
        export_settings=export_settings,
    )

    return {
        "positions": positions,
        "summary": summary,
        "sector_exposure": sector_exposure,
        "strategy_exposure": strategy_exposure,
        "autopilot_lookthrough": autopilot_lookthrough,
    }


if __name__ == "__main__":
    outputs = run_portfolio_analysis("config/universe_config.yaml")

    positions = outputs["positions"]
    summary = outputs["summary"]
    strategy = outputs["strategy_exposure"]
    lookthrough = outputs["autopilot_lookthrough"]

    print("\nPortfolio positions preview:")
    preview_columns = [
        "account",
        "ticker",
        "strategy",
        "market_value",
        "unrealized_gain",
        "unrealized_return",
        "portfolio_weight",
        "composite_research_score",
        "holding_analysis_flag",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in positions.columns
    ]

    print(positions[available_preview_columns].head(30))

    print("\nPortfolio summary:")
    print(summary)

    print("\nStrategy exposure:")
    print(strategy)

    if not lookthrough.empty:
        print("\nAutopilot lookthrough preview:")
        print(lookthrough.head(30))
