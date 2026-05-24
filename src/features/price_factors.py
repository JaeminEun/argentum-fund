from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.universe.config import load_config


REQUIRED_PRICE_COLUMNS = [
    "ticker",
    "date",
    "adjusted_close",
]


def load_price_factor_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load the price_factors section from the project configuration file.
    """
    config = load_config(config_path)

    if "price_factors" not in config:
        raise ValueError("Missing 'price_factors' section in config file.")

    factor_config = config["price_factors"]

    if not factor_config.get("enabled", False):
        raise ValueError("price_factors is disabled in the config file.")

    return factor_config


def load_price_history(price_path: str | Path) -> pd.DataFrame:
    """
    Load standardized price history created by the price history builder.
    """
    price_path = Path(price_path)

    if not price_path.exists():
        raise FileNotFoundError(
            f"Price history file not found: {price_path}. "
            "Run the price history builder first."
        )

    prices = pd.read_csv(price_path)

    return prices


def validate_price_history(
    prices: pd.DataFrame,
    price_column: str,
) -> None:
    """
    Validate that the price history contains the required columns.
    """
    required_columns = set(REQUIRED_PRICE_COLUMNS + [price_column])
    missing_columns = required_columns - set(prices.columns)

    if missing_columns:
        raise ValueError(
            f"Price history is missing required columns: {missing_columns}"
        )

    if prices.empty:
        raise ValueError("Price history is empty.")

    if prices["ticker"].isna().any():
        raise ValueError("Price history contains missing tickers.")

    if prices["date"].isna().any():
        raise ValueError("Price history contains missing dates.")

    if prices[price_column].isna().all():
        raise ValueError(
            f"Price column '{price_column}' contains only missing values."
        )


def prepare_price_history(
    prices: pd.DataFrame,
    price_column: str,
) -> pd.DataFrame:
    """
    Clean and prepare price history for factor calculation.
    """
    prices = prices.copy()

    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    prices["date"] = pd.to_datetime(prices["date"])
    prices[price_column] = pd.to_numeric(prices[price_column], errors="coerce")

    prices = prices.dropna(subset=["ticker", "date", price_column])

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    return prices


def calculate_return_factors(
    prices: pd.DataFrame,
    price_column: str,
    return_windows: Dict[str, int],
) -> pd.DataFrame:
    """
    Calculate return factors for configured lookback windows.
    """
    prices = prices.copy()

    grouped_prices = prices.groupby("ticker")[price_column]

    for label, window in return_windows.items():
        column_name = f"return_{label}"
        prices[column_name] = grouped_prices.pct_change(periods=window)

    return prices


def calculate_high_factors(
    prices: pd.DataFrame,
    price_column: str,
    high_windows: Dict[str, int],
) -> pd.DataFrame:
    """
    Calculate rolling highs and distance below those highs.
    """
    prices = prices.copy()

    grouped_prices = prices.groupby("ticker")[price_column]

    for label, window in high_windows.items():
        high_column = f"high_{label}"
        below_high_column = f"below_{label}_high"

        prices[high_column] = grouped_prices.transform(
            lambda series: series.rolling(window=window, min_periods=1).max()
        )

        prices[below_high_column] = (
            prices[price_column] / prices[high_column] - 1
        )

    return prices


def calculate_moving_average_factors(
    prices: pd.DataFrame,
    price_column: str,
    moving_average_windows: list[int],
) -> pd.DataFrame:
    """
    Calculate moving averages and distance from moving averages.
    """
    prices = prices.copy()

    grouped_prices = prices.groupby("ticker")[price_column]

    for window in moving_average_windows:
        ma_column = f"ma_{window}d"
        distance_column = f"distance_from_ma_{window}d"

        prices[ma_column] = grouped_prices.transform(
            lambda series: series.rolling(window=window, min_periods=1).mean()
        )

        prices[distance_column] = prices[price_column] / prices[ma_column] - 1

    return prices


def calculate_volatility_factors(
    prices: pd.DataFrame,
    price_column: str,
    volatility_windows: list[int],
    trading_days_per_year: int,
) -> pd.DataFrame:
    """
    Calculate annualized rolling volatility based on daily returns.
    """
    prices = prices.copy()

    prices["daily_return"] = (
        prices.groupby("ticker")[price_column]
        .pct_change()
    )

    grouped_returns = prices.groupby("ticker")["daily_return"]

    for window in volatility_windows:
        volatility_column = f"volatility_{window}d"

        prices[volatility_column] = grouped_returns.transform(
            lambda series: (
                series.rolling(window=window, min_periods=2).std()
                * np.sqrt(trading_days_per_year)
            )
        )

    return prices


def calculate_drawdown_factors(
    prices: pd.DataFrame,
    price_column: str,
) -> pd.DataFrame:
    """
    Calculate drawdown from each ticker's running all-time high
    within the available price history.
    """
    prices = prices.copy()

    prices["running_high"] = (
        prices.groupby("ticker")[price_column]
        .cummax()
    )

    prices["drawdown_from_running_high"] = (
        prices[price_column] / prices["running_high"] - 1
    )

    return prices


def calculate_price_factors(
    prices: pd.DataFrame,
    factor_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate all configured price-based factors.
    """
    price_column = factor_config.get("price_column", "adjusted_close")
    trading_days_per_year = factor_config.get("trading_days_per_year", 252)

    return_windows = factor_config.get(
        "return_windows",
        {
            "1w": 5,
            "4w": 20,
            "13w": 63,
            "26w": 126,
            "52w": 252,
        },
    )

    high_windows = factor_config.get(
        "high_windows",
        {
            "13w": 63,
            "52w": 252,
        },
    )

    moving_average_windows = factor_config.get(
        "moving_average_windows",
        [20, 50, 200],
    )

    volatility_windows = factor_config.get(
        "volatility_windows",
        [20, 60],
    )

    validate_price_history(prices, price_column=price_column)

    prices = prepare_price_history(prices, price_column=price_column)

    prices = calculate_return_factors(
        prices=prices,
        price_column=price_column,
        return_windows=return_windows,
    )

    prices = calculate_high_factors(
        prices=prices,
        price_column=price_column,
        high_windows=high_windows,
    )

    prices = calculate_moving_average_factors(
        prices=prices,
        price_column=price_column,
        moving_average_windows=moving_average_windows,
    )

    prices = calculate_volatility_factors(
        prices=prices,
        price_column=price_column,
        volatility_windows=volatility_windows,
        trading_days_per_year=trading_days_per_year,
    )

    prices = calculate_drawdown_factors(
        prices=prices,
        price_column=price_column,
    )

    return prices


def create_latest_snapshot(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Create a latest available factor snapshot for each ticker.
    """
    factors = factors.copy()
    factors["date"] = pd.to_datetime(factors["date"])

    latest = (
        factors.sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    latest = latest.rename(columns={"date": "latest_date"})

    return latest


def save_factor_outputs(
    factors: pd.DataFrame,
    latest: pd.DataFrame,
    history_output_path: str | Path,
    latest_output_path: str | Path,
) -> None:
    """
    Save historical and latest factor outputs.
    """
    history_output_path = Path(history_output_path)
    latest_output_path = Path(latest_output_path)

    history_output_path.parent.mkdir(parents=True, exist_ok=True)
    latest_output_path.parent.mkdir(parents=True, exist_ok=True)

    factors.to_csv(history_output_path, index=False)
    latest.to_csv(latest_output_path, index=False)

    print(f"Saved historical price factors to {history_output_path}")
    print(f"Historical rows written: {len(factors)}")

    print(f"Saved latest price factors to {latest_output_path}")
    print(f"Latest rows written: {len(latest)}")


def build_price_factors(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build price-based factor outputs from historical price data.
    """
    factor_config = load_price_factor_config(config_path)

    input_price_path = factor_config["input_price_path"]
    output_history_path = factor_config["output_history_path"]
    output_latest_path = factor_config["output_latest_path"]

    prices = load_price_history(input_price_path)

    factors = calculate_price_factors(
        prices=prices,
        factor_config=factor_config,
    )

    latest = create_latest_snapshot(factors)

    save_factor_outputs(
        factors=factors,
        latest=latest,
        history_output_path=output_history_path,
        latest_output_path=output_latest_path,
    )

    return factors, latest


if __name__ == "__main__":
    factors_history, latest_factors = build_price_factors(
        "config/universe_config.yaml"
    )

    print("\nLatest factor preview:")
    preview_columns = [
        "ticker",
        "latest_date",
        "adjusted_close",
        "return_4w",
        "return_13w",
        "below_13w_high",
        "below_52w_high",
        "distance_from_ma_200d",
        "volatility_60d",
        "drawdown_from_running_high",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in latest_factors.columns
    ]

    print(latest_factors[available_preview_columns].head(20))
