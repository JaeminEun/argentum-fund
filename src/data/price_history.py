from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yfinance as yf

from src.universe.config import load_config


STANDARD_PRICE_COLUMNS = [
    "ticker",
    "price_ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "source",
]


def load_price_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load the price_history section from the project configuration file.
    """
    config = load_config(config_path)

    if "price_history" not in config:
        raise ValueError("Missing 'price_history' section in config file.")

    price_config = config["price_history"]

    if not price_config.get("enabled", False):
        raise ValueError("price_history is disabled in the config file.")

    return price_config


def load_current_universe(universe_path: str | Path) -> pd.DataFrame:
    """
    Load the standardized universe file created by the universe builder.
    """
    universe_path = Path(universe_path)

    if not universe_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}. "
            "Run the universe builder first."
        )

    return pd.read_csv(universe_path)


def get_tradable_tickers(
    universe: pd.DataFrame,
    tradable_asset_types: Iterable[str],
) -> List[str]:
    """
    Extract unique tradable tickers from the current universe.

    Non-market identifiers like CASH_USD, BURRY_AUTOPILOT, and
    SIMONS_AUTOPILOT should be excluded by asset_type.
    """
    required_columns = {"ticker", "asset_type"}

    missing_columns = required_columns - set(universe.columns)

    if missing_columns:
        raise ValueError(
            f"Universe file is missing required columns: {missing_columns}"
        )

    tradable_asset_types = {asset.lower() for asset in tradable_asset_types}

    universe = universe.copy()
    universe["asset_type"] = universe["asset_type"].astype(str).str.lower()
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()

    tradable = universe[
        universe["asset_type"].isin(tradable_asset_types)
    ].copy()

    tickers = sorted(tradable["ticker"].dropna().unique().tolist())

    # Defensive filtering for internal identifiers.
    excluded_prefixes = ("CASH_",)
    excluded_exact = {
        "BURRY_AUTOPILOT",
        "SIMONS_AUTOPILOT",
    }

    tickers = [
        ticker
        for ticker in tickers
        if ticker
        and not ticker.startswith(excluded_prefixes)
        and ticker not in excluded_exact
    ]

    return tickers

def normalize_yfinance_prices(
    prices: pd.DataFrame,
    ticker: str,
    source: str = "yfinance",
) -> pd.DataFrame:
    """
    Convert yfinance output into the standard price history schema.

    Handles:
    - regular single-level columns
    - yfinance MultiIndex columns
    - date stored as Date, Datetime, or index
    """
    if prices.empty:
        return pd.DataFrame(columns=STANDARD_PRICE_COLUMNS)

    prices = prices.copy()
    prices = prices.reset_index()

    # yfinance may return MultiIndex columns like:
    # ("Adj Close", "PLTD"), ("Close", "PLTD"), etc.
    # After reset_index(), the date column may appear as ("index", "").
    if isinstance(prices.columns, pd.MultiIndex):
        flattened_columns = []

        for col in prices.columns:
            # col is usually a tuple, e.g. ("Adj Close", "PLTD")
            first_level = str(col[0]).strip()

            flattened_columns.append(first_level)

        prices.columns = flattened_columns

    # Normalize column names.
    prices.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in prices.columns
    ]

    # yfinance may return the date column under different names.
    if "date" in prices.columns:
        date_column = "date"
    elif "datetime" in prices.columns:
        date_column = "datetime"
    elif "index" in prices.columns:
        date_column = "index"
    else:
        raise ValueError(
            f"No date column found for {ticker}. "
            f"Available columns: {list(prices.columns)}"
        )

    rename_map = {
        date_column: "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adjusted_close",
        "adjusted_close": "adjusted_close",
        "volume": "volume",
    }

    prices = prices.rename(columns=rename_map)

    required_columns = ["date", "open", "high", "low", "close", "volume"]

    missing_columns = [
        column for column in required_columns if column not in prices.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected price columns for {ticker}: {missing_columns}. "
            f"Available columns: {list(prices.columns)}"
        )

    if "adjusted_close" not in prices.columns:
        prices["adjusted_close"] = prices["close"]

    prices["ticker"] = ticker
    prices["price_ticker"] = ticker
    prices["source"] = source

    prices["date"] = pd.to_datetime(prices["date"]).dt.date

    prices = prices[STANDARD_PRICE_COLUMNS]

    return prices

def map_ticker_for_provider(ticker: str, provider: str = "yfinance") -> str:
    """
    Map internal ticker symbols to provider-specific ticker symbols.

    Some providers use different conventions for share classes.
    Example:
        BF.B -> BF-B on Yahoo Finance
        BRK.B -> BRK-B on Yahoo Finance
    """
    ticker = str(ticker).strip().upper()

    if provider == "yfinance":
        manual_map = {
            "BF.B": "BF-B",
            "BRK.B": "BRK-B",
        }

        if ticker in manual_map:
            return manual_map[ticker]

        # General Yahoo Finance convention for U.S. share classes.
        if "." in ticker:
            return ticker.replace(".", "-")

    return ticker

def download_price_history_for_ticker(
    ticker: str,
    start_date: str,
    end_date: str | None,
    interval: str,
    auto_adjust: bool,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """
    Download historical price data for one ticker from yfinance.

    Uses provider-specific ticker mapping while preserving the original ticker
    in the standardized output.
    """
    price_ticker = map_ticker_for_provider(ticker, provider=provider)

    try:
        prices = yf.download(
            tickers=price_ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=False,
        )

        frame = normalize_yfinance_prices(prices, ticker=ticker)

        if not frame.empty:
            frame["price_ticker"] = price_ticker

        return frame

    except Exception as error:
        print(
            f"Warning: failed to download prices for {ticker} "
            f"using {price_ticker}: {error}"
        )
        return pd.DataFrame(columns=STANDARD_PRICE_COLUMNS)


def build_price_history(config_path: str | Path) -> pd.DataFrame:
    """
    Build standardized historical price data for all tradable tickers
    in the current universe.
    """
    price_config = load_price_config(config_path)

    universe_path = price_config["input_universe_path"]
    output_path = Path(price_config["output_path"])
    start_date = price_config.get("start_date", "2020-01-01")
    end_date = price_config.get("end_date")
    interval = price_config.get("interval", "1d")
    auto_adjust = price_config.get("auto_adjust", False)
    tradable_asset_types = price_config.get(
        "tradable_asset_types",
        ["stock", "etf", "fund"],
    )

    universe = load_current_universe(universe_path)

    tickers = get_tradable_tickers(
        universe=universe,
        tradable_asset_types=tradable_asset_types,
    )

    if not tickers:
        raise ValueError("No tradable tickers found in current universe.")

    print(f"Found {len(tickers)} tradable tickers.")
    print(f"Downloading price history from {start_date}...")

    frames = []

    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{len(tickers)}] Downloading {ticker}")

        frame = download_price_history_for_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            auto_adjust=auto_adjust,
        )

        if not frame.empty:
            frames.append(frame)
        else:
            print(f"Warning: no price data returned for {ticker}")

    if not frames:
        raise ValueError("No price history was downloaded.")

    price_history = pd.concat(frames, ignore_index=True)

    price_history = price_history.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    price_history.to_csv(output_path, index=False)

    print(f"Saved price history to {output_path}")
    print(f"Rows written: {len(price_history)}")

    return price_history


if __name__ == "__main__":
    prices = build_price_history("config/universe_config.yaml")

    print("\nPrice history preview:")
    print(prices.head())
