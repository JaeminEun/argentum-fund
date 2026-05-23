from typing import Iterable

import pandas as pd


STANDARD_COLUMNS = [
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
    "cusip",
    "cik",
    "date_added",
    "as_of_date",
    "notes",
    "active",
]


REQUIRED_COLUMNS = [
    "ticker",
    "universe_name",
    "source_type",
    "strategy_role",
    "account_target",
    "active",
]


def normalize_ticker(value: object) -> str:
    """
    Normalize ticker symbols to uppercase strings.

    Handles common whitespace issues and placeholder values.
    """
    if pd.isna(value):
        return ""

    ticker = str(value).strip().upper()

    placeholder_values = {
        "",
        "-",
        "--",
        "N/A",
        "NA",
        "NONE",
        "NULL",
        "NAN",
    }

    if ticker in placeholder_values:
        return ""

    return ticker

def normalize_active(value: object) -> bool:
    """
    Convert common active/inactive labels to boolean.
    """
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return True

    text = str(value).strip().lower()

    if text in {"true", "yes", "y", "1", "active"}:
        return True

    if text in {"false", "no", "n", "0", "inactive"}:
        return False

    raise ValueError(f"Could not parse active value: {value}")

def parse_weight(value: object) -> float | None:
    """
    Parse portfolio weights into decimal form.

    Supports:
    - 0.10
    - 0,10
    - 10%
    - 10,5%
    - blank/null values

    Returns
    -------
    float | None
        Decimal weight, e.g. 10% becomes 0.10.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        # If user enters 10, assume 10%.
        # If user enters 0.10, assume decimal weight.
        if value > 1:
            return float(value) / 100
        return float(value)

    text = str(value).strip()

    if text == "":
        return None

    # Normalize international decimal comma.
    text = text.replace(",", ".")

    if text.endswith("%"):
        number = text.replace("%", "").strip()
        return float(number) / 100

    number = float(text)

    # Interpret values greater than 1 as percentages.
    # Example: 10 becomes 0.10.
    if number > 1:
        return number / 100

    return number

def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Ensure a DataFrame has all expected columns.
    Missing columns are added with blank values.
    """
    df = df.copy()

    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA

    return df

def map_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common source-specific weight columns to target_weight.

    This supports files where weights are stored under names such as:
    - target_weight
    - weight_decimal
    - weight_percent
    - weight

    Notes
    -----
    Some index files store weight_percent as a scaled integer.
    Example:
        7765389 = 7.765389% = 0.07765389 decimal weight
    """
    df = df.copy()

    # If target_weight already exists and has values, keep it.
    if "target_weight" in df.columns and df["target_weight"].notna().any():
        return df

    if "weight_decimal" in df.columns:
        df["target_weight"] = df["weight_decimal"]
        return df

    if "weight_percent" in df.columns:
        # Handles the State Street-style scaled integer format.
        # Example: 7765389 -> 0.07765389
        df["target_weight"] = pd.to_numeric(
            df["weight_percent"], errors="coerce"
        ) / 100_000_000
        return df

    if "weight" in df.columns:
        # Fallback for files where weight is even more scaled.
        # Example: 776538900 -> 0.07765389
        df["target_weight"] = pd.to_numeric(
            df["weight"], errors="coerce"
        ) / 10_000_000_000
        return df

    return df

def standardize_universe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a universe DataFrame to the canonical schema.
    """
    df = map_weight_columns(df)

    df = ensure_columns(df, STANDARD_COLUMNS)

    df["ticker"] = df["ticker"].apply(normalize_ticker)
    df["active"] = df["active"].apply(normalize_active)

    if "target_weight" in df.columns:
        df["target_weight"] = df["target_weight"].apply(parse_weight)

    blank_ticker_count = (df["ticker"] == "").sum()

    if blank_ticker_count > 0:
        print(
            f"Warning: dropping {blank_ticker_count} rows "
            f"with blank or placeholder tickers."
        )

    df = df[df["ticker"] != ""].copy()

    df = df[STANDARD_COLUMNS]

    return df

def validate_universe_frame(df: pd.DataFrame) -> None:
    """
    Validate the standardized universe DataFrame.

    Raises
    ------
    ValueError
        If required fields are missing or invalid.
    """
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    null_required = [
        col for col in REQUIRED_COLUMNS if df[col].isna().any()
    ]

    if null_required:
        raise ValueError(f"Required columns contain null values: {null_required}")

    blank_tickers = df["ticker"].astype(str).str.strip().eq("").any()

    if blank_tickers:
        raise ValueError("Universe contains blank tickers.")

    # Validate target weights, if present.
    if "target_weight" in df.columns:
        weights = df["target_weight"].dropna()

        if not weights.between(0, 1).all():
            bad_weights = df.loc[
                df["target_weight"].notna()
                & ~df["target_weight"].between(0, 1),
                ["ticker", "target_weight"]
            ]

            raise ValueError(
                f"Target weights must be between 0 and 1:\n{bad_weights}"
            )

    duplicate_rows = df.duplicated(
        subset=["ticker", "universe_name", "account_target"]
    )

    if duplicate_rows.any():
        duplicates = df.loc[
            duplicate_rows, ["ticker", "universe_name", "account_target"]
        ]

        raise ValueError(f"Duplicate universe rows detected:\n{duplicates}")
