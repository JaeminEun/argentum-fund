from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SECURITY_MASTER_PATH = Path("data/reference/security_master.csv")
DEFAULT_SEC_REFERENCE_PATH = Path("data/sec/reference/company_tickers.csv")


SECURITY_MASTER_COLUMNS = [
    "ticker",
    "company_name",
    "cik",
    "cik_padded",
    "asset_type",
    "sector",
    "industry",
    "exchange",
    "currency",
    "source",
    "last_updated",
    "notes",
]


def normalize_ticker(value: object) -> str:
    """
    Normalize ticker symbols for lookup and storage.
    """
    if pd.isna(value):
        return ""

    ticker = str(value).strip().upper()

    return ticker


def ensure_parent_directory(path: str | Path) -> None:
    """
    Create parent directory for a file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def empty_security_master() -> pd.DataFrame:
    """
    Create an empty security master with the standard schema.
    """
    return pd.DataFrame(columns=SECURITY_MASTER_COLUMNS)


def load_security_master(path: str | Path = DEFAULT_SECURITY_MASTER_PATH) -> pd.DataFrame:
    """
    Load existing security master or return an empty standard frame.
    """
    path = Path(path)

    if not path.exists():
        return empty_security_master()

    frame = pd.read_csv(path)

    return standardize_security_master(frame)


def load_sec_ticker_reference(
    path: str | Path = DEFAULT_SEC_REFERENCE_PATH,
) -> pd.DataFrame:
    """
    Load SEC ticker/CIK reference file.

    Expected columns:
        ticker, cik, cik_padded, company_name
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"SEC ticker reference not found: {path}. "
            "Run python -m src.sec.cik first."
        )

    frame = pd.read_csv(path)

    required_columns = {"ticker", "cik", "cik_padded", "company_name"}
    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"SEC ticker reference missing columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["ticker"] = frame["ticker"].apply(normalize_ticker)
    frame = frame.dropna(subset=["ticker"])
    frame = frame[frame["ticker"] != ""]
    frame = frame.drop_duplicates(subset=["ticker"], keep="first")

    return frame


def standardize_security_master(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize security master columns and ticker formatting.
    """
    frame = frame.copy()

    for column in SECURITY_MASTER_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[SECURITY_MASTER_COLUMNS].copy()

    frame["ticker"] = frame["ticker"].apply(normalize_ticker)

    frame = frame.dropna(subset=["ticker"])
    frame = frame[frame["ticker"] != ""]

    frame = frame.drop_duplicates(subset=["ticker"], keep="first")

    return frame.reset_index(drop=True)


def infer_basic_asset_type(ticker: str) -> str:
    """
    Infer simple asset type for common placeholders.

    This is intentionally conservative.
    """
    ticker = normalize_ticker(ticker)

    if ticker in {"CASH", "CASH_USD", "USD"}:
        return "cash"

    return "stock"


def add_tickers_to_security_master(
    security_master: pd.DataFrame,
    tickers: Iterable[str],
    default_source: str = "user_ticker_list",
) -> pd.DataFrame:
    """
    Add missing tickers to the security master with minimal metadata.
    """
    master = standardize_security_master(security_master)

    existing = set(master["ticker"])
    new_rows = []

    today = date.today().isoformat()

    for ticker in tickers:
        normalized = normalize_ticker(ticker)

        if not normalized or normalized in existing:
            continue

        new_rows.append(
            {
                "ticker": normalized,
                "company_name": pd.NA,
                "cik": pd.NA,
                "cik_padded": pd.NA,
                "asset_type": infer_basic_asset_type(normalized),
                "sector": pd.NA,
                "industry": pd.NA,
                "exchange": pd.NA,
                "currency": "USD",
                "source": default_source,
                "last_updated": today,
                "notes": "Added from ticker list.",
            }
        )

        existing.add(normalized)

    if not new_rows:
        return master

    return standardize_security_master(
        pd.concat([master, pd.DataFrame(new_rows)], ignore_index=True)
    )


def enrich_from_sec_reference(
    security_master: pd.DataFrame,
    sec_reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing company_name, cik, and cik_padded using SEC ticker reference.
    """
    master = standardize_security_master(security_master)
    sec = sec_reference.copy()

    sec["ticker"] = sec["ticker"].apply(normalize_ticker)

    merged = master.merge(
        sec[["ticker", "cik", "cik_padded", "company_name"]],
        on="ticker",
        how="left",
        suffixes=("", "_sec"),
    )

    for column in ["company_name", "cik", "cik_padded"]:
        sec_column = f"{column}_sec"

        if sec_column not in merged.columns:
            continue

        missing_or_blank = (
            merged[column].isna()
            | merged[column].astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
        )

        merged.loc[missing_or_blank, column] = merged.loc[
            missing_or_blank,
            sec_column,
        ]

    merged = merged.drop(
        columns=[column for column in merged.columns if column.endswith("_sec")]
    )

    today = date.today().isoformat()

    has_sec_match = merged["cik"].notna() | merged["cik_padded"].notna()

    source_missing_or_blank = (
        merged["source"].isna()
        | merged["source"].astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
    )

    merged.loc[has_sec_match & source_missing_or_blank, "source"] = (
        "sec_company_tickers"
    )

    last_updated_missing_or_blank = (
        merged["last_updated"].isna()
        | merged["last_updated"].astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
    )

    merged.loc[has_sec_match & last_updated_missing_or_blank, "last_updated"] = today

    return standardize_security_master(merged)


def update_security_master(
    tickers: Iterable[str] | None = None,
    security_master_path: str | Path = DEFAULT_SECURITY_MASTER_PATH,
    sec_reference_path: str | Path = DEFAULT_SEC_REFERENCE_PATH,
) -> pd.DataFrame:
    """
    Load, update, enrich, and save the security master.
    """
    tickers = list(tickers or [])

    security_master = load_security_master(security_master_path)

    if tickers:
        security_master = add_tickers_to_security_master(
            security_master=security_master,
            tickers=tickers,
        )

    sec_reference = load_sec_ticker_reference(sec_reference_path)

    security_master = enrich_from_sec_reference(
        security_master=security_master,
        sec_reference=sec_reference,
    )

    ensure_parent_directory(security_master_path)

    security_master.to_csv(security_master_path, index=False)

    return security_master


def build_security_master_from_current_universe(
    current_universe_path: str | Path = "data/processed/current_universe.csv",
    security_master_path: str | Path = DEFAULT_SECURITY_MASTER_PATH,
    sec_reference_path: str | Path = DEFAULT_SEC_REFERENCE_PATH,
) -> pd.DataFrame:
    """
    Build or update security master using tickers and metadata from current_universe.csv.
    """
    current_universe_path = Path(current_universe_path)

    if not current_universe_path.exists():
        raise FileNotFoundError(
            f"Current universe file not found: {current_universe_path}. "
            "Run python -m src.universe.build_universe first, or pass tickers directly."
        )

    universe = pd.read_csv(current_universe_path)

    if "ticker" not in universe.columns:
        raise ValueError(
            f"Current universe missing required column 'ticker': {current_universe_path}"
        )

    tickers = universe["ticker"].dropna().astype(str).tolist()

    security_master = load_security_master(security_master_path)

    security_master = add_tickers_to_security_master(
        security_master=security_master,
        tickers=tickers,
        default_source="current_universe",
    )

    metadata_columns = [
        "ticker",
        "company_name",
        "asset_type",
        "sector",
        "industry",
    ]

    available_metadata = [
        column for column in metadata_columns if column in universe.columns
    ]

    universe_metadata = universe[available_metadata].copy()
    universe_metadata["ticker"] = universe_metadata["ticker"].apply(normalize_ticker)
    universe_metadata = universe_metadata.drop_duplicates(
        subset=["ticker"],
        keep="first",
    )

    merged = security_master.merge(
        universe_metadata,
        on="ticker",
        how="left",
        suffixes=("", "_universe"),
    )

    for column in ["company_name", "asset_type", "sector", "industry"]:
        universe_column = f"{column}_universe"

        if universe_column not in merged.columns:
            continue

        missing_or_blank = (
            merged[column].isna()
            | merged[column].astype(str).str.strip().isin(["", "nan", "None", "<NA>"])
        )

        merged.loc[missing_or_blank, column] = merged.loc[
            missing_or_blank,
            universe_column,
        ]

    merged = merged.drop(
        columns=[column for column in merged.columns if column.endswith("_universe")]
    )

    sec_reference = load_sec_ticker_reference(sec_reference_path)

    merged = enrich_from_sec_reference(
        security_master=merged,
        sec_reference=sec_reference,
    )

    ensure_parent_directory(security_master_path)

    merged.to_csv(security_master_path, index=False)

    return standardize_security_master(merged)


if __name__ == "__main__":
    security_master = build_security_master_from_current_universe()

    print("Security master updated.")
    print(f"Rows: {len(security_master)}")
    print(f"Output: {DEFAULT_SECURITY_MASTER_PATH}")
    print("")
    print(security_master.head(20))
