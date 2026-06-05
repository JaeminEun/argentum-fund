from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.sec.client import SecClient
from src.universe.config import load_config


def normalize_ticker_for_sec(value: object) -> str:
    """
    Normalize ticker strings for SEC ticker mapping.

    SEC company_tickers.json generally uses symbols like:
    BRK-B rather than BRK.B.
    """
    if pd.isna(value):
        return ""

    ticker = str(value).strip().upper()

    if ticker in {"", "-", "N/A", "NA", "NONE", "NULL", "NAN"}:
        return ""

    # Common conversion for share classes.
    ticker = ticker.replace(".", "-")

    return ticker


def pad_cik(cik: int | str) -> str:
    """
    Pad a CIK to the 10-digit format required by many SEC API URLs.

    Example:
        320193 -> 0000320193
    """
    return str(int(cik)).zfill(10)


def load_sec_reference_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load SEC API and reference configuration.
    """
    config = load_config(config_path)

    if "sec_api" not in config:
        raise ValueError("Missing 'sec_api' section in config file.")

    if "sec_reference" not in config:
        raise ValueError("Missing 'sec_reference' section in config file.")

    return config


def company_tickers_json_to_frame(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert SEC company_tickers.json data into a clean DataFrame.

    The SEC file is keyed by numeric strings. Each entry normally contains:
    - cik_str
    - ticker
    - title
    """
    records = []

    for _, record in data.items():
        records.append(record)

    df = pd.DataFrame(records)

    required_columns = {"cik_str", "ticker", "title"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"SEC company tickers file missing expected columns: {missing_columns}"
        )

    df["ticker"] = df["ticker"].apply(normalize_ticker_for_sec)
    df["cik"] = df["cik_str"].astype(int)
    df["cik_padded"] = df["cik"].apply(pad_cik)
    df["company_name"] = df["title"].astype(str).str.strip()

    df = df[
        [
            "ticker",
            "cik",
            "cik_padded",
            "company_name",
        ]
    ].sort_values("ticker")

    df = df[df["ticker"] != ""].copy()
    df = df.drop_duplicates(subset=["ticker"], keep="first")

    return df.reset_index(drop=True)


def build_company_tickers_reference(config_path: str | Path) -> pd.DataFrame:
    """
    Download or load SEC company_tickers.json and save a standardized
    ticker-to-CIK reference table.
    """
    config = load_sec_reference_config(config_path)

    sec_api_config = config["sec_api"]
    reference_config = config["sec_reference"]

    if not reference_config.get("enabled", False):
        raise ValueError("sec_reference is disabled in the config file.")

    force_refresh = bool(reference_config.get("force_refresh", False))
    company_tickers_url = sec_api_config["company_tickers_url"]
    output_path = Path(reference_config["company_tickers_output_path"])

    client = SecClient.from_config(config_path)

    data = client.get_json(
        company_tickers_url,
        force_refresh=force_refresh,
        host="www.sec.gov",
    )

    tickers = company_tickers_json_to_frame(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tickers.to_csv(output_path, index=False)

    print(f"Saved SEC company ticker reference to {output_path}")
    print(f"Rows written: {len(tickers)}")

    return tickers


def load_company_tickers_reference(
    reference_path: str | Path,
) -> pd.DataFrame:
    """
    Load a previously built ticker-to-CIK reference table.
    """
    reference_path = Path(reference_path)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"SEC ticker reference not found: {reference_path}. "
            "Run build_company_tickers_reference first."
        )

    df = pd.read_csv(reference_path, dtype={"cik_padded": str})
    df["ticker"] = df["ticker"].apply(normalize_ticker_for_sec)
    df["cik_padded"] = df["cik_padded"].astype(str).str.zfill(10)

    return df


def map_tickers_to_ciks(
    tickers: list[str],
    company_tickers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map a list of ticker symbols to SEC CIKs.

    Returns one row per requested ticker with mapping status.
    """
    requested = pd.DataFrame({"ticker_original": tickers})
    requested["ticker"] = requested["ticker_original"].apply(normalize_ticker_for_sec)

    reference = company_tickers.copy()
    reference["ticker"] = reference["ticker"].apply(normalize_ticker_for_sec)

    mapped = requested.merge(
        reference,
        on="ticker",
        how="left",
    )

    mapped["mapping_status"] = mapped["cik_padded"].apply(
        lambda value: "mapped" if pd.notna(value) else "unmapped"
    )

    return mapped[
        [
            "ticker_original",
            "ticker",
            "cik",
            "cik_padded",
            "company_name",
            "mapping_status",
        ]
    ]


if __name__ == "__main__":
    reference = build_company_tickers_reference("config/universe_config.yaml")

    print("\nSEC ticker reference preview:")
    print(reference.head(20))

    test_tickers = ["AAPL", "MSFT", "BRK.B", "BF.B", "OXY", "JD", "VTI"]
    mapped = map_tickers_to_ciks(test_tickers, reference)

    print("\nTest ticker mapping:")
    print(mapped)
