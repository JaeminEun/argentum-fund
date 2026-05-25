from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.sec.cik import load_company_tickers_reference, map_tickers_to_ciks
from src.sec.client import SecClient
from src.universe.config import load_config


def load_sec_company_facts_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load the full project config and validate the sec_company_facts block.
    """
    config = load_config(config_path)

    if "sec_api" not in config:
        raise ValueError("Missing 'sec_api' section in config file.")

    if "sec_company_facts" not in config:
        raise ValueError("Missing 'sec_company_facts' section in config file.")

    if not config["sec_company_facts"].get("enabled", False):
        raise ValueError("sec_company_facts is disabled in config file.")

    return config


def load_current_universe(input_path: str | Path) -> pd.DataFrame:
    """
    Load the standardized universe created by the universe builder.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Current universe file not found: {input_path}. "
            "Run the universe builder first."
        )

    return pd.read_csv(input_path)


def get_stock_universe(
    universe: pd.DataFrame,
    tradable_asset_types: list[str],
) -> pd.DataFrame:
    """
    Filter universe to stock-like securities eligible for SEC company facts.
    """
    required_columns = {"ticker", "asset_type"}

    missing_columns = required_columns - set(universe.columns)

    if missing_columns:
        raise ValueError(f"Universe file missing columns: {missing_columns}")

    universe = universe.copy()
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    universe["asset_type"] = universe["asset_type"].astype(str).str.lower().str.strip()

    tradable_asset_types = {
        asset_type.lower().strip() for asset_type in tradable_asset_types
    }

    stocks = universe[universe["asset_type"].isin(tradable_asset_types)].copy()

    # One row per ticker for SEC facts.
    stocks = stocks.drop_duplicates(subset=["ticker"], keep="first")

    stocks = stocks[stocks["ticker"] != ""].copy()

    return stocks.reset_index(drop=True)


def build_companyfacts_url(
    companyfacts_base_url: str,
    cik_padded: str,
) -> str:
    """
    Build SEC companyfacts URL.

    Example:
        https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json
    """
    cik_padded = str(cik_padded).zfill(10)

    return f"{companyfacts_base_url}/CIK{cik_padded}.json"


def companyfacts_cache_path(
    facts_cache_dir: str | Path,
    cik_padded: str,
) -> Path:
    """
    Local cache path for one companyfacts JSON file.
    """
    facts_cache_dir = Path(facts_cache_dir)
    cik_padded = str(cik_padded).zfill(10)

    return facts_cache_dir / f"CIK{cik_padded}.json"


def load_cached_companyfacts(
    facts_cache_dir: str | Path,
    cik_padded: str,
) -> Dict[str, Any] | None:
    """
    Load cached companyfacts JSON if it exists.
    """
    path = companyfacts_cache_path(facts_cache_dir, cik_padded)

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_cached_companyfacts(
    facts: Dict[str, Any],
    facts_cache_dir: str | Path,
    cik_padded: str,
) -> None:
    """
    Save companyfacts JSON to local cache.
    """
    path = companyfacts_cache_path(facts_cache_dir, cik_padded)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(facts, file)


def fetch_companyfacts(
    client: SecClient,
    companyfacts_base_url: str,
    facts_cache_dir: str | Path,
    cik_padded: str,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fetch SEC companyfacts JSON for a padded CIK.

    Uses a dedicated companyfacts cache folder so these files are easy
    to inspect and reuse later.
    """
    if not force_refresh:
        cached = load_cached_companyfacts(facts_cache_dir, cik_padded)

        if cached is not None:
            return cached

    url = build_companyfacts_url(
        companyfacts_base_url=companyfacts_base_url,
        cik_padded=cik_padded,
    )

    facts = client.get_json(
        url=url,
        force_refresh=force_refresh,
        host="data.sec.gov",
    )

    save_cached_companyfacts(
        facts=facts,
        facts_cache_dir=facts_cache_dir,
        cik_padded=cik_padded,
    )

    return facts


def summarize_companyfacts(
    facts: Dict[str, Any],
    ticker: str,
    cik_padded: str,
) -> Dict[str, Any]:
    """
    Summarize a raw SEC companyfacts JSON response.

    This does not extract financial metrics yet. It summarizes coverage.
    """
    entity_name = facts.get("entityName")
    cik = facts.get("cik")

    facts_root = facts.get("facts", {})

    us_gaap = facts_root.get("us-gaap", {})
    dei = facts_root.get("dei", {})

    us_gaap_concepts = list(us_gaap.keys())
    dei_concepts = list(dei.keys())

    return {
        "ticker": ticker,
        "cik": cik,
        "cik_padded": cik_padded,
        "entity_name": entity_name,
        "facts_available": bool(facts_root),
        "us_gaap_concepts_count": len(us_gaap_concepts),
        "dei_concepts_count": len(dei_concepts),
        "sample_us_gaap_concepts": "; ".join(sorted(us_gaap_concepts)[:15]),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "error_message": "",
    }


def summarize_missing_mapping(
    ticker: str,
) -> Dict[str, Any]:
    """
    Summary row for a ticker that could not be mapped to a CIK.
    """
    return {
        "ticker": ticker,
        "cik": pd.NA,
        "cik_padded": pd.NA,
        "entity_name": pd.NA,
        "facts_available": False,
        "us_gaap_concepts_count": 0,
        "dei_concepts_count": 0,
        "sample_us_gaap_concepts": "",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unmapped_cik",
        "error_message": "Ticker could not be mapped to SEC CIK.",
    }


def summarize_failed_fetch(
    ticker: str,
    cik: object,
    cik_padded: str,
    error: Exception,
) -> Dict[str, Any]:
    """
    Summary row for a ticker whose SEC companyfacts request failed.
    """
    return {
        "ticker": ticker,
        "cik": cik,
        "cik_padded": cik_padded,
        "entity_name": pd.NA,
        "facts_available": False,
        "us_gaap_concepts_count": 0,
        "dei_concepts_count": 0,
        "sample_us_gaap_concepts": "",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fetch_failed",
        "error_message": str(error),
    }


def build_sec_company_facts_summary(
    config_path: str | Path,
) -> pd.DataFrame:
    """
    Build SEC companyfacts coverage summary for stock tickers in the universe.
    """
    config = load_sec_company_facts_config(config_path)

    sec_api_config = config["sec_api"]
    facts_config = config["sec_company_facts"]

    universe_path = facts_config["input_universe_path"]
    cik_reference_path = facts_config["input_cik_reference_path"]
    output_summary_path = Path(facts_config["output_summary_path"])

    facts_cache_dir = facts_config.get(
        "facts_cache_dir",
        "data/sec/cache/companyfacts",
    )

    force_refresh = bool(facts_config.get("force_refresh", False))

    tradable_asset_types = facts_config.get(
        "tradable_asset_types",
        ["stock"],
    )

    companyfacts_base_url = sec_api_config.get(
        "companyfacts_base_url",
        "https://data.sec.gov/api/xbrl/companyfacts",
    )

    universe = load_current_universe(universe_path)
    stocks = get_stock_universe(
        universe=universe,
        tradable_asset_types=tradable_asset_types,
    )

    print(f"Found {len(stocks)} stock tickers for SEC company facts.")

    company_tickers = load_company_tickers_reference(cik_reference_path)

    mapped = map_tickers_to_ciks(
        tickers=stocks["ticker"].tolist(),
        company_tickers=company_tickers,
    )

    client = SecClient.from_config(config_path)

    summaries = []

    for index, row in mapped.iterrows():
        ticker = row["ticker_original"]
        cik = row.get("cik")
        cik_padded = row.get("cik_padded")
        status = row.get("mapping_status")

        print(f"[{index + 1}/{len(mapped)}] {ticker}")

        if status != "mapped" or pd.isna(cik_padded):
            summaries.append(summarize_missing_mapping(ticker))
            print(f"  Warning: no CIK mapping for {ticker}")
            continue

        try:
            facts = fetch_companyfacts(
                client=client,
                companyfacts_base_url=companyfacts_base_url,
                facts_cache_dir=facts_cache_dir,
                cik_padded=str(cik_padded),
                force_refresh=force_refresh,
            )

            summary = summarize_companyfacts(
                facts=facts,
                ticker=ticker,
                cik_padded=str(cik_padded),
            )

            summaries.append(summary)

        except Exception as error:
            print(f"  Warning: failed to fetch companyfacts for {ticker}: {error}")

            summaries.append(
                summarize_failed_fetch(
                    ticker=ticker,
                    cik=cik,
                    cik_padded=str(cik_padded),
                    error=error,
                )
            )

    summary_df = pd.DataFrame(summaries)

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_summary_path, index=False)

    print(f"\nSaved SEC company facts summary to {output_summary_path}")
    print(f"Rows written: {len(summary_df)}")

    print("\nStatus counts:")
    print(summary_df["status"].value_counts(dropna=False))

    return summary_df


if __name__ == "__main__":
    summary = build_sec_company_facts_summary("config/universe_config.yaml")

    print("\nSEC company facts summary preview:")
    preview_columns = [
        "ticker",
        "cik_padded",
        "entity_name",
        "facts_available",
        "us_gaap_concepts_count",
        "status",
        "error_message",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in summary.columns
    ]

    print(summary[available_preview_columns].head(30))
