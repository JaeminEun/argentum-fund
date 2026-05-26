from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.universe.config import load_config


def load_sec_accounting_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load project config and validate the sec_accounting_concepts block.
    """
    config = load_config(config_path)

    if "sec_accounting_concepts" not in config:
        raise ValueError("Missing 'sec_accounting_concepts' section in config file.")

    if not config["sec_accounting_concepts"].get("enabled", False):
        raise ValueError("sec_accounting_concepts is disabled in config file.")

    return config


def load_company_facts_summary(input_path: str | Path) -> pd.DataFrame:
    """
    Load the company facts summary produced by sec_company_facts.py.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Company facts summary not found: {input_path}. "
            "Run the SEC company facts downloader first."
        )

    return pd.read_csv(input_path, dtype={"cik_padded": str})


def companyfacts_cache_path(
    facts_cache_dir: str | Path,
    cik_padded: str,
) -> Path:
    """
    Build local cache path for a companyfacts JSON file.
    """
    cik_padded = str(cik_padded).zfill(10)
    return Path(facts_cache_dir) / f"CIK{cik_padded}.json"


def load_companyfacts_json(
    facts_cache_dir: str | Path,
    cik_padded: str,
) -> Dict[str, Any]:
    """
    Load cached SEC companyfacts JSON.
    """
    path = companyfacts_cache_path(facts_cache_dir, cik_padded)

    if not path.exists():
        raise FileNotFoundError(f"Cached companyfacts JSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_concept_units(
    facts: Dict[str, Any],
    taxonomy: str,
    concept_name: str,
) -> Dict[str, List[Dict[str, Any]]] | None:
    """
    Return units dictionary for a given SEC concept, if available.
    """
    return (
        facts.get("facts", {})
        .get(taxonomy, {})
        .get(concept_name, {})
        .get("units")
    )


def choose_best_unit(
    units: Dict[str, List[Dict[str, Any]]],
) -> str | None:
    """
    Choose the most useful unit key from a SEC units dictionary.

    For financial statement values, USD is usually what we want.
    """
    if not units:
        return None

    preferred_units = ["USD", "shares", "USD/shares", "pure"]

    for unit in preferred_units:
        if unit in units:
            return unit

    return next(iter(units.keys()))


def normalize_fact_records(
    records: List[Dict[str, Any]],
    ticker: str,
    cik_padded: str,
    entity_name: str | None,
    metric_name: str,
    taxonomy: str,
    concept_name: str,
    unit: str,
) -> pd.DataFrame:
    """
    Convert SEC fact records into a normalized DataFrame.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["ticker"] = ticker
    df["cik_padded"] = cik_padded
    df["entity_name"] = entity_name
    df["metric_name"] = metric_name
    df["taxonomy"] = taxonomy
    df["concept_name"] = concept_name
    df["unit"] = unit

    # Some records may not have all fields.
    expected_columns = [
        "ticker",
        "cik_padded",
        "entity_name",
        "metric_name",
        "taxonomy",
        "concept_name",
        "unit",
        "val",
        "fy",
        "fp",
        "form",
        "filed",
        "end",
        "start",
        "frame",
        "accn",
    ]

    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA

    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")

    return df[expected_columns]


def extract_candidate_concept_records(
    facts: Dict[str, Any],
    ticker: str,
    cik_padded: str,
    metric_name: str,
    concept_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Extract records for the first available candidate concept for a metric.

    Example:
        metric_name = revenue
        candidates = Revenues, RevenueFromContract...
    """
    taxonomy = concept_config.get("taxonomy", "us-gaap")
    candidates = concept_config.get("candidates", [])

    entity_name = facts.get("entityName")

    for concept_name in candidates:
        units = get_concept_units(
            facts=facts,
            taxonomy=taxonomy,
            concept_name=concept_name,
        )

        if not units:
            continue

        unit = choose_best_unit(units)

        if unit is None:
            continue

        records = units.get(unit, [])

        if records:
            return normalize_fact_records(
                records=records,
                ticker=ticker,
                cik_padded=cik_padded,
                entity_name=entity_name,
                metric_name=metric_name,
                taxonomy=taxonomy,
                concept_name=concept_name,
                unit=unit,
            )

    return pd.DataFrame()


def filter_preferred_forms(
    records: pd.DataFrame,
    preferred_forms: list[str],
) -> pd.DataFrame:
    """
    Keep only preferred SEC filing forms when form is available.
    """
    if records.empty:
        return records

    records = records.copy()

    if "form" not in records.columns:
        return records

    preferred_forms = {form.upper() for form in preferred_forms}

    records["form"] = records["form"].astype(str).str.upper()

    filtered = records[records["form"].isin(preferred_forms)].copy()

    if filtered.empty:
        return records

    return filtered


def select_latest_fact_record(records: pd.DataFrame) -> pd.DataFrame:
    """
    Select the latest fact record for each ticker and metric.

    This is intentionally simple for the first version:
    - sort by period end date and filing date
    - keep the latest record
    """
    if records.empty:
        return records

    records = records.copy()

    records = records.sort_values(
        ["ticker", "metric_name", "end", "filed"],
        ascending=[True, True, True, True],
    )

    latest = (
        records.groupby(["ticker", "metric_name"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return latest


def extract_accounting_concepts_for_company(
    facts: Dict[str, Any],
    ticker: str,
    cik_padded: str,
    concepts_config: Dict[str, Any],
    preferred_forms: list[str],
) -> pd.DataFrame:
    """
    Extract all configured accounting concept records for one company.
    """
    frames = []

    for metric_name, concept_config in concepts_config.items():
        records = extract_candidate_concept_records(
            facts=facts,
            ticker=ticker,
            cik_padded=cik_padded,
            metric_name=metric_name,
            concept_config=concept_config,
        )

        if records.empty:
            continue

        records = filter_preferred_forms(
            records=records,
            preferred_forms=preferred_forms,
        )

        frames.append(records)

    if not frames:
        return pd.DataFrame()

    all_records = pd.concat(frames, ignore_index=True)

    latest_records = select_latest_fact_record(all_records)

    return latest_records


def build_sec_accounting_concepts(
    config_path: str | Path,
) -> pd.DataFrame:
    """
    Extract latest configured accounting concept values from cached
    SEC companyfacts JSON files.
    """
    config = load_sec_accounting_config(config_path)
    accounting_config = config["sec_accounting_concepts"]

    input_summary_path = accounting_config["input_company_facts_summary_path"]
    facts_cache_dir = accounting_config["facts_cache_dir"]
    output_path = Path(accounting_config["output_path"])

    preferred_forms = accounting_config.get(
        "preferred_forms",
        ["10-K", "10-Q", "20-F", "40-F"],
    )

    concepts_config = accounting_config.get("concepts", {})

    if not concepts_config:
        raise ValueError("No accounting concepts configured.")

    summary = load_company_facts_summary(input_summary_path)

    success = summary[summary["status"] == "success"].copy()

    if success.empty:
        raise ValueError("No successful companyfacts rows found in summary.")

    frames = []

    for index, row in success.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        cik_padded = str(row["cik_padded"]).zfill(10)

        print(f"[{index + 1}/{len(success)}] Extracting concepts for {ticker}")

        try:
            facts = load_companyfacts_json(
                facts_cache_dir=facts_cache_dir,
                cik_padded=cik_padded,
            )

            company_records = extract_accounting_concepts_for_company(
                facts=facts,
                ticker=ticker,
                cik_padded=cik_padded,
                concepts_config=concepts_config,
                preferred_forms=preferred_forms,
            )

            if not company_records.empty:
                frames.append(company_records)
            else:
                print(f"  Warning: no configured concepts found for {ticker}")

        except Exception as error:
            print(f"  Warning: failed to extract concepts for {ticker}: {error}")

    if not frames:
        raise ValueError("No accounting concept records were extracted.")

    concepts = pd.concat(frames, ignore_index=True)

    concepts = concepts.sort_values(
        ["ticker", "metric_name"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    concepts.to_csv(output_path, index=False)

    print(f"\nSaved SEC accounting concepts to {output_path}")
    print(f"Rows written: {len(concepts)}")

    print("\nMetric counts:")
    print(concepts["metric_name"].value_counts(dropna=False))

    return concepts


if __name__ == "__main__":
    concepts = build_sec_accounting_concepts("config/universe_config.yaml")

    print("\nSEC accounting concepts preview:")
    preview_columns = [
        "ticker",
        "metric_name",
        "concept_name",
        "unit",
        "val",
        "fy",
        "fp",
        "form",
        "filed",
        "end",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in concepts.columns
    ]

    print(concepts[available_preview_columns].head(40))
