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

    df["form"] = df["form"].astype(str).str.upper().str.strip()
    df["fp"] = df["fp"].astype(str).str.upper().str.strip()

    return df[expected_columns]


def add_period_classification(records: pd.DataFrame) -> pd.DataFrame:
    """
    Add period classification fields to SEC fact records.

    SEC facts include both:
    - instant values, usually balance sheet facts
    - duration values, usually income statement or cash flow facts

    Duration length is approximate because companies have different
    fiscal calendars.
    """
    records = records.copy()

    records["has_start"] = records["start"].notna()
    records["has_end"] = records["end"].notna()

    records["period_days"] = (
        records["end"] - records["start"]
    ).dt.days

    records["period_kind"] = "unknown"

    records.loc[
        records["has_end"] & ~records["has_start"],
        "period_kind",
    ] = "instant"

    records.loc[
        records["has_end"] & records["has_start"],
        "period_kind",
    ] = "duration"

    records["duration_type"] = pd.NA

    records.loc[
        records["period_kind"].eq("duration")
        & records["period_days"].between(75, 115, inclusive="both"),
        "duration_type",
    ] = "quarterly"

    records.loc[
        records["period_kind"].eq("duration")
        & records["period_days"].between(170, 210, inclusive="both"),
        "duration_type",
    ] = "semiannual"

    records.loc[
        records["period_kind"].eq("duration")
        & records["period_days"].between(250, 290, inclusive="both"),
        "duration_type",
    ] = "nine_months"

    records.loc[
        records["period_kind"].eq("duration")
        & records["period_days"].between(330, 380, inclusive="both"),
        "duration_type",
    ] = "annual"

    return records


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
            normalized = normalize_fact_records(
                records=records,
                ticker=ticker,
                cik_padded=cik_padded,
                entity_name=entity_name,
                metric_name=metric_name,
                taxonomy=taxonomy,
                concept_name=concept_name,
                unit=unit,
            )

            return add_period_classification(normalized)

    return pd.DataFrame()


def filter_preferred_forms(
    records: pd.DataFrame,
    preferred_forms: list[str],
) -> pd.DataFrame:
    """
    Keep only preferred SEC filing forms when possible.
    """
    if records.empty:
        return records

    records = records.copy()

    preferred_forms = {form.upper() for form in preferred_forms}

    filtered = records[records["form"].isin(preferred_forms)].copy()

    if filtered.empty:
        return records

    return filtered


def select_latest_instant_record(records: pd.DataFrame) -> pd.DataFrame:
    """
    Select latest instant record, intended for balance sheet metrics.
    """
    if records.empty:
        return records

    records = records.copy()

    instant = records[records["period_kind"].eq("instant")].copy()

    if instant.empty:
        # Some concepts may not classify cleanly. Fall back to all records.
        instant = records.copy()

    instant = instant.sort_values(
        ["ticker", "metric_name", "end", "filed"],
        ascending=[True, True, True, True],
    )

    latest = (
        instant.groupby(["ticker", "metric_name"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return latest


def select_latest_annual_duration_record(
    records: pd.DataFrame,
    annual_forms: list[str],
) -> pd.DataFrame:
    """
    Select latest annual duration record, intended for income statement
    and cash flow metrics.

    Preference order:
    1. annual duration records from annual forms
    2. annual duration records from any form
    3. records with FY fiscal period from annual forms
    4. latest available duration record
    """
    if records.empty:
        return records

    records = records.copy()
    annual_forms = {form.upper() for form in annual_forms}

    annual_duration = records[
        records["duration_type"].eq("annual")
        & records["form"].isin(annual_forms)
    ].copy()

    if annual_duration.empty:
        annual_duration = records[
            records["duration_type"].eq("annual")
        ].copy()

    if annual_duration.empty:
        annual_duration = records[
            records["fp"].eq("FY")
            & records["form"].isin(annual_forms)
        ].copy()

    if annual_duration.empty:
        annual_duration = records[
            records["period_kind"].eq("duration")
        ].copy()

    if annual_duration.empty:
        annual_duration = records.copy()

    annual_duration = annual_duration.sort_values(
        ["ticker", "metric_name", "end", "filed"],
        ascending=[True, True, True, True],
    )

    latest = (
        annual_duration.groupby(["ticker", "metric_name"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return latest


def select_record_for_metric(
    records: pd.DataFrame,
    metric_name: str,
    metric_period_types: Dict[str, str],
    annual_forms: list[str],
) -> pd.DataFrame:
    """
    Select an appropriate record based on the configured metric period type.
    """
    period_type = metric_period_types.get(metric_name, "duration_annual")

    if period_type == "instant_latest":
        return select_latest_instant_record(records)

    if period_type == "duration_annual":
        return select_latest_annual_duration_record(
            records=records,
            annual_forms=annual_forms,
        )

    raise ValueError(
        f"Unsupported period type '{period_type}' for metric '{metric_name}'."
    )


def extract_accounting_concepts_for_company(
    facts: Dict[str, Any],
    ticker: str,
    cik_padded: str,
    concepts_config: Dict[str, Any],
    preferred_forms: list[str],
    annual_forms: list[str],
    metric_period_types: Dict[str, str],
) -> pd.DataFrame:
    """
    Extract configured accounting concepts for one company.
    """
    selected_records = []

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

        selected = select_record_for_metric(
            records=records,
            metric_name=metric_name,
            metric_period_types=metric_period_types,
            annual_forms=annual_forms,
        )

        if not selected.empty:
            selected_records.append(selected)

    if not selected_records:
        return pd.DataFrame()

    return pd.concat(selected_records, ignore_index=True)


def build_sec_accounting_concepts(
    config_path: str | Path,
) -> pd.DataFrame:
    """
    Extract configured accounting concept values from cached SEC companyfacts
    JSON files with period-aware selection logic.
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

    annual_forms = accounting_config.get(
        "annual_forms",
        ["10-K", "20-F", "40-F"],
    )

    metric_period_types = accounting_config.get(
        "metric_period_types",
        {},
    )

    concepts_config = accounting_config.get("concepts", {})

    if not concepts_config:
        raise ValueError("No accounting concepts configured.")

    summary = load_company_facts_summary(input_summary_path)

    success = summary[summary["status"] == "success"].copy()

    if success.empty:
        raise ValueError("No successful companyfacts rows found in summary.")

    frames = []

    total = len(success)

    for count, (_, row) in enumerate(success.iterrows(), start=1):
        ticker = str(row["ticker"]).upper().strip()
        cik_padded = str(row["cik_padded"]).zfill(10)

        print(f"[{count}/{total}] Extracting concepts for {ticker}")

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
                annual_forms=annual_forms,
                metric_period_types=metric_period_types,
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

    print("\nSelected period types:")
    period_summary_columns = [
        "metric_name",
        "form",
        "fp",
        "period_kind",
        "duration_type",
    ]

    available_columns = [
        column for column in period_summary_columns if column in concepts.columns
    ]

    print(
        concepts[available_columns]
        .value_counts(dropna=False)
        .head(30)
    )

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
        "start",
        "end",
        "period_kind",
        "duration_type",
        "period_days",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in concepts.columns
    ]

    print(concepts[available_preview_columns].head(40))