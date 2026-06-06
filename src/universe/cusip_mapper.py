from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.universe.security_master import (
    DEFAULT_SECURITY_MASTER_PATH,
    normalize_ticker,
)


DEFAULT_PARSED_HOLDINGS_PATH = Path(
    "data/13f/parsed/latest_13f_holdings_raw_combined.csv"
)
DEFAULT_MAPPED_HOLDINGS_PATH = Path(
    "data/13f/parsed/latest_13f_holdings_mapped.csv"
)
DEFAULT_MAPPING_REVIEW_PATH = Path(
    "data/13f/parsed/cusip_mapping_review.csv"
)
DEFAULT_CUSIP_LOOKUP_PATH = Path(
    "data/13f/reference/cusip_ticker_lookup.csv"
)
DEFAULT_CUSIP_OVERRIDES_PATH = Path(
    "data/13f/reference/cusip_manual_overrides.csv"
)


CUSIP_LOOKUP_COLUMNS = [
    "cusip",
    "ticker",
    "company_name",
    "asset_type",
    "source",
    "confidence",
    "last_updated",
    "notes",
]


MAPPING_COLUMNS = [
    "ticker",
    "mapped_company_name",
    "mapped_asset_type",
    "mapping_status",
    "mapping_source",
    "mapping_confidence",
    "mapping_notes",
]


def normalize_cusip(value: object) -> str:
    """
    Normalize CUSIP values for lookup.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()
    text = "".join(character for character in text if character.isalnum())

    return text


def normalize_text(value: object) -> str:
    """
    Normalize text fields.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def ensure_parent_directory(path: str | Path) -> None:
    """
    Create parent directory for a file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def is_option_position(row: pd.Series) -> bool:
    """
    Identify option rows using the put_call field.
    """
    put_call = normalize_text(row.get("put_call", "")).upper()

    return put_call in {"PUT", "CALL"}


def empty_cusip_lookup() -> pd.DataFrame:
    """
    Create an empty CUSIP lookup table.
    """
    return pd.DataFrame(columns=CUSIP_LOOKUP_COLUMNS)


def standardize_cusip_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize CUSIP lookup / override table.
    """
    frame = frame.copy()

    for column in CUSIP_LOOKUP_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[CUSIP_LOOKUP_COLUMNS].copy()

    frame["cusip"] = frame["cusip"].apply(normalize_cusip)
    frame["ticker"] = frame["ticker"].apply(normalize_ticker)
    frame["company_name"] = frame["company_name"].apply(normalize_text)
    frame["asset_type"] = frame["asset_type"].apply(normalize_text)
    frame["source"] = frame["source"].apply(normalize_text)
    frame["notes"] = frame["notes"].apply(normalize_text)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")

    frame = frame[frame["cusip"] != ""].copy()
    frame = frame[frame["ticker"] != ""].copy()

    frame = frame.drop_duplicates(subset=["cusip"], keep="first")

    return frame.reset_index(drop=True)


def load_cusip_lookup(path: str | Path) -> pd.DataFrame:
    """
    Load a CUSIP lookup table if it exists.
    """
    path = Path(path)

    if not path.exists():
        return empty_cusip_lookup()

    frame = pd.read_csv(path)

    return standardize_cusip_lookup(frame)


def load_parsed_holdings(path: str | Path = DEFAULT_PARSED_HOLDINGS_PATH) -> pd.DataFrame:
    """
    Load raw parsed 13F holdings.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Parsed 13F holdings not found: {path}. "
            "Run python -m src.universe.sec_13f_parser first."
        )

    frame = pd.read_csv(path)

    required_columns = {
        "manager_id",
        "issuer_name",
        "class_title",
        "cusip",
        "value_usd",
        "shares",
        "put_call",
        "parse_status",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"Parsed 13F holdings missing required columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["cusip"] = frame["cusip"].apply(normalize_cusip)

    return frame


def load_security_master_metadata(
    path: str | Path = DEFAULT_SECURITY_MASTER_PATH,
) -> pd.DataFrame:
    """
    Load ticker metadata from security master if available.
    """
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "ticker",
                "company_name",
                "asset_type",
                "sector",
                "industry",
            ]
        )

    frame = pd.read_csv(path)

    for column in ["ticker", "company_name", "asset_type", "sector", "industry"]:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[
        ["ticker", "company_name", "asset_type", "sector", "industry"]
    ].copy()

    frame["ticker"] = frame["ticker"].apply(normalize_ticker)
    frame = frame[frame["ticker"] != ""].copy()
    frame = frame.drop_duplicates(subset=["ticker"], keep="first")

    return frame.reset_index(drop=True)


def build_mapping_dictionary(
    manual_overrides: pd.DataFrame,
    local_lookup: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Build CUSIP mapping dictionary.

    Priority:
    1. local lookup
    2. manual override overwrites local lookup

    This produces final priority:
    manual override > local lookup
    """
    mapping: dict[str, dict[str, Any]] = {}

    today = date.today().isoformat()

    for source_frame, default_status in [
        (local_lookup, "mapped_local_lookup"),
        (manual_overrides, "mapped_manual_override"),
    ]:
        standardized = standardize_cusip_lookup(source_frame)

        for _, row in standardized.iterrows():
            cusip = row["cusip"]

            mapping[cusip] = {
                "ticker": row["ticker"],
                "mapped_company_name": row["company_name"],
                "mapped_asset_type": row["asset_type"],
                "mapping_status": default_status,
                "mapping_source": row["source"] or default_status,
                "mapping_confidence": row["confidence"],
                "mapping_notes": row["notes"],
                "last_updated": row.get("last_updated", today),
            }

    return mapping


def map_single_holding(
    row: pd.Series,
    mapping_dict: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Map one parsed 13F holding row.
    """
    cusip = normalize_cusip(row.get("cusip", ""))

    if is_option_position(row):
        return {
            "ticker": pd.NA,
            "mapped_company_name": pd.NA,
            "mapped_asset_type": pd.NA,
            "mapping_status": "excluded_option_position",
            "mapping_source": "put_call",
            "mapping_confidence": 0.0,
            "mapping_notes": "Option position excluded from active universe mapping.",
        }

    if not cusip:
        return {
            "ticker": pd.NA,
            "mapped_company_name": pd.NA,
            "mapped_asset_type": pd.NA,
            "mapping_status": "excluded_missing_cusip",
            "mapping_source": "parsed_13f",
            "mapping_confidence": 0.0,
            "mapping_notes": "Missing CUSIP.",
        }

    if cusip in mapping_dict:
        return mapping_dict[cusip]

    return {
        "ticker": pd.NA,
        "mapped_company_name": pd.NA,
        "mapped_asset_type": pd.NA,
        "mapping_status": "unmapped",
        "mapping_source": "none",
        "mapping_confidence": 0.0,
        "mapping_notes": "No manual override or local lookup mapping found.",
    }


def apply_cusip_mappings(
    holdings: pd.DataFrame,
    manual_overrides: pd.DataFrame,
    local_lookup: pd.DataFrame,
    security_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply conservative CUSIP-to-ticker mappings to parsed holdings.
    """
    holdings = holdings.copy()

    mapping_dict = build_mapping_dictionary(
        manual_overrides=manual_overrides,
        local_lookup=local_lookup,
    )

    mapping_rows = [
        map_single_holding(row, mapping_dict)
        for _, row in holdings.iterrows()
    ]

    mapping_frame = pd.DataFrame(mapping_rows)

    mapped = pd.concat(
        [
            holdings.reset_index(drop=True),
            mapping_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    if security_master is not None and not security_master.empty:
        security_master = security_master.copy()
        security_master["ticker"] = security_master["ticker"].apply(normalize_ticker)

        mapped = mapped.merge(
            security_master,
            on="ticker",
            how="left",
            suffixes=("", "_security_master"),
        )

        for column in ["company_name", "asset_type"]:
            mapped_column = f"mapped_{column}"

            if mapped_column not in mapped.columns:
                continue

            security_column = f"{column}_security_master"

            if security_column not in mapped.columns:
                continue

            missing_or_blank = (
                mapped[mapped_column].isna()
                | mapped[mapped_column].astype(str).str.strip().isin(
                    ["", "nan", "None", "<NA>"]
                )
            )

            mapped.loc[missing_or_blank, mapped_column] = mapped.loc[
                missing_or_blank,
                security_column,
            ]

    return mapped


def build_mapping_review(mapped_holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated review file for unmapped CUSIPs.
    """
    review_source = mapped_holdings[
        mapped_holdings["mapping_status"].eq("unmapped")
    ].copy()

    if review_source.empty:
        return pd.DataFrame(
            columns=[
                "cusip",
                "issuer_name",
                "class_title",
                "manager_ids",
                "holding_count",
                "total_value_usd",
                "suggested_ticker",
                "mapping_status",
                "notes",
            ]
        )

    grouped = (
        review_source.groupby(["cusip", "issuer_name", "class_title"], dropna=False)
        .agg(
            manager_ids=("manager_id", lambda values: ";".join(sorted(set(values)))),
            holding_count=("cusip", "size"),
            total_value_usd=("value_usd", "sum"),
        )
        .reset_index()
    )

    grouped["suggested_ticker"] = ""
    grouped["mapping_status"] = "review_required"
    grouped["notes"] = "Add confirmed mapping to cusip_manual_overrides.csv."

    grouped = grouped.sort_values(
        "total_value_usd",
        ascending=False,
    ).reset_index(drop=True)

    return grouped


def save_lookup_from_manual_overrides(
    manual_overrides: pd.DataFrame,
    local_lookup: pd.DataFrame,
    output_path: str | Path = DEFAULT_CUSIP_LOOKUP_PATH,
) -> pd.DataFrame:
    """
    Save accumulated local lookup table.

    Manual overrides are merged into the local lookup so confirmed mappings
    persist for future runs.
    """
    combined = pd.concat(
        [
            local_lookup,
            manual_overrides,
        ],
        ignore_index=True,
    )

    combined = standardize_cusip_lookup(combined)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_path, index=False)

    return combined


def run_cusip_mapping(
    parsed_holdings_path: str | Path = DEFAULT_PARSED_HOLDINGS_PATH,
    manual_overrides_path: str | Path = DEFAULT_CUSIP_OVERRIDES_PATH,
    local_lookup_path: str | Path = DEFAULT_CUSIP_LOOKUP_PATH,
    security_master_path: str | Path = DEFAULT_SECURITY_MASTER_PATH,
    mapped_output_path: str | Path = DEFAULT_MAPPED_HOLDINGS_PATH,
    review_output_path: str | Path = DEFAULT_MAPPING_REVIEW_PATH,
) -> dict[str, pd.DataFrame]:
    """
    Run conservative CUSIP mapping workflow.
    """
    holdings = load_parsed_holdings(parsed_holdings_path)

    manual_overrides = load_cusip_lookup(manual_overrides_path)
    local_lookup = load_cusip_lookup(local_lookup_path)
    security_master = load_security_master_metadata(security_master_path)

    mapped = apply_cusip_mappings(
        holdings=holdings,
        manual_overrides=manual_overrides,
        local_lookup=local_lookup,
        security_master=security_master,
    )

    review = build_mapping_review(mapped)

    updated_lookup = save_lookup_from_manual_overrides(
        manual_overrides=manual_overrides,
        local_lookup=local_lookup,
        output_path=local_lookup_path,
    )

    ensure_parent_directory(mapped_output_path)
    ensure_parent_directory(review_output_path)

    mapped.to_csv(mapped_output_path, index=False)
    review.to_csv(review_output_path, index=False)

    return {
        "mapped_holdings": mapped,
        "mapping_review": review,
        "cusip_lookup": updated_lookup,
    }


def print_mapping_summary(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Print useful mapping summary.
    """
    mapped = outputs["mapped_holdings"]
    review = outputs["mapping_review"]

    print("")
    print("CUSIP mapping complete.")
    print(f"Mapped holdings rows: {len(mapped)}")
    print(f"Unique CUSIPs: {mapped['cusip'].nunique()}")
    print("")
    print("Rows by mapping status:")
    print(mapped["mapping_status"].value_counts(dropna=False))
    print("")
    print("Rows by manager and mapping status:")
    print(mapped.groupby(["manager_id", "mapping_status"]).size())
    print("")
    print(f"Unmapped review rows: {len(review)}")

    if not review.empty:
        print("")
        print("Top unmapped CUSIPs by total value:")
        print(
            review[
                [
                    "cusip",
                    "issuer_name",
                    "class_title",
                    "manager_ids",
                    "total_value_usd",
                    "holding_count",
                ]
            ].head(20)
        )


if __name__ == "__main__":
    outputs = run_cusip_mapping()
    print_mapping_summary(outputs)
