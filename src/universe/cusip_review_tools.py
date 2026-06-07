from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from src.universe.cusip_mapper import (
    DEFAULT_CUSIP_OVERRIDES_PATH,
    DEFAULT_MAPPING_REVIEW_PATH,
    CUSIP_LOOKUP_COLUMNS,
    normalize_cusip,
    normalize_text,
    standardize_cusip_lookup,
)


DEFAULT_OVERRIDE_DRAFT_PATH = Path(
    "data/13f/parsed/cusip_manual_override_draft.csv"
)


DRAFT_COLUMNS = [
    "cusip",
    "ticker",
    "company_name",
    "asset_type",
    "source",
    "confidence",
    "notes",
    "issuer_name",
    "class_title",
    "manager_ids",
    "total_value_usd",
    "holding_count",
]


def normalize_ticker(value: object) -> str:
    """
    Normalize ticker symbols.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().upper()

def read_csv_auto(
    path: str | Path,
    required_columns: set[str],
) -> pd.DataFrame:
    """
    Read a CSV file using common delimiter/decimal combinations.

    This is useful for files edited in Excel under different regional settings.
    The selected format is the first one that produces the required columns.
    """
    path = Path(path)

    attempts = [
        {"delimiter": ",", "decimal": ".", "label": "comma delimiter, period decimal"},
        {"delimiter": ";", "decimal": ",", "label": "semicolon delimiter, comma decimal"},
        {"delimiter": ";", "decimal": ".", "label": "semicolon delimiter, period decimal"},
        {"delimiter": "\t", "decimal": ".", "label": "tab delimiter, period decimal"},
    ]

    diagnostics = []

    for attempt in attempts:
        try:
            frame = pd.read_csv(
                path,
                sep=attempt["delimiter"],
                decimal=attempt["decimal"],
            )

            columns = set(frame.columns)

            if required_columns.issubset(columns):
                print(
                    "Detected CSV format for "
                    f"{path}: {attempt['label']}"
                )
                return frame

            diagnostics.append(
                {
                    "format": attempt["label"],
                    "columns": list(frame.columns),
                }
            )

        except Exception as error:
            diagnostics.append(
                {
                    "format": attempt["label"],
                    "error": str(error),
                }
            )

    raise ValueError(
        f"Could not read {path} with required columns: {sorted(required_columns)}. "
        f"Diagnostics: {diagnostics}"
    )


def load_mapping_review(
    path: str | Path = DEFAULT_MAPPING_REVIEW_PATH,
) -> pd.DataFrame:
    """
    Load CUSIP mapping review file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CUSIP mapping review file not found: {path}. "
            "Run python -m src.universe.cusip_mapper first."
        )

    review = pd.read_csv(path)

    required_columns = {
        "cusip",
        "issuer_name",
        "class_title",
        "manager_ids",
        "holding_count",
        "total_value_usd",
    }

    missing = required_columns - set(review.columns)

    if missing:
        raise ValueError(
            f"CUSIP mapping review file missing columns: {sorted(missing)}"
        )

    review = review.copy()
    review["cusip"] = review["cusip"].apply(normalize_cusip)
    review["total_value_usd"] = pd.to_numeric(
        review["total_value_usd"],
        errors="coerce",
    )

    review = review[review["cusip"] != ""].copy()

    return review.reset_index(drop=True)


def load_existing_overrides(
    path: str | Path = DEFAULT_CUSIP_OVERRIDES_PATH,
) -> pd.DataFrame:
    """
    Load existing manual overrides, or return an empty standardized frame.
    """
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=CUSIP_LOOKUP_COLUMNS)

    overrides = pd.read_csv(path)

    return standardize_cusip_lookup(overrides)


def create_override_draft(
    review_path: str | Path = DEFAULT_MAPPING_REVIEW_PATH,
    overrides_path: str | Path = DEFAULT_CUSIP_OVERRIDES_PATH,
    output_path: str | Path = DEFAULT_OVERRIDE_DRAFT_PATH,
    top_n: int = 50,
    min_value_usd: float | None = None,
) -> pd.DataFrame:
    """
    Create a manual override draft from highest-value unmapped CUSIPs.

    Existing overrides are excluded from the draft.
    """
    review = load_mapping_review(review_path)
    overrides = load_existing_overrides(overrides_path)

    existing_override_cusips = set(overrides["cusip"]) if not overrides.empty else set()

    draft_source = review[~review["cusip"].isin(existing_override_cusips)].copy()

    if min_value_usd is not None:
        draft_source = draft_source[
            draft_source["total_value_usd"].fillna(0) >= min_value_usd
        ].copy()

    draft_source = draft_source.sort_values(
        "total_value_usd",
        ascending=False,
    ).head(top_n)

    today = date.today().isoformat()

    draft = pd.DataFrame(
        {
            "cusip": draft_source["cusip"],
            "ticker": "",
            "company_name": "",
            "asset_type": "stock",
            "source": "manual_override",
            "confidence": 1.0,
            "notes": (
                "Review required. Generated from CUSIP mapping review on "
                + today
                + "."
            ),
            "issuer_name": draft_source["issuer_name"],
            "class_title": draft_source["class_title"],
            "manager_ids": draft_source["manager_ids"],
            "total_value_usd": draft_source["total_value_usd"],
            "holding_count": draft_source["holding_count"],
        }
    )

    for column in DRAFT_COLUMNS:
        if column not in draft.columns:
            draft[column] = pd.NA

    draft = draft[DRAFT_COLUMNS].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    draft.to_csv(output_path, index=False)

    return draft


def load_completed_draft(
    draft_path: str | Path = DEFAULT_OVERRIDE_DRAFT_PATH,
    auto_detect_csv: bool = True,
    delimiter: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """
    Load completed draft rows where ticker is filled.
    """
    draft_path = Path(draft_path)

    if not draft_path.exists():
        raise FileNotFoundError(f"Override draft not found: {draft_path}")

    required_columns = {
        "cusip",
        "ticker",
        "company_name",
        "asset_type",
        "source",
        "confidence",
        "notes",
    }

    if auto_detect_csv:
        draft = read_csv_auto(
            path=draft_path,
            required_columns=required_columns,
        ) 
    else:
        draft = pd.read_csv(
            draft_path,
            sep=delimiter,
            decimal=decimal,
        )

        missing = required_columns - set(draft.columns)

        if missing:
            raise ValueError(
                f"Override draft missing required columns: {sorted(missing)}"
            )

    completed = draft.copy()
    completed["cusip"] = completed["cusip"].apply(normalize_cusip)
    completed["ticker"] = completed["ticker"].apply(normalize_ticker)
    completed["company_name"] = completed["company_name"].apply(normalize_text)
    completed["asset_type"] = completed["asset_type"].apply(normalize_text)
    completed["source"] = completed["source"].apply(normalize_text)
    completed["notes"] = completed["notes"].apply(normalize_text)
    completed["confidence"] = pd.to_numeric(
        completed["confidence"],
        errors="coerce",
    )

    completed = completed[
        (completed["cusip"] != "")
        & (completed["ticker"] != "")
    ].copy()

    completed = completed[
        [
            "cusip",
            "ticker",
            "company_name",
            "asset_type",
            "source",
            "confidence",
            "notes",
        ]
    ].copy()

    completed["last_updated"] = date.today().isoformat()

    for column in CUSIP_LOOKUP_COLUMNS:
        if column not in completed.columns:
            completed[column] = pd.NA

    completed = completed[CUSIP_LOOKUP_COLUMNS].copy()

    return standardize_cusip_lookup(completed)


def merge_completed_draft_into_overrides(
    draft_path: str | Path = DEFAULT_OVERRIDE_DRAFT_PATH,
    overrides_path: str | Path = DEFAULT_CUSIP_OVERRIDES_PATH,
    auto_detect_csv: bool = True,
    delimiter: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """
    Merge completed draft rows into manual overrides.

    Completed draft rows take priority over existing override rows for the same CUSIP.
    """
    completed = load_completed_draft(
        draft_path=draft_path,
        auto_detect_csv=auto_detect_csv,
        delimiter=delimiter,
        decimal=decimal,
    )
    existing = load_existing_overrides(overrides_path)

    merged = pd.concat(
        [
            completed,
            existing,
        ],
        ignore_index=True,
    )

    merged = standardize_cusip_lookup(merged)

    overrides_path = Path(overrides_path)
    overrides_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(overrides_path, index=False)

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or merge CUSIP manual override drafts."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create-draft",
        help="Create a manual override draft from unmapped CUSIPs.",
    )
    create_parser.add_argument("--top-n", type=int, default=50)
    create_parser.add_argument("--min-value-usd", type=float, default=None)
    create_parser.add_argument(
        "--review-path",
        default=str(DEFAULT_MAPPING_REVIEW_PATH),
    )
    create_parser.add_argument(
        "--overrides-path",
        default=str(DEFAULT_CUSIP_OVERRIDES_PATH),
    )
    create_parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OVERRIDE_DRAFT_PATH),
    )

    merge_parser = subparsers.add_parser(
        "merge-draft",
        help="Merge completed draft rows into manual overrides.",
    )
    merge_parser.add_argument(
        "--draft-path",
        default=str(DEFAULT_OVERRIDE_DRAFT_PATH),
    )
    merge_parser.add_argument(
        "--overrides-path",
        default=str(DEFAULT_CUSIP_OVERRIDES_PATH),
    )
    merge_parser.add_argument(
        "--no-auto-detect-csv",
        action="store_true",
        help="Disable automatic CSV format detection for the completed draft.",
    )
    merge_parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter to use when auto-detect is disabled.",
    )
    merge_parser.add_argument(
        "--decimal",
        default=".",
        help="Decimal marker to use when auto-detect is disabled.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "create-draft":
        draft = create_override_draft(
            review_path=args.review_path,
            overrides_path=args.overrides_path,
            output_path=args.output_path,
            top_n=args.top_n,
            min_value_usd=args.min_value_usd,
        )

        print("CUSIP manual override draft created.")
        print(f"Rows: {len(draft)}")
        print(f"Output: {args.output_path}")
        print("")
        print(
            draft[
                [
                    "cusip",
                    "ticker",
                    "company_name",
                    "issuer_name",
                    "class_title",
                    "manager_ids",
                    "total_value_usd",
                ]
            ].head(20)
        )

    elif args.command == "merge-draft":
        overrides = merge_completed_draft_into_overrides(
            draft_path=args.draft_path,
            overrides_path=args.overrides_path,
            auto_detect_csv=not args.no_auto_detect_csv,
            delimiter=args.delimiter,
            decimal=args.decimal,
        )

        print("Completed CUSIP draft rows merged into manual overrides.")
        print(f"Rows in overrides: {len(overrides)}")
        print(f"Output: {args.overrides_path}")


if __name__ == "__main__":
    main()
