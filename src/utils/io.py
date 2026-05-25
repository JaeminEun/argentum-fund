from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_export_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load export settings from the project config.

    Provides defaults if the config block is missing.
    """
    return config.get(
        "export_settings",
        {
            "standard_csv": {
                "delimiter": ",",
                "decimal": ".",
                "encoding": "utf-8",
            },
            "excel_csv": {
                "enabled": False,
                "delimiter": ";",
                "decimal": ",",
                "encoding": "utf-8-sig",
                "suffix": "_excel",
            },
        },
    )


def build_excel_output_path(
    standard_output_path: str | Path,
    suffix: str = "_excel",
) -> Path:
    """
    Build an Excel-friendly CSV path from a standard CSV path.

    Example:
        data/scores/latest_price_scores.csv
        -> data/scores/latest_price_scores_excel.csv
    """
    standard_output_path = Path(standard_output_path)

    return standard_output_path.with_name(
        f"{standard_output_path.stem}{suffix}{standard_output_path.suffix}"
    )


def save_csv_outputs(
    df: pd.DataFrame,
    output_path: str | Path,
    export_settings: Dict[str, Any] | None = None,
) -> None:
    """
    Save a standard CSV and optionally an Excel-friendly CSV.

    The standard CSV should remain the canonical pipeline output.
    The Excel CSV is for user inspection.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_settings is None:
        export_settings = load_export_settings({})

    standard_settings = export_settings.get("standard_csv", {})
    standard_delimiter = standard_settings.get("delimiter", ",")
    standard_decimal = standard_settings.get("decimal", ".")
    standard_encoding = standard_settings.get("encoding", "utf-8")

    df.to_csv(
        output_path,
        index=False,
        sep=standard_delimiter,
        decimal=standard_decimal,
        encoding=standard_encoding,
    )

    print(f"Saved standard CSV to {output_path}")

    excel_settings = export_settings.get("excel_csv", {})

    if excel_settings.get("enabled", False):
        suffix = excel_settings.get("suffix", "_excel")
        excel_output_path = build_excel_output_path(
            standard_output_path=output_path,
            suffix=suffix,
        )

        excel_delimiter = excel_settings.get("delimiter", ";")
        excel_decimal = excel_settings.get("decimal", ",")
        excel_encoding = excel_settings.get("encoding", "utf-8-sig")

        df.to_csv(
            excel_output_path,
            index=False,
            sep=excel_delimiter,
            decimal=excel_decimal,
            encoding=excel_encoding,
        )

        print(f"Saved Excel-friendly CSV to {excel_output_path}")
