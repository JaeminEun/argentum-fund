from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_MANAGER_REGISTRY_PATH = Path("data/13f/reference/manager_registry.csv")
DEFAULT_MANAGER_REGISTRY_TEMPLATE_PATH = Path("examples/manager_registry_template.csv")


MANAGER_REGISTRY_COLUMNS = [
    "manager_id",
    "manager_name",
    "filing_entity_name",
    "cik",
    "cik_padded",
    "strategy_family",
    "enabled",
    "notes",
]


REQUIRED_COLUMNS = [
    "manager_id",
    "manager_name",
    "filing_entity_name",
    "cik",
    "enabled",
]


def normalize_manager_id(value: object) -> str:
    """
    Normalize manager IDs for stable filenames and lookups.
    """
    if pd.isna(value):
        return ""

    manager_id = str(value).strip().lower()
    manager_id = manager_id.replace(" ", "_")
    manager_id = manager_id.replace("-", "_")

    cleaned = []

    for character in manager_id:
        if character.isalnum() or character == "_":
            cleaned.append(character)

    return "".join(cleaned)


def normalize_bool(value: object) -> bool:
    """
    Normalize TRUE/FALSE-style values.
    """
    if pd.isna(value):
        return False

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_cik(value: object) -> str:
    """
    Normalize CIK values as unpadded numeric strings.
    """
    if pd.isna(value):
        return ""

    cik = str(value).strip()

    if cik.endswith(".0"):
        cik = cik[:-2]

    cik = "".join(character for character in cik if character.isdigit())

    return cik


def pad_cik(cik: object) -> str:
    """
    Pad CIK to SEC's 10-digit format.
    """
    normalized = normalize_cik(cik)

    if not normalized:
        return ""

    return normalized.zfill(10)


def empty_manager_registry() -> pd.DataFrame:
    """
    Create an empty manager registry with standard columns.
    """
    return pd.DataFrame(columns=MANAGER_REGISTRY_COLUMNS)


def standardize_manager_registry(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize manager registry schema and values.
    """
    frame = frame.copy()

    for column in MANAGER_REGISTRY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[MANAGER_REGISTRY_COLUMNS].copy()

    frame["manager_id"] = frame["manager_id"].apply(normalize_manager_id)
    frame["manager_name"] = frame["manager_name"].astype(str).str.strip()
    frame["filing_entity_name"] = frame["filing_entity_name"].astype(str).str.strip()
    frame["cik"] = frame["cik"].apply(normalize_cik)
    frame["cik_padded"] = frame["cik"].apply(pad_cik)
    frame["strategy_family"] = frame["strategy_family"].astype(str).str.strip()
    frame["enabled"] = frame["enabled"].apply(normalize_bool)
    frame["notes"] = frame["notes"].fillna("").astype(str).str.strip()

    frame = frame[frame["manager_id"] != ""].copy()

    return frame.reset_index(drop=True)


def validate_manager_registry(frame: pd.DataFrame) -> None:
    """
    Validate manager registry contents.

    Raises:
        ValueError if required fields or uniqueness checks fail.
    """
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Manager registry missing required columns: {sorted(missing_columns)}"
        )

    registry = standardize_manager_registry(frame)

    if registry.empty:
        raise ValueError("Manager registry is empty after standardization.")

    missing_required_rows = []

    for column in REQUIRED_COLUMNS:
        if column == "enabled":
            continue

        missing_mask = registry[column].isna() | registry[column].astype(str).str.strip().eq("")

        if missing_mask.any():
            missing_required_rows.append(
                {
                    "column": column,
                    "rows": registry.loc[missing_mask, "manager_id"].tolist(),
                }
            )

    if missing_required_rows:
        raise ValueError(
            f"Manager registry has missing required values: {missing_required_rows}"
        )

    duplicate_manager_ids = registry[
        registry.duplicated(subset=["manager_id"], keep=False)
    ]

    if not duplicate_manager_ids.empty:
        raise ValueError(
            "Duplicate manager_id values found: "
            f"{duplicate_manager_ids['manager_id'].tolist()}"
        )

    duplicate_ciks = registry[
        registry["cik"].ne("")
        & registry.duplicated(subset=["cik"], keep=False)
    ]

    if not duplicate_ciks.empty:
        raise ValueError(
            "Duplicate CIK values found: "
            f"{duplicate_ciks[['manager_id', 'cik']].to_dict(orient='records')}"
        )

    invalid_cik_mask = registry["cik"].astype(str).str.len().gt(10)

    if invalid_cik_mask.any():
        raise ValueError(
            "CIK values longer than 10 digits found: "
            f"{registry.loc[invalid_cik_mask, ['manager_id', 'cik']].to_dict(orient='records')}"
        )


def load_manager_registry(
    path: str | Path = DEFAULT_MANAGER_REGISTRY_PATH,
    fallback_template_path: str | Path | None = DEFAULT_MANAGER_REGISTRY_TEMPLATE_PATH,
    enabled_only: bool = True,
) -> pd.DataFrame:
    """
    Load, standardize, and validate the manager registry.

    If the local registry does not exist and a fallback template exists,
    load the template instead. This makes first-run testing easier.
    """
    path = Path(path)

    if path.exists():
        frame = pd.read_csv(path)
        source_path = path
    elif fallback_template_path is not None and Path(fallback_template_path).exists():
        frame = pd.read_csv(fallback_template_path)
        source_path = Path(fallback_template_path)
    else:
        raise FileNotFoundError(
            f"Manager registry not found: {path}. "
            f"Fallback template also missing: {fallback_template_path}"
        )

    validate_manager_registry(frame)

    registry = standardize_manager_registry(frame)

    if enabled_only:
        registry = registry[registry["enabled"] == True].copy()

    registry["registry_source_path"] = str(source_path)

    return registry.reset_index(drop=True)


def save_manager_registry(
    registry: pd.DataFrame,
    path: str | Path = DEFAULT_MANAGER_REGISTRY_PATH,
) -> None:
    """
    Save a standardized manager registry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    standardized = standardize_manager_registry(registry)
    validate_manager_registry(standardized)

    standardized.to_csv(path, index=False)


def initialize_manager_registry(
    output_path: str | Path = DEFAULT_MANAGER_REGISTRY_PATH,
    template_path: str | Path = DEFAULT_MANAGER_REGISTRY_TEMPLATE_PATH,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Initialize a local manager registry from the example template.
    """
    output_path = Path(output_path)
    template_path = Path(template_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manager registry already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )

    if not template_path.exists():
        raise FileNotFoundError(f"Template registry not found: {template_path}")

    registry = pd.read_csv(template_path)
    registry = standardize_manager_registry(registry)
    validate_manager_registry(registry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_path, index=False)

    return registry


if __name__ == "__main__":
    registry = load_manager_registry(enabled_only=False)

    print("Manager registry loaded.")
    print(f"Rows: {len(registry)}")
    print("")
    print(registry)
