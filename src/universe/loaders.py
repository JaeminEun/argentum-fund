from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .schema import standardize_universe_frame


def load_manual_universe(universe_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load a manually maintained universe CSV and enrich it with config metadata.
    """
    input_path = Path(universe_config["input_path"])

    if not input_path.exists():
        raise FileNotFoundError(f"Manual universe file not found: {input_path}")

    input_format = universe_config.get("input_format", {})
    delimiter = input_format.get("delimiter", ",")
    decimal = input_format.get("decimal", ".")

    df = pd.read_csv(input_path, sep=delimiter, decimal=decimal)

    df["universe_name"] = universe_config["universe_name"]
    df["source_type"] = universe_config["source_type"]
    df["source_name"] = universe_config.get("source_name")
    df["strategy_role"] = universe_config.get("strategy_role")
    df["account_target"] = universe_config.get("account_target")

    if "asset_type" not in df.columns or df["asset_type"].isna().all():
        df["asset_type"] = universe_config.get("default_asset_type", "stock")

    if "active" not in df.columns:
        df["active"] = True

    return standardize_universe_frame(df)

def load_sec_13f_universe(universe_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Placeholder for future SEC 13F ingestion.

    Future implementation:
    - Use manager CIK from config.
    - Query SEC submissions endpoint.
    - Find most recent 13F-HR filing.
    - Parse information table XML or use SEC flattened 13F datasets.
    - Map CUSIPs to tickers.
    - Return standardized universe DataFrame.
    """
    manager_cik = universe_config.get("manager_cik")
    universe_name = universe_config.get("universe_name")

    raise NotImplementedError(
        f"SEC 13F loading is not implemented yet for "
        f"{universe_name=} and {manager_cik=}."
    )


def load_universe(universe_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Dispatch to the correct universe loader based on source_type.
    """
    source_type = universe_config.get("source_type")

    if source_type == "manual":
        return load_manual_universe(universe_config)

    if source_type == "sec_13f":
        return load_sec_13f_universe(universe_config)

    raise ValueError(f"Unsupported source_type: {source_type}")
