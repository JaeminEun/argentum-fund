from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.universe.config import load_config
from src.universe.security_master import (
    DEFAULT_SEC_REFERENCE_PATH,
    DEFAULT_SECURITY_MASTER_PATH,
    SECURITY_MASTER_COLUMNS,
    add_tickers_to_security_master,
    enrich_from_sec_reference,
    load_sec_ticker_reference,
    load_security_master,
    normalize_ticker,
    standardize_security_master,
)


DEFAULT_CONFIG_PATH = "config/universe_config.yaml"


OUTPUT_COLUMNS = [
    "ticker",
    "company_name",
    "asset_type",
    "target_weight",
    "sector",
    "industry",
    "universe_name",
    "source_type",
    "source_name",
    "strategy_role",
    "account_target",
    "active",
    "notes",
]


def load_ticker_list_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load project config and validate ticker_list_universes block.
    """
    config = load_config(config_path)

    if "ticker_list_universes" not in config:
        raise ValueError("Missing 'ticker_list_universes' section in config file.")

    return config


def read_ticker_list(
    input_path: str | Path,
    delimiter: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """
    Read user-provided ticker list.

    Supports either:
        ticker
    or optional extra columns such as:
        ticker,target_weight,notes,sector,industry,asset_type
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Ticker list file not found: {input_path}")

    frame = pd.read_csv(input_path, sep=delimiter, decimal=decimal)

    if "ticker" not in frame.columns:
        raise ValueError(
            f"Ticker list file must contain a 'ticker' column: {input_path}. "
            f"Detected columns: {list(frame.columns)}"
        )

    frame = frame.copy()
    frame["ticker"] = frame["ticker"].apply(normalize_ticker)

    frame = frame.dropna(subset=["ticker"])
    frame = frame[frame["ticker"] != ""]
    frame = frame.drop_duplicates(subset=["ticker"], keep="first")

    return frame.reset_index(drop=True)


def infer_asset_type_from_ticker(ticker: str, default_asset_type: str = "stock") -> str:
    """
    Infer simple asset type from ticker.

    This is intentionally conservative.
    """
    ticker = normalize_ticker(ticker)

    if ticker in {"CASH", "CASH_USD", "USD"}:
        return "cash"

    known_treasury_etfs = {"SGOV", "BIL", "SHV", "TBIL"}
    broad_index_etfs = {"VTI", "VOO", "SPY", "IVV", "QQQ", "VEA", "VXUS"}

    if ticker in known_treasury_etfs or ticker in broad_index_etfs:
        return "ETF"

    return default_asset_type


def calculate_target_weights(
    frame: pd.DataFrame,
    weighting_method: str = "equal_weight",
) -> pd.Series:
    """
    Calculate or preserve target weights.
    """
    frame = frame.copy()

    if "target_weight" in frame.columns:
        weights = pd.to_numeric(frame["target_weight"], errors="coerce")

        if weights.notna().any():
            total = weights.sum(skipna=True)

            if total > 0:
                return weights / total

    if weighting_method == "equal_weight":
        if len(frame) == 0:
            return pd.Series(dtype=float)

        return pd.Series([1.0 / len(frame)] * len(frame), index=frame.index)

    if weighting_method == "blank":
        return pd.Series([pd.NA] * len(frame), index=frame.index)

    raise ValueError(
        f"Unsupported weighting_method={weighting_method!r}. "
        "Supported values: 'equal_weight', 'blank'."
    )


def build_metadata_lookup(
    security_master_path: str | Path = DEFAULT_SECURITY_MASTER_PATH,
    sec_reference_path: str | Path = DEFAULT_SEC_REFERENCE_PATH,
) -> pd.DataFrame:
    """
    Load and enrich security master so ticker metadata is available.
    """
    security_master = load_security_master(security_master_path)

    sec_reference = load_sec_ticker_reference(sec_reference_path)

    security_master = enrich_from_sec_reference(
        security_master=security_master,
        sec_reference=sec_reference,
    )

    security_master = standardize_security_master(security_master)

    return security_master


def update_security_master_with_ticker_list(
    ticker_list: pd.DataFrame,
    security_master_path: str | Path = DEFAULT_SECURITY_MASTER_PATH,
    sec_reference_path: str | Path = DEFAULT_SEC_REFERENCE_PATH,
) -> pd.DataFrame:
    """
    Add ticker-list tickers to the security master and enrich them from SEC reference.
    """
    tickers = ticker_list["ticker"].dropna().astype(str).tolist()

    security_master = load_security_master(security_master_path)

    security_master = add_tickers_to_security_master(
        security_master=security_master,
        tickers=tickers,
        default_source="ticker_list_builder",
    )

    sec_reference = load_sec_ticker_reference(sec_reference_path)

    security_master = enrich_from_sec_reference(
        security_master=security_master,
        sec_reference=sec_reference,
    )

    Path(security_master_path).parent.mkdir(parents=True, exist_ok=True)
    security_master.to_csv(security_master_path, index=False)

    return security_master


def merge_ticker_list_with_metadata(
    ticker_list: pd.DataFrame,
    metadata: pd.DataFrame,
    universe_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build full manual-style universe rows from a simple ticker list.
    """
    universe_name = universe_config["universe_name"]
    account_target = universe_config.get("account_target", "paper")
    source_name = universe_config.get("default_source_name", universe_name)
    strategy_role = universe_config.get("default_strategy_role", "manual_watchlist")
    default_asset_type = universe_config.get("default_asset_type", "stock")
    weighting_method = universe_config.get("weighting_method", "equal_weight")

    ticker_list = ticker_list.copy()
    metadata = metadata.copy()

    metadata["ticker"] = metadata["ticker"].apply(normalize_ticker)

    universe = ticker_list.merge(
        metadata[
            [
                "ticker",
                "company_name",
                "asset_type",
                "sector",
                "industry",
            ]
        ],
        on="ticker",
        how="left",
        suffixes=("", "_security_master"),
    )

    # User-provided columns should override metadata columns where present.
    for column in ["company_name", "asset_type", "sector", "industry"]:
        if column in ticker_list.columns:
            user_values = ticker_list.set_index("ticker")[column]
            universe[column] = universe["ticker"].map(user_values).combine_first(
                universe[column]
            )

    universe["asset_type"] = universe.apply(
        lambda row: (
            row["asset_type"]
            if pd.notna(row.get("asset_type")) and str(row.get("asset_type")).strip()
            else infer_asset_type_from_ticker(
                row["ticker"],
                default_asset_type=default_asset_type,
            )
        ),
        axis=1,
    )

    universe["target_weight"] = calculate_target_weights(
        universe,
        weighting_method=weighting_method,
    )

    universe["universe_name"] = universe_name
    universe["source_type"] = "ticker_list"
    universe["source_name"] = source_name
    universe["strategy_role"] = strategy_role
    universe["account_target"] = account_target
    universe["active"] = True

    if "notes" not in universe.columns:
        universe["notes"] = ""

    today = date.today().isoformat()

    missing_company = universe["company_name"].isna()

    universe.loc[missing_company, "notes"] = (
        universe.loc[missing_company, "notes"].fillna("").astype(str)
        + f" Missing company_name as of {today}; review security master."
    )

    for column in OUTPUT_COLUMNS:
        if column not in universe.columns:
            universe[column] = pd.NA

    universe = universe[OUTPUT_COLUMNS].copy()

    return universe


def save_ticker_list_universe(
    universe: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save generated manual-style universe.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    universe.to_csv(output_path, index=False)


def build_single_ticker_list_universe(
    universe_config: dict[str, Any],
    security_master_path: str | Path = DEFAULT_SECURITY_MASTER_PATH,
    sec_reference_path: str | Path = DEFAULT_SEC_REFERENCE_PATH,
) -> pd.DataFrame:
    """
    Build one ticker-list universe from config.
    """
    input_path = universe_config["input_path"]
    output_path = universe_config["output_path"]
    delimiter = universe_config.get("delimiter", ",")
    decimal = universe_config.get("decimal", ".")

    ticker_list = read_ticker_list(
        input_path=input_path,
        delimiter=delimiter,
        decimal=decimal,
    )

    security_master = update_security_master_with_ticker_list(
        ticker_list=ticker_list,
        security_master_path=security_master_path,
        sec_reference_path=sec_reference_path,
    )

    universe = merge_ticker_list_with_metadata(
        ticker_list=ticker_list,
        metadata=security_master,
        universe_config=universe_config,
    )

    save_ticker_list_universe(
        universe=universe,
        output_path=output_path,
    )

    return universe


def build_ticker_list_universes(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, pd.DataFrame]:
    """
    Build all enabled ticker-list universes.
    """
    config = load_ticker_list_config(config_path)

    security_master_config = config.get("security_master", {})

    security_master_path = security_master_config.get(
        "path",
        DEFAULT_SECURITY_MASTER_PATH,
    )

    sec_reference_path = security_master_config.get(
        "sec_ticker_reference_path",
        DEFAULT_SEC_REFERENCE_PATH,
    )

    ticker_list_configs = config.get("ticker_list_universes", [])

    outputs: dict[str, pd.DataFrame] = {}

    enabled_configs = [
        item for item in ticker_list_configs if item.get("enabled", False)
    ]

    if not enabled_configs:
        print("No enabled ticker-list universes found.")
        return outputs

    for universe_config in enabled_configs:
        universe_name = universe_config["universe_name"]

        print(f"Building ticker-list universe: {universe_name}")

        universe = build_single_ticker_list_universe(
            universe_config=universe_config,
            security_master_path=security_master_path,
            sec_reference_path=sec_reference_path,
        )

        outputs[universe_name] = universe

        print(f"Rows: {len(universe)}")
        print(f"Output: {universe_config['output_path']}")
        print("")

    return outputs


if __name__ == "__main__":
    outputs = build_ticker_list_universes()

    if outputs:
        print("Ticker-list universe build complete.")
        for universe_name, frame in outputs.items():
            print(f"{universe_name}: {len(frame)} rows")
