"""
Refresh index-tracking universes from official ETF holdings files.

The S&P 500 Value universe is taken from State Street's daily SPYV
holdings file. Each refresh is kept as a dated snapshot with a diff
against the previous universe, so changes to the investable universe
are visible and runs made on different dates stay comparable.

Usage:
    python -m src.universe.etf_holdings --universe sp500_value
    python -m src.universe.etf_holdings --universe sp500_value --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import requests

from .config import load_config
from .schema import map_weight_columns, normalize_ticker


# Common stock tickers, including share classes such as BF.B.
# Cash lines ("-") and placeholder identifiers for rights or CVRs
# (e.g. "2602335D") do not match and are skipped.
STOCK_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")

SSGA_COLUMN_MAP = {
    "Name": "company_name",
    "Ticker": "ticker",
    "Identifier": "cusip",
    "SEDOL": "sedol",
    "Weight": "weight_pct",
    "Shares Held": "shares_held",
    "Local Currency": "currency",
}

# Columns written to the universe input file, in order.
UNIVERSE_FILE_COLUMNS = [
    "ticker",
    "company_name",
    "asset_type",
    "target_weight",
    "sector",
    "industry",
    "cusip",
    "sedol",
    "shares_held",
    "currency",
    "date_added",
    "as_of_date",
    "source_url",
    "notes",
    "active",
]

DOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0 (ArgentumFund universe refresh)"}


def parse_ssga_as_of_date(raw: pd.DataFrame) -> date:
    """
    Read the holdings date from the header block, e.g. 'As of 23-Oct-2026'.
    """
    for _, row in raw.head(10).iterrows():
        if str(row.iloc[0]).strip().lower().startswith("holdings"):
            text = str(row.iloc[1]).replace("As of", "").strip()
            return datetime.strptime(text, "%d-%b-%Y").date()

    raise ValueError("Could not find the 'Holdings: As of' date in the file.")


def parse_ssga_holdings(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, date]:
    """
    Parse a State Street daily holdings xlsx file.

    Returns
    -------
    holdings:
        Stock holdings with standardized columns and decimal weights.
    skipped:
        Cash, money market and placeholder lines that are not stocks.
    as_of_date:
        Holdings date stated in the file.
    """
    raw = pd.read_excel(path, header=None, dtype=str)
    as_of_date = parse_ssga_as_of_date(raw)

    header_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Name"]

    if len(header_rows) == 0:
        raise ValueError("Could not find the holdings table header row.")

    header_row = header_rows[0]
    table = raw.iloc[header_row + 1 :].copy()
    table.columns = [str(c).strip() for c in raw.iloc[header_row]]

    missing = [c for c in SSGA_COLUMN_MAP if c not in table.columns]
    if missing:
        raise ValueError(f"Holdings file is missing expected columns: {missing}")

    table = table[list(SSGA_COLUMN_MAP)].rename(columns=SSGA_COLUMN_MAP)

    # The table ends at the first fully blank row; disclaimers follow.
    blank = table.isna().all(axis=1)
    if blank.any():
        table = table.loc[: blank.idxmax()].iloc[:-1]

    table["ticker"] = table["ticker"].apply(normalize_ticker)
    table["company_name"] = table["company_name"].str.split().str.join(" ")
    table["weight_pct"] = pd.to_numeric(table["weight_pct"], errors="coerce")
    table["shares_held"] = pd.to_numeric(table["shares_held"], errors="coerce")

    is_stock = table["ticker"].str.match(STOCK_TICKER_PATTERN)

    holdings = table[is_stock].copy()
    holdings["target_weight"] = holdings["weight_pct"] / 100
    holdings = holdings.drop(columns="weight_pct").reset_index(drop=True)

    skipped = table[~is_stock].reset_index(drop=True)

    return holdings, skipped, as_of_date


def previous_cusips(previous: pd.DataFrame) -> pd.Series:
    """
    CUSIP column of a previous universe file.

    Older hand-made files stored the CUSIP under 'identifier'.
    """
    for column in ("cusip", "identifier"):
        if column in previous.columns:
            return previous[column].astype("string").str.strip()

    return pd.Series(pd.NA, index=previous.index, dtype="string")


def diff_universe(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each ticker as kept, added, removed or renamed.

    A rename is a removed ticker whose CUSIP reappears under a new ticker.
    Kept tickers whose CUSIP changed are flagged as 'identifier_changed',
    which usually means a corporate reorganization worth checking.
    """
    # Older files also listed cash and placeholder lines; compare stocks only.
    previous = previous[
        previous["ticker"].apply(normalize_ticker).str.match(STOCK_TICKER_PATTERN)
    ]

    before = pd.DataFrame(
        {
            "ticker": previous["ticker"].apply(normalize_ticker),
            "cusip": previous_cusips(previous),
            "company_name": previous.get("company_name", previous.get("name")),
            "weight": map_weight_columns(previous).get("target_weight"),
        }
    )
    after = current[["ticker", "cusip", "company_name", "target_weight"]].rename(
        columns={"target_weight": "weight"}
    )

    merged = before.merge(
        after, on="ticker", how="outer", suffixes=("_before", "_after"), indicator=True
    )
    merged["change"] = merged["_merge"].astype(str).map(
        {"both": "kept", "left_only": "removed", "right_only": "added"}
    )
    merged["new_ticker"] = pd.NA

    identifier_changed = (merged["change"] == "kept") & (
        merged["cusip_before"].fillna("") != merged["cusip_after"].fillna("")
    )
    merged.loc[identifier_changed, "change"] = "identifier_changed"

    added_by_cusip = (
        merged.loc[merged["change"] == "added"]
        .dropna(subset=["cusip_after"])
        .set_index("cusip_after")["ticker"]
    )

    for index, row in merged[merged["change"] == "removed"].iterrows():
        new_ticker = added_by_cusip.get(row["cusip_before"])
        if pd.isna(row["cusip_before"]) or new_ticker is None:
            continue

        merged.loc[index, "change"] = "renamed"
        merged.loc[index, "new_ticker"] = new_ticker
        merged = merged[~((merged["ticker"] == new_ticker) & (merged["change"] == "added"))]

    merged = merged.drop(columns="_merge")
    order = ["removed", "added", "renamed", "identifier_changed", "kept"]
    merged["change"] = pd.Categorical(merged["change"], categories=order, ordered=True)

    return merged.sort_values(["change", "ticker"]).reset_index(drop=True)


def carry_forward_metadata(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Fill sector, industry and date_added from earlier records.

    The holdings file has no sector data, so sectors come from the
    previous universe file (matched by ticker, then by CUSIP to follow
    renames) and then from the security master, if given.
    """
    current = current.copy()
    previous = previous.copy()
    previous["ticker"] = previous["ticker"].apply(normalize_ticker)
    previous["cusip"] = previous_cusips(previous)

    for column in ("sector", "industry", "date_added"):
        if column not in previous.columns:
            previous[column] = pd.NA

        by_ticker = previous.drop_duplicates("ticker").set_index("ticker")[column]
        by_cusip = (
            previous.dropna(subset=["cusip"]).drop_duplicates("cusip").set_index("cusip")[column]
        )

        values = current["ticker"].map(by_ticker)
        values = values.fillna(current["cusip"].map(by_cusip))

        if reference is not None and column in reference.columns:
            by_reference = reference.drop_duplicates("ticker").set_index("ticker")[column]
            values = values.fillna(current["ticker"].map(by_reference))

        # The old file used '-' for unknown sectors.
        current[column] = values.replace({"-": pd.NA, "": pd.NA})

    return current


def download_holdings(url: str, destination: Path) -> Path:
    """
    Download a holdings file to destination.
    """
    response = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=60)
    response.raise_for_status()

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)

    return destination


def read_universe_file(path: Path, input_format: Dict[str, Any]) -> pd.DataFrame:
    """
    Read a universe input file, keeping identifiers as text.
    """
    return pd.read_csv(
        path,
        sep=input_format.get("delimiter", ","),
        decimal=input_format.get("decimal", "."),
        dtype={"cusip": str, "identifier": str, "sedol": str, "date_added": str},
    )


def previous_as_of_date(previous: pd.DataFrame) -> str | None:
    """
    ISO holdings date of a previous universe file, if it records one.
    """
    if "as_of_date" not in previous.columns:
        return None

    values = previous["as_of_date"].dropna().astype(str)
    if values.empty:
        return None

    text = values.iloc[0]
    # Hand-made files used dd/mm/yyyy.
    parsed = pd.to_datetime(text, dayfirst="/" in text, errors="coerce")

    return None if pd.isna(parsed) else parsed.date().isoformat()


def find_universe_config(config: Dict[str, Any], universe_name: str) -> Dict[str, Any]:
    for universe in config.get("universes", []):
        if universe.get("universe_name") == universe_name:
            if "holdings_refresh" not in universe:
                raise ValueError(
                    f"Universe '{universe_name}' has no holdings_refresh settings."
                )
            return universe

    raise ValueError(f"Universe '{universe_name}' not found in configuration.")


def build_universe_file(
    holdings: pd.DataFrame,
    previous: pd.DataFrame,
    as_of_date: date,
    source_url: str,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Turn parsed holdings into the universe input file layout.
    """
    universe = carry_forward_metadata(holdings, previous, reference)

    # Names already in the previous file were there at least since its date.
    in_previous = universe["ticker"].isin(
        previous["ticker"].apply(normalize_ticker)
    ) | universe["cusip"].isin(previous_cusips(previous).dropna())
    universe.loc[in_previous, "date_added"] = universe.loc[
        in_previous, "date_added"
    ].fillna(previous_as_of_date(previous))

    universe["asset_type"] = "stock"
    universe["target_weight"] = universe["target_weight"].round(8)
    universe["date_added"] = universe["date_added"].fillna(as_of_date.isoformat())
    universe["as_of_date"] = as_of_date.isoformat()
    universe["source_url"] = source_url
    universe["notes"] = pd.NA
    universe["active"] = True

    return universe[UNIVERSE_FILE_COLUMNS]


def refresh_holdings_universe(
    config_path: str | Path,
    universe_name: str,
    apply: bool = False,
    holdings_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Download the latest holdings, save a dated snapshot and diff, and,
    with apply=True, replace the universe input file.

    The previous input file is archived as a snapshot under its own
    holdings date before it is replaced, so no universe is lost.
    """
    config = load_config(config_path)
    universe_config = find_universe_config(config, universe_name)
    refresh_config = universe_config["holdings_refresh"]

    input_path = Path(universe_config["input_path"])
    input_format = universe_config.get("input_format", {})
    snapshot_dir = Path(refresh_config["snapshot_dir"])
    raw_dir = snapshot_dir / "raw"
    url = refresh_config["url"]

    if holdings_path is None:
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = download_holdings(url, Path(tmp) / "holdings.xlsx")
            holdings, skipped, as_of_date = parse_ssga_holdings(downloaded)
            raw_path = raw_dir / f"{as_of_date.isoformat()}.xlsx"
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, raw_path)
    else:
        holdings, skipped, as_of_date = parse_ssga_holdings(holdings_path)

    previous = read_universe_file(input_path, input_format)

    reference = None
    reference_path = refresh_config.get("sector_reference_path")
    if reference_path and Path(reference_path).exists():
        reference = pd.read_csv(reference_path, dtype=str)

    universe = build_universe_file(holdings, previous, as_of_date, url, reference)
    diff = diff_universe(previous, universe)

    stamp = as_of_date.isoformat()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{stamp}.csv"
    diff_path = snapshot_dir / f"{stamp}_diff.csv"

    universe.to_csv(snapshot_path, index=False)
    diff.to_csv(diff_path, index=False)

    result = {
        "universe_name": universe_name,
        "as_of_date": stamp,
        "previous_as_of_date": previous_as_of_date(previous),
        "stocks": len(universe),
        "skipped_lines": len(skipped),
        "missing_sector": int(universe["sector"].isna().sum()),
        "changes": diff["change"].value_counts().reindex(diff["change"].cat.categories, fill_value=0).to_dict(),
        "snapshot_path": str(snapshot_path),
        "diff_path": str(diff_path),
        "applied": False,
    }

    if apply:
        previous_stamp = result["previous_as_of_date"] or "previous"
        archive_path = snapshot_dir / f"{previous_stamp}.csv"
        if not archive_path.exists():
            previous.to_csv(archive_path, index=False)

        universe.to_csv(
            input_path,
            sep=input_format.get("delimiter", ","),
            decimal=input_format.get("decimal", "."),
            index=False,
        )
        result["applied"] = True

    return result


def print_refresh_summary(result: Dict[str, Any], diff_path: str | Path) -> None:
    diff = pd.read_csv(diff_path)

    print(
        f"{result['universe_name']}: holdings as of {result['as_of_date']} "
        f"(current file: {result['previous_as_of_date']})"
    )
    print(
        f"  {result['stocks']} stocks, {result['skipped_lines']} non-stock lines skipped, "
        f"{result['missing_sector']} without a sector"
    )

    for change in ("removed", "added", "renamed", "identifier_changed"):
        rows = diff[diff["change"] == change]
        if rows.empty:
            continue

        if change == "renamed":
            items = [f"{r.ticker}->{r.new_ticker}" for r in rows.itertuples()]
        else:
            items = rows["ticker"].tolist()

        print(f"  {change} ({len(rows)}): {', '.join(items)}")

    print(f"  Snapshot: {result['snapshot_path']}")
    print(f"  Diff:     {result['diff_path']}")

    if result["applied"]:
        print("  Applied: universe input file replaced.")
    else:
        print("  Not applied. Review the diff, then rerun with --apply.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh an index universe from its ETF holdings file."
    )
    parser.add_argument("--config", default="config/universe_config.yaml")
    parser.add_argument("--universe", default="sp500_value")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace the universe input file after saving the snapshot.",
    )
    parser.add_argument(
        "--holdings-file",
        default=None,
        help="Use a local holdings xlsx instead of downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = refresh_holdings_universe(
        args.config,
        args.universe,
        apply=args.apply,
        holdings_path=args.holdings_file,
    )
    print_refresh_summary(result, result["diff_path"])


if __name__ == "__main__":
    main()
