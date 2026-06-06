from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RAW_INPUT_DIR = Path("data/13f/raw")
DEFAULT_PARSED_OUTPUT_DIR = Path("data/13f/parsed")
DEFAULT_DOWNLOAD_STATUS_PATH = Path("data/13f/raw/download_status.csv")


PARSED_HOLDINGS_COLUMNS = [
    "manager_id",
    "manager_name",
    "cik",
    "cik_padded",
    "form",
    "accession_number",
    "accession_number_nodashes",
    "filing_date",
    "report_date",
    "source_file",
    "issuer_name",
    "class_title",
    "cusip",
    "value_usd_thousands",
    "value_usd",
    "shares",
    "share_type",
    "put_call",
    "investment_discretion",
    "other_manager",
    "voting_sole",
    "voting_shared",
    "voting_none",
    "parse_status",
    "notes",
]


def strip_namespace(tag: str) -> str:
    """
    Strip XML namespace from tag name.
    """
    if "}" in tag:
        return tag.split("}", 1)[1]

    return tag


def normalize_text(value: object) -> str:
    """
    Normalize text values from XML.
    """
    if value is None:
        return ""

    return str(value).strip()


def normalize_cusip(value: object) -> str:
    """
    Normalize CUSIP values.
    """
    text = normalize_text(value)
    text = re.sub(r"[^A-Za-z0-9]", "", text)

    return text.upper()


def parse_numeric(value: object) -> float | None:
    """
    Parse numeric values safely.
    """
    text = normalize_text(value)

    if not text:
        return None

    text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None

def extract_xml_candidates(text: str) -> list[str]:
    """
    Extract possible XML fragments from EDGAR filing text.

    Some SEC .txt files contain several <XML>...</XML> blocks. The first block
    is often the cover page, not the holdings table, so return all plausible
    candidates.
    """
    text = text.strip()

    candidates: list[str] = []

    xml_wrappers = re.findall(
        r"<XML>([\s\S]*?)</XML>",
        text,
        flags=re.IGNORECASE,
    )

    for fragment in xml_wrappers:
        fragment = fragment.strip()
        if fragment:
            candidates.append(fragment)

    information_table_blocks = re.findall(
        r"(<[^>]*informationTable[\s\S]*?</[^>]*informationTable>)",
        text,
        flags=re.IGNORECASE,
    )

    for fragment in information_table_blocks:
        fragment = fragment.strip()
        if fragment:
            candidates.append(fragment)

    if not candidates and (
        text.startswith("<?xml")
        or "<informationTable" in text
        or "informationTable" in text
    ):
        candidates.append(text)

    deduped = []
    seen = set()

    for candidate in candidates:
        key = candidate[:500]
        if key in seen:
            continue

        seen.add(key)
        deduped.append(candidate)

    return deduped


def nested_child_text(
    element: ET.Element,
    parent_name: str,
    child_name: str,
) -> str:
    """
    Return text for a nested child by local tag names, ignoring namespaces.
    """
    for child in list(element):
        if strip_namespace(child.tag) != parent_name:
            continue

        for nested in list(child):
            if strip_namespace(nested.tag) == child_name:
                return normalize_text(nested.text)

    return ""

def child_text(element: ET.Element, child_name: str) -> str:
    """
    Return text for a child by local tag name, ignoring namespaces.
    """
    for child in list(element):
        if strip_namespace(child.tag) == child_name:
            return normalize_text(child.text)

    return ""


def nested_child_text(
    element: ET.Element,
    parent_name: str,
    child_name: str,
) -> str:
    """
    Return text for a nested child by local tag names, ignoring namespaces.
    """
    for child in list(element):
        if strip_namespace(child.tag) != parent_name:
            continue

        for nested in list(child):
            if strip_namespace(nested.tag) == child_name:
                return normalize_text(nested.text)

    return ""

def find_info_table_elements(root: ET.Element) -> list[ET.Element]:
    """
    Find all infoTable elements regardless of namespace.
    """
    info_tables = []

    for element in root.iter():
        if strip_namespace(element.tag) == "infoTable":
            info_tables.append(element)

    return info_tables

def parsed_table_from_root(root: ET.Element, source_file: str) -> pd.DataFrame:
    """
    Convert an XML root containing 13F infoTable rows into a DataFrame.
    """
    info_table_elements = find_info_table_elements(root)

    if strip_namespace(root.tag) == "infoTable":
        info_table_elements = [root]

    if not info_table_elements:
        raise ValueError("No infoTable elements found.")

    rows = [
        parse_info_table_element(element)
        for element in info_table_elements
    ]

    frame = pd.DataFrame(rows)
    frame["source_file"] = source_file
    frame["parse_status"] = "parsed"
    frame["notes"] = ""

    return frame

def parse_info_table_element(element: ET.Element) -> dict[str, Any]:
    """
    Parse one 13F infoTable XML element.
    """
    value_usd_thousands = parse_numeric(child_text(element, "value"))
    shares = parse_numeric(nested_child_text(element, "shrsOrPrnAmt", "sshPrnamt"))

    value_usd = (
        value_usd_thousands * 1000
        if value_usd_thousands is not None
        else None
    )

    return {
        "issuer_name": child_text(element, "nameOfIssuer"),
        "class_title": child_text(element, "titleOfClass"),
        "cusip": normalize_cusip(child_text(element, "cusip")),
        "value_usd_thousands": value_usd_thousands,
        "value_usd": value_usd,
        "shares": shares,
        "share_type": nested_child_text(element, "shrsOrPrnAmt", "sshPrnamtType"),
        "put_call": child_text(element, "putCall"),
        "investment_discretion": child_text(element, "investmentDiscretion"),
        "other_manager": child_text(element, "otherManager"),
        "voting_sole": parse_numeric(nested_child_text(element, "votingAuthority", "Sole")),
        "voting_shared": parse_numeric(nested_child_text(element, "votingAuthority", "Shared")),
        "voting_none": parse_numeric(nested_child_text(element, "votingAuthority", "None")),
    }


def parse_information_table_file(file_path: str | Path) -> pd.DataFrame:
    """
    Parse one 13F information table file.

    Strategy:
    1. For XML files, first try ElementTree.parse directly.
    2. If that fails, try extracted XML fragments from text.
    3. Return the parsed table with the most infoTable rows.
    """
    file_path = Path(file_path)

    parsed_tables = []
    errors = []

    # Best path for clean SEC XML files.
    if file_path.suffix.lower() == ".xml":
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            parsed = parsed_table_from_root(root, file_path.name)
            parsed_tables.append(parsed)
        except Exception as error:
            errors.append(f"direct ET.parse failed: {error}")

    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    xml_candidates = extract_xml_candidates(raw_text)

    for xml_text in xml_candidates:
        try:
            # Encode to bytes so XML declarations with encoding do not break parsing.
            root = ET.fromstring(xml_text.encode("utf-8"))
            parsed = parsed_table_from_root(root, file_path.name)
            parsed_tables.append(parsed)
        except Exception as error:
            errors.append(f"fragment parse failed: {error}")

    if not parsed_tables:
        raise ValueError(
            f"No parseable information table found in file: {file_path}. "
            f"Tried {len(xml_candidates)} extracted XML candidates. "
            f"Errors: {' | '.join(errors[:10])}"
        )

    parsed_tables = sorted(parsed_tables, key=len, reverse=True)

    return parsed_tables[0]


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """
    Load per-filing metadata JSON from downloader output.
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def metadata_to_columns(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Convert downloader metadata to parsed holdings columns.
    """
    return {
        "manager_id": metadata.get("manager_id"),
        "manager_name": metadata.get("manager_name"),
        "cik": metadata.get("cik"),
        "cik_padded": metadata.get("cik_padded"),
        "form": metadata.get("form"),
        "accession_number": metadata.get("accession_number"),
        "accession_number_nodashes": metadata.get("accession_number_nodashes"),
        "filing_date": metadata.get("filing_date"),
        "report_date": metadata.get("report_date"),
    }


def get_candidate_information_table_files(filing_dir: str | Path) -> list[Path]:
    """
    Identify candidate information table files.

    Prioritize metadata-listed information table files, but also try all
    XML/TXT/HTML files because SEC filing structures vary.
    """
    filing_dir = Path(filing_dir)
    metadata_path = filing_dir / "metadata.json"

    candidates: list[Path] = []

    if metadata_path.exists():
        metadata = load_metadata(metadata_path)

        for filename in metadata.get("information_table_files", []):
            candidate = filing_dir / filename

            if candidate.exists():
                candidates.append(candidate)

    for pattern in ["*.xml", "*.txt", "*.html", "*.htm"]:
        for candidate in sorted(filing_dir.glob(pattern)):
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


def choose_best_parsed_table(parsed_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Choose the best parsed table from candidate XML files.

    The best table is currently the one with the most rows.
    """
    if not parsed_tables:
        return pd.DataFrame()

    parsed_tables = sorted(parsed_tables, key=len, reverse=True)

    return parsed_tables[0]


def parse_single_filing_directory(filing_dir: str | Path) -> pd.DataFrame:
    """
    Parse one downloaded 13F filing directory.
    """
    filing_dir = Path(filing_dir)
    metadata = load_metadata(filing_dir / "metadata.json")
    metadata_columns = metadata_to_columns(metadata)

    candidate_files = get_candidate_information_table_files(filing_dir)

    parsed_tables = []
    errors = []

    for candidate_file in candidate_files:
        try:
            parsed = parse_information_table_file(candidate_file)
            parsed_tables.append(parsed)
        except Exception as error:
            errors.append(f"{candidate_file.name}: {error}")

    if not parsed_tables:
        row = {
            **metadata_columns,
            "source_file": "",
            "issuer_name": "",
            "class_title": "",
            "cusip": "",
            "value_usd_thousands": None,
            "value_usd": None,
            "shares": None,
            "share_type": "",
            "put_call": "",
            "investment_discretion": "",
            "other_manager": "",
            "voting_sole": None,
            "voting_shared": None,
            "voting_none": None,
            "parse_status": "error",
            "notes": "No parseable information table found. " + " | ".join(errors),
        }

        return pd.DataFrame([row])

    best = choose_best_parsed_table(parsed_tables)

    for column, value in metadata_columns.items():
        best[column] = value

    for column in PARSED_HOLDINGS_COLUMNS:
        if column not in best.columns:
            best[column] = pd.NA

    return best[PARSED_HOLDINGS_COLUMNS].copy()


def load_download_status(
    path: str | Path = DEFAULT_DOWNLOAD_STATUS_PATH,
) -> pd.DataFrame:
    """
    Load downloader status file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Download status file not found: {path}. "
            "Run python -m src.universe.sec_13f_downloader first."
        )

    status = pd.read_csv(path)

    required_columns = {"manager_id", "local_filing_dir", "status"}
    missing = required_columns - set(status.columns)

    if missing:
        raise ValueError(
            f"Download status file missing required columns: {sorted(missing)}"
        )

    status = status[status["status"].astype(str).eq("downloaded")].copy()

    if status.empty:
        raise ValueError("No downloaded filings found in download status.")

    return status.reset_index(drop=True)


def save_parsed_holdings(
    holdings: pd.DataFrame,
    output_dir: str | Path = DEFAULT_PARSED_OUTPUT_DIR,
) -> Path:
    """
    Save parsed holdings for one manager/accession.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if holdings.empty:
        raise ValueError("Cannot save empty parsed holdings frame.")

    manager_id = str(holdings["manager_id"].iloc[0]).strip()
    report_date = str(holdings["report_date"].iloc[0]).strip()
    accession = str(holdings["accession_number_nodashes"].iloc[0]).strip()

    safe_report_date = report_date.replace("-", "")
    filename = f"{manager_id}_{safe_report_date}_{accession}_holdings_raw.csv"

    output_path = output_dir / filename

    holdings.to_csv(output_path, index=False)

    return output_path


def parse_downloaded_13f_filings(
    download_status_path: str | Path = DEFAULT_DOWNLOAD_STATUS_PATH,
    output_dir: str | Path = DEFAULT_PARSED_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Parse all downloaded 13F filings listed in download_status.csv.
    """
    status = load_download_status(download_status_path)

    summary_rows = []

    all_holdings = []

    for _, row in status.iterrows():
        manager_id = row["manager_id"]
        filing_dir = Path(row["local_filing_dir"])

        print(f"Parsing 13F information table for {manager_id}: {filing_dir}")

        try:
            holdings = parse_single_filing_directory(filing_dir)
            output_path = save_parsed_holdings(
                holdings=holdings,
                output_dir=output_dir,
            )

            parsed_rows = len(holdings)
            parse_status = (
                "parsed"
                if (holdings["parse_status"] == "parsed").any()
                else "error"
            )

            notes = ""

            all_holdings.append(holdings)

        except Exception as error:
            parsed_rows = 0
            parse_status = "error"
            output_path = pd.NA
            notes = str(error)

        summary_rows.append(
            {
                "manager_id": manager_id,
                "local_filing_dir": str(filing_dir),
                "parsed_rows": parsed_rows,
                "output_path": output_path,
                "status": parse_status,
                "notes": notes,
            }
        )

    summary = pd.DataFrame(summary_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "parse_status.csv"
    summary.to_csv(summary_path, index=False)

    if all_holdings:
        combined = pd.concat(all_holdings, ignore_index=True)
        combined_path = output_dir / "latest_13f_holdings_raw_combined.csv"
        combined.to_csv(combined_path, index=False)

    return summary


if __name__ == "__main__":
    summary = parse_downloaded_13f_filings()

    print("")
    print("13F information table parsing complete.")
    print(f"Rows: {len(summary)}")
    print(f"Output: {DEFAULT_PARSED_OUTPUT_DIR / 'parse_status.csv'}")
    print("")
    print(summary)
