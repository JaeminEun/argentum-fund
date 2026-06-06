from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_FILING_INDEX_PATH = Path("data/13f/reference/13f_filing_index.csv")
DEFAULT_RAW_OUTPUT_DIR = Path("data/13f/raw")

SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"


DOWNLOAD_STATUS_COLUMNS = [
    "manager_id",
    "manager_name",
    "cik",
    "cik_padded",
    "accession_number",
    "accession_number_nodashes",
    "filing_date",
    "report_date",
    "form",
    "filing_directory_url",
    "filing_index_json_url",
    "local_filing_dir",
    "primary_document",
    "primary_document_downloaded",
    "information_table_files",
    "downloaded_file_count",
    "downloaded_at",
    "status",
    "notes",
]


def get_sec_user_agent() -> str:
    """
    Get SEC User-Agent from environment.
    """
    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. Set it with something like:\n"
            'export SEC_USER_AGENT="ArgentumFund/0.1 contact: your_email@example.com"'
        )

    return user_agent


def normalize_accession_number(value: object) -> str:
    """
    Normalize accession number string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def accession_without_dashes(accession_number: str) -> str:
    """
    Remove dashes from accession number.
    """
    return normalize_accession_number(accession_number).replace("-", "")


def normalize_cik_unpadded(value: object) -> str:
    """
    Normalize CIK for SEC archive URL path.

    SEC archive URLs generally use unpadded CIK path segments.
    """
    if pd.isna(value):
        return ""

    cik = str(value).strip()

    if cik.endswith(".0"):
        cik = cik[:-2]

    digits = "".join(character for character in cik if character.isdigit())

    if not digits:
        return ""

    return str(int(digits))


def build_filing_directory_url(
    cik: object,
    accession_number: object,
) -> str:
    """
    Build base filing directory URL in EDGAR archives.
    """
    cik_unpadded = normalize_cik_unpadded(cik)
    accession_nodashes = accession_without_dashes(str(accession_number))

    return f"{SEC_ARCHIVES_BASE_URL}/{cik_unpadded}/{accession_nodashes}"


def build_filing_index_json_url(
    cik: object,
    accession_number: object,
) -> str:
    """
    Build filing directory index.json URL.
    """
    return f"{build_filing_directory_url(cik, accession_number)}/index.json"


def sec_get(
    url: str,
    request_delay_seconds: float = 0.15,
    timeout_seconds: int = 30,
    max_retries: int = 3,
) -> requests.Response:
    """
    GET request with SEC User-Agent, simple retry, and polite delay.
    """
    headers = {
        "User-Agent": get_sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
    }

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(request_delay_seconds)

            response = requests.get(
                url,
                headers=headers,
                timeout=timeout_seconds,
            )

            response.raise_for_status()

            return response

        except Exception as error:
            last_error = error

            print(
                f"Warning: SEC request failed on attempt "
                f"{attempt}/{max_retries}: {url}"
            )
            print(f"Reason: {error}")

            if attempt < max_retries:
                time.sleep(request_delay_seconds * attempt)

    raise RuntimeError(
        f"Failed to download SEC URL after {max_retries} attempts: {url}"
    ) from last_error


def load_filing_index(
    path: str | Path = DEFAULT_FILING_INDEX_PATH,
) -> pd.DataFrame:
    """
    Load local 13F filing index.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"13F filing index not found: {path}. "
            "Run python -m src.universe.sec_13f_index first."
        )

    frame = pd.read_csv(path)

    required_columns = {
        "manager_id",
        "manager_name",
        "cik",
        "cik_padded",
        "form",
        "accession_number",
        "accession_number_nodashes",
        "filing_date",
        "report_date",
        "primary_document",
        "status",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"13F filing index missing required columns: {sorted(missing)}"
        )

    frame = frame.copy()

    frame = frame[frame["status"].astype(str).eq("latest_13f_found")].copy()

    if frame.empty:
        raise ValueError(
            "No downloadable 13F rows found in filing index. "
            "Expected status='latest_13f_found'."
        )

    return frame.reset_index(drop=True)


def load_existing_download_status(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load existing download status if present.
    """
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=DOWNLOAD_STATUS_COLUMNS)

    frame = pd.read_csv(path)

    for column in DOWNLOAD_STATUS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    return frame[DOWNLOAD_STATUS_COLUMNS].copy()


def local_filing_directory(
    row: pd.Series,
    raw_output_dir: str | Path = DEFAULT_RAW_OUTPUT_DIR,
) -> Path:
    """
    Build local filing directory path.
    """
    manager_id = str(row["manager_id"]).strip()
    accession_nodashes = str(row["accession_number_nodashes"]).strip()

    return Path(raw_output_dir) / manager_id / accession_nodashes


def download_filing_index_json(
    row: pd.Series,
    output_dir: Path,
    request_delay_seconds: float,
    timeout_seconds: int,
    max_retries: int,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Download archive folder index.json for one filing.
    """
    index_path = output_dir / "filing_index.json"

    if index_path.exists() and not force_refresh:
        return json.loads(index_path.read_text(encoding="utf-8"))

    url = build_filing_index_json_url(
        cik=row["cik"],
        accession_number=row["accession_number"],
    )

    response = sec_get(
        url=url,
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )

    index_json = response.json()

    index_path.write_text(
        json.dumps(index_json, indent=2),
        encoding="utf-8",
    )

    return index_json


def extract_document_items(index_json: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract filing document items from archive index JSON.
    """
    directory = index_json.get("directory", {})
    items = directory.get("item", [])

    if isinstance(items, dict):
        items = [items]

    if not isinstance(items, list):
        return []

    return items


def should_download_document(
    filename: str,
    primary_document: str | None = None,
) -> bool:
    """
    Decide which documents to download for 13F parsing.

    Download:
    - primary document
    - XML files
    - TXT files
    - HTML/HTM files

    Skip obvious images and miscellaneous binary files.
    """
    filename_lower = filename.lower()

    if primary_document and filename == primary_document:
        return True

    useful_extensions = (
        ".xml",
        ".txt",
        ".html",
        ".htm",
    )

    return filename_lower.endswith(useful_extensions)


def identify_information_table_files(document_names: list[str]) -> list[str]:
    """
    Identify likely 13F information table files.

    This is intentionally broad. The parser will decide which file is usable.
    """
    candidates = []

    for name in document_names:
        lowered = name.lower()

        if not lowered.endswith(".xml"):
            continue

        if any(
            token in lowered
            for token in [
                "infotable",
                "info_table",
                "informationtable",
                "form13f",
                "primary_doc",
            ]
        ):
            candidates.append(name)

    # If no obvious candidate, keep all XMLs as parser candidates.
    if not candidates:
        candidates = [
            name for name in document_names if name.lower().endswith(".xml")
        ]

    return candidates


def download_document(
    base_directory_url: str,
    filename: str,
    output_dir: Path,
    request_delay_seconds: float,
    timeout_seconds: int,
    max_retries: int,
    force_refresh: bool = False,
) -> Path:
    """
    Download one filing document to local output directory.
    """
    output_path = output_dir / filename

    if output_path.exists() and not force_refresh:
        return output_path

    url = f"{base_directory_url}/{filename}"

    response = sec_get(
        url=url,
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )

    output_path.write_bytes(response.content)

    return output_path


def write_metadata(
    row: pd.Series,
    output_dir: Path,
    filing_directory_url: str,
    filing_index_json_url: str,
    downloaded_files: list[str],
    information_table_files: list[str],
) -> None:
    """
    Write filing metadata JSON.
    """
    metadata = {
        "manager_id": row.get("manager_id"),
        "manager_name": row.get("manager_name"),
        "cik": row.get("cik"),
        "cik_padded": row.get("cik_padded"),
        "form": row.get("form"),
        "accession_number": row.get("accession_number"),
        "accession_number_nodashes": row.get("accession_number_nodashes"),
        "filing_date": row.get("filing_date"),
        "report_date": row.get("report_date"),
        "primary_document": row.get("primary_document"),
        "filing_directory_url": filing_directory_url,
        "filing_index_json_url": filing_index_json_url,
        "downloaded_files": downloaded_files,
        "information_table_files": information_table_files,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = output_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def download_single_13f_filing(
    row: pd.Series,
    raw_output_dir: str | Path = DEFAULT_RAW_OUTPUT_DIR,
    request_delay_seconds: float = 0.15,
    timeout_seconds: int = 30,
    max_retries: int = 3,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Download all useful files for one indexed 13F filing.
    """
    output_dir = local_filing_directory(
        row=row,
        raw_output_dir=raw_output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    filing_directory_url = build_filing_directory_url(
        cik=row["cik"],
        accession_number=row["accession_number"],
    )

    filing_index_json_url = build_filing_index_json_url(
        cik=row["cik"],
        accession_number=row["accession_number"],
    )

    index_json = download_filing_index_json(
        row=row,
        output_dir=output_dir,
        request_delay_seconds=request_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        force_refresh=force_refresh,
    )

    document_items = extract_document_items(index_json)

    primary_document = str(row.get("primary_document", "")).strip()

    document_names = [
        str(item.get("name", "")).strip()
        for item in document_items
        if str(item.get("name", "")).strip()
    ]

    downloadable_names = [
        name
        for name in document_names
        if should_download_document(
            filename=name,
            primary_document=primary_document,
        )
    ]

    downloaded_files = []

    for filename in downloadable_names:
        try:
            downloaded_path = download_document(
                base_directory_url=filing_directory_url,
                filename=filename,
                output_dir=output_dir,
                request_delay_seconds=request_delay_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                force_refresh=force_refresh,
            )

            downloaded_files.append(downloaded_path.name)

        except Exception as error:
            print(f"Warning: failed to download document {filename}: {error}")

    information_table_files = identify_information_table_files(downloaded_files)

    write_metadata(
        row=row,
        output_dir=output_dir,
        filing_directory_url=filing_directory_url,
        filing_index_json_url=filing_index_json_url,
        downloaded_files=downloaded_files,
        information_table_files=information_table_files,
    )

    primary_document_downloaded = (
        primary_document in downloaded_files
        if primary_document
        else False
    )

    return {
        "manager_id": row.get("manager_id"),
        "manager_name": row.get("manager_name"),
        "cik": row.get("cik"),
        "cik_padded": row.get("cik_padded"),
        "accession_number": row.get("accession_number"),
        "accession_number_nodashes": row.get("accession_number_nodashes"),
        "filing_date": row.get("filing_date"),
        "report_date": row.get("report_date"),
        "form": row.get("form"),
        "filing_directory_url": filing_directory_url,
        "filing_index_json_url": filing_index_json_url,
        "local_filing_dir": str(output_dir),
        "primary_document": primary_document,
        "primary_document_downloaded": primary_document_downloaded,
        "information_table_files": ";".join(information_table_files),
        "downloaded_file_count": len(downloaded_files),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "downloaded",
        "notes": "",
    }


def download_13f_filings(
    filing_index_path: str | Path = DEFAULT_FILING_INDEX_PATH,
    raw_output_dir: str | Path = DEFAULT_RAW_OUTPUT_DIR,
    status_output_path: str | Path | None = None,
    request_delay_seconds: float = 0.15,
    timeout_seconds: int = 30,
    max_retries: int = 3,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download all filings listed in the local 13F filing index.
    """
    filing_index = load_filing_index(filing_index_path)

    if status_output_path is None:
        status_output_path = Path(raw_output_dir) / "download_status.csv"

    rows = []

    for _, row in filing_index.iterrows():
        manager_id = row["manager_id"]
        accession = row["accession_number"]

        print(f"Downloading 13F filing for {manager_id}: {accession}")

        try:
            status_row = download_single_13f_filing(
                row=row,
                raw_output_dir=raw_output_dir,
                request_delay_seconds=request_delay_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                force_refresh=force_refresh,
            )

        except Exception as error:
            status_row = {
                "manager_id": row.get("manager_id"),
                "manager_name": row.get("manager_name"),
                "cik": row.get("cik"),
                "cik_padded": row.get("cik_padded"),
                "accession_number": row.get("accession_number"),
                "accession_number_nodashes": row.get("accession_number_nodashes"),
                "filing_date": row.get("filing_date"),
                "report_date": row.get("report_date"),
                "form": row.get("form"),
                "filing_directory_url": build_filing_directory_url(
                    cik=row.get("cik"),
                    accession_number=row.get("accession_number"),
                ),
                "filing_index_json_url": build_filing_index_json_url(
                    cik=row.get("cik"),
                    accession_number=row.get("accession_number"),
                ),
                "local_filing_dir": str(
                    local_filing_directory(row, raw_output_dir=raw_output_dir)
                ),
                "primary_document": row.get("primary_document"),
                "primary_document_downloaded": False,
                "information_table_files": "",
                "downloaded_file_count": 0,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "notes": str(error),
            }

        rows.append(status_row)

    status = pd.DataFrame(rows)

    for column in DOWNLOAD_STATUS_COLUMNS:
        if column not in status.columns:
            status[column] = pd.NA

    status = status[DOWNLOAD_STATUS_COLUMNS].copy()

    status_output_path = Path(status_output_path)
    status_output_path.parent.mkdir(parents=True, exist_ok=True)

    status.to_csv(status_output_path, index=False)

    return status


if __name__ == "__main__":
    status = download_13f_filings()

    print("")
    print("13F filing download complete.")
    print(f"Rows: {len(status)}")
    print(f"Output: {DEFAULT_RAW_OUTPUT_DIR / 'download_status.csv'}")
    print("")
    print(
        status[
            [
                "manager_id",
                "accession_number",
                "downloaded_file_count",
                "information_table_files",
                "status",
            ]
        ]
    )
