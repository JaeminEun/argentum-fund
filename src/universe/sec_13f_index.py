from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.universe.manager_registry import load_manager_registry, pad_cik


DEFAULT_MANAGER_REGISTRY_PATH = Path("data/13f/reference/manager_registry.csv")
DEFAULT_FILING_INDEX_PATH = Path("data/13f/reference/13f_filing_index.csv")
SEC_SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"


FILING_INDEX_COLUMNS = [
    "manager_id",
    "manager_name",
    "filing_entity_name",
    "cik",
    "cik_padded",
    "strategy_family",
    "form",
    "accession_number",
    "accession_number_nodashes",
    "filing_date",
    "report_date",
    "primary_document",
    "filing_detail_url",
    "filing_submissions_url",
    "is_amendment",
    "is_new_latest",
    "last_checked",
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
    Normalize accession number as a string.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def accession_without_dashes(accession_number: str) -> str:
    """
    Remove dashes from SEC accession number for archive URLs.
    """
    return normalize_accession_number(accession_number).replace("-", "")


def build_submissions_url(cik: str | int) -> str:
    """
    Build SEC submissions JSON URL for a padded CIK.
    """
    cik_padded = pad_cik(cik)
    return f"{SEC_SUBMISSIONS_BASE_URL}/CIK{cik_padded}.json"


def build_filing_detail_url(
    cik: str | int,
    accession_number: str,
) -> str:
    """
    Build SEC archive filing detail URL.
    """
    cik_unpadded = str(int(str(cik)))
    accession_nodashes = accession_without_dashes(accession_number)

    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_unpadded}/{accession_nodashes}/"
    )


def fetch_company_submissions(
    cik: str | int,
    request_delay_seconds: float = 0.15,
    timeout_seconds: int = 30,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Fetch SEC company submissions JSON for one CIK.
    """
    url = build_submissions_url(cik)
    headers = {
        "User-Agent": get_sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
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

            return response.json()

        except Exception as error:
            last_error = error
            print(
                f"Warning: SEC submissions request failed on attempt "
                f"{attempt}/{max_retries}: {url}"
            )
            print(f"Reason: {error}")

            if attempt < max_retries:
                time.sleep(request_delay_seconds * attempt)

    raise RuntimeError(
        f"Failed to fetch SEC submissions after {max_retries} attempts: {url}"
    ) from last_error


def recent_filings_to_frame(submissions_json: dict[str, Any]) -> pd.DataFrame:
    """
    Convert SEC submissions recent filings object to a DataFrame.
    """
    recent = submissions_json.get("filings", {}).get("recent", {})

    if not recent:
        return pd.DataFrame()

    frame = pd.DataFrame(recent)

    return frame


def filter_13f_filings(filings: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SEC submissions to 13F-HR and 13F-HR/A filings.
    """
    if filings.empty:
        return filings

    filings = filings.copy()

    if "form" not in filings.columns:
        return pd.DataFrame()

    filings["form"] = filings["form"].astype(str).str.strip()

    filings = filings[filings["form"].isin(["13F-HR", "13F-HR/A"])].copy()

    if filings.empty:
        return filings

    filings["filingDate"] = pd.to_datetime(
        filings["filingDate"],
        errors="coerce",
    )

    filings = filings.sort_values(
        ["filingDate", "accessionNumber"],
        ascending=[False, False],
    )

    return filings.reset_index(drop=True)


def select_latest_13f_filing(filings: pd.DataFrame) -> pd.Series | None:
    """
    Select latest 13F filing from filtered filings.
    """
    if filings.empty:
        return None

    return filings.iloc[0]


def load_existing_filing_index(
    path: str | Path = DEFAULT_FILING_INDEX_PATH,
) -> pd.DataFrame:
    """
    Load existing local filing index if available.
    """
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=FILING_INDEX_COLUMNS)

    frame = pd.read_csv(path)

    for column in FILING_INDEX_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    return frame[FILING_INDEX_COLUMNS].copy()


def get_previous_latest_accessions(existing_index: pd.DataFrame) -> dict[str, str]:
    """
    Get previous latest accession per manager from existing index.
    """
    if existing_index.empty:
        return {}

    previous = {}

    for _, row in existing_index.iterrows():
        manager_id = str(row.get("manager_id", "")).strip()
        accession = normalize_accession_number(row.get("accession_number", ""))

        if manager_id and accession:
            previous[manager_id] = accession

    return previous


def build_filing_index_row(
    manager: pd.Series,
    latest_filing: pd.Series | None,
    previous_accessions: dict[str, str],
    submissions_url: str,
) -> dict[str, Any]:
    """
    Build one filing index row.
    """
    checked_at = datetime.now(timezone.utc).isoformat()

    manager_id = manager["manager_id"]
    cik = str(manager["cik"])
    cik_padded = pad_cik(cik)

    if latest_filing is None:
        return {
            "manager_id": manager_id,
            "manager_name": manager.get("manager_name", ""),
            "filing_entity_name": manager.get("filing_entity_name", ""),
            "cik": cik,
            "cik_padded": cik_padded,
            "strategy_family": manager.get("strategy_family", ""),
            "form": pd.NA,
            "accession_number": pd.NA,
            "accession_number_nodashes": pd.NA,
            "filing_date": pd.NA,
            "report_date": pd.NA,
            "primary_document": pd.NA,
            "filing_detail_url": pd.NA,
            "filing_submissions_url": submissions_url,
            "is_amendment": pd.NA,
            "is_new_latest": False,
            "last_checked": checked_at,
            "status": "no_13f_found",
            "notes": "No 13F-HR or 13F-HR/A filing found in recent SEC submissions.",
        }

    accession_number = normalize_accession_number(
        latest_filing.get("accessionNumber", "")
    )
    accession_nodashes = accession_without_dashes(accession_number)

    previous_accession = previous_accessions.get(manager_id)
    is_new_latest = previous_accession != accession_number

    form = str(latest_filing.get("form", "")).strip()

    return {
        "manager_id": manager_id,
        "manager_name": manager.get("manager_name", ""),
        "filing_entity_name": manager.get("filing_entity_name", ""),
        "cik": cik,
        "cik_padded": cik_padded,
        "strategy_family": manager.get("strategy_family", ""),
        "form": form,
        "accession_number": accession_number,
        "accession_number_nodashes": accession_nodashes,
        "filing_date": latest_filing.get("filingDate", pd.NA),
        "report_date": latest_filing.get("reportDate", pd.NA),
        "primary_document": latest_filing.get("primaryDocument", pd.NA),
        "filing_detail_url": build_filing_detail_url(
            cik=cik,
            accession_number=accession_number,
        ),
        "filing_submissions_url": submissions_url,
        "is_amendment": form == "13F-HR/A",
        "is_new_latest": is_new_latest,
        "last_checked": checked_at,
        "status": "latest_13f_found",
        "notes": (
            "New latest filing detected."
            if is_new_latest
            else "Latest filing unchanged from previous index."
        ),
    }


def update_13f_filing_index(
    manager_registry_path: str | Path = DEFAULT_MANAGER_REGISTRY_PATH,
    output_path: str | Path = DEFAULT_FILING_INDEX_PATH,
    request_delay_seconds: float = 0.15,
    timeout_seconds: int = 30,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Build or update latest 13F filing index for enabled managers.
    """
    managers = load_manager_registry(
        path=manager_registry_path,
        enabled_only=True,
    )

    existing_index = load_existing_filing_index(output_path)
    previous_accessions = get_previous_latest_accessions(existing_index)

    rows = []

    for _, manager in managers.iterrows():
        manager_id = manager["manager_id"]
        cik = str(manager["cik"])
        submissions_url = build_submissions_url(cik)

        print(f"Checking latest 13F filing for {manager_id} ({manager['cik_padded']})")

        try:
            submissions_json = fetch_company_submissions(
                cik=cik,
                request_delay_seconds=request_delay_seconds,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )

            filings = recent_filings_to_frame(submissions_json)
            filings_13f = filter_13f_filings(filings)
            latest_filing = select_latest_13f_filing(filings_13f)

            row = build_filing_index_row(
                manager=manager,
                latest_filing=latest_filing,
                previous_accessions=previous_accessions,
                submissions_url=submissions_url,
            )

        except Exception as error:
            row = {
                "manager_id": manager_id,
                "manager_name": manager.get("manager_name", ""),
                "filing_entity_name": manager.get("filing_entity_name", ""),
                "cik": cik,
                "cik_padded": pad_cik(cik),
                "strategy_family": manager.get("strategy_family", ""),
                "form": pd.NA,
                "accession_number": pd.NA,
                "accession_number_nodashes": pd.NA,
                "filing_date": pd.NA,
                "report_date": pd.NA,
                "primary_document": pd.NA,
                "filing_detail_url": pd.NA,
                "filing_submissions_url": submissions_url,
                "is_amendment": pd.NA,
                "is_new_latest": False,
                "last_checked": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "notes": str(error),
            }

        rows.append(row)

    filing_index = pd.DataFrame(rows)

    for column in FILING_INDEX_COLUMNS:
        if column not in filing_index.columns:
            filing_index[column] = pd.NA

    filing_index = filing_index[FILING_INDEX_COLUMNS].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filing_index.to_csv(output_path, index=False)

    return filing_index


if __name__ == "__main__":
    index = update_13f_filing_index()

    print("")
    print("13F filing index updated.")
    print(f"Rows: {len(index)}")
    print(f"Output: {DEFAULT_FILING_INDEX_PATH}")
    print("")
    print(
        index[
            [
                "manager_id",
                "form",
                "filing_date",
                "report_date",
                "accession_number",
                "is_new_latest",
                "status",
            ]
        ]
    )
