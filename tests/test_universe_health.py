import pandas as pd

from src.evaluation.data_health import (
    check_cik_changes,
    check_name_mismatches,
    check_new_registrants,
    check_stale_universe_files,
    check_tickers_not_in_sec,
    name_tokens,
)
from src.fundamentals.sec_company_facts import summarize_annual_reports

AS_OF = pd.Timestamp("2026-10-23")

SEC_TICKERS = pd.DataFrame(
    {
        "ticker": ["AAPL", "XOM", "LOW", "WAB"],
        "cik": [320193, 2115436, 60667, 943452],
        "company_name": [
            "Apple Inc.",
            "ExxonMobil Holdings Corp",
            "LOWES COMPANIES INC",
            "WESTINGHOUSE AIR BRAKE TECHNOLOGIES CORP",
        ],
    }
)


def stocks(tickers, names=None):
    frame = pd.DataFrame(
        {
            "ticker": tickers,
            "company_name": names or tickers,
            "universe_name": "sp500_value",
            "asset_type": "stock",
        }
    )
    frame["sec_ticker"] = frame["ticker"]
    return frame


def test_stale_universe_files_reads_iso_and_day_first_dates():
    universe = pd.DataFrame(
        {
            "ticker": ["AAPL", "LULU"],
            "universe_name": ["sp500_value", "burry_autopilot"],
            "as_of_date": ["2026-10-23", "23/05/2026"],
        }
    )

    issues = check_stale_universe_files(universe, AS_OF, max_age_days=45)

    assert issues["ticker"].tolist() == ["LULU"]
    assert "2026-05-23 (153 days old)" in issues["detail"].iloc[0]


def test_tickers_not_in_sec_are_flagged():
    issues = check_tickers_not_in_sec(stocks(["AAPL", "EA"]), SEC_TICKERS)

    assert issues["ticker"].tolist() == ["EA"]


def test_cik_change_is_flagged_against_security_master():
    security_master = pd.DataFrame(
        {"ticker": ["AAPL", "XOM"], "cik": ["320193.0", "34088.0"]}
    )

    issues = check_cik_changes(stocks(["AAPL", "XOM"]), SEC_TICKERS, security_master)

    assert issues["ticker"].tolist() == ["XOM"]
    assert "recorded CIK 34088, SEC now lists 2115436" in issues["detail"].iloc[0]


def test_name_tokens_ignore_punctuation_and_legal_suffixes():
    assert name_tokens("LOWE S COS INC") & name_tokens("LOWES COMPANIES INC")
    assert name_tokens("P G + E CORP") == name_tokens("PG&E Corp") == {"PGE"}
    assert name_tokens("BROWN FORMAN CORP CLASS B") == {"BROWN", "FORMAN"}


def test_name_mismatch_flags_unrelated_names_only():
    universe = stocks(
        ["AAPL", "LOW", "WAB"],
        ["APPLE INC", "LOWE S COS INC", "WABTEC CORP"],
    )

    issues = check_name_mismatches(universe, SEC_TICKERS)

    assert issues["ticker"].tolist() == ["WAB"]


def test_new_registrants_flag_recent_or_missing_annual_reports():
    summary = pd.DataFrame(
        {
            "ticker": ["AAPL", "FDXF", "HONA", "GONE"],
            "cik": [320193, 2082247, 2089271, 1],
            "status": ["success", "success", "success", "unmapped_cik"],
            "first_annual_report_filed": ["2009-10-27", "2026-08-05", None, None],
        }
    )

    issues = check_new_registrants(
        stocks(["AAPL", "FDXF", "HONA", "GONE"]), summary, AS_OF, 365
    )

    assert issues["ticker"].tolist() == ["FDXF", "HONA"]
    assert issues["detail"].tolist() == [
        "CIK 2082247: first annual report filed 2026-08-05",
        "CIK 2089271: no annual report filed yet",
    ]


def test_summarize_annual_reports_counts_distinct_filings():
    facts_root = {
        "us-gaap": {
            "Assets": {
                "units": {
                    "USD": [
                        {"form": "10-Q", "filed": "2025-05-01", "accn": "q1"},
                        {"form": "10-K", "filed": "2026-02-25", "accn": "k1"},
                        {"form": "10-K", "filed": "2026-02-25", "accn": "k1"},
                    ]
                }
            },
            "Revenues": {
                "units": {"USD": [{"form": "10-K", "filed": "2027-02-24", "accn": "k2"}]}
            },
        }
    }

    assert summarize_annual_reports(facts_root) == {
        "first_annual_report_filed": "2026-02-25",
        "annual_reports_filed": 2,
    }

    assert summarize_annual_reports({}) == {
        "first_annual_report_filed": "",
        "annual_reports_filed": 0,
    }
