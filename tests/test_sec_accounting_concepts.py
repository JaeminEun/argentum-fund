import pandas as pd

from src.fundamentals.sec_accounting_concepts import (
    extract_accounting_concepts_for_company,
    extract_candidate_concept_records,
)


def duration_fact(start, end, val, form="10-K", fp="FY", filed=None):
    return {
        "start": start,
        "end": end,
        "val": val,
        "form": form,
        "fp": fp,
        "filed": filed or end,
    }


def instant_fact(end, val, form="10-K", fp="FY"):
    return {"end": end, "val": val, "form": form, "fp": fp, "filed": end}


def make_facts(concepts):
    return {
        "entityName": "Example Corp",
        "facts": {
            "us-gaap": {
                name: {"units": {"USD": records}}
                for name, records in concepts.items()
            }
        },
    }


REVENUE_CONFIG = {
    "taxonomy": "us-gaap",
    "candidates": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
}


def test_later_candidate_with_newer_data_is_selected():
    # Revenues stops in 2017, as it did for many companies after ASC 606.
    facts = make_facts(
        {
            "Revenues": [duration_fact("2017-01-01", "2017-12-31", 100.0)],
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                duration_fact("2024-01-01", "2024-12-31", 250.0),
            ],
        }
    )

    selected = extract_accounting_concepts_for_company(
        facts=facts,
        ticker="EX",
        cik_padded="0000000001",
        concepts_config={"revenue": REVENUE_CONFIG},
        preferred_forms=["10-K", "10-Q"],
        annual_forms=["10-K"],
        metric_period_types={"revenue": "duration_annual"},
    )

    assert len(selected) == 1
    row = selected.iloc[0]
    assert row["concept_name"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert row["end"] == pd.Timestamp("2024-12-31")
    assert row["val"] == 250.0


def test_earlier_candidate_wins_when_both_report_same_period():
    facts = make_facts(
        {
            "Revenues": [duration_fact("2024-01-01", "2024-12-31", 260.0)],
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                duration_fact("2024-01-01", "2024-12-31", 250.0, filed="2025-03-01"),
            ],
        }
    )

    records = extract_candidate_concept_records(
        facts=facts,
        ticker="EX",
        cik_padded="0000000001",
        metric_name="revenue",
        concept_config=REVENUE_CONFIG,
    )

    assert records["concept_name"].unique().tolist() == ["Revenues"]
    assert records["val"].tolist() == [260.0]


def test_proxy_statement_values_do_not_displace_annual_filings():
    # Proxy pay-versus-performance tables report NetIncomeLoss for periods
    # whose 10-K values are only under ProfitLoss.
    facts = make_facts(
        {
            "NetIncomeLoss": [
                duration_fact("2025-01-01", "2025-12-31", 2204.0, form="DEF 14A", fp=None),
                duration_fact("2009-01-01", "2009-03-31", 207.0, form="10-Q", fp="Q1"),
            ],
            "ProfitLoss": [
                duration_fact("2020-01-01", "2020-12-31", 865.0),
                duration_fact("2025-01-01", "2025-12-31", 4152.0),
            ],
        }
    )

    selected = extract_accounting_concepts_for_company(
        facts=facts,
        ticker="EX",
        cik_padded="0000000001",
        concepts_config={
            "net_income": {
                "taxonomy": "us-gaap",
                "candidates": ["NetIncomeLoss", "ProfitLoss"],
            }
        },
        preferred_forms=["10-K", "10-Q"],
        annual_forms=["10-K"],
        metric_period_types={"net_income": "duration_annual"},
    )

    row = selected.iloc[0]
    assert row["concept_name"] == "ProfitLoss"
    assert row["end"] == pd.Timestamp("2025-12-31")
    assert row["val"] == 4152.0


def test_instant_facts_survive_candidate_merge():
    # Balance sheet facts have no start date; they must not be dropped.
    facts = make_facts(
        {
            "StockholdersEquity": [instant_fact("2011-09-30", 26.0)],
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
                instant_fact("2011-09-30", 27.0),
                instant_fact("2025-09-30", 38.0),
            ],
        }
    )

    records = extract_candidate_concept_records(
        facts=facts,
        ticker="EX",
        cik_padded="0000000001",
        metric_name="stockholders_equity",
        concept_config={
            "taxonomy": "us-gaap",
            "candidates": [
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ],
        },
    )

    by_end = records.set_index("end")["val"].to_dict()

    assert by_end == {
        pd.Timestamp("2011-09-30"): 26.0,
        pd.Timestamp("2025-09-30"): 38.0,
    }
    assert records["period_kind"].eq("instant").all()


def test_missing_concepts_return_empty_frame():
    records = extract_candidate_concept_records(
        facts=make_facts({}),
        ticker="EX",
        cik_padded="0000000001",
        metric_name="revenue",
        concept_config=REVENUE_CONFIG,
    )

    assert records.empty


BALANCE_SHEET_CONFIG = {
    "assets": {"taxonomy": "us-gaap", "candidates": ["Assets"]},
    "liabilities": {"taxonomy": "us-gaap", "candidates": ["Liabilities"]},
}

DERIVED_LIABILITIES_CONFIG = {
    "enabled": True,
    "equity_candidates": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity",
    ],
}


def extract_balance_sheet(concepts, **kwargs):
    return extract_accounting_concepts_for_company(
        facts=make_facts(concepts),
        ticker="EX",
        cik_padded="0000000001",
        concepts_config=BALANCE_SHEET_CONFIG,
        preferred_forms=["10-K", "10-Q"],
        annual_forms=["10-K"],
        metric_period_types={
            "assets": "instant_latest",
            "liabilities": "instant_latest",
        },
        derived_liabilities_config=DERIVED_LIABILITIES_CONFIG,
        **kwargs,
    ).set_index("metric_name")


def test_liabilities_derived_from_assets_minus_total_equity():
    selected = extract_balance_sheet(
        {
            "Assets": [instant_fact("2025-12-31", 1000.0)],
            "StockholdersEquity": [instant_fact("2025-12-31", 350.0)],
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
                instant_fact("2025-12-31", 400.0),
            ],
        }
    )

    liabilities = selected.loc["liabilities"]
    assert liabilities["val"] == 600.0
    assert liabilities["taxonomy"] == "derived"
    assert liabilities["end"] == pd.Timestamp("2025-12-31")


def test_reported_liabilities_kept_when_current():
    selected = extract_balance_sheet(
        {
            "Assets": [instant_fact("2025-12-31", 1000.0)],
            "Liabilities": [instant_fact("2025-12-31", 580.0)],
            "StockholdersEquity": [instant_fact("2025-12-31", 350.0)],
        }
    )

    assert selected.loc["liabilities", "val"] == 580.0
    assert selected.loc["liabilities", "taxonomy"] == "us-gaap"


def test_stale_reported_liabilities_replaced_by_derived_value():
    selected = extract_balance_sheet(
        {
            "Assets": [instant_fact("2025-12-31", 1000.0)],
            "Liabilities": [instant_fact("2016-12-31", 300.0)],
            "StockholdersEquity": [instant_fact("2025-12-31", 350.0)],
        }
    )

    assert selected.loc["liabilities", "val"] == 650.0
    assert selected.loc["liabilities", "end"] == pd.Timestamp("2025-12-31")


def test_no_derivation_without_equity_on_assets_date():
    selected = extract_balance_sheet(
        {
            "Assets": [instant_fact("2025-12-31", 1000.0)],
            "StockholdersEquity": [instant_fact("2024-12-31", 350.0)],
        }
    )

    assert "liabilities" not in selected.index


def test_values_far_older_than_latest_filing_are_dropped():
    selected = extract_balance_sheet(
        {
            "Assets": [instant_fact("2025-12-31", 1000.0)],
            "Liabilities": [instant_fact("2016-12-31", 300.0)],
        },
        max_metric_age_days=548,
    )

    # No equity to derive from, so the 2016 value is dropped, not scored.
    assert selected.index.tolist() == ["assets"]
