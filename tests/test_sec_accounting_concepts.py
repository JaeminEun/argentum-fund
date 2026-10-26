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
