import pandas as pd
import pytest

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


def extract_ttm(concepts, candidates):
    return extract_accounting_concepts_for_company(
        facts=make_facts(concepts),
        ticker="EX",
        cik_padded="0000000001",
        concepts_config={"revenue": {"taxonomy": "us-gaap", "candidates": candidates}},
        preferred_forms=["10-K", "10-Q"],
        annual_forms=["10-K"],
        metric_period_types={"revenue": "duration_ttm"},
    ).iloc[0]


def test_ttm_combines_annual_and_year_to_date_with_52_53_week_calendar():
    # Apple-style fiscal calendar, values in billions.
    row = extract_ttm(
        {
            "Revenue": [
                duration_fact("2023-10-01", "2024-09-28", 391.035),
                duration_fact("2024-09-29", "2024-12-28", 124.300, "10-Q", "Q1"),
                duration_fact("2024-09-29", "2025-03-29", 219.659, "10-Q", "Q2"),
                duration_fact("2024-09-29", "2025-09-27", 416.161),
                duration_fact("2025-09-28", "2025-12-27", 143.756, "10-Q", "Q1"),
                duration_fact("2025-12-28", "2026-03-28", 111.184, "10-Q", "Q2"),
                duration_fact("2025-09-28", "2026-03-28", 254.940, "10-Q", "Q2"),
            ]
        },
        ["Revenue"],
    )

    assert row["val"] == pytest.approx(416.161 + 254.940 - 219.659)
    assert row["end"] == pd.Timestamp("2026-03-28")
    assert row["duration_type"] == "ttm"
    assert 360 <= row["period_days"] <= 370


def test_ttm_uses_year_to_date_cash_flow_values():
    # Cash flow statements only report year-to-date in 10-Qs.
    row = extract_ttm(
        {
            "Capex": [
                duration_fact("2024-01-01", "2024-12-31", 425.0),
                duration_fact("2025-01-01", "2025-03-31", 117.0, "10-Q", "Q1"),
                duration_fact("2025-01-01", "2025-12-31", 527.0),
                duration_fact("2026-01-01", "2026-03-31", 114.0, "10-Q", "Q1"),
            ]
        },
        ["Capex"],
    )

    assert row["val"] == pytest.approx(527.0 + 114.0 - 117.0)
    assert row["end"] == pd.Timestamp("2026-03-31")


def test_ttm_falls_back_to_annual_after_a_10k():
    row = extract_ttm(
        {
            "Revenue": [
                duration_fact("2025-01-01", "2025-09-30", 300.0, "10-Q", "Q3"),
                duration_fact("2025-01-01", "2025-12-31", 400.0),
            ]
        },
        ["Revenue"],
    )

    assert row["val"] == 400.0
    assert row["duration_type"] == "annual"


def test_ttm_without_prior_year_to_date_falls_back_to_annual():
    row = extract_ttm(
        {
            "Revenue": [
                duration_fact("2025-01-01", "2025-12-31", 400.0),
                duration_fact("2026-01-01", "2026-03-31", 110.0, "10-Q", "Q1"),
            ]
        },
        ["Revenue"],
    )

    assert row["val"] == 400.0
    assert row["duration_type"] == "annual"


def test_old_concept_ttm_does_not_beat_newer_annual_value():
    row = extract_ttm(
        {
            "Revenues": [
                duration_fact("2016-01-01", "2016-12-31", 90.0),
                duration_fact("2016-01-01", "2016-03-31", 20.0, "10-Q", "Q1"),
                duration_fact("2017-01-01", "2017-12-31", 100.0),
                duration_fact("2018-01-01", "2018-03-31", 25.0, "10-Q", "Q1"),
                duration_fact("2017-01-01", "2017-03-31", 22.0, "10-Q", "Q1"),
            ],
            "RevenueFromContract": [
                duration_fact("2025-01-01", "2025-12-31", 250.0),
            ],
        },
        ["Revenues", "RevenueFromContract"],
    )

    assert row["concept_name"] == "RevenueFromContract"
    assert row["end"] == pd.Timestamp("2025-12-31")


def test_ttm_combines_concepts_when_10k_and_10q_use_different_tags():
    # Annual net income under NetIncomeLoss, quarterly only under ProfitLoss.
    row = extract_ttm(
        {
            "NetIncomeLoss": [
                duration_fact("2025-01-01", "2025-03-31", 148.2, "10-Q", "Q1"),
                duration_fact("2025-01-01", "2025-12-31", 643.8),
            ],
            "ProfitLoss": [
                duration_fact("2026-01-01", "2026-03-31", 138.1, "10-Q", "Q1"),
            ],
        },
        ["NetIncomeLoss", "ProfitLoss"],
    )

    assert row["val"] == pytest.approx(643.8 + 138.1 - 148.2)
    assert row["end"] == pd.Timestamp("2026-03-31")
    assert row["concept_name"] == "NetIncomeLoss + ProfitLoss"
