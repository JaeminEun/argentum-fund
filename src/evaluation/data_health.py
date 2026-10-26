from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.scoring.price_scores import REQUIRED_SCORE_INPUT_COLUMNS
from src.sec.cik import normalize_ticker_for_sec
from src.universe.config import load_config


ISSUE_COLUMNS = ["area", "check", "ticker", "detail"]

# Words that carry no identity when comparing company names.
NAME_STOP_WORDS = {
    "INC", "CORP", "CORPORATION", "CO", "COS", "COMPANY", "COMPANIES", "LTD",
    "PLC", "SA", "NV", "AG", "LP", "LLC", "HOLDINGS", "HOLDING", "GROUP",
    "THE", "CLASS", "CL", "OF", "AND", "DE", "NEW", "INTERNATIONAL", "INTL",
    "TRUST", "COM", "SHS", "ORD", "ADR",
}

DEFAULT_IMPLAUSIBLE_RATIO_BOUNDS = {
    "net_margin": {"upper": 1.0},
    "operating_margin": {"upper": 1.0},
    "fcf_margin": {"upper": 1.0},
    "return_on_equity": {"lower": -3.0, "upper": 3.0},
}


def load_data_health_settings(config_path: str | Path) -> Dict[str, Any]:
    """
    Load input paths and thresholds for the data health report.
    """
    config = load_config(config_path)

    health_config = config.get("evaluation", {}).get("data_health", {})
    accounting_config = config.get("sec_accounting_concepts", {})
    facts_config = config.get("sec_company_facts", {})
    factor_config = config.get("sec_fundamental_factors", {})
    price_factor_config = config.get("price_factors", {})

    metric_period_types = accounting_config.get("metric_period_types", {})

    return {
        "accounting_concepts_path": accounting_config.get("output_path"),
        "fundamental_factors_path": factor_config.get("output_path"),
        "price_history_path": price_factor_config.get("input_price_path"),
        "latest_price_factors_path": price_factor_config.get("output_latest_path"),
        "universe_path": config.get("project", {}).get("output_path"),
        "sec_ticker_reference_path": config.get("sec_reference", {}).get(
            "company_tickers_output_path"
        ),
        "security_master_path": config.get("security_master", {}).get("path"),
        "companyfacts_summary_path": facts_config.get("output_summary_path"),
        "max_universe_age_days": int(health_config.get("max_universe_age_days", 45)),
        "new_registrant_days": int(health_config.get("new_registrant_days", 365)),
        "duration_metrics": [
            metric
            for metric, period_type in metric_period_types.items()
            if period_type in ("duration_annual", "duration_ttm")
        ],
        "accounting_metrics": list(accounting_config.get("concepts", {}).keys()),
        "reference_metric": health_config.get("reference_metric", "net_income"),
        "max_metric_age_days": int(health_config.get("max_metric_age_days", 548)),
        "period_mismatch_tolerance_days": int(
            health_config.get("period_mismatch_tolerance_days", 45)
        ),
        "stale_price_days": int(health_config.get("stale_price_days", 7)),
        "min_price_history_days": int(
            health_config.get("min_price_history_days", 252)
        ),
        "implausible_ratio_bounds": health_config.get(
            "implausible_ratio_bounds",
            DEFAULT_IMPLAUSIBLE_RATIO_BOUNDS,
        ),
    }


def read_optional_csv(path: str | Path | None, **kwargs: Any) -> pd.DataFrame | None:
    """
    Read a CSV if it exists; missing inputs skip their checks.
    """
    if path is None or not Path(path).exists():
        print(f"  Warning: {path} not found; skipping related checks.")
        return None

    frame = pd.read_csv(path, **kwargs)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()

    return frame


def make_issues(
    rows: pd.DataFrame,
    area: str,
    check: str,
    detail: pd.Series,
) -> pd.DataFrame:
    """
    Build issue records for the rows that failed a check.
    """
    return pd.DataFrame(
        {
            "area": area,
            "check": check,
            "ticker": rows["ticker"].values,
            "detail": detail.values,
        },
        columns=ISSUE_COLUMNS,
    )


def check_stale_metrics(
    concepts: pd.DataFrame,
    as_of: pd.Timestamp,
    max_age_days: int,
) -> pd.DataFrame:
    """
    Flag accounting metrics whose period ended long before the as-of date.
    """
    age_days = (as_of - concepts["end"]).dt.days
    stale = concepts[age_days > max_age_days]

    detail = (
        stale["metric_name"]
        + " from "
        + stale["concept_name"].astype(str)
        + " ends "
        + stale["end"].dt.strftime("%Y-%m-%d")
    )

    return make_issues(stale, "fundamentals", "stale_metric", detail)


def check_missing_metrics(
    concepts: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Flag configured accounting metrics with no value for a ticker.

    Stale values may be dropped during extraction, so coverage has to be
    tracked alongside staleness to see the full picture.
    """
    present = set(zip(concepts["ticker"], concepts["metric_name"]))

    missing = pd.DataFrame(
        [
            {"ticker": ticker, "metric_name": metric}
            for ticker in sorted(concepts["ticker"].unique())
            for metric in metrics
            if (ticker, metric) not in present
        ],
        columns=["ticker", "metric_name"],
    )

    detail = missing["metric_name"] + " missing"

    return make_issues(missing, "fundamentals", "missing_metric", detail)


def check_period_mismatch(
    concepts: pd.DataFrame,
    duration_metrics: list[str],
    reference_metric: str,
    tolerance_days: int,
) -> pd.DataFrame:
    """
    Flag income and cash flow metrics that cover a different period than
    the reference metric, which makes ratios between them meaningless.
    """
    duration = concepts[concepts["metric_name"].isin(duration_metrics)]

    reference = (
        duration[duration["metric_name"] == reference_metric]
        .set_index("ticker")["end"]
        .rename("reference_end")
    )

    compared = duration[duration["metric_name"] != reference_metric].join(
        reference,
        on="ticker",
        how="inner",
    )

    gap_days = (compared["end"] - compared["reference_end"]).dt.days.abs()
    mismatched = compared[gap_days > tolerance_days]

    detail = (
        mismatched["metric_name"]
        + " ends "
        + mismatched["end"].dt.strftime("%Y-%m-%d")
        + f" vs {reference_metric} "
        + mismatched["reference_end"].dt.strftime("%Y-%m-%d")
    )

    return make_issues(mismatched, "fundamentals", "period_mismatch", detail)


def check_non_annual_durations(
    concepts: pd.DataFrame,
    duration_metrics: list[str],
) -> pd.DataFrame:
    """
    Flag metrics meant to be annual that fell back to another period length.
    Trailing twelve month values cover a full year and count as annual.
    """
    if "duration_type" not in concepts.columns:
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    duration = concepts[concepts["metric_name"].isin(duration_metrics)]
    non_annual = duration[
        ~duration["duration_type"].astype(str).isin(["annual", "ttm"])
    ]

    detail = (
        non_annual["metric_name"]
        + " period is "
        + non_annual["duration_type"].astype(str)
    )

    return make_issues(non_annual, "fundamentals", "non_annual_duration", detail)


def check_implausible_ratios(
    factors: pd.DataFrame,
    bounds: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Flag ratios outside plausible ranges, which usually signal mismatched
    inputs rather than real business results.
    """
    frames = []

    for metric, metric_bounds in bounds.items():
        if metric not in factors.columns:
            continue

        values = pd.to_numeric(factors[metric], errors="coerce")
        outside = pd.Series(False, index=factors.index)

        if metric_bounds.get("lower") is not None:
            outside |= values < float(metric_bounds["lower"])

        if metric_bounds.get("upper") is not None:
            outside |= values > float(metric_bounds["upper"])

        flagged = factors[outside]
        detail = metric + " = " + values[outside].round(3).astype(str)

        frames.append(
            make_issues(flagged, "fundamentals", "implausible_ratio", detail)
        )

    if not frames:
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def check_stale_prices(
    latest_factors: pd.DataFrame,
    as_of: pd.Timestamp,
    stale_days: int,
) -> pd.DataFrame:
    """
    Flag tickers whose latest price is older than the rest of the universe.
    """
    latest_dates = pd.to_datetime(latest_factors["latest_date"])
    stale = latest_factors[(as_of - latest_dates).dt.days > stale_days]

    detail = "last price " + pd.to_datetime(stale["latest_date"]).dt.strftime(
        "%Y-%m-%d"
    )

    return make_issues(stale, "prices", "stale_price", detail)


def check_short_history(
    prices: pd.DataFrame,
    min_days: int,
) -> pd.DataFrame:
    """
    Flag tickers with too little history for long-window factors.
    """
    counts = prices.groupby("ticker").size().rename("days").reset_index()
    short = counts[counts["days"] < min_days]

    detail = short["days"].astype(str) + " trading days of history"

    return make_issues(short, "prices", "short_history", detail)


def check_missing_score_inputs(latest_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Flag tickers missing a factor the price score depends on.

    The scoring step labels missing values with default flags such as
    'normal_volatility', so these would otherwise go unnoticed.
    """
    input_columns = [
        column
        for column in REQUIRED_SCORE_INPUT_COLUMNS
        if column in latest_factors.columns
    ]

    missing = latest_factors[input_columns].isna()
    flagged = latest_factors[missing.any(axis=1)]

    detail = missing[missing.any(axis=1)].apply(
        lambda row: "missing " + ", ".join(row.index[row]),
        axis=1,
    )

    return make_issues(flagged, "prices", "missing_score_input", detail)


def parse_universe_dates(values: pd.Series) -> pd.Series:
    """
    Parse universe as_of dates written either as ISO or as dd/mm/yyyy.
    """
    text = values.astype("string").str.strip()
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    day_first = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")

    return iso.fillna(day_first)


def check_stale_universe_files(
    universe: pd.DataFrame,
    as_of: pd.Timestamp,
    max_age_days: int,
) -> pd.DataFrame:
    """
    Flag holdings in universe files that have not been refreshed recently.

    Index and Autopilot holdings change over time; an old file scores
    companies that may no longer belong in the universe.
    """
    file_dates = parse_universe_dates(universe["as_of_date"])
    age_days = (as_of - file_dates).dt.days
    stale = universe[age_days > max_age_days]
    stale_dates = file_dates[stale.index]

    detail = (
        stale["universe_name"]
        + " file as of "
        + stale_dates.dt.strftime("%Y-%m-%d")
        + " ("
        + age_days[stale.index].astype(int).astype(str)
        + " days old)"
    )

    return make_issues(stale, "universe", "stale_universe_file", detail)


def check_tickers_not_in_sec(
    stocks: pd.DataFrame,
    sec_tickers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag stocks whose ticker is no longer in the SEC ticker list.

    Usually the company was acquired, delisted or renamed; its SEC data
    can no longer be found by ticker.
    """
    missing = stocks[~stocks["sec_ticker"].isin(sec_tickers["ticker"])]
    detail = (
        missing["universe_name"]
        + ": not in SEC ticker list (acquired, delisted, renamed or not an SEC filer)"
    )

    return make_issues(missing, "universe", "ticker_not_in_sec", detail)


def check_cik_changes(
    stocks: pd.DataFrame,
    sec_tickers: pd.DataFrame,
    security_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag tickers whose SEC registrant differs from the one on record.

    A new CIK behind the same ticker means a new legal entity, such as a
    holding company reorganization; its filings start without the
    predecessor's history.
    """
    recorded = (
        security_master.assign(
            sec_ticker=security_master["ticker"].apply(normalize_ticker_for_sec),
            recorded_cik=pd.to_numeric(security_master["cik"], errors="coerce"),
        )
        .dropna(subset=["recorded_cik"])
        .drop_duplicates("sec_ticker")[["sec_ticker", "recorded_cik"]]
    )
    current = sec_tickers.rename(
        columns={"ticker": "sec_ticker", "cik": "sec_cik", "company_name": "sec_name"}
    )[["sec_ticker", "sec_cik", "sec_name"]]

    merged = (
        stocks.drop_duplicates("ticker")
        .merge(recorded, on="sec_ticker")
        .merge(current, on="sec_ticker")
    )
    changed = merged[
        merged["recorded_cik"].astype("int64") != pd.to_numeric(merged["sec_cik"]).astype("int64")
    ]

    detail = (
        "recorded CIK "
        + changed["recorded_cik"].astype("int64").astype(str)
        + ", SEC now lists "
        + pd.to_numeric(changed["sec_cik"]).astype("int64").astype(str)
        + " ("
        + changed["sec_name"].astype(str)
        + ")"
    )

    return make_issues(changed, "universe", "cik_changed", detail)


def name_tokens(name: object) -> set[str]:
    """
    Identifying words in a company name.

    Single letters are joined to the previous word so that "LOWE S"
    matches "LOWES" and "P G + E" matches "PG&E".
    """
    words = re.sub(r"[^A-Z0-9]+", " ", str(name).upper().replace("'", "")).split()
    merged: list[str] = []

    for word in words:
        if len(word) == 1 and merged and (word == "S" or len(merged[-1]) <= 2):
            merged[-1] += word
        else:
            merged.append(word)

    return {word for word in merged if word not in NAME_STOP_WORDS and len(word) > 1}


def check_name_mismatches(
    stocks: pd.DataFrame,
    sec_tickers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag stocks whose name shares no identifying word with the SEC name.

    This catches a ticker that now belongs to a different company.
    Abbreviated names (e.g. WABTEC) can also be flagged; check the detail.
    """
    merged = stocks.drop_duplicates("ticker").merge(
        sec_tickers.rename(columns={"ticker": "sec_ticker", "company_name": "sec_name"})[
            ["sec_ticker", "sec_name"]
        ],
        on="sec_ticker",
    )
    shares_word = [
        bool(name_tokens(own) & name_tokens(sec))
        for own, sec in zip(merged["company_name"], merged["sec_name"])
    ]
    mismatched = merged[~pd.Series(shares_word, index=merged.index) & merged["company_name"].notna()]

    detail = (
        "universe name '"
        + mismatched["company_name"].astype(str)
        + "' vs SEC name '"
        + mismatched["sec_name"].astype(str)
        + "'"
    )

    return make_issues(mismatched, "universe", "name_mismatch", detail)


def check_new_registrants(
    stocks: pd.DataFrame,
    facts_summary: pd.DataFrame,
    as_of: pd.Timestamp,
    new_registrant_days: int,
) -> pd.DataFrame:
    """
    Flag stocks whose SEC registrant has less than a year of annual reports.

    Spin-offs and new holding companies start a fresh CIK, so trailing
    fundamentals are missing or cover only the new entity.
    """
    if "first_annual_report_filed" not in facts_summary.columns:
        return pd.DataFrame(columns=ISSUE_COLUMNS)

    summary = facts_summary[facts_summary["status"] == "success"].copy()
    summary = summary[summary["ticker"].isin(stocks["ticker"])]
    first_filed = pd.to_datetime(summary["first_annual_report_filed"], errors="coerce")

    no_annual = first_filed.isna()
    recent = (as_of - first_filed).dt.days < new_registrant_days
    flagged = summary[no_annual | recent]
    flagged_first = first_filed[flagged.index]

    detail = pd.Series(
        [
            f"CIK {cik}: no annual report filed yet"
            if pd.isna(first)
            else f"CIK {cik}: first annual report filed {first.date()}"
            for cik, first in zip(flagged["cik"], flagged_first)
        ],
        index=flagged.index,
        dtype="object",
    )

    return make_issues(flagged, "universe", "new_sec_registrant", detail)


def load_universe_for_checks(universe_path: str | Path | None) -> pd.DataFrame | None:
    """
    Load the built universe with identifiers as text.
    """
    universe = read_optional_csv(universe_path, dtype=str)

    if universe is None:
        return None

    universe["sec_ticker"] = universe["ticker"].apply(normalize_ticker_for_sec)

    return universe


def summarize_issues(
    issues: pd.DataFrame,
    tickers_checked: Dict[str, int],
    checks: Dict[str, str],
) -> pd.DataFrame:
    """
    Count flagged tickers per check, including checks with zero issues.
    """
    records = []

    for check, area in checks.items():
        flagged = issues.loc[issues["check"] == check, "ticker"].nunique()
        checked = tickers_checked[area]

        records.append(
            {
                "area": area,
                "check": check,
                "tickers_flagged": flagged,
                "tickers_checked": checked,
                "pct_flagged": flagged / checked if checked else float("nan"),
            }
        )

    return pd.DataFrame(records)


def run_data_health(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all data health checks on current pipeline inputs and outputs.

    Returns:
        summary: one row per check
        issues: one row per flagged ticker and problem
    """
    settings = load_data_health_settings(config_path)

    concepts = read_optional_csv(
        settings["accounting_concepts_path"],
        dtype={"cik_padded": str},
        parse_dates=["end", "filed"],
    )
    factors = read_optional_csv(settings["fundamental_factors_path"])
    prices = read_optional_csv(
        settings["price_history_path"],
        usecols=["ticker", "date"],
        parse_dates=["date"],
    )
    latest_factors = read_optional_csv(settings["latest_price_factors_path"])

    # Measure staleness against the data snapshot rather than today's date
    # so the same inputs always produce the same report.
    if prices is not None:
        as_of = prices["date"].max()
    elif latest_factors is not None:
        as_of = pd.to_datetime(latest_factors["latest_date"]).max()
    else:
        as_of = pd.Timestamp.today().normalize()

    print(f"Data health as of {as_of.date()}")

    issue_frames = []
    checks: Dict[str, str] = {}
    tickers_checked = {"fundamentals": 0, "prices": 0, "universe": 0}

    if concepts is not None:
        tickers_checked["fundamentals"] = concepts["ticker"].nunique()
        issue_frames += [
            check_stale_metrics(concepts, as_of, settings["max_metric_age_days"]),
            check_missing_metrics(concepts, settings["accounting_metrics"]),
            check_period_mismatch(
                concepts,
                settings["duration_metrics"],
                settings["reference_metric"],
                settings["period_mismatch_tolerance_days"],
            ),
            check_non_annual_durations(concepts, settings["duration_metrics"]),
        ]
        checks.update(
            {
                "stale_metric": "fundamentals",
                "missing_metric": "fundamentals",
                "period_mismatch": "fundamentals",
                "non_annual_duration": "fundamentals",
            }
        )

    if factors is not None:
        tickers_checked["fundamentals"] = max(
            tickers_checked["fundamentals"],
            factors["ticker"].nunique(),
        )
        issue_frames.append(
            check_implausible_ratios(factors, settings["implausible_ratio_bounds"])
        )
        checks["implausible_ratio"] = "fundamentals"

    if latest_factors is not None:
        tickers_checked["prices"] = latest_factors["ticker"].nunique()
        issue_frames += [
            check_stale_prices(latest_factors, as_of, settings["stale_price_days"]),
            check_missing_score_inputs(latest_factors),
        ]
        checks.update(
            {
                "stale_price": "prices",
                "missing_score_input": "prices",
            }
        )

    if prices is not None:
        tickers_checked["prices"] = max(
            tickers_checked["prices"],
            prices["ticker"].nunique(),
        )
        issue_frames.append(
            check_short_history(prices, settings["min_price_history_days"])
        )
        checks["short_history"] = "prices"

    universe = load_universe_for_checks(settings["universe_path"])

    if universe is not None:
        tickers_checked["universe"] = universe["ticker"].nunique()
        issue_frames.append(
            check_stale_universe_files(universe, as_of, settings["max_universe_age_days"])
        )
        checks["stale_universe_file"] = "universe"

        stocks = universe[universe["asset_type"] == "stock"]
        sec_tickers = read_optional_csv(
            settings["sec_ticker_reference_path"], dtype={"cik_padded": str}
        )

        if sec_tickers is not None:
            sec_tickers["ticker"] = sec_tickers["ticker"].apply(normalize_ticker_for_sec)
            issue_frames += [
                check_tickers_not_in_sec(stocks, sec_tickers),
                check_name_mismatches(stocks, sec_tickers),
            ]
            checks.update(
                {"ticker_not_in_sec": "universe", "name_mismatch": "universe"}
            )

            security_master = read_optional_csv(settings["security_master_path"], dtype=str)
            if security_master is not None:
                issue_frames.append(check_cik_changes(stocks, sec_tickers, security_master))
                checks["cik_changed"] = "universe"

        facts_summary = read_optional_csv(
            settings["companyfacts_summary_path"], dtype={"cik_padded": str}
        )
        if facts_summary is not None:
            issue_frames.append(
                check_new_registrants(
                    stocks, facts_summary, as_of, settings["new_registrant_days"]
                )
            )
            checks["new_sec_registrant"] = "universe"

    issues = (
        pd.concat(issue_frames, ignore_index=True)
        if issue_frames
        else pd.DataFrame(columns=ISSUE_COLUMNS)
    )
    issues = issues.sort_values(["area", "check", "ticker"]).reset_index(drop=True)

    summary = summarize_issues(issues, tickers_checked, checks)

    return summary, issues


def save_data_health(
    summary: pd.DataFrame,
    issues: pd.DataFrame,
    run_dir: Path,
) -> None:
    """
    Save data health outputs into an evaluation run folder.
    """
    summary.to_csv(run_dir / "data_health_summary.csv", index=False)
    issues.to_csv(run_dir / "data_health_issues.csv", index=False)

    print(f"Saved data health report to {run_dir}")
