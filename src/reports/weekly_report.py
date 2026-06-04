from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.universe.config import load_config


def load_weekly_report_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load full project config and validate weekly_report block.
    """
    config = load_config(config_path)

    if "weekly_report" not in config:
        raise ValueError("Missing 'weekly_report' section in config file.")

    if not config["weekly_report"].get("enabled", False):
        raise ValueError("weekly_report is disabled in config file.")

    return config


def load_optional_csv(path: str | Path, label: str) -> pd.DataFrame:
    """
    Load a CSV if it exists. Return empty DataFrame if missing.

    Weekly reports should be resilient, because users may not run
    every optional analysis module.
    """
    path = Path(path)

    if not path.exists():
        print(f"Warning: {label} file not found: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def format_number(value: object, decimals: int = 2) -> str:
    """
    Format numeric values for the report.
    """
    if pd.isna(value):
        return "n/a"

    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return str(value)


def format_currency(value: object) -> str:
    """
    Format currency values.
    """
    if pd.isna(value):
        return "n/a"

    try:
        return f"${float(value):,.2f}"
    except Exception:
        return str(value)


def format_percent(value: object) -> str:
    """
    Format decimal percentages.
    """
    if pd.isna(value):
        return "n/a"

    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return str(value)


def dataframe_to_markdown_table(
    df: pd.DataFrame,
    columns: list[str],
    rename_map: Dict[str, str] | None = None,
    max_rows: int | None = None,
) -> str:
    """
    Convert selected DataFrame columns to a Markdown table.
    """
    if df.empty:
        return "_No data available._"

    available_columns = [column for column in columns if column in df.columns]

    if not available_columns:
        return "_No matching columns available._"

    table = df[available_columns].copy()

    if max_rows is not None:
        table = table.head(max_rows)

    if rename_map:
        table = table.rename(columns=rename_map)

    return table.to_markdown(index=False)


def get_summary_value(summary: pd.DataFrame, metric: str) -> object:
    """
    Retrieve a metric value from the portfolio summary table.
    """
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return pd.NA

    match = summary[summary["metric"] == metric]

    if match.empty:
        return pd.NA

    return match["value"].iloc[0]


def build_executive_summary(
    composite: pd.DataFrame,
    deployment: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    strategy_exposure: pd.DataFrame,
    autopilot_lookthrough: pd.DataFrame,
) -> str:
    """
    Build concise executive summary paragraph.
    """
    total_candidates = len(composite) if not composite.empty else 0

    aligned_count = 0
    top_candidate_count = 0

    if not composite.empty:
        if "composite_signal" in composite.columns:
            aligned_count = int((composite["composite_signal"] == "aligned_candidate").sum())

        if "composite_bucket" in composite.columns:
            top_candidate_count = int((composite["composite_bucket"] == "top_candidate").sum())

    deployment_count = 0
    deployment_total = 0.0

    if not deployment.empty and "suggested_amount" in deployment.columns:
        active = deployment[pd.to_numeric(deployment["suggested_amount"], errors="coerce") > 0]
        deployment_count = len(active)
        deployment_total = pd.to_numeric(active["suggested_amount"], errors="coerce").sum()

    total_market_value = get_summary_value(portfolio_summary, "total_market_value")
    cash_weight = get_summary_value(portfolio_summary, "cash_weight")

    weak_autopilot_count = 0

    if not autopilot_lookthrough.empty and "lookthrough_flag" in autopilot_lookthrough.columns:
        weak_autopilot_count = int(
            autopilot_lookthrough["lookthrough_flag"]
            .astype(str)
            .isin(["fundamental_caution", "missing_composite_score"])
            .sum()
        )

    lines = [
        f"This report reviewed {total_candidates} composite research candidates.",
        f"The model identified {aligned_count} aligned candidates and {top_candidate_count} top-candidate bucket entries.",
        f"The deployment planner suggested {deployment_count} active tranche actions totaling {format_currency(deployment_total)}.",
        f"Current tracked portfolio value is {format_currency(total_market_value)}, with cash weight at {format_percent(cash_weight)}.",
        f"Autopilot look-through flagged {weak_autopilot_count} holdings requiring closer review.",
    ]

    if not strategy_exposure.empty and "strategy" in strategy_exposure.columns:
        lines.append(
            "Strategy exposure is summarized below to compare manual composite positions, Autopilot holdings, reserves, and cash."
        )

    return " ".join(lines)


def build_top_composite_section(composite: pd.DataFrame, top_n: int) -> str:
    """
    Build top composite candidates section.
    """
    if composite.empty:
        return "_Composite scores are unavailable._"

    columns = [
        "composite_rank",
        "ticker",
        "sector",
        "composite_research_score",
        "composite_signal",
        "price_score",
        "fundamental_score",
        "price_signal",
        "fundamental_signal",
    ]

    table = composite.sort_values("composite_rank").copy()

    rename_map = {
        "composite_rank": "Rank",
        "ticker": "Ticker",
        "sector": "Sector",
        "composite_research_score": "Composite",
        "composite_signal": "Composite Signal",
        "price_score": "Price",
        "fundamental_score": "Fundamental",
        "price_signal": "Price Signal",
        "fundamental_signal": "Fundamental Signal",
    }

    return dataframe_to_markdown_table(
        table,
        columns=columns,
        rename_map=rename_map,
        max_rows=top_n,
    )


def build_deployment_section(deployment: pd.DataFrame, top_n: int) -> str:
    """
    Build deployment suggestions section.
    """
    if deployment.empty:
        return "_Deployment plan is unavailable._"

    deployment = deployment.copy()

    if "suggested_amount" not in deployment.columns:
        return "_Deployment plan does not contain suggested amounts._"

    deployment["suggested_amount"] = pd.to_numeric(
        deployment["suggested_amount"],
        errors="coerce",
    )

    active = deployment[deployment["suggested_amount"] > 0].copy()

    if active.empty:
        return "_No active deployment actions were suggested under current rules._"

    columns = [
        "composite_rank",
        "ticker",
        "sector",
        "deployment_action",
        "suggested_amount",
        "composite_research_score",
        "price_score",
        "fundamental_score",
        "deployment_reason",
    ]

    rename_map = {
        "composite_rank": "Rank",
        "ticker": "Ticker",
        "sector": "Sector",
        "deployment_action": "Action",
        "suggested_amount": "Amount",
        "composite_research_score": "Composite",
        "price_score": "Price",
        "fundamental_score": "Fundamental",
        "deployment_reason": "Reason",
    }

    return dataframe_to_markdown_table(
        active.sort_values("composite_rank"),
        columns=columns,
        rename_map=rename_map,
        max_rows=top_n,
    )


def build_manual_review_section(deployment: pd.DataFrame, top_n: int) -> str:
    """
    Build concise manual review section from deployment plan.
    """
    if deployment.empty or "manual_review_required" not in deployment.columns:
        return "_No manual review data available._"

    review = deployment[
        deployment["manual_review_required"].astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()

    if review.empty:
        return "_No deployment rows require manual review._"

    columns = [
        "composite_rank",
        "ticker",
        "sector",
        "deployment_action",
        "composite_signal",
        "deployment_reason",
    ]

    rename_map = {
        "composite_rank": "Rank",
        "ticker": "Ticker",
        "sector": "Sector",
        "deployment_action": "Action",
        "composite_signal": "Composite Signal",
        "deployment_reason": "Reason",
    }

    return dataframe_to_markdown_table(
        review.sort_values("composite_rank"),
        columns=columns,
        rename_map=rename_map,
        max_rows=top_n,
    )


def build_portfolio_snapshot_section(
    positions: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    top_n: int,
) -> str:
    """
    Build portfolio snapshot section.
    """
    if positions.empty:
        return "_Portfolio positions are unavailable._"

    summary_lines = []

    total_market_value = get_summary_value(portfolio_summary, "total_market_value")
    total_unrealized_gain = get_summary_value(portfolio_summary, "total_unrealized_gain")
    total_unrealized_return = get_summary_value(portfolio_summary, "total_unrealized_return")
    cash_weight = get_summary_value(portfolio_summary, "cash_weight")

    summary_lines.append(f"- Total market value: **{format_currency(total_market_value)}**")
    summary_lines.append(f"- Unrealized gain/loss: **{format_currency(total_unrealized_gain)}**")
    summary_lines.append(f"- Unrealized return: **{format_percent(total_unrealized_return)}**")
    summary_lines.append(f"- Cash weight: **{format_percent(cash_weight)}**")

    columns = [
        "account",
        "ticker",
        "strategy",
        "market_value",
        "unrealized_return",
        "portfolio_weight",
        "composite_research_score",
        "holding_analysis_flag",
    ]

    rename_map = {
        "account": "Account",
        "ticker": "Ticker",
        "strategy": "Strategy",
        "market_value": "Value",
        "unrealized_return": "Return",
        "portfolio_weight": "Weight",
        "composite_research_score": "Composite",
        "holding_analysis_flag": "Flag",
    }

    positions = positions.copy()
    if "market_value" in positions.columns:
        positions["market_value"] = pd.to_numeric(positions["market_value"], errors="coerce")
        positions = positions.sort_values("market_value", ascending=False)

    table = dataframe_to_markdown_table(
        positions,
        columns=columns,
        rename_map=rename_map,
        max_rows=top_n,
    )

    return "\n".join(summary_lines) + "\n\n" + table


def build_strategy_exposure_section(strategy_exposure: pd.DataFrame) -> str:
    """
    Build strategy exposure section.
    """
    if strategy_exposure.empty:
        return "_Strategy exposure is unavailable._"

    columns = [
        "strategy",
        "market_value",
        "portfolio_weight",
        "unrealized_return",
        "avg_composite_score",
        "avg_price_score",
        "avg_fundamental_score",
    ]

    rename_map = {
        "strategy": "Strategy",
        "market_value": "Value",
        "portfolio_weight": "Weight",
        "unrealized_return": "Return",
        "avg_composite_score": "Avg Composite",
        "avg_price_score": "Avg Price",
        "avg_fundamental_score": "Avg Fundamental",
    }

    return dataframe_to_markdown_table(
        strategy_exposure,
        columns=columns,
        rename_map=rename_map,
    )


def build_sector_exposure_section(sector_exposure: pd.DataFrame) -> str:
    """
    Build sector exposure section.
    """
    if sector_exposure.empty:
        return "_Sector exposure is unavailable._"

    columns = [
        "sector",
        "market_value",
        "portfolio_weight",
        "holding_count",
    ]

    rename_map = {
        "sector": "Sector",
        "market_value": "Value",
        "portfolio_weight": "Weight",
        "holding_count": "Holdings",
    }

    return dataframe_to_markdown_table(
        sector_exposure,
        columns=columns,
        rename_map=rename_map,
    )


def build_autopilot_lookthrough_section(
    lookthrough: pd.DataFrame,
    top_n: int,
) -> str:
    """
    Build concise Autopilot look-through section.
    """
    if lookthrough.empty:
        return "_Autopilot look-through is unavailable._"

    columns = [
        "universe_name",
        "ticker",
        "sector",
        "target_weight",
        "composite_rank",
        "composite_research_score",
        "composite_signal",
        "lookthrough_flag",
    ]

    rename_map = {
        "universe_name": "Universe",
        "ticker": "Ticker",
        "sector": "Sector",
        "target_weight": "Target Weight",
        "composite_rank": "Rank",
        "composite_research_score": "Composite",
        "composite_signal": "Signal",
        "lookthrough_flag": "Flag",
    }

    if "composite_rank" in lookthrough.columns:
        lookthrough = lookthrough.sort_values(["universe_name", "composite_rank"])

    return dataframe_to_markdown_table(
        lookthrough,
        columns=columns,
        rename_map=rename_map,
        max_rows=top_n,
    )


def build_caution_section(
    composite: pd.DataFrame,
    positions: pd.DataFrame,
    top_n: int,
) -> str:
    """
    Build caution section from composite scores and portfolio holdings.
    """
    caution_frames = []

    if not composite.empty and "composite_signal" in composite.columns:
        caution = composite[
            composite["composite_signal"].astype(str).isin(
                ["possible_value_trap", "partial_data_review_required"]
            )
        ].copy()

        if not caution.empty:
            caution["caution_source"] = "composite_score"
            caution_frames.append(caution)

    if not positions.empty and "holding_analysis_flag" in positions.columns:
        holding_caution = positions[
            positions["holding_analysis_flag"].astype(str).isin(
                ["weak_holding", "missing_model_score"]
            )
        ].copy()

        if not holding_caution.empty:
            holding_caution["caution_source"] = "portfolio_holding"
            caution_frames.append(holding_caution)

    if not caution_frames:
        return "_No major caution rows were identified under current report rules._"

    caution_all = pd.concat(caution_frames, ignore_index=True, sort=False)

    if "composite_rank" in caution_all.columns:
        caution_all["composite_rank"] = pd.to_numeric(
            caution_all["composite_rank"],
            errors="coerce",
        )
        caution_all = caution_all.sort_values("composite_rank", na_position="last")

    columns = [
        "caution_source",
        "ticker",
        "sector",
        "composite_rank",
        "composite_research_score",
        "composite_signal",
        "holding_analysis_flag",
    ]

    rename_map = {
        "caution_source": "Source",
        "ticker": "Ticker",
        "sector": "Sector",
        "composite_rank": "Rank",
        "composite_research_score": "Composite",
        "composite_signal": "Signal",
        "holding_analysis_flag": "Holding Flag",
    }

    return dataframe_to_markdown_table(
        caution_all,
        columns=columns,
        rename_map=rename_map,
        max_rows=top_n,
    )


def build_report_summary_csv(
    composite: pd.DataFrame,
    deployment: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Build and save compact report summary metrics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    active_deployment_total = 0.0
    active_deployment_count = 0

    if not deployment.empty and "suggested_amount" in deployment.columns:
        suggested = pd.to_numeric(deployment["suggested_amount"], errors="coerce")
        active = deployment[suggested > 0]
        active_deployment_count = len(active)
        active_deployment_total = suggested[suggested > 0].sum()

    metrics = [
        {
            "metric": "report_date",
            "value": datetime.now().date().isoformat(),
        },
        {
            "metric": "composite_rows",
            "value": len(composite),
        },
        {
            "metric": "aligned_candidates",
            "value": (
                int((composite["composite_signal"] == "aligned_candidate").sum())
                if not composite.empty and "composite_signal" in composite.columns
                else 0
            ),
        },
        {
            "metric": "active_deployment_count",
            "value": active_deployment_count,
        },
        {
            "metric": "active_deployment_total",
            "value": active_deployment_total,
        },
        {
            "metric": "portfolio_market_value",
            "value": get_summary_value(portfolio_summary, "total_market_value"),
        },
        {
            "metric": "portfolio_cash_weight",
            "value": get_summary_value(portfolio_summary, "cash_weight"),
        },
    ]

    summary = pd.DataFrame(metrics)
    summary.to_csv(output_path, index=False)

    return summary


def build_weekly_report(config_path: str | Path) -> str:
    """
    Build Markdown weekly research memo.
    """
    full_config = load_weekly_report_config(config_path)
    report_config = full_config["weekly_report"]

    report_date = datetime.now().date().isoformat()

    composite = load_optional_csv(
        report_config["input_composite_scores_path"],
        "Composite scores",
    )

    deployment = load_optional_csv(
        report_config["input_deployment_plan_path"],
        "Deployment plan",
    )

    positions = load_optional_csv(
        report_config["input_portfolio_positions_path"],
        "Portfolio positions",
    )

    portfolio_summary = load_optional_csv(
        report_config["input_portfolio_summary_path"],
        "Portfolio summary",
    )

    strategy_exposure = load_optional_csv(
        report_config["input_strategy_exposure_path"],
        "Strategy exposure",
    )

    sector_exposure = load_optional_csv(
        report_config["input_sector_exposure_path"],
        "Sector exposure",
    )

    autopilot_lookthrough = load_optional_csv(
        report_config["input_autopilot_lookthrough_path"],
        "Autopilot look-through",
    )

    top_n_candidates = int(report_config.get("top_n_candidates", 15))
    top_n_deployment_actions = int(report_config.get("top_n_deployment_actions", 10))
    top_n_holdings = int(report_config.get("top_n_holdings", 15))
    top_n_autopilot_holdings = int(report_config.get("top_n_autopilot_holdings", 10))
    top_n_cautions = int(report_config.get("top_n_cautions", 10))

    executive_summary = build_executive_summary(
        composite=composite,
        deployment=deployment,
        portfolio_summary=portfolio_summary,
        strategy_exposure=strategy_exposure,
        autopilot_lookthrough=autopilot_lookthrough,
    )

    sections = [
        f"# Argentum Fund Weekly Research Memo",
        "",
        f"**Report date:** {report_date}",
        "",
        "## Executive Summary",
        "",
        executive_summary,
        "",
        "## Top Composite Research Candidates",
        "",
        build_top_composite_section(composite, top_n_candidates),
        "",
        "## Active Deployment Suggestions",
        "",
        build_deployment_section(deployment, top_n_deployment_actions),
        "",
        "## Manual Review and Caution Items",
        "",
        build_manual_review_section(deployment, top_n_cautions),
        "",
        build_caution_section(composite, positions, top_n_cautions),
        "",
        "## Portfolio Snapshot",
        "",
        build_portfolio_snapshot_section(
            positions=positions,
            portfolio_summary=portfolio_summary,
            top_n=top_n_holdings,
        ),
        "",
        "## Strategy Exposure",
        "",
        build_strategy_exposure_section(strategy_exposure),
        "",
        "## Sector Exposure",
        "",
        build_sector_exposure_section(sector_exposure),
        "",
        "## Autopilot Look-Through",
        "",
        build_autopilot_lookthrough_section(
            autopilot_lookthrough,
            top_n_autopilot_holdings,
        ),
    ]

    if report_config.get("include_disclaimer", True):
        sections.extend(
            [
                "",
                "## Disclaimer",
                "",
                (
                    "This report is generated for research and educational purposes only. "
                    "It is not financial advice, investment advice, or a recommendation to buy or sell securities. "
                    "Outputs should be reviewed manually before any investment decision."
                ),
            ]
        )

    report_text = "\n".join(sections)

    latest_report_path = Path(report_config["output_latest_report_path"])
    dated_report_dir = Path(report_config["output_dated_report_dir"])
    dated_report_path = dated_report_dir / f"argentum_weekly_report_{report_date}.md"

    latest_report_path.parent.mkdir(parents=True, exist_ok=True)
    dated_report_dir.mkdir(parents=True, exist_ok=True)

    latest_report_path.write_text(report_text, encoding="utf-8")
    dated_report_path.write_text(report_text, encoding="utf-8")

    build_report_summary_csv(
        composite=composite,
        deployment=deployment,
        portfolio_summary=portfolio_summary,
        output_path=report_config["output_summary_path"],
    )

    print(f"Saved latest weekly report to {latest_report_path}")
    print(f"Saved dated weekly report to {dated_report_path}")
    print(f"Saved summary metrics to {report_config['output_summary_path']}")

    return report_text


if __name__ == "__main__":
    report = build_weekly_report("config/universe_config.yaml")

    print("\nWeekly report preview:")
    print(report[:2000])
