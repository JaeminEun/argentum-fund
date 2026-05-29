from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.universe.config import load_config
from src.utils.io import load_export_settings, save_csv_outputs


REQUIRED_COMPOSITE_COLUMNS = [
    "ticker",
    "composite_research_score",
    "composite_rank",
    "composite_signal",
    "composite_data_status",
    "price_score",
    "fundamental_score",
]


def load_cash_deployment_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load full project config and validate the cash_deployment block.
    """
    config = load_config(config_path)

    if "cash_deployment" not in config:
        raise ValueError("Missing 'cash_deployment' section in config file.")

    if not config["cash_deployment"].get("enabled", False):
        raise ValueError("cash_deployment is disabled in config file.")

    return config


def load_composite_scores(input_path: str | Path) -> pd.DataFrame:
    """
    Load latest composite score output.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Composite score file not found: {input_path}. "
            "Run composite_scores first."
        )

    return pd.read_csv(input_path)


def validate_composite_scores(scores: pd.DataFrame) -> None:
    """
    Validate required columns for cash deployment planning.
    """
    missing_columns = set(REQUIRED_COMPOSITE_COLUMNS) - set(scores.columns)

    if missing_columns:
        raise ValueError(
            f"Composite scores are missing required columns: {missing_columns}"
        )

    if scores.empty:
        raise ValueError("Composite score file is empty.")


def prepare_composite_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare composite score table for deployment logic.
    """
    scores = scores.copy()

    scores["ticker"] = scores["ticker"].astype(str).str.upper().str.strip()

    numeric_columns = [
        "composite_research_score",
        "composite_rank",
        "price_score",
        "fundamental_score",
        "adjusted_close",
        "below_13w_high",
        "below_52w_high",
        "distance_from_ma_200d",
        "return_13w",
        "volatility_60d",
        "risk_penalty",
        "fundamental_penalty",
    ]

    for column in numeric_columns:
        if column in scores.columns:
            scores[column] = pd.to_numeric(scores[column], errors="coerce")

    return scores


def is_fundamentals_led_candidate(
    row: pd.Series,
    deployment_config: Dict[str, Any],
) -> bool:
    """
    Determine whether a row qualifies as a fundamentals-led test candidate.

    This allows strong fundamental profiles with acceptable price context
    to receive smaller test-tranche suggestions.
    """
    rules = deployment_config.get("fundamentals_led_rules", {})

    if not rules.get("enabled", True):
        return False

    min_fundamental_score = float(rules.get("min_fundamental_score", 65.0))
    min_price_score = float(rules.get("min_price_score", 50.0))
    min_composite_score = float(rules.get("min_composite_score", 60.0))

    return (
        row.get("fundamental_score", np.nan) >= min_fundamental_score
        and row.get("price_score", np.nan) >= min_price_score
        and row.get("composite_research_score", np.nan) >= min_composite_score
    )


def determine_base_action(
    row: pd.Series,
    deployment_config: Dict[str, Any],
) -> str:
    """
    Determine the initial deployment action from composite signal and
    fundamentals-led rules.
    """
    actions = deployment_config.get("actions", {})
    signal = str(row.get("composite_signal", "mixed_signal_review_required"))

    if is_fundamentals_led_candidate(row, deployment_config):
        return "test_tranche"

    return actions.get(signal, "watch_only")


def apply_eligibility_rules(
    row: pd.Series,
    base_action: str,
    deployment_config: Dict[str, Any],
) -> tuple[str, bool, str]:
    """
    Apply score and data-quality constraints.

    Returns:
        final_action
        manual_review_required
        reason
    """
    constraints = deployment_config.get("constraints", {})

    min_composite_score = float(constraints.get("min_composite_score", 60.0))
    min_price_score = float(constraints.get("min_price_score", 45.0))
    min_fundamental_score = float(constraints.get("min_fundamental_score", 55.0))
    require_complete_data = bool(constraints.get("require_complete_data", True))
    block_possible_value_traps = bool(
        constraints.get("block_possible_value_traps", True)
    )

    composite_score = row.get("composite_research_score", np.nan)
    price_score = row.get("price_score", np.nan)
    fundamental_score = row.get("fundamental_score", np.nan)
    composite_signal = str(row.get("composite_signal", ""))
    data_status = str(row.get("composite_data_status", ""))

    reasons = []
    manual_review_required = False

    if require_complete_data and data_status != "complete":
        return (
            "watch_only",
            True,
            "Incomplete composite data; manual review required.",
        )

    if block_possible_value_traps and composite_signal == "possible_value_trap":
        return (
            "watch_only",
            True,
            "Possible value trap; price signal is not supported by fundamentals.",
        )

    if pd.isna(composite_score) or composite_score < min_composite_score:
        return (
            "watch_only",
            False,
            f"Composite score below threshold ({composite_score:.1f} < {min_composite_score:.1f})."
            if pd.notna(composite_score)
            else "Composite score missing.",
        )

    if pd.isna(price_score) or price_score < min_price_score:
        reasons.append(
            f"Price score below deployment threshold ({price_score:.1f} < {min_price_score:.1f})."
            if pd.notna(price_score)
            else "Price score missing."
        )
        manual_review_required = True

    if pd.isna(fundamental_score) or fundamental_score < min_fundamental_score:
        reasons.append(
            f"Fundamental score below deployment threshold ({fundamental_score:.1f} < {min_fundamental_score:.1f})."
            if pd.notna(fundamental_score)
            else "Fundamental score missing."
        )
        manual_review_required = True

    if reasons:
        return (
            "watch_only",
            manual_review_required,
            " ".join(reasons),
        )

    if base_action == "watch_only":
        return (
            "watch_only",
            False,
            "Composite signal does not qualify for deployment under current rules.",
        )

    return (
        base_action,
        manual_review_required,
        "Candidate passes deployment score and data constraints.",
    )


def calculate_suggested_amount(
    action: str,
    deployment_config: Dict[str, Any],
) -> float:
    """
    Calculate suggested deployment amount before portfolio-level constraints.
    """
    base_tranche_amount = float(deployment_config.get("base_tranche_amount", 100.0))
    tranche_multipliers = deployment_config.get("tranche_multipliers", {})

    multiplier = float(tranche_multipliers.get(action, 0.0))

    amount = base_tranche_amount * multiplier

    max_single_position_amount = float(
        deployment_config.get("constraints", {}).get(
            "max_single_position_amount",
            base_tranche_amount,
        )
    )

    return min(amount, max_single_position_amount)


def build_initial_deployment_candidates(
    composite_scores: pd.DataFrame,
    deployment_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build candidate table with preliminary actions and suggested amounts.
    """
    scores = prepare_composite_scores(composite_scores)

    rows = []

    account = deployment_config.get("account", "research_portfolio")

    for _, row in scores.iterrows():
        base_action = determine_base_action(
            row=row,
            deployment_config=deployment_config,
        )

        final_action, manual_review_required, reason = apply_eligibility_rules(
            row=row,
            base_action=base_action,
            deployment_config=deployment_config,
        )

        suggested_amount = calculate_suggested_amount(
            action=final_action,
            deployment_config=deployment_config,
        )

        record = row.to_dict()
        record["account"] = account
        record["base_action"] = base_action
        record["deployment_action"] = final_action
        record["suggested_amount"] = suggested_amount
        record["manual_review_required"] = manual_review_required
        record["deployment_reason"] = reason

        rows.append(record)

    candidates = pd.DataFrame(rows)

    candidates = candidates.sort_values(
        ["composite_rank", "ticker"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return candidates


def apply_sector_candidate_limit(
    candidates: pd.DataFrame,
    deployment_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Limit number of active deployment candidates from the same sector.

    This is a simple concentration control. It does not yet know current
    portfolio holdings, only the proposed deployment plan.
    """
    candidates = candidates.copy()

    constraints = deployment_config.get("constraints", {})
    max_sector_candidates = int(constraints.get("max_sector_candidates", 2))

    if "sector" not in candidates.columns:
        return candidates

    deploy_actions = {"test_tranche", "normal_tranche"}

    candidates["sector_deployment_count"] = 0

    sector_counts: dict[str, int] = {}

    for index, row in candidates.iterrows():
        action = row["deployment_action"]

        if action not in deploy_actions:
            continue

        sector = str(row.get("sector", "Unknown")).strip()

        if sector == "" or sector.lower() in {"nan", "none", "<na>"}:
            sector = "Unknown"

        current_count = sector_counts.get(sector, 0)

        if current_count >= max_sector_candidates:
            candidates.loc[index, "deployment_action"] = "watch_only"
            candidates.loc[index, "suggested_amount"] = 0.0
            candidates.loc[index, "manual_review_required"] = True
            candidates.loc[index, "deployment_reason"] = (
                f"Sector candidate limit reached for {sector}."
            )
            candidates.loc[index, "sector_deployment_count"] = current_count
            continue

        sector_counts[sector] = current_count + 1
        candidates.loc[index, "sector_deployment_count"] = current_count + 1

    return candidates


def apply_period_budget_limit(
    candidates: pd.DataFrame,
    deployment_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Enforce deployable cash and max candidate limits.
    """
    candidates = candidates.copy()

    deployable_cash = float(deployment_config.get("deployable_cash_this_period", 0.0))
    available_cash = float(deployment_config.get("available_cash", deployable_cash))
    max_candidates = int(deployment_config.get("max_candidates", 5))

    budget = min(deployable_cash, available_cash)

    candidates["period_budget"] = budget
    candidates["cumulative_suggested_amount"] = 0.0

    deploy_actions = {"test_tranche", "normal_tranche"}

    used_budget = 0.0
    deployed_count = 0

    for index, row in candidates.iterrows():
        action = row["deployment_action"]
        amount = float(row["suggested_amount"])

        if action not in deploy_actions or amount <= 0:
            continue

        if deployed_count >= max_candidates:
            candidates.loc[index, "deployment_action"] = "watch_only"
            candidates.loc[index, "suggested_amount"] = 0.0
            candidates.loc[index, "manual_review_required"] = True
            candidates.loc[index, "deployment_reason"] = (
                "Maximum candidate count reached for this deployment period."
            )
            continue

        if used_budget + amount > budget:
            remaining_budget = max(budget - used_budget, 0.0)

            if remaining_budget > 0:
                candidates.loc[index, "suggested_amount"] = remaining_budget
                used_budget += remaining_budget
                deployed_count += 1
                candidates.loc[index, "deployment_reason"] = (
                    f"Suggested amount reduced to fit remaining period budget "
                    f"(${remaining_budget:.2f})."
                )
                candidates.loc[index, "cumulative_suggested_amount"] = used_budget
            else:
                candidates.loc[index, "deployment_action"] = "watch_only"
                candidates.loc[index, "suggested_amount"] = 0.0
                candidates.loc[index, "manual_review_required"] = True
                candidates.loc[index, "deployment_reason"] = (
                    "No remaining deployable cash for this period."
                )

            continue

        used_budget += amount
        deployed_count += 1
        candidates.loc[index, "cumulative_suggested_amount"] = used_budget

    return candidates


def add_deployment_interpretation(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Add plain-language deployment interpretation.
    """
    candidates = candidates.copy()

    def fmt_amount(value: float | int | None) -> str:
        if pd.isna(value):
            return "$0.00"
        return f"${float(value):.2f}"

    def interpret_row(row: pd.Series) -> str:
        return (
            f"{row['deployment_action']}: "
            f"{fmt_amount(row.get('suggested_amount'))} suggested for "
            f"{row.get('ticker', 'n/a')} based on composite rank "
            f"{row.get('composite_rank', 'n/a')}. "
            f"{row.get('deployment_reason', '')}"
        )

    candidates["deployment_interpretation"] = candidates.apply(
        interpret_row,
        axis=1,
    )

    return candidates


def select_output_columns(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order deployment plan columns.
    """
    output_columns = [
        "account",
        "composite_rank",
        "ticker",
        "company_name",
        "sector",
        "industry",
        "deployment_action",
        "suggested_amount",
        "base_action",
        "manual_review_required",
        "deployment_reason",
        "composite_research_score",
        "composite_signal",
        "composite_bucket",
        "price_score",
        "price_rank",
        "price_signal",
        "fundamental_score",
        "fundamental_rank",
        "fundamental_signal",
        "adjusted_close",
        "below_13w_high",
        "distance_from_ma_200d",
        "return_13w",
        "volatility_60d",
        "risk_flag",
        "fundamental_penalty",
        "negative_net_income_flag",
        "negative_free_cash_flow_flag",
        "high_liabilities_flag",
        "negative_equity_flag",
        "sanity_filter_flag",
        "sanity_filter_notes",
        "period_budget",
        "cumulative_suggested_amount",
        "deployment_interpretation",
        "composite_interpretation",
    ]

    available_columns = [
        column for column in output_columns if column in candidates.columns
    ]

    return candidates[available_columns]


def build_deployment_plan(
    composite_scores: pd.DataFrame,
    deployment_config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build cash deployment plan from composite scores.
    """
    validate_composite_scores(composite_scores)

    candidates = build_initial_deployment_candidates(
        composite_scores=composite_scores,
        deployment_config=deployment_config,
    )

    candidates = apply_sector_candidate_limit(
        candidates=candidates,
        deployment_config=deployment_config,
    )

    candidates = apply_period_budget_limit(
        candidates=candidates,
        deployment_config=deployment_config,
    )

    candidates = add_deployment_interpretation(candidates)

    candidates = select_output_columns(candidates)

    return candidates


def save_deployment_plan(
    deployment_plan: pd.DataFrame,
    output_path: str | Path,
    export_settings: Dict[str, Any] | None = None,
) -> None:
    """
    Save deployment plan.
    """
    save_csv_outputs(
        df=deployment_plan,
        output_path=output_path,
        export_settings=export_settings,
    )

    print(f"Rows written: {len(deployment_plan)}")


def run_cash_deployment(config_path: str | Path) -> pd.DataFrame:
    """
    Run cash deployment planner from config.
    """
    full_config = load_cash_deployment_config(config_path)

    deployment_config = full_config["cash_deployment"]
    export_settings = load_export_settings(full_config)

    input_path = deployment_config["input_composite_scores_path"]
    output_path = deployment_config["output_deployment_plan_path"]

    composite_scores = load_composite_scores(input_path)

    deployment_plan = build_deployment_plan(
        composite_scores=composite_scores,
        deployment_config=deployment_config,
    )

    save_deployment_plan(
        deployment_plan=deployment_plan,
        output_path=output_path,
        export_settings=export_settings,
    )

    return deployment_plan


if __name__ == "__main__":
    plan = run_cash_deployment("config/universe_config.yaml")

    print("\nDeployment plan preview:")
    preview_columns = [
        "composite_rank",
        "ticker",
        "sector",
        "deployment_action",
        "suggested_amount",
        "composite_research_score",
        "composite_signal",
        "price_score",
        "fundamental_score",
        "deployment_reason",
    ]

    available_preview_columns = [
        column for column in preview_columns if column in plan.columns
    ]

    print(plan[available_preview_columns].head(30))

    print("\nDeployment action counts:")
    print(plan["deployment_action"].value_counts(dropna=False))

    print("\nSuggested deployment total:")
    print(plan["suggested_amount"].sum())
