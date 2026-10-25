from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.evaluation.runs import rank_correlation
from src.features.price_factors import (
    calculate_price_factors,
    load_price_history,
)
from src.scoring.price_scores import calculate_price_scores
from src.universe.config import load_config


DEFAULT_FORWARD_HORIZONS = {
    "4w": 20,
    "13w": 63,
    "26w": 126,
}

# Approximate trading days between weekly rebalance dates. Used to
# thin out overlapping forward-return windows for the t-statistic.
TRADING_DAYS_PER_WEEK = 5


def load_backtest_settings(config_path: str | Path) -> Dict[str, Any]:
    """
    Load the config sections the price backtest depends on.
    """
    config = load_config(config_path)

    for section in ("price_factors", "price_scores"):
        if section not in config:
            raise ValueError(f"Missing '{section}' section in config file.")

    evaluation_config = config.get("evaluation", {})

    return {
        "factor_config": config["price_factors"],
        "score_config": config["price_scores"],
        "backtest_config": evaluation_config.get("price_backtest", {}),
    }


def add_forward_returns(
    factors: pd.DataFrame,
    price_column: str,
    horizons: Dict[str, int],
) -> pd.DataFrame:
    """
    Add forward returns over each horizon, measured in trading rows
    per ticker.

    These columns look into the future and must only be used to
    evaluate scores, never to calculate them.
    """
    factors = factors.sort_values(["ticker", "date"]).reset_index(drop=True)

    grouped_prices = factors.groupby("ticker")[price_column]

    for label, periods in horizons.items():
        future_price = grouped_prices.shift(-periods)
        factors[f"fwd_return_{label}"] = future_price / factors[price_column] - 1

    return factors


def select_rebalance_dates(
    dates: pd.Series,
    warmup_days: int,
    frequency: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[pd.Timestamp]:
    """
    Pick the last trading date in each period after a warmup.

    The warmup skips the early history where long-window factors
    (52-week highs, 200-day averages) are not yet meaningful.
    """
    calendar = pd.Series(pd.to_datetime(dates).unique()).sort_values()
    calendar = calendar.iloc[warmup_days:]

    if start_date is not None:
        calendar = calendar[calendar >= pd.Timestamp(start_date)]

    if end_date is not None:
        calendar = calendar[calendar < pd.Timestamp(end_date)]

    if calendar.empty:
        return []

    periods = calendar.dt.to_period(frequency)

    return calendar.groupby(periods).max().tolist()


def score_cross_section(
    factors: pd.DataFrame,
    date: pd.Timestamp,
    score_config: Dict[str, Any],
    forward_columns: list[str],
) -> pd.DataFrame:
    """
    Score all tickers as of one date using the production scoring code,
    then attach their forward returns.
    """
    snapshot = factors[factors["date"] == date].copy()
    snapshot = snapshot.rename(columns={"date": "latest_date"})

    scores = calculate_price_scores(
        factors=snapshot,
        score_config=score_config,
    )

    scores = scores.merge(
        snapshot[["ticker"] + forward_columns],
        on="ticker",
        how="left",
    )

    scores["rebalance_date"] = date

    return scores


def evaluate_date(
    scores: pd.DataFrame,
    score_column: str,
    bucket_column: str,
    horizon: str,
    min_names: int,
) -> Dict[str, Any] | None:
    """
    Measure how well one date's scores lined up with later returns.
    """
    forward_column = f"fwd_return_{horizon}"

    valid = scores[scores[score_column].notna() & scores[forward_column].notna()]

    if len(valid) < min_names:
        return None

    # Excess return versus an equal-weighted basket of the same universe.
    excess = valid[forward_column] - valid[forward_column].mean()

    record: Dict[str, Any] = {
        "rebalance_date": valid["rebalance_date"].iloc[0],
        "horizon": horizon,
        "names": len(valid),
        "ic": rank_correlation(valid[score_column], valid[forward_column]),
        "universe_return": valid[forward_column].mean(),
    }

    bucket_excess = excess.groupby(valid[bucket_column].astype(str)).mean()

    for bucket, value in bucket_excess.items():
        record[f"excess_{bucket}"] = value

    top = valid[bucket_column].astype(str) == "top_candidate"
    weak = valid[bucket_column].astype(str) == "weak"

    if top.any() and weak.any():
        record["top_minus_weak"] = (
            valid.loc[top, forward_column].mean()
            - valid.loc[weak, forward_column].mean()
        )

    if top.any():
        record["top_hit_rate"] = (excess[top] > 0).mean()

    return record


def summarize_period(
    by_date: pd.DataFrame,
    horizons: Dict[str, int],
) -> pd.DataFrame:
    """
    Summarize per-date results into one row per horizon.

    Forward windows from weekly dates overlap, which makes naive
    t-statistics too optimistic. The t-statistic here only uses dates
    spaced at least one full horizon apart.
    """
    records = []

    for label, periods in horizons.items():
        rows = by_date[by_date["horizon"] == label].sort_values("rebalance_date")

        if rows.empty:
            continue

        step = max(1, math.ceil(periods / TRADING_DAYS_PER_WEEK))
        independent_ic = rows["ic"].iloc[::step].dropna()

        if len(independent_ic) > 1 and independent_ic.std() > 0:
            ic_t_stat = (
                independent_ic.mean()
                / independent_ic.std()
                * math.sqrt(len(independent_ic))
            )
        else:
            ic_t_stat = np.nan

        record: Dict[str, Any] = {
            "horizon": label,
            "dates": len(rows),
            "independent_dates": len(independent_ic),
            "mean_ic": rows["ic"].mean(),
            "ic_t_stat": ic_t_stat,
            "pct_dates_ic_positive": (rows["ic"] > 0).mean(),
        }

        for column in ("top_minus_weak", "top_hit_rate"):
            if column in rows.columns:
                record[f"mean_{column}"] = rows[column].mean()

        for column in sorted(c for c in rows.columns if c.startswith("excess_")):
            record[f"mean_{column}"] = rows[column].mean()

        records.append(record)

    return pd.DataFrame(records)


def run_price_backtest(
    config_path: str | Path,
    include_holdout: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest the price score point-in-time on weekly rebalance dates.

    Returns:
        summary: one row per period and horizon
        by_date: one row per rebalance date and horizon
    """
    settings = load_backtest_settings(config_path)
    factor_config = settings["factor_config"]
    score_config = settings["score_config"]
    backtest_config = settings["backtest_config"]

    price_column = factor_config.get("price_column", "adjusted_close")
    horizons = backtest_config.get("forward_horizons", DEFAULT_FORWARD_HORIZONS)
    score_column = backtest_config.get("score_column", "risk_adjusted_timing_score")
    bucket_column = backtest_config.get("bucket_column", "score_bucket")
    frequency = backtest_config.get("rebalance_frequency", "W-FRI")
    warmup_days = int(backtest_config.get("warmup_days", 252))
    min_names = int(backtest_config.get("min_names_per_date", 20))
    holdout_start = backtest_config.get("holdout_start", "2025-01-01")

    print("Calculating point-in-time price factors...")
    prices = load_price_history(factor_config["input_price_path"])
    factors = calculate_price_factors(prices, factor_config)
    factors = add_forward_returns(factors, price_column, horizons)

    forward_columns = [f"fwd_return_{label}" for label in horizons]

    periods = {"in_sample": (None, holdout_start)}

    if include_holdout:
        periods["holdout"] = (holdout_start, None)
    else:
        print(
            f"Holdout period from {holdout_start} is sealed. "
            "Pass --include-holdout only for a final check of a settled design."
        )

    date_records = []

    for period, (start, end) in periods.items():
        rebalance_dates = select_rebalance_dates(
            factors["date"],
            warmup_days=warmup_days,
            frequency=frequency,
            start_date=start,
            end_date=end,
        )

        print(f"Scoring {len(rebalance_dates)} {period} rebalance dates...")

        for date in rebalance_dates:
            scores = score_cross_section(
                factors,
                date,
                score_config,
                forward_columns,
            )

            for label in horizons:
                record = evaluate_date(
                    scores,
                    score_column=score_column,
                    bucket_column=bucket_column,
                    horizon=label,
                    min_names=min_names,
                )

                if record is not None:
                    record["period"] = period
                    date_records.append(record)

    if not date_records:
        raise ValueError("No rebalance dates had enough data to evaluate.")

    by_date = pd.DataFrame(date_records)

    summary = pd.concat(
        [
            summarize_period(by_date[by_date["period"] == period], horizons)
            .assign(period=period)
            for period in by_date["period"].unique()
        ],
        ignore_index=True,
    )

    leading_columns = ["period", "horizon"]
    summary = summary[
        leading_columns + [c for c in summary.columns if c not in leading_columns]
    ]

    return summary, by_date


def save_price_backtest(
    summary: pd.DataFrame,
    by_date: pd.DataFrame,
    run_dir: Path,
) -> None:
    """
    Save backtest outputs into an evaluation run folder.
    """
    summary.to_csv(run_dir / "price_backtest_summary.csv", index=False)
    by_date.to_csv(run_dir / "price_backtest_by_date.csv", index=False)

    print(f"Saved price backtest to {run_dir}")
