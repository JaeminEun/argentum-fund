import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluation.data_health import (
    check_implausible_ratios,
    check_missing_score_inputs,
    check_period_mismatch,
    check_stale_metrics,
)
from src.evaluation.price_backtest import (
    add_forward_returns,
    evaluate_date,
    run_price_backtest,
    select_rebalance_dates,
    summarize_period,
)
from src.evaluation.runs import compare_label_distribution, compare_top_membership


def test_forward_returns_use_future_prices_per_ticker():
    factors = pd.DataFrame(
        {
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "adjusted_close": [100.0, 110.0, 121.0, 50.0, 40.0, 30.0],
        }
    )

    result = add_forward_returns(factors, "adjusted_close", {"1d": 1})

    a = result[result["ticker"] == "A"]["fwd_return_1d"].tolist()
    b = result[result["ticker"] == "B"]["fwd_return_1d"].tolist()

    assert a[:2] == pytest.approx([0.10, 0.10])
    assert b[:2] == pytest.approx([-0.20, -0.25])
    # The last row of each ticker must not borrow the next ticker's price.
    assert np.isnan(a[2]) and np.isnan(b[2])


def test_rebalance_dates_skip_warmup_and_respect_end_date():
    dates = pd.Series(pd.bdate_range("2024-01-01", "2024-03-29"))

    rebalance = select_rebalance_dates(
        dates,
        warmup_days=10,
        frequency="W-FRI",
        end_date="2024-03-02",
    )

    assert rebalance[0] >= dates.iloc[10]
    assert max(rebalance) < pd.Timestamp("2024-03-02")
    assert all(date.dayofweek == 4 for date in rebalance)


def make_scored_cross_section(predictive: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    scores = np.linspace(0, 100, 40)
    forward = scores / 1000 if predictive else rng.normal(0, 0.05, 40)
    buckets = pd.cut(
        pd.Series(scores).rank(pct=True),
        bins=[0, 0.30, 0.70, 0.85, 0.95, 1.00],
        labels=["weak", "neutral", "watchlist", "strong_candidate", "top_candidate"],
        include_lowest=True,
    )

    return pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(40)],
            "risk_adjusted_timing_score": scores,
            "score_bucket": buckets,
            "fwd_return_4w": forward,
            "rebalance_date": pd.Timestamp("2024-01-05"),
        }
    )


def test_perfect_score_has_ic_of_one_and_positive_spread():
    record = evaluate_date(
        make_scored_cross_section(predictive=True),
        score_column="risk_adjusted_timing_score",
        bucket_column="score_bucket",
        horizon="4w",
        min_names=20,
    )

    assert record["ic"] == pytest.approx(1.0)
    assert record["top_minus_weak"] > 0
    assert record["top_hit_rate"] == 1.0
    assert record["excess_top_candidate"] > 0 > record["excess_weak"]


def test_evaluate_date_skips_thin_cross_sections():
    record = evaluate_date(
        make_scored_cross_section(predictive=True).head(10),
        score_column="risk_adjusted_timing_score",
        bucket_column="score_bucket",
        horizon="4w",
        min_names=20,
    )

    assert record is None


def test_t_stat_only_uses_non_overlapping_dates():
    by_date = pd.DataFrame(
        {
            "rebalance_date": pd.date_range("2024-01-05", periods=26, freq="W-FRI"),
            "horizon": "13w",
            "ic": np.linspace(0.0, 0.1, 26),
        }
    )

    summary = summarize_period(by_date, {"13w": 63})

    # 63 trading days is 13 weeks, so only every 13th date is independent.
    assert summary.loc[0, "dates"] == 26
    assert summary.loc[0, "independent_dates"] == 2


def make_concepts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "metric_name": ["revenue", "net_income", "revenue", "net_income"],
            "concept_name": ["Revenues", "NetIncomeLoss"] * 2,
            "end": pd.to_datetime(
                ["2018-09-29", "2025-09-27", "2025-06-30", "2025-06-30"]
            ),
            "duration_type": "annual",
        }
    )


def test_stale_metric_and_period_mismatch_catch_old_revenue_tag():
    concepts = make_concepts()

    stale = check_stale_metrics(concepts, pd.Timestamp("2026-10-14"), 548)
    mismatch = check_period_mismatch(concepts, ["revenue", "net_income"], "net_income", 45)

    assert stale["ticker"].tolist() == ["AAPL"]
    assert mismatch["ticker"].tolist() == ["AAPL"]
    assert "2018-09-29" in mismatch.loc[0, "detail"]


def test_implausible_ratios_respect_bounds():
    factors = pd.DataFrame(
        {
            "ticker": ["AAPL", "ALGN", "XOM"],
            "net_margin": [0.24, 1.06, 0.10],
            "return_on_equity": [1.5, 0.1, -4.0],
        }
    )

    issues = check_implausible_ratios(
        factors,
        {"net_margin": {"upper": 1.0}, "return_on_equity": {"lower": -3.0, "upper": 3.0}},
    )

    assert sorted(issues["ticker"]) == ["ALGN", "XOM"]


def test_missing_score_inputs_are_flagged():
    latest = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "latest_date": ["2026-10-14"] * 2,
            "adjusted_close": [10.0, 20.0],
            "below_13w_high": [-0.1, -0.1],
            "below_52w_high": [-0.2, -0.2],
            "distance_from_ma_200d": [0.0, 0.0],
            "return_13w": [0.05, 0.05],
            "volatility_60d": [0.3, np.nan],
        }
    )

    issues = check_missing_score_inputs(latest)

    assert issues["ticker"].tolist() == ["B"]
    assert "volatility_60d" in issues.loc[0, "detail"]


def test_run_comparison_tables():
    before = pd.DataFrame(
        {"ticker": ["A", "B", "C"], "signal": ["x", "x", "y"], "score": [3, 2, 1]}
    )
    after = pd.DataFrame(
        {"ticker": ["A", "B", "C"], "signal": ["x", "y", "y"], "score": [1, 2, 3]}
    )

    distribution = compare_label_distribution(before, after, "signal")
    membership = compare_top_membership(before, after, "score", top_n=1)

    assert distribution.set_index("signal")["change"].to_dict() == {"x": -1, "y": 1}
    assert membership.to_dict("records") == [
        {"ticker": "C", "change": "entered_top"},
        {"ticker": "A", "change": "left_top"},
    ]


def test_backtest_runs_end_to_end_and_keeps_holdout_sealed(tmp_path):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", "2025-06-30")
    frames = []

    for i in range(25):
        returns = rng.normal(0.0003, 0.015, len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": f"T{i:02d}",
                    "date": dates,
                    "adjusted_close": 100 * np.cumprod(1 + returns),
                }
            )
        )

    price_path = tmp_path / "prices.csv"
    pd.concat(frames).to_csv(price_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "price_factors": {"input_price_path": str(price_path)},
                "price_scores": {},
                "evaluation": {
                    "price_backtest": {
                        "holdout_start": "2025-01-01",
                        "forward_horizons": {"4w": 20},
                    }
                },
            }
        )
    )

    summary, by_date = run_price_backtest(config_path)

    assert summary["period"].tolist() == ["in_sample"]
    assert by_date["rebalance_date"].max() < pd.Timestamp("2025-01-01")
    assert by_date["ic"].between(-1, 1).all()

    _, with_holdout = run_price_backtest(config_path, include_holdout=True)

    assert set(with_holdout["period"]) == {"in_sample", "holdout"}
