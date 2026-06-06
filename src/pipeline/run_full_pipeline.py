from __future__ import annotations

from src import __version__

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = "config/universe_config.yaml"


@dataclass
class PipelineStep:
    """
    One executable pipeline step.
    """
    name: str
    module: str
    required: bool = True


PIPELINE_STEPS = {
    "universe": [
        PipelineStep("Build universe", "src.universe.build_universe"),
    ],
    "prices": [
        PipelineStep("Download price history", "src.data.price_history"),
        PipelineStep("Calculate price factors", "src.features.price_factors"),
        PipelineStep("Build price scores", "src.scoring.price_scores"),
    ],
    "sec": [
        PipelineStep("Build SEC CIK reference", "src.sec.cik"),
        PipelineStep("Download SEC company facts", "src.fundamentals.sec_company_facts"),
        PipelineStep("Extract SEC accounting concepts", "src.fundamentals.sec_accounting_concepts"),
        PipelineStep("Build SEC fundamental factors", "src.fundamentals.sec_fundamental_factors"),
        PipelineStep("Build fundamental scores", "src.scoring.fundamental_scores"),
    ],
    "composite": [
        PipelineStep("Build composite scores", "src.scoring.composite_scores"),
    ],
    "deployment": [
        PipelineStep("Build cash deployment plan", "src.deployment.cash_deployment"),
    ],
    "portfolio": [
        PipelineStep("Run portfolio analyzer", "src.portfolio.portfolio_analyzer"),
    ],
    "report": [
        PipelineStep("Build weekly research memo", "src.reports.weekly_report"),
    ],
}


PIPELINE_MODES = {
    "full": [
        "universe",
        "prices",
        "sec",
        "composite",
        "deployment",
        "portfolio",
        "report",
    ],
    "scores": [
        "universe",
        "prices",
        "sec",
        "composite",
    ],
    "market": [
        "universe",
        "prices",
        "sec",
        "composite",
        "deployment",
    ],
    "portfolio": [
        "deployment",
        "portfolio",
        "report",
    ],
    "report": [
        "report",
    ],
    "sec": [
        "sec",
    ],
}


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Argentum Fund MVP pipeline."
    )

    parser.add_argument(
        "--mode",
        choices=sorted(PIPELINE_MODES.keys()),
        default="full",
        help=(
            "Pipeline mode to run. "
            "'full' runs everything. "
            "'scores' rebuilds scoring outputs. "
            "'market' rebuilds scores and deployment. "
            "'portfolio' rebuilds deployment, portfolio, and report. "
            "'report' rebuilds only the report. "
            "'sec' reruns only SEC/fundamental steps."
        ),
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to project YAML config.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining steps if one step fails.",
    )

    parser.add_argument(
        "--skip-sec-user-agent-check",
        action="store_true",
        help="Skip check for SEC_USER_AGENT environment variable.",
    )

    return parser.parse_args()


def check_repo_root() -> None:
    """
    Warn if the command does not appear to be running from repo root.
    """
    expected = Path("src")
    config = Path(DEFAULT_CONFIG_PATH)

    if not expected.exists() or not config.exists():
        raise RuntimeError(
            "This command should be run from the repository root. "
            "Expected to find 'src/' and 'config/universe_config.yaml'."
        )


def check_config_exists(config_path: str | Path) -> None:
    """
    Validate that the config file exists.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")


def mode_requires_sec(mode: str) -> bool:
    """
    Return True if a pipeline mode includes SEC API steps.
    """
    groups = PIPELINE_MODES[mode]
    return "sec" in groups or mode == "full"


def check_sec_user_agent(mode: str, skip_check: bool = False) -> None:
    """
    Check SEC_USER_AGENT when SEC steps are included.

    The SEC client itself will also validate this. This preflight check
    just gives a clearer message before the pipeline starts.
    """
    if skip_check:
        return

    if not mode_requires_sec(mode):
        return

    if os.getenv("SEC_USER_AGENT"):
        return

    raise RuntimeError(
        "SEC_USER_AGENT is not set, but this pipeline mode includes SEC steps.\n"
        "Set it with something like:\n"
        'export SEC_USER_AGENT="ArgentumFund/0.1 contact: your_email@example.com"\n'
        "Or rerun with --skip-sec-user-agent-check if you know SEC steps are cached "
        "and your config does not require the environment variable."
    )


def build_steps_for_mode(mode: str) -> list[PipelineStep]:
    """
    Expand a pipeline mode into ordered steps.
    """
    steps: list[PipelineStep] = []

    for group_name in PIPELINE_MODES[mode]:
        steps.extend(PIPELINE_STEPS[group_name])

    return steps


def run_step(
    step: PipelineStep,
    step_number: int,
    total_steps: int,
) -> None:
    """
    Run one pipeline step as a Python module.
    """
    print("")
    print("=" * 90)
    print(f"[{step_number}/{total_steps}] {step.name}")
    print(f"Module: python -m {step.module}")
    print("=" * 90)

    start = time.time()

    result = subprocess.run(
        [sys.executable, "-m", step.module],
        check=False,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline step failed: {step.name} "
            f"(module={step.module}, exit_code={result.returncode})"
        )

    print(f"Completed: {step.name} in {elapsed:.1f} seconds")


def run_pipeline(
    mode: str,
    config_path: str | Path,
    continue_on_error: bool = False,
    skip_sec_user_agent_check: bool = False,
) -> None:
    """
    Run selected pipeline mode.
    """
    check_repo_root()
    check_config_exists(config_path)
    check_sec_user_agent(
        mode=mode,
        skip_check=skip_sec_user_agent_check,
    )

    steps = build_steps_for_mode(mode)

    print("")
    print("ARGENTUM FUND PIPELINE")
    print(f"Version: v{__version__}")
    print(f"Mode: {mode}")
    print(f"Config: {config_path}")
    print(f"Steps: {len(steps)}")

    pipeline_start = time.time()
    failed_steps = []

    for index, step in enumerate(steps, start=1):
        try:
            run_step(
                step=step,
                step_number=index,
                total_steps=len(steps),
            )
        except Exception as error:
            failed_steps.append((step, error))

            print("")
            print("ERROR")
            print(f"Step failed: {step.name}")
            print(f"Reason: {error}")

            if not continue_on_error:
                print("")
                print("Pipeline stopped. Rerun with --continue-on-error to keep going.")
                raise

    total_elapsed = time.time() - pipeline_start

    print("")
    print("=" * 90)
    print("PIPELINE COMPLETE")
    print(f"Mode: {mode}")
    print(f"Elapsed time: {total_elapsed:.1f} seconds")
    print("=" * 90)

    if failed_steps:
        print("")
        print("Completed with failed steps:")
        for step, error in failed_steps:
            print(f"- {step.name}: {error}")
    else:
        print("All steps completed successfully.")


def main() -> None:
    args = parse_args()

    run_pipeline(
        mode=args.mode,
        config_path=args.config,
        continue_on_error=args.continue_on_error,
        skip_sec_user_agent_check=args.skip_sec_user_agent_check,
    )


if __name__ == "__main__":
    main()
