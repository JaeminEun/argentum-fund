from __future__ import annotations

from src import __version__

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.universe.config import load_config


DEFAULT_CONFIG_PATH = "config/universe_config.yaml"


ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_MAGENTA = "\033[95m"
ANSI_CYAN = "\033[96m"
ANSI_WHITE = "\033[97m"
ANSI_DIM = "\033[2m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"

STATUS_ICONS = {
    "PASS": "✓",
    "WARNING": "!",
    "ERROR": "✗",
}


@dataclass
class CheckResult:
    """
    One preflight check result.
    """
    level: str
    check: str
    message: str
    downstream_impact: str | None = None


def color_text(text: str, color: str, use_color: bool = True) -> str:
    """
    Add terminal color when enabled.
    """
    if not use_color:
        return text

    return f"{color}{text}{ANSI_RESET}"

def bold_text(text: str, use_color: bool = True) -> str:
    """
    Bold terminal text when enabled.
    """
    return color_text(text, ANSI_BOLD, use_color=use_color)


def dim_text(text: str, use_color: bool = True) -> str:
    """
    Dim terminal text when enabled.
    """
    return color_text(text, ANSI_DIM, use_color=use_color)


def print_banner(title: str, subtitle: str | None = None, use_color: bool = True) -> None:
    """
    Print a visually distinct preflight banner.
    """
    line = "═" * 92

    print("")
    print(color_text(line, ANSI_CYAN, use_color))
    print(color_text(f"  {title}", ANSI_BOLD + ANSI_CYAN, use_color))

    if subtitle:
        print(color_text(f"  {subtitle}", ANSI_WHITE, use_color))

    print(color_text(line, ANSI_CYAN, use_color))


def print_section_break(use_color: bool = True) -> None:
    """
    Print a soft divider between check rows.
    """
    print(color_text("─" * 92, ANSI_DIM, use_color))


def pass_result(check: str, message: str) -> CheckResult:
    return CheckResult(level="PASS", check=check, message=message)


def warn_result(
    check: str,
    message: str,
    downstream_impact: str | None = None,
) -> CheckResult:
    return CheckResult(
        level="WARNING",
        check=check,
        message=message,
        downstream_impact=downstream_impact,
    )


def error_result(
    check: str,
    message: str,
    downstream_impact: str | None = None,
) -> CheckResult:
    return CheckResult(
        level="ERROR",
        check=check,
        message=message,
        downstream_impact=downstream_impact,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Argentum Fund preflight checks."
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to project YAML config.",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored terminal output.",
    )

    parser.add_argument(
        "--allow-warnings",
        action="store_true",
        help=(
            "Allow warnings without changing behavior. This flag is currently "
            "informational because warnings do not cause failure by default."
        ),
    )

    return parser.parse_args()


def check_repo_root() -> list[CheckResult]:
    """
    Check that command is running from the repository root.
    """
    required_paths = [
        Path("src"),
        Path("config"),
        Path(DEFAULT_CONFIG_PATH),
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        return [
            error_result(
                check="Repository root",
                message=(
                    "This command does not appear to be running from the "
                    "repository root. Missing: "
                    + ", ".join(missing)
                ),
                downstream_impact=(
                    "Most pipeline modules use relative paths. Running from the "
                    "wrong directory can cause missing config, input, or output files."
                ),
            )
        ]

    return [
        pass_result(
            check="Repository root",
            message="Repository structure looks valid.",
        )
    ]


def check_config_load(config_path: str | Path) -> tuple[list[CheckResult], dict[str, Any] | None]:
    """
    Check that YAML config exists and can be loaded.
    """
    results = []
    config_path = Path(config_path)

    if not config_path.exists():
        return [
            error_result(
                check="Config file",
                message=f"Config file not found: {config_path}",
                downstream_impact=(
                    "The pipeline cannot run without the YAML configuration file."
                ),
            )
        ], None

    try:
        config = load_config(config_path)
    except Exception as error:
        return [
            error_result(
                check="Config file",
                message=f"Failed to load config file: {config_path}. Error: {error}",
                downstream_impact=(
                    "YAML syntax or indentation errors will prevent all modules "
                    "from reading project settings."
                ),
            )
        ], None

    results.append(
        pass_result(
            check="Config file",
            message=f"Config loaded successfully: {config_path}",
        )
    )

    return results, config


def read_csv_with_settings(
    path: str | Path,
    delimiter: str = ",",
    decimal: str = ".",
) -> pd.DataFrame:
    """
    Read CSV using configured delimiter and decimal markers.
    """
    return pd.read_csv(path, sep=delimiter, decimal=decimal)


def check_manual_universe_files(config: dict[str, Any]) -> list[CheckResult]:
    """
    Check enabled manual universe files and basic columns.
    """
    results = []

    manual_universes = config.get("manual_universes", [])

    if not manual_universes:
        results.append(
            warn_result(
                check="Manual universes",
                message="No manual_universes block found or no manual universes configured.",
                downstream_impact=(
                    "Universe building may still work if other universe sources exist, "
                    "but no manual CSV universes will be loaded."
                ),
            )
        )
        return results

    enabled_count = 0

    for index, universe in enumerate(manual_universes, start=1):
        if not universe.get("enabled", False):
            continue

        enabled_count += 1

        universe_name = universe.get("universe_name", f"manual_universe_{index}")
        path = Path(universe.get("path", ""))
        delimiter = universe.get("delimiter", ",")
        decimal = universe.get("decimal", ".")

        if not path.exists():
            results.append(
                error_result(
                    check=f"Manual universe: {universe_name}",
                    message=f"Enabled manual universe file not found: {path}",
                    downstream_impact=(
                        "Universe builder will fail or the intended securities "
                        "will be missing from price, fundamental, and composite scores."
                    ),
                )
            )
            continue

        try:
            frame = read_csv_with_settings(
                path=path,
                delimiter=delimiter,
                decimal=decimal,
            )
        except Exception as error:
            results.append(
                error_result(
                    check=f"Manual universe: {universe_name}",
                    message=(
                        f"Failed to read {path} using delimiter={delimiter!r}, "
                        f"decimal={decimal!r}. Error: {error}"
                    ),
                    downstream_impact=(
                        "Check delimiter/decimal settings. Excel exports often use "
                        "semicolon delimiters and comma decimals in international locales."
                    ),
                )
            )
            continue

        columns = set(frame.columns)

        if "ticker" not in columns:
            results.append(
                error_result(
                    check=f"Manual universe: {universe_name}",
                    message=(
                        f"Missing required column 'ticker' in {path}. "
                        f"Detected columns: {list(frame.columns)}"
                    ),
                    downstream_impact=(
                        "Tickers are required for price downloads, SEC mapping, "
                        "scoring, and portfolio comparison."
                    ),
                )
            )
        else:
            results.append(
                pass_result(
                    check=f"Manual universe: {universe_name}",
                    message=f"Found ticker column in {path}.",
                )
            )

        recommended_columns = {"asset_type", "target_weight"}
        missing_recommended = recommended_columns - columns

        if missing_recommended:
            results.append(
                warn_result(
                    check=f"Manual universe: {universe_name}",
                    message=(
                        f"Missing recommended columns in {path}: "
                        f"{sorted(missing_recommended)}"
                    ),
                    downstream_impact=(
                        "The model can still run, but strategy weighting, cash "
                        "handling, or asset-type filtering may be less informative."
                    ),
                )
            )

    if enabled_count == 0:
        results.append(
            error_result(
                check="Manual universes",
                message="No enabled manual universes found.",
                downstream_impact=(
                    "The universe builder may produce no securities unless another "
                    "universe source is enabled."
                ),
            )
        )

    return results


def check_portfolio_holdings(config: dict[str, Any]) -> list[CheckResult]:
    """
    Check portfolio holdings file and required columns when enabled.
    """
    results = []

    portfolio_config = config.get("portfolio_analysis", {})

    if not portfolio_config.get("enabled", False):
        results.append(
            warn_result(
                check="Portfolio holdings",
                message="portfolio_analysis is disabled.",
                downstream_impact=(
                    "Portfolio analyzer and weekly report portfolio sections may "
                    "not reflect current holdings."
                ),
            )
        )
        return results

    input_path = Path(portfolio_config.get("input_holdings_path", ""))

    if not input_path.exists():
        results.append(
            error_result(
                check="Portfolio holdings",
                message=f"Portfolio holdings file not found: {input_path}",
                downstream_impact=(
                    "Portfolio analysis, strategy exposure, sector exposure, and "
                    "weekly portfolio memo sections cannot be built."
                ),
            )
        )
        return results

    delimiter = portfolio_config.get("holdings_delimiter", ",")
    decimal = portfolio_config.get("holdings_decimal", ".")

    try:
        holdings = read_csv_with_settings(
            path=input_path,
            delimiter=delimiter,
            decimal=decimal,
        )
    except Exception as error:
        results.append(
            error_result(
                check="Portfolio holdings",
                message=(
                    f"Failed to read holdings file {input_path} using "
                    f"holdings_delimiter={delimiter!r}, "
                    f"holdings_decimal={decimal!r}. Error: {error}"
                ),
                downstream_impact=(
                    "Check portfolio_analysis.holdings_delimiter and "
                    "portfolio_analysis.holdings_decimal in the YAML config."
                ),
            )
        )
        return results

    required_columns = {
        "account",
        "ticker",
        "shares",
        "average_cost",
        "strategy",
        "active",
    }

    columns = set(holdings.columns)
    missing = required_columns - columns

    if missing:
        results.append(
            error_result(
                check="Portfolio holdings",
                message=(
                    f"Missing required columns in {input_path}: {sorted(missing)}. "
                    f"Detected columns: {list(holdings.columns)}"
                ),
                downstream_impact=(
                    "If the detected columns look merged into one column, the "
                    "holdings delimiter is probably wrong. Portfolio analysis "
                    "will fail until this is fixed."
                ),
            )
        )
    else:
        results.append(
            pass_result(
                check="Portfolio holdings",
                message=f"Required holdings columns found in {input_path}.",
            )
        )

    optional_columns = {"current_value_override", "notes"}
    missing_optional = optional_columns - columns

    if missing_optional:
        results.append(
            warn_result(
                check="Portfolio holdings",
                message=(
                    f"Missing optional columns in {input_path}: "
                    f"{sorted(missing_optional)}"
                ),
                downstream_impact=(
                    "The analyzer can still run, but synthetic strategy sleeves, "
                    "cash overrides, or notes may be less useful."
                ),
            )
        )

    return results


def check_sec_user_agent() -> list[CheckResult]:
    """
    Check SEC_USER_AGENT environment variable.
    """
    if os.getenv("SEC_USER_AGENT"):
        return [
            pass_result(
                check="SEC User-Agent",
                message="SEC_USER_AGENT is set.",
            )
        ]

    return [
        error_result(
            check="SEC User-Agent",
            message="SEC_USER_AGENT is not set.",
            downstream_impact=(
                "SEC CIK mapping and company facts downloads require a User-Agent. "
                "Set it with: export SEC_USER_AGENT=\"ArgentumFund/0.1 contact: your_email@example.com\""
            ),
        )
    ]


def check_gitignore_patterns() -> list[CheckResult]:
    """
    Check that private/generated files are likely protected by .gitignore.
    """
    gitignore_path = Path(".gitignore")

    required_patterns = [
        "data/scores/",
        "data/portfolio/",
        "data/deployment/",
        "data/fundamentals/",
        "data/sec/cache/",
        "data/reports/",
        "reports/weekly/",
        "__pycache__/",
        "*.pyc",
        ".env",
        "*.swp",
    ]

    if not gitignore_path.exists():
        return [
            error_result(
                check=".gitignore privacy safeguards",
                message=".gitignore file not found.",
                downstream_impact=(
                    "Private holdings, generated reports, SEC cache files, and "
                    "temporary files could be committed accidentally."
                ),
            )
        ]

    text = gitignore_path.read_text(encoding="utf-8", errors="ignore")

    missing = [pattern for pattern in required_patterns if pattern not in text]

    if missing:
        return [
            warn_result(
                check=".gitignore privacy safeguards",
                message=f".gitignore may be missing patterns: {missing}",
                downstream_impact=(
                    "Review .gitignore before committing to avoid leaking generated "
                    "outputs or personal portfolio data."
                ),
            )
        ]

    return [
        pass_result(
            check=".gitignore privacy safeguards",
            message="Key generated/private file patterns are present.",
        )
    ]


def check_core_source_files() -> list[CheckResult]:
    """
    Check that important source files exist.
    """
    required_files = [
        "src/universe/build_universe.py",
        "src/data/price_history.py",
        "src/features/price_factors.py",
        "src/scoring/price_scores.py",
        "src/sec/cik.py",
        "src/sec/client.py",
        "src/fundamentals/sec_company_facts.py",
        "src/fundamentals/sec_accounting_concepts.py",
        "src/fundamentals/sec_fundamental_factors.py",
        "src/scoring/fundamental_scores.py",
        "src/scoring/composite_scores.py",
        "src/deployment/cash_deployment.py",
        "src/portfolio/portfolio_analyzer.py",
        "src/reports/weekly_report.py",
        "src/pipeline/run_full_pipeline.py",
        "src/pipeline/preflight.py",
    ]

    missing = [path for path in required_files if not Path(path).exists()]

    if missing:
        return [
            error_result(
                check="Core source files",
                message=f"Missing core source files: {missing}",
                downstream_impact=(
                    "The pipeline may fail with 'No module named ...' or missing "
                    "file errors. Make sure source files are committed and present "
                    "on the active branch."
                ),
            )
        ]

    return [
        pass_result(
            check="Core source files",
            message="Core source files are present.",
        )
    ]


def check_expected_output_directories() -> list[CheckResult]:
    """
    Warn if common output parent directories are missing.

    These are usually created automatically, so missing directories are warnings,
    not errors.
    """
    expected_dirs = [
        "data",
        "data/manual",
        "data/processed",
    ]

    missing = [path for path in expected_dirs if not Path(path).exists()]

    if missing:
        return [
            warn_result(
                check="Expected local directories",
                message=f"Some common local directories are missing: {missing}",
                downstream_impact=(
                    "Most modules create output directories automatically, but missing "
                    "manual input directories may indicate setup is incomplete."
                ),
            )
        ]

    return [
        pass_result(
            check="Expected local directories",
            message="Common local directories are present.",
        )
    ]


def collect_results(config_path: str | Path) -> list[CheckResult]:
    """
    Run all preflight checks and collect results.
    """
    results: list[CheckResult] = []

    results.extend(check_repo_root())

    config_results, config = check_config_load(config_path)
    results.extend(config_results)

    results.extend(check_core_source_files())
    results.extend(check_gitignore_patterns())
    results.extend(check_expected_output_directories())
    results.extend(check_sec_user_agent())

    if config is not None:
        results.extend(check_manual_universe_files(config))
        results.extend(check_portfolio_holdings(config))

    return results


def print_results(results: list[CheckResult], use_color: bool = True) -> None:
    """
    Print preflight results in a readable, colorful format.
    """
    print_banner(
    title="ARGENTUM FUND PREFLIGHT CHECKS",
    subtitle=(
        f"Argentum Fund v{__version__} | "
        "Validating repo structure, config, inputs, privacy safeguards, and source files."
    ),
    use_color=use_color,
)

    for result in results:
        icon = STATUS_ICONS.get(result.level, "-")

        if result.level == "PASS":
            label = color_text(f"{icon} PASS", ANSI_GREEN + ANSI_BOLD, use_color)
        elif result.level == "WARNING":
            label = color_text(f"{icon} WARNING", ANSI_YELLOW + ANSI_BOLD, use_color)
        elif result.level == "ERROR":
            label = color_text(f"{icon} ERROR", ANSI_RED + ANSI_BOLD, use_color)
        else:
            label = result.level

        check_name = bold_text(result.check, use_color=use_color)

        print(f"{label}  {check_name}")
        print(f"   {result.message}")

        if result.downstream_impact:
            impact_label = color_text(
                "Downstream impact:",
                ANSI_BLUE + ANSI_BOLD,
                use_color,
            )

            if result.level == "ERROR":
                impact_text = color_text(
                    result.downstream_impact,
                    ANSI_RED,
                    use_color,
                )
            elif result.level == "WARNING":
                impact_text = color_text(
                    result.downstream_impact,
                    ANSI_YELLOW,
                    use_color,
                )
            else:
                impact_text = result.downstream_impact

            print(f"   {impact_label} {impact_text}")

        print_section_break(use_color=use_color)

    pass_count = sum(result.level == "PASS" for result in results)
    warning_count = sum(result.level == "WARNING" for result in results)
    error_count = sum(result.level == "ERROR" for result in results)

    print("")
    print(color_text("SUMMARY", ANSI_BOLD + ANSI_MAGENTA, use_color))
    print(color_text("─" * 92, ANSI_MAGENTA, use_color))

    print(f"  {color_text('Passed:', ANSI_GREEN + ANSI_BOLD, use_color)}   {pass_count}")
    print(f"  {color_text('Warnings:', ANSI_YELLOW + ANSI_BOLD, use_color)} {warning_count}")
    print(f"  {color_text('Errors:', ANSI_RED + ANSI_BOLD, use_color)}   {error_count}")

    print("")

    if error_count > 0:
        status = color_text("NOT READY", ANSI_RED + ANSI_BOLD, use_color)
        message = (
            "Fix blocking errors before running the pipeline. "
            "These issues may cause missing outputs, broken scoring, or misleading reports."
        )
        print(f"  Status: {status}")
        print(f"  {color_text(message, ANSI_RED, use_color)}")

    elif warning_count > 0:
        status = color_text("READY WITH WARNINGS", ANSI_YELLOW + ANSI_BOLD, use_color)
        message = (
            "The pipeline can run, but review warnings because outputs may be less complete "
            "or less informative."
        )
        print(f"  Status: {status}")
        print(f"  {color_text(message, ANSI_YELLOW, use_color)}")

    else:
        status = color_text("READY", ANSI_GREEN + ANSI_BOLD, use_color)
        message = "All core checks passed. The project appears ready to run."
        print(f"  Status: {status}")
        print(f"  {color_text(message, ANSI_GREEN, use_color)}")

    print(color_text("═" * 92, ANSI_CYAN, use_color))


def run_preflight(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    use_color: bool = True,
) -> int:
    """
    Run preflight checks.

    Returns:
        0 if there are no errors
        1 if one or more errors were found
    """
    results = collect_results(config_path)
    print_results(results, use_color=use_color)

    error_count = sum(result.level == "ERROR" for result in results)

    if error_count > 0:
        return 1

    return 0


def main() -> None:
    args = parse_args()

    exit_code = run_preflight(
        config_path=args.config,
        use_color=not args.no_color,
    )

    if exit_code == 0:
        print("")
        print(
            color_text(
                "Next step: python -m src.pipeline.run_full_pipeline --mode full",
                ANSI_CYAN + ANSI_BOLD,
                use_color=not args.no_color,
            )
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
