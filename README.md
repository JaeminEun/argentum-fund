# Argentum Fund

> **The youngest you're ever going to be is today.**

Argentum Fund is a Python-based investment research project built around a simple idea: it is never too late to begin investing more thoughtfully. People often say they wish they had invested earlier, bought strong companies sooner, or taken advantage of market weakness when they had the chance. This project is designed around a more constructive view: the time available today still matters, and disciplined tools can help turn today's research into better long-term decisions.

Argentum Fund is not designed to chase short-term hype or produce automatic buy/sell commands. Its core purpose is to help users identify **when a security may deserve closer research**, using price behavior, SEC-derived fundamentals, portfolio context, and clear weekly reporting.

The project began as a way to supplement Autopilot-style investing based on 13F filings. It has since evolved into a broader research engine that can evaluate professional-investor-inspired universes, value-oriented stock pools, manual watchlists, and current portfolio holdings.

---

## Project Philosophy

Argentum Fund is built around four principles:

1. **Timing matters.**  
   The central thesis of the model is not only *what* to buy, but *when* a security may deserve attention.

2. **Fundamentals provide context.**  
   Price behavior can identify attractive setups, but fundamentals help distinguish stronger businesses from weaker or riskier candidates.

3. **Research should be disciplined and repeatable.**  
   A user should be able to run the same pipeline regularly and receive consistent, explainable outputs.

4. **The model should explain uncertainty.**  
   A good research system should not only highlight candidates. It should also flag missing data, weak fundamentals, possible value traps, and manual review items.

---

## What Argentum Fund Does

Argentum Fund currently supports a full MVP research workflow:

```text
Universe Builder
        ↓
Price History
        ↓
Price Factors
        ↓
Price Scores
        ↓
SEC Fundamentals
        ↓
Fundamental Scores
        ↓
Composite Scores
        ↓
Cash Deployment Plan
        ↓
Portfolio Analyzer
        ↓
Weekly Research Memo
```

At a high level, the model asks:

- Which securities have attractive price timing?
- Which securities have supportive accounting fundamentals?
- Which candidates combine price and fundamentals well?
- Which holdings in my portfolio look strong, weak, or in need of review?
- Which Autopilot or 13F-inspired holdings actually look attractive under my own model?
- What should I review in this week's research memo?

---

## What Argentum Fund Does Not Do

Argentum Fund does **not** provide financial advice, investment advice, or automatic recommendations to buy or sell securities.

It does not currently:

- Place trades
- Connect to brokerage APIs
- Guarantee returns
- Predict future market behavior
- Replace manual investment judgment
- Fully model macroeconomic conditions
- Fully model sector-specific accounting differences
- Fully calculate valuation ratios such as P/E, P/B, P/S, or FCF yield

Outputs should be treated as a **research queue**, not as investment instructions.

---

## Current MVP Features

### Universe Builder

Builds a standardized research universe from manual CSV files.

Current supported use cases:

- S&P 500 Value-style watchlists
- Manual Autopilot-style holdings
- Manual 13F-inspired portfolios
- Paper portfolios
- Portfolio-specific universes
- Cash and reserve positions

The universe builder standardizes tickers, weights, account targets, asset types, and metadata.

---

### Price History Module

Downloads historical price data for tradable tickers.

The price history module provides the foundation for:

- Return calculations
- Moving averages
- Drawdown measures
- Volatility estimates
- Dip and trend detection

---

### Price Factor Calculator

Calculates price-derived features such as:

- Recent returns
- Distance from moving averages
- Distance from 13-week and 52-week highs
- Volatility
- Trend flags
- Dip flags
- Momentum flags

This module is central to the project's original thesis: **price behavior helps identify when a security may deserve attention.**

---

### Price Scoring Module

Converts price factors into a timing-oriented score.

The price score asks:

> Does this security currently show an attractive market-behavior setup?

It is not a prediction. It is a structured timing screen.

---

### SEC CIK Mapper

Maps tickers to SEC CIK identifiers.

The SEC primarily identifies companies by CIK rather than ticker. This module allows Argentum Fund to connect tradable securities to SEC company data.

---

### SEC Company Facts Downloader

Downloads SEC company facts from EDGAR's public XBRL company facts API.

This provides official filing-derived accounting data such as:

- Revenue
- Net income
- Assets
- Liabilities
- Stockholders' equity
- Operating income
- Operating cash flow
- Capital expenditures

The downloader uses local caching to avoid unnecessary repeated requests.

---

### SEC Accounting Concept Extraction

Extracts selected accounting concepts from cached SEC company facts.

The extraction layer includes improved period handling:

- Annual duration metrics for income statement and cash-flow items
- Latest instant metrics for balance-sheet items

This helps avoid mixing quarterly revenue with annual net income or cash flow.

---

### Fundamental Factor Calculator

Calculates accounting-based fundamental factors such as:

- Free cash flow
- Return on equity
- Net margin
- Operating margin
- FCF margin
- Asset turnover
- Liabilities to assets
- Equity to assets
- Operating cash flow to net income

This module currently focuses on accounting strength and financial quality, not market valuation.

---

### Fundamental Scoring Module

Builds sector-relative fundamental scores.

The fundamental score asks:

> Does this company show a supportive accounting profile relative to peers?

It includes:

- Quality score
- Cash-flow score
- Balance-sheet score
- Sanity filters for extreme ratios
- Penalties for weak or suspicious fundamentals
- Sector-relative percentile scoring

---

### Composite Scoring Module

Combines price and fundamental scores into a composite research score.

Current default weighting:

```text
60% price score
40% fundamental score
```

This reflects the project's thesis that timing is central, while fundamentals provide context and quality control.

The composite module classifies securities into labels such as:

- `aligned_candidate`
- `timing_candidate_fundamentals_neutral`
- `quality_watchlist_wait_for_timing`
- `possible_value_trap`
- `mixed_signal_review_required`
- `low_priority`

The composite score is best interpreted as a **research priority score**, not a buy score.

---

### Cash Deployment Planner

Creates a non-executing cash deployment plan.

The deployment planner suggests possible tranche actions such as:

- `normal_tranche`
- `test_tranche`
- `watch_only`

It considers:

- Composite score
- Price score
- Fundamental score
- Candidate signal
- Period deployment budget
- Maximum candidates
- Sector candidate limits
- Manual review flags

This module does not execute trades.

---

### Portfolio Analyzer

Analyzes current holdings from a manual holdings CSV.

It calculates:

- Market value
- Cost basis
- Unrealized gain/loss
- Unrealized return
- Portfolio weight
- Account weight
- Strategy exposure
- Sector exposure
- Holding-level model score
- Holding review flags

It also supports synthetic strategy sleeves and cash positions.

---

### Autopilot Look-Through

Analyzes individual securities inside Autopilot-style universes.

This is useful because Autopilot may select holdings based on account size, whole-share constraints, fractional-share availability, or implementation rules. Argentum Fund can instead examine the underlying securities directly.

The look-through analysis asks:

> Which holdings inside a professional-investor-inspired universe actually look attractive under Argentum's model?

---

### Weekly Research Memo

Generates a Markdown weekly research memo.

The memo summarizes:

- Top composite research candidates
- Active deployment suggestions
- Manual review items
- Portfolio snapshot
- Strategy exposure
- Sector exposure
- Autopilot look-through
- Caution items

The report is designed to read like a concise research memo rather than a trading dashboard.

---

## Repository Structure

```text
config/
    universe_config.yaml

examples/
    README.md
    manual_universe_template.csv
    portfolio_holdings_template.csv
    universe_config_template.yaml

src/
    data/
        price_history.py

    deployment/
        cash_deployment.py

    features/
        price_factors.py

    fundamentals/
        sec_accounting_concepts.py
        sec_company_facts.py
        sec_fundamental_factors.py

    pipeline/
        run_full_pipeline.py

    portfolio/
        portfolio_analyzer.py

    reports/
        weekly_report.py

    scoring/
        price_scores.py
        fundamental_scores.py
        composite_scores.py

    sec/
        cik.py
        client.py

    universe/
        build_universe.py
        config.py
        loaders.py
        schema.py

    utils/
        io.py
```

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd argentum-fund
```

### 2. Create or activate your Python environment

Example with Conda:

```bash
conda create -n argentum python=3.13
conda activate argentum
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set SEC User-Agent

The SEC data tools require a User-Agent. Do not hardcode your personal email into the repository.

Set it locally:

```bash
export SEC_USER_AGENT="ArgentumFund/0.1 contact: your_email@example.com"
```

To persist this in WSL:

```bash
echo 'export SEC_USER_AGENT="ArgentumFund/0.1 contact: your_email@example.com"' >> ~/.bashrc
source ~/.bashrc
```

---

## Configuration

The main configuration file is:

```text
config/universe_config.yaml
```

This file controls:

- Enabled universes
- Manual CSV paths
- CSV delimiter settings
- SEC API behavior
- Price data settings
- Fundamental scoring settings
- Composite score weights
- Cash deployment settings
- Portfolio analysis settings
- Weekly report paths

For public or example setup, see:

```text
examples/universe_config_template.yaml
```

---

## CSV Formatting

Argentum Fund supports both standard Python-style CSV files and international Excel-style CSV files.

### Standard Python CSV

```yaml
delimiter: ","
decimal: "."
```

For portfolio holdings:

```yaml
holdings_delimiter: ","
holdings_decimal: "."
```

### International Excel CSV

```yaml
delimiter: ";"
decimal: ","
```

For portfolio holdings:

```yaml
holdings_delimiter: ";"
holdings_decimal: ","
```

---

## Running the Pipeline

The easiest way to run the project is through the pipeline runner.

### Full Pipeline

Run everything:

```bash
python -m src.pipeline.run_full_pipeline --mode full
```

### Scores Only

Rebuild universe, prices, SEC fundamentals, and composite scores:

```bash
python -m src.pipeline.run_full_pipeline --mode scores
```

### Market Mode

Rebuild scores and cash deployment plan:

```bash
python -m src.pipeline.run_full_pipeline --mode market
```

### Portfolio Mode

Use this when holdings changed but the scoring universe did not:

```bash
python -m src.pipeline.run_full_pipeline --mode portfolio
```

This runs:

```text
cash deployment → portfolio analyzer → weekly report
```

### Report Only

Regenerate only the weekly memo:

```bash
python -m src.pipeline.run_full_pipeline --mode report
```

### SEC/Fundamentals Only

```bash
python -m src.pipeline.run_full_pipeline --mode sec
```

---

## Manual Pipeline Order

If you prefer to run modules manually:

```bash
python -m src.universe.build_universe
python -m src.data.price_history
python -m src.features.price_factors
python -m src.scoring.price_scores
python -m src.sec.cik
python -m src.fundamentals.sec_company_facts
python -m src.fundamentals.sec_accounting_concepts
python -m src.fundamentals.sec_fundamental_factors
python -m src.scoring.fundamental_scores
python -m src.scoring.composite_scores
python -m src.deployment.cash_deployment
python -m src.portfolio.portfolio_analyzer
python -m src.reports.weekly_report
```

---

## Viewing the Weekly Report

After running the report module, open:

```text
reports/weekly/latest_weekly_report.md
```

In VS Code:

- Markdown preview: `Ctrl + Shift + V`
- Preview to side: `Ctrl + K`, then `V`

Generated reports are private and should not be committed.

---

## Git and Private Files

This project is designed so users can fork the repo without leaking personal financial data.

Do not commit:

- Personal holdings
- Generated score files
- Portfolio outputs
- Deployment plans
- Weekly reports
- SEC cache files
- Environment variables
- `.env` files
- `__pycache__`
- `.pyc` files
- editor swap files

Private/generated folders should remain ignored:

```text
data/scores/
data/portfolio/
data/deployment/
data/fundamentals/
data/sec/cache/
reports/weekly/
data/reports/
```

---

## Development Workflow

Recommended branch workflow:

```bash
git checkout main
git pull
git status
git checkout -b feature/my-new-feature
```

After making changes:

```bash
git status
git diff --stat
git add <intended-files>
git commit -m "Describe the change"
git push -u origin feature/my-new-feature
```

Then open a pull request into `main`.

Before merging, check the GitHub **Files changed** tab carefully. A PR should only contain files related to that branch's purpose.

---

## Versioning

Argentum Fund uses early-stage semantic versioning.

Suggested meaning:

```text
v0.1.0   First MVP
v0.1.1   MVP cleanup and bug fixes
v0.2.0   13F universe automation foundation
v0.3.0   Valuation ratios and market-cap integration
v0.4.0   Score history and model validation
v0.5.0   Sector-specific scoring profiles
v0.6.0   Sell/trim review logic
v1.0.0   Stable public release
```

The current MVP should be treated as a working research system, not a stable final product.

---

## Roadmap

### Near-Term Cleanup

- Pipeline runner improvements
- Template and example files
- Better README and onboarding
- Preflight checks
- Improved report formatting
- Version display in reports
- Better warnings for missing or stale data

### Next Major Direction: Universe Automation

The next major development direction is automated universe creation, especially from SEC 13F filings.

Planned capabilities:

- Manager registry
- 13F filing downloader
- 13F holdings parser
- CUSIP-to-ticker mapping
- Portfolio weight extraction
- Automatic universe CSV generation
- 13F-based universe comparison

This will allow Argentum Fund to evaluate professional-investor-inspired universes directly, without depending on Autopilot implementation constraints.

### Model Accuracy Improvements

Future model improvements may include:

- Market cap integration
- P/E, P/B, P/S, and FCF yield
- Earnings yield
- Multi-year fundamental trends
- TTM accounting metrics
- Sector-specific scoring profiles
- Macro regime filters
- Score history archive
- Candidate persistence tracking
- Sell and trim review logic

---

## Disclaimer

Argentum Fund is a research and educational software project. It is not financial advice, investment advice, or a recommendation to buy or sell securities.

The software is intended to support disciplined research, not replace independent judgment. Always review outputs manually before making investment decisions.
