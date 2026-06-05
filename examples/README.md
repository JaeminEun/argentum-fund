# Argentum Fund Example Files

This folder contains public-safe templates for setting up Argentum Fund.

## Files

- `manual_universe_template.csv`: example universe input file
- `portfolio_holdings_template.csv`: example current holdings file
- `universe_config_template.yaml`: simplified example configuration

## Important

Do not commit your personal holdings, generated reports, SEC cache, score outputs, or portfolio analysis files.

Generated and private folders should remain ignored, including:

- `data/scores/`
- `data/portfolio/`
- `data/deployment/`
- `data/fundamentals/`
- `data/sec/cache/`
- `reports/weekly/`

## CSV formatting

Argentum Fund supports both standard Python-style CSV files and international Excel-style CSV files.

### Standard Python CSV

Use comma delimiters and period decimals:

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

Use semicolon delimiters and comma decimals:

```yaml
delimiter: ";"
decimal: ","
```

For portfolio holdings:

```yaml
holdings_delimiter: ";"
holdings_decimal: ","
```

## SEC User-Agent

The SEC data tools require a User-Agent. Set it locally rather than committing personal contact details to GitHub.

Example:

```bash
export SEC_USER_AGENT="ArgentumFund/0.1 contact: your_email@example.com"
```

If you want this to persist in WSL, add it to `~/.bashrc`.

## Recommended setup flow

1. Copy the template files into your own working configuration.
2. Update tickers, holdings, account names, and strategy labels.
3. Keep private files out of Git.
4. Run the pipeline from the repository root.

Example:

```bash
python -m src.pipeline.run_full_pipeline --mode full
```

For portfolio-only updates:

```bash
python -m src.pipeline.run_full_pipeline --mode portfolio
```

## Disclaimer

Argentum Fund is a research and educational software project. It does not provide financial advice, investment advice, or buy/sell recommendations. Always review outputs manually before making investment decisions.
