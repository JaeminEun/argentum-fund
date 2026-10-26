import pandas as pd
import pytest
import yaml

from src.evaluation.runs import compare_universes
from src.universe.etf_holdings import (
    diff_universe,
    parse_ssga_holdings,
    refresh_holdings_universe,
)
from src.universe.loaders import load_manual_universe


def write_ssga_file(path, rows, as_of="23-Oct-2026"):
    """
    Write a holdings file with State Street's layout: a header block,
    the holdings table, a blank row, then disclaimer text.
    """
    header = [
        ["Fund Name:", "Example S&P 500 Value ETF"] + [None] * 6,
        ["Ticker Symbol:", "SPYV"] + [None] * 6,
        ["Holdings:", f"As of {as_of}"] + [None] * 6,
        [None] * 8,
        ["Name", "Ticker", "Identifier", "SEDOL", "Weight", "Sector",
         "Shares Held", "Local Currency"],
    ]
    footer = [[None] * 8, ["Distributor: example disclaimer"] + [None] * 7]
    pd.DataFrame(header + rows + footer).to_excel(path, header=False, index=False)


HOLDINGS_ROWS = [
    ["APPLE INC", "AAPL", "037833100", "2046251", "60.0", "-", "100", "USD"],
    ["SKYDANCE CORP", "SKYD", "69932A204", "BSNMNQ5", "20.0", "-", "50", "USD"],
    ["EXXONMOBIL HOLDINGS CORP", "XOM", "30233Q108", "BVSRPD7", "15.0", "-", "40", "USD"],
    ["BROWN FORMAN CORP CLASS B", "BF.B", "115637209", "2146838", "4.9", "-", "10", "USD"],
    ["US DOLLAR", "-", "999USDZ92", "-", "0.09", "-", "1000", "USD"],
    ["CONTRA HOLOGIC INCORPO", "2602335D", "436CVR021", "-", "0.01", "-", "5", "USD"],
]

PREVIOUS = pd.DataFrame(
    {
        "ticker": ["AAPL", "PSKY", "XOM", "EA", "CASH_USD"],
        "name": ["APPLE INC", "PARAMOUNT SKYDANCE", "EXXON MOBIL CORP",
                 "ELECTRONIC ARTS INC", "US DOLLAR"],
        "asset_type": ["equity", "equity", "equity", "equity", "cash"],
        "identifier": ["037833100", "69932A204", "30231G102", "285512109", "999USDZ92"],
        "sector": ["Tech Hardware", "Media", "Oil & Gas", "Software", "-"],
        "as_of_date": ["21/05/2026"] * 5,
        "active": [True] * 5,
    }
)


def test_parse_keeps_stocks_and_skips_cash_and_placeholders(tmp_path):
    path = tmp_path / "holdings.xlsx"
    write_ssga_file(path, HOLDINGS_ROWS)

    holdings, skipped, as_of = parse_ssga_holdings(path)

    assert as_of.isoformat() == "2026-10-23"
    assert holdings["ticker"].tolist() == ["AAPL", "SKYD", "XOM", "BF.B"]
    assert holdings["cusip"].iloc[0] == "037833100"
    assert holdings["target_weight"].iloc[0] == pytest.approx(0.60)
    assert set(skipped["company_name"]) == {"US DOLLAR", "CONTRA HOLOGIC INCORPO"}


def test_diff_detects_renames_by_cusip_and_identifier_changes(tmp_path):
    path = tmp_path / "holdings.xlsx"
    write_ssga_file(path, HOLDINGS_ROWS)
    holdings, _, _ = parse_ssga_holdings(path)

    diff = diff_universe(PREVIOUS, holdings).set_index("ticker")

    assert diff.loc["PSKY", "change"] == "renamed"
    assert diff.loc["PSKY", "new_ticker"] == "SKYD"
    assert "SKYD" not in diff.index
    assert diff.loc["XOM", "change"] == "identifier_changed"
    assert diff.loc["EA", "change"] == "removed"
    assert diff.loc["BF.B", "change"] == "added"
    assert diff.loc["AAPL", "change"] == "kept"
    # Cash lines in the old file are not reported as removed stocks.
    assert "CASH_USD" not in diff.index


@pytest.fixture
def refresh_setup(tmp_path):
    holdings_path = tmp_path / "holdings.xlsx"
    write_ssga_file(holdings_path, HOLDINGS_ROWS)

    input_path = tmp_path / "sp500_value_manual.csv"
    PREVIOUS.to_csv(input_path, sep=";", decimal=",", index=False)

    universe_config = {
        "universe_name": "sp500_value",
        "enabled": True,
        "source_type": "manual",
        "input_path": str(input_path),
        "input_format": {"delimiter": ";", "decimal": ","},
        "strategy_role": "value",
        "account_target": "paper",
        "holdings_refresh": {
            "provider": "ssga",
            "url": "https://example.com/holdings.xlsx",
            "snapshot_dir": str(tmp_path / "snapshots"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"universes": [universe_config]}))

    return config_path, holdings_path, input_path, universe_config, tmp_path


def test_refresh_without_apply_leaves_input_file_unchanged(refresh_setup):
    config_path, holdings_path, input_path, _, tmp_path = refresh_setup
    before = input_path.read_text()

    result = refresh_holdings_universe(
        config_path, "sp500_value", holdings_path=holdings_path
    )

    assert not result["applied"]
    assert input_path.read_text() == before
    assert (tmp_path / "snapshots" / "2026-10-23.csv").exists()
    assert result["changes"]["renamed"] == 1


def test_refresh_apply_archives_previous_and_writes_loadable_universe(refresh_setup):
    config_path, holdings_path, input_path, universe_config, tmp_path = refresh_setup

    result = refresh_holdings_universe(
        config_path, "sp500_value", apply=True, holdings_path=holdings_path
    )

    assert result["applied"]
    assert result["previous_as_of_date"] == "2026-05-21"
    assert (tmp_path / "snapshots" / "2026-05-21.csv").exists()

    universe = load_manual_universe(universe_config).set_index("ticker")

    assert list(universe.index) == ["AAPL", "SKYD", "XOM", "BF.B"]
    assert universe.loc["AAPL", "cusip"] == "037833100"
    assert universe.loc["AAPL", "target_weight"] == pytest.approx(0.60)
    # Sectors follow renames through the CUSIP; new names stay blank.
    assert universe.loc["SKYD", "sector"] == "Media"
    assert pd.isna(universe.loc["BF.B", "sector"])
    assert universe.loc["AAPL", "date_added"] == "2026-05-21"
    assert universe.loc["BF.B", "date_added"] == "2026-10-23"


def test_compare_universes_reports_membership_changes(tmp_path):
    for label, tickers, as_of in (
        ("before", ["AAPL", "EA"], "2026-05-21"),
        ("after", ["AAPL", "BF.B"], "2026-10-23"),
    ):
        outputs = tmp_path / label / "outputs"
        outputs.mkdir(parents=True)
        pd.DataFrame(
            {"ticker": tickers, "universe_name": "sp500_value", "as_of_date": as_of}
        ).to_csv(outputs / "universe.csv", index=False)

    changes = compare_universes(tmp_path / "before", tmp_path / "after").iloc[0]

    assert changes["tickers_added"] == 1 and changes["added"] == "BF.B"
    assert changes["tickers_removed"] == 1 and changes["removed"] == "EA"
    assert changes["as_of_after"] == "2026-10-23"
