from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.price_history import drop_incomplete_sessions, last_completed_session_date

NEW_YORK = ZoneInfo("America/New_York")


def test_cutoff_is_yesterday_while_market_is_open():
    now = datetime(2026, 10, 26, 11, 0, tzinfo=NEW_YORK)

    assert last_completed_session_date(now) == date(2026, 10, 25)


def test_cutoff_is_today_once_close_has_settled():
    assert last_completed_session_date(
        datetime(2026, 10, 26, 16, 15, tzinfo=NEW_YORK)
    ) == date(2026, 10, 25)
    assert last_completed_session_date(
        datetime(2026, 10, 26, 16, 30, tzinfo=NEW_YORK)
    ) == date(2026, 10, 26)


def test_cutoff_uses_market_timezone_not_local_time():
    # 18:00 in Israel is 12:00 in New York: the session is still open.
    now = datetime(2026, 10, 26, 18, 0, tzinfo=ZoneInfo("Asia/Jerusalem"))

    assert last_completed_session_date(now) == date(2026, 10, 25)


def test_drop_incomplete_sessions_keeps_closed_days_only():
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "date": [date(2026, 10, 23), date(2026, 10, 26), date(2026, 10, 27)],
            "close": [1.0, 2.0, 3.0],
        }
    )

    kept = drop_incomplete_sessions(prices, cutoff=date(2026, 10, 25))

    assert kept["date"].tolist() == [date(2026, 10, 23)]
