import json
import os
import time

import pytest

import src.sec.client as client_module
from src.fundamentals.sec_company_facts import companyfacts_cache_path, fetch_companyfacts
from src.sec.client import SecClient

URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def requests_log(monkeypatch):
    """
    Replace network calls; set requests_log.fail to simulate an outage.
    """
    class Log:
        calls = []
        fail = False

    def fake_get(url, headers, timeout):
        Log.calls.append(url)
        if Log.fail:
            raise ConnectionError("SEC unreachable")
        return FakeResponse({"source": "network"})

    monkeypatch.setattr(client_module.requests, "get", fake_get)
    return Log


def make_client(tmp_path, max_cache_age_days=7):
    return SecClient(
        user_agent="ArgentumFund/test contact: test@example.com",
        request_delay_seconds=0,
        max_retries=2,
        cache_dir=tmp_path / "client",
        max_cache_age_days=max_cache_age_days,
    )


def write_cache(path, payload, age_days):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    mtime = time.time() - age_days * 86400
    os.utime(path, (mtime, mtime))


def test_fresh_cache_served_without_network(tmp_path, requests_log):
    client = make_client(tmp_path)
    write_cache(client._cache_path_for_url(URL), {"source": "cache"}, age_days=2)

    assert client.get_json(URL) == {"source": "cache"}
    assert requests_log.calls == []
    assert client.last_response_from_cache


def test_stale_cache_is_refetched(tmp_path, requests_log):
    client = make_client(tmp_path)
    write_cache(client._cache_path_for_url(URL), {"source": "cache"}, age_days=30)

    assert client.get_json(URL) == {"source": "network"}
    assert requests_log.calls
    assert not client.last_response_from_cache


def test_stale_cache_used_when_refetch_fails(tmp_path, requests_log):
    client = make_client(tmp_path)
    write_cache(client._cache_path_for_url(URL), {"source": "cache"}, age_days=30)
    requests_log.fail = True

    assert client.get_json(URL) == {"source": "cache"}
    assert client.last_response_from_cache


def test_missing_cache_and_failed_fetch_raises(tmp_path, requests_log):
    client = make_client(tmp_path)
    requests_log.fail = True

    with pytest.raises(RuntimeError):
        client.get_json(URL)


def test_no_max_age_keeps_cache_forever(tmp_path, requests_log):
    client = make_client(tmp_path, max_cache_age_days=None)
    write_cache(client._cache_path_for_url(URL), {"source": "cache"}, age_days=400)

    assert client.get_json(URL) == {"source": "cache"}
    assert requests_log.calls == []


def test_companyfacts_refetch_failure_keeps_original_download_time(
    tmp_path,
    requests_log,
):
    client = make_client(tmp_path)
    facts_dir = tmp_path / "companyfacts"
    facts_path = companyfacts_cache_path(facts_dir, "0000000001")

    write_cache(client._cache_path_for_url(URL), {"source": "cache"}, age_days=30)
    write_cache(facts_path, {"source": "cache"}, age_days=30)
    requests_log.fail = True

    facts = fetch_companyfacts(
        client=client,
        companyfacts_base_url="https://data.sec.gov/api/xbrl/companyfacts",
        facts_cache_dir=facts_dir,
        cik_padded="0000000001",
    )

    assert facts == {"source": "cache"}
    # Re-saving must not make 30-day-old data look freshly downloaded.
    assert client_module.cache_age_days(facts_path) > 29


def test_companyfacts_refreshed_when_stale(tmp_path, requests_log):
    client = make_client(tmp_path)
    facts_dir = tmp_path / "companyfacts"
    facts_path = companyfacts_cache_path(facts_dir, "0000000001")
    write_cache(facts_path, {"source": "cache"}, age_days=30)

    facts = fetch_companyfacts(
        client=client,
        companyfacts_base_url="https://data.sec.gov/api/xbrl/companyfacts",
        facts_cache_dir=facts_dir,
        cik_padded="0000000001",
    )

    assert facts == {"source": "network"}
    assert json.loads(facts_path.read_text()) == {"source": "network"}
    assert client_module.cache_age_days(facts_path) < 1
