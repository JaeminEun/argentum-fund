from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote_plus

import requests

from src.universe.config import load_config

def resolve_user_agent(sec_config: Dict[str, Any]) -> str:
    """
    Resolve the SEC User-Agent from either a direct config value
    or an environment variable.

    Preferred public-repo pattern:
        user_agent_env: SEC_USER_AGENT

    Local shell:
        export SEC_USER_AGENT="ArgentumFund/0.1 contact: email@example.com"
    """
    if "user_agent_env" in sec_config:
        env_name = sec_config["user_agent_env"]
        user_agent = os.getenv(env_name)

        if not user_agent:
            raise ValueError(
                f"Environment variable '{env_name}' is not set. "
                "Set it before using the SEC client. Example:\n"
                f"export {env_name}='ArgentumFund/0.1 contact: email@example.com'"
            )

        return user_agent

    if "user_agent" in sec_config:
        return sec_config["user_agent"]

    raise ValueError(
        "SEC config must define either 'user_agent_env' or 'user_agent'."
    )

class SecClient:
    """
    Lightweight SEC EDGAR API client.

    Responsibilities:
    - manage SEC request headers
    - respect request delay
    - retry failed requests
    - optionally cache JSON responses
    """

    def __init__(
        self,
        user_agent: str,
        request_delay_seconds: float = 0.15,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        cache_enabled: bool = True,
        cache_dir: str | Path = "data/sec/cache",
    ) -> None:
        if not user_agent or "contact:" not in user_agent.lower():
            raise ValueError(
                "SEC user_agent should identify the project and include contact info. "
                "Example: 'ArgentumFund/0.1 contact: your_email@example.com'"
            )

        self.user_agent = user_agent
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir)

        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        }

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SecClient":
        """
        Build a SecClient from the project's YAML config.
        """
        config = load_config(config_path)

        if "sec_api" not in config:
            raise ValueError("Missing 'sec_api' section in config file.")

        sec_config = config["sec_api"]

        return cls(
            user_agent=resolve_user_agent(sec_config),
            request_delay_seconds=float(sec_config.get("request_delay_seconds", 0.15)),
            max_retries=int(sec_config.get("max_retries", 3)),
            timeout_seconds=int(sec_config.get("timeout_seconds", 30)),
            cache_enabled=bool(sec_config.get("cache_enabled", True)),
            cache_dir=sec_config.get("cache_dir", "data/sec/cache"),
        )

    def _cache_path_for_url(self, url: str) -> Path:
        """
        Convert a URL into a safe local cache path.
        """
        safe_name = quote_plus(url)
        return self.cache_dir / f"{safe_name}.json"

    def get_json(
        self,
        url: str,
        force_refresh: bool = False,
        host: str | None = None,
    ) -> Dict[str, Any]:
        """
        Retrieve JSON from the SEC, using cache if enabled.

        Parameters
        ----------
        url:
            URL to retrieve.
        force_refresh:
            If True, ignore any cached copy.
        host:
            Optional override for Host header. Useful because
            company_tickers.json is hosted on www.sec.gov while many
            API endpoints are on data.sec.gov.
        """
        cache_path = self._cache_path_for_url(url)

        if self.cache_enabled and cache_path.exists() and not force_refresh:
            with cache_path.open("r", encoding="utf-8") as file:
                return json.load(file)

        headers = self.headers.copy()

        if host is not None:
            headers["Host"] = host
        elif "www.sec.gov" in url:
            headers["Host"] = "www.sec.gov"
        else:
            headers["Host"] = "data.sec.gov"

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(self.request_delay_seconds)

                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )

                response.raise_for_status()

                data = response.json()

                if self.cache_enabled:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("w", encoding="utf-8") as file:
                        json.dump(data, file)

                return data

            except Exception as error:
                last_error = error
                wait_seconds = self.request_delay_seconds * attempt * 2

                print(
                    f"Warning: SEC request failed on attempt "
                    f"{attempt}/{self.max_retries}: {url}"
                )
                print(f"Reason: {error}")
                time.sleep(wait_seconds)

        raise RuntimeError(
            f"SEC request failed after {self.max_retries} attempts: {url}"
        ) from last_error
