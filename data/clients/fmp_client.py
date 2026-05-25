from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any

import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 86400  # 24 hours
_REQUEST_COUNT = 0
_REQUEST_LIMIT = 240  # stay safely under FMP free tier 250/day


def _cached(key: str, fn):
    now = time.time()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < _CACHE_TTL:
            return val
    result = fn()
    _CACHE[key] = (now, result)
    return result


class FMPClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._api_key = cfg.fmp_api_key
        self._base = cfg.fmp_base_url

    def _get(self, path: str, params: dict | None = None) -> Any:
        global _REQUEST_COUNT
        if _REQUEST_COUNT >= _REQUEST_LIMIT:
            raise RuntimeError(
                f"FMP daily request limit ({_REQUEST_LIMIT}) reached. Restart tomorrow."
            )
        p = params or {}
        p["apikey"] = self._api_key
        url = f"{self._base}{path}"
        resp = requests.get(url, params=p, timeout=15)
        resp.raise_for_status()
        _REQUEST_COUNT += 1
        return resp.json()

    def get_sp500_constituents(self) -> list[str]:
        key = "sp500_constituents"
        data = _cached(key, lambda: self._get("/sp500_constituent"))
        if isinstance(data, list):
            return [item["symbol"] for item in data if "symbol" in item]
        return []

    def get_tradeable_symbols(self) -> list[dict]:
        """Returns list of all tradeable symbols with exchange/type info."""
        key = "tradeable_symbols"
        return _cached(key, lambda: self._get("/available-traded/list") or [])

    def get_analyst_estimates(self, symbol: str, limit: int = 8) -> list[dict]:
        """Quarterly analyst EPS + revenue estimates with date stamps."""
        key = f"analyst_est_{symbol}_{limit}"
        return _cached(
            key,
            lambda: self._get(f"/analyst-estimates/{symbol}", {"limit": limit, "period": "quarter"}) or [],
        )

    def get_key_metrics(self, symbol: str, period: str = "annual", limit: int = 4) -> list[dict]:
        """Key metrics: ROIC, FCF per share, etc."""
        key = f"key_metrics_{symbol}_{period}"
        return _cached(
            key,
            lambda: self._get(f"/key-metrics/{symbol}", {"period": period, "limit": limit}) or [],
        )

    def get_income_statement(self, symbol: str, period: str = "annual", limit: int = 8) -> list[dict]:
        """Income statements: revenue, gross profit, net income."""
        key = f"income_{symbol}_{period}_{limit}"
        return _cached(
            key,
            lambda: self._get(f"/income-statement/{symbol}", {"period": period, "limit": limit}) or [],
        )

    def get_cash_flow(self, symbol: str, period: str = "annual", limit: int = 4) -> list[dict]:
        """Cash flow statements: FCF."""
        key = f"cashflow_{symbol}_{period}"
        return _cached(
            key,
            lambda: self._get(f"/cash-flow-statement/{symbol}", {"period": period, "limit": limit}) or [],
        )

    def get_profile(self, symbol: str) -> dict:
        """Company profile: sector, market cap, exchange."""
        key = f"profile_{symbol}"
        data = _cached(key, lambda: self._get(f"/profile/{symbol}") or [])
        return data[0] if data else {}

    @staticmethod
    def reset_daily_counter() -> None:
        global _REQUEST_COUNT
        _REQUEST_COUNT = 0

    @staticmethod
    def request_count() -> int:
        return _REQUEST_COUNT
