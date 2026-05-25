"""Shared pytest fixtures — set env vars before any module-level Settings instantiation."""
from __future__ import annotations

import os
import pytest


def pytest_configure(config):
    """Set required env vars before any test module is imported."""
    os.environ.setdefault("ALPACA_API_KEY", "test_alpaca_key")
    os.environ.setdefault("ALPACA_SECRET_KEY", "test_alpaca_secret")
    os.environ.setdefault("FMP_API_KEY", "test_fmp_key")
    os.environ.setdefault("DB_PATH", "/tmp/test_investment_agent.db")


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset the settings singleton between tests to avoid state leakage."""
    import config.settings as s
    s._settings = None
    yield
    s._settings = None


@pytest.fixture(autouse=True)
def patch_store_no_op(monkeypatch):
    """
    Redirect DuckDB store to an in-memory DB for all tests.
    Prevents tests from touching the real market.db file.
    """
    import duckdb

    def _test_conn():
        return duckdb.connect(":memory:")

    monkeypatch.setattr("data.store._conn", _test_conn)
