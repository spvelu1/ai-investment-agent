"""Tests for DuckDB store — uses a temp DB, no real file state."""
import os
import tempfile

import pandas as pd
import pytest

from datetime import date


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Redirect DuckDB to a temp file for each test."""
    db_path = str(tmp_path / "test_market.db")
    monkeypatch.setenv("DB_PATH", db_path)
    # Force settings reload
    import config.settings as s
    s._settings = None
    monkeypatch.setattr(s.Settings.model_fields["db_path"], "default", db_path)
    yield db_path
    s._settings = None


def _make_bars_df() -> pd.DataFrame:
    symbols = ["AAPL", "MSFT"]
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    rows = []
    for sym in symbols:
        for d in dates:
            rows.append({
                "symbol": sym, "date": d,
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.5, "volume": 1e6,
            })
    return pd.DataFrame(rows)


class TestBarsStore:
    def test_write_and_read_roundtrip(self, tmp_path, monkeypatch):
        import config.settings as s
        s._settings = None
        cfg = s.get_settings.__wrapped__() if hasattr(s.get_settings, "__wrapped__") else None
        # Patch db_path directly
        import data.store as store_mod
        monkeypatch.setattr("data.store._conn", lambda: __import__("duckdb").connect(str(tmp_path / "t.db")))

        df_in = _make_bars_df()
        store_mod.write_bars(df_in)
        df_out = store_mod.read_bars(["AAPL", "MSFT"])
        assert not df_out.empty
        assert set(df_out.index.get_level_values("symbol")) == {"AAPL", "MSFT"}


class TestPositionsStore:
    def test_write_and_read(self, tmp_path, monkeypatch):
        import duckdb
        import data.store as store_mod
        monkeypatch.setattr("data.store._conn", lambda: duckdb.connect(str(tmp_path / "p.db")))

        positions = [
            {"symbol": "AAPL", "entry_date": date.today(), "entry_price": 150.0, "qty": 10.0, "below_50dma_days": 0},
            {"symbol": "MSFT", "entry_date": date.today(), "entry_price": 300.0, "qty": 5.0, "below_50dma_days": 1},
        ]
        store_mod.write_positions(positions)
        result = store_mod.read_positions()
        symbols = {r["symbol"] for r in result}
        assert symbols == {"AAPL", "MSFT"}

    def test_write_empty_clears(self, tmp_path, monkeypatch):
        import duckdb
        import data.store as store_mod
        monkeypatch.setattr("data.store._conn", lambda: duckdb.connect(str(tmp_path / "p2.db")))

        positions = [
            {"symbol": "AAPL", "entry_date": date.today(), "entry_price": 150.0, "qty": 10.0, "below_50dma_days": 0},
        ]
        store_mod.write_positions(positions)
        store_mod.write_positions([])
        result = store_mod.read_positions()
        assert result == []
