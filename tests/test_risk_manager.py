"""Tests for sell-rule evaluation — no live API calls."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from portfolio.risk_manager import evaluate, SellSignal


def _make_position(
    symbol: str = "AAPL",
    entry_price: float = 100.0,
    entry_date: str | None = None,
    below_50dma_days: int = 0,
    qty: float = 10.0,
) -> dict:
    if entry_date is None:
        entry_date = (date.today() - timedelta(days=10)).isoformat()
    return {
        "symbol": symbol,
        "entry_price": entry_price,
        "entry_date": entry_date,
        "qty": qty,
        "below_50dma_days": below_50dma_days,
    }


def _make_bars(symbol: str, prices: list[float]) -> pd.DataFrame:
    """Build MultiIndex bars from a simple price list."""
    n = len(prices)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    rows = []
    for d, p in zip(dates, prices):
        rows.append({
            "symbol": symbol,
            "date": d,
            "open": p,
            "high": p * 1.01,
            "low": p * 0.99,
            "close": p,
            "volume": 1e6,
        })
    df = pd.DataFrame(rows).set_index(["symbol", "date"])
    return df


class TestSellRules:
    def _prices(self, n: int = 260, trend: float = 0.0) -> list[float]:
        """Flat price history at 100 by default."""
        return [100.0 + trend * i for i in range(n)]

    def test_no_signal_when_all_ok(self):
        pos = _make_position(entry_price=90.0)
        bars = _make_bars("AAPL", self._prices(260, trend=0.02))
        signals = evaluate([pos], bars)
        assert signals == []

    def test_stop_loss_triggers(self):
        pos = _make_position(symbol="AAPL", entry_price=100.0)
        # Prices fall to 90 — 10% loss, exceeds 8% stop
        prices = [100.0] * 200 + [90.0] * 60
        bars = _make_bars("AAPL", prices)
        signals = evaluate([pos], bars)
        exits = [s for s in signals if s.action == "EXIT" and "stop_loss" in s.reason]
        assert len(exits) == 1

    def test_time_stop_triggers(self):
        old_date = (date.today() - timedelta(days=45)).isoformat()
        pos = _make_position(entry_price=95.0, entry_date=old_date)
        bars = _make_bars("AAPL", self._prices(260))
        signals = evaluate([pos], bars)
        exits = [s for s in signals if "time_stop" in s.reason]
        assert len(exits) == 1

    def test_below_50dma_triggers_after_2_days(self):
        # below_50dma_days=1 → one more day below MA50 should trigger EXIT.
        # MA50 uses last 50 closes. Use 230 days at 100, then 30 at 70 so
        # MA50 ≈ (20*100 + 30*70) / 50 = 62 ... wait, easier:
        # 220 prices at 100, then 30 at 70: MA50 = (20*100 + 30*70)/50 = 86, last = 70 < 86. ✓
        pos = _make_position(entry_price=95.0, below_50dma_days=1)
        prices = [100.0] * 220 + [70.0] * 40
        bars = _make_bars("AAPL", prices)
        signals = evaluate([pos], bars)
        exits = [s for s in signals if s.reason == "close_below_50dma_2d"]
        assert len(exits) == 1

    def test_profit_trim_triggers(self):
        # Gain > 15%, last close below 10DMA.
        # entry_price=70, prices rise to 105 then last bar drops to 85.
        # gain = (85-70)/70 = 21.4% > 15%. MA10 = avg of 9 bars at 105 + 1 at 85 = 103. 85 < 103. ✓
        pos = _make_position(entry_price=70.0)
        prices = [70.0] * 200 + [105.0] * 9 + [85.0]
        bars = _make_bars("AAPL", prices)
        signals = evaluate([pos], bars)
        trims = [s for s in signals if s.action == "TRIM"]
        assert len(trims) == 1
        assert trims[0].trim_pct == 0.5

    def test_multiple_positions(self):
        pos1 = _make_position("AAPL", entry_price=100.0)
        pos2 = _make_position("MSFT", entry_price=100.0)
        bars_aapl = _make_bars("AAPL", [100.0] * 200 + [88.0] * 60)  # stop loss
        bars_msft = _make_bars("MSFT", self._prices(260))               # no signal
        combined = pd.concat([bars_aapl, bars_msft])
        signals = evaluate([pos1, pos2], combined)
        assert any(s.symbol == "AAPL" for s in signals)
        assert not any(s.symbol == "MSFT" for s in signals)
