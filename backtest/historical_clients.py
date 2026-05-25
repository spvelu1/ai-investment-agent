from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from backtest.paper_portfolio import PaperPortfolio
from data.clients.macro_client import MacroClient


class HistoricalAlpacaClient:
    """
    Serves pre-loaded historical bars filtered to the current backtest date.
    Routes order submissions to PaperPortfolio instead of Alpaca's live API.
    """

    def __init__(self, all_bars: pd.DataFrame, portfolio: PaperPortfolio) -> None:
        self._bars = all_bars
        self._portfolio = portfolio
        self._as_of: date = date.today()
        self._prices: dict[str, float] = {}

    def set_date(self, d: date, prices: dict[str, float]) -> None:
        self._as_of = d
        self._prices = prices

    def get_bars(
        self,
        symbols: list[str],
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        # Always use a 365-day lookback capped at as_of to prevent look-ahead.
        # The `start` passed by signal modules uses date.today() as the anchor,
        # which would be in the future relative to the backtest date — ignore it.
        effective_start = self._as_of - timedelta(days=365)
        effective_end = self._as_of

        sym_idx = self._bars.index.get_level_values("symbol")
        date_idx = self._bars.index.get_level_values("date")
        mask = (
            sym_idx.isin(symbols)
            & (date_idx >= pd.Timestamp(effective_start))
            & (date_idx <= pd.Timestamp(effective_end))
        )
        return self._bars[mask]

    def get_bars_lookback(self, symbols: list[str], days: int = 260) -> pd.DataFrame:
        return self.get_bars(symbols)

    def get_positions(self) -> list[dict]:
        return self._portfolio.get_positions_alpaca_format(self._prices)

    def get_account(self) -> dict:
        nav = self._portfolio.nav(self._prices)
        return {
            "equity": nav,
            "cash": self._portfolio.cash,
            "buying_power": self._portfolio.cash * 2,
            "portfolio_value": nav,
        }

    def submit_order(self, symbol: str, qty: float, side: str, reason: str = "") -> dict:
        price = self._prices.get(symbol, 0.0)
        if price > 0:
            self._portfolio.fill(symbol, side, int(abs(qty)), price, self._as_of, reason)
        return {
            "id": f"bt-{symbol}-{self._as_of}",
            "symbol": symbol,
            "qty": qty,
            "side": side,
        }

    def get_clock(self) -> dict:
        return {"is_open": True, "next_open": None, "next_close": None}


class HistoricalMacroClient(MacroClient):
    """
    Returns pre-loaded yfinance macro data sliced to the current backtest date.
    Inherits compute_macro_regime() from MacroClient unchanged.
    """

    def __init__(self, macro_data_full: dict[str, pd.DataFrame]) -> None:
        self._data_full = macro_data_full
        self._as_of: date = date.today()

    def set_date(self, d: date) -> None:
        self._as_of = d

    def get_macro_data(self, lookback_days: int = 252) -> dict[str, pd.DataFrame]:
        # Return all pre-loaded data up to as_of — no lookback window.
        # The 252+30=282 calendar day window only yields ~196 trading days,
        # which is less than the 200 needed for SPY's 200DMA, causing
        # spy_ma200=None and spy_trend=0 (false "below 200DMA") on every date.
        # The engine pre-loads 400 days of buffer, so we always have >200 bars.
        end = pd.Timestamp(self._as_of)
        return {
            ticker: df[df.index <= end]
            for ticker, df in self._data_full.items()
        }
