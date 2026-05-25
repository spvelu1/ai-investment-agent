from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config.settings import get_settings

logger = logging.getLogger(__name__)


class AlpacaClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._data_client = StockHistoricalDataClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
        )
        self._trading_client = TradingClient(
            api_key=cfg.alpaca_api_key,
            secret_key=cfg.alpaca_secret_key,
            paper=True,
        )

    def get_bars(
        self,
        symbols: list[str],
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Return daily OHLCV DataFrame with MultiIndex (symbol, timestamp)."""
        if end is None:
            end = date.today()

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
            end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc),
            adjustment="split",
        )
        bars = self._data_client.get_stock_bars(request)
        df = bars.df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.rename(columns={"timestamp": "date"})
        df = df.set_index(["symbol", "date"]).sort_index()
        return df

    def get_bars_lookback(self, symbols: list[str], days: int = 252) -> pd.DataFrame:
        """Convenience wrapper: last N calendar days of daily bars."""
        start = date.today() - timedelta(days=days + 60)
        return self.get_bars(symbols, start=start)

    def get_account(self) -> dict:
        acct = self._trading_client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
        }

    def get_positions(self) -> list[dict]:
        positions = self._trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        reason: str = "",
    ) -> dict:
        """Submit a market order. side must be 'buy' or 'sell'."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._trading_client.submit_order(request)
        logger.info(
            "ORDER submitted | %s %s %.4f | reason=%s | id=%s",
            side.upper(),
            symbol,
            abs(qty),
            reason,
            order.id,
        )
        return {"id": str(order.id), "symbol": symbol, "qty": qty, "side": side}

    def get_clock(self) -> dict:
        clock = self._trading_client.get_clock()
        return {"is_open": clock.is_open, "next_open": clock.next_open, "next_close": clock.next_close}
