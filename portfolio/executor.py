from __future__ import annotations

import logging
import math
from datetime import date

import pandas as pd

from config.settings import get_settings
from data.clients.alpaca_client import AlpacaClient
from data import store
from portfolio.risk_manager import SellSignal

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, alpaca: AlpacaClient | None = None) -> None:
        self._alpaca = alpaca or AlpacaClient()

    def execute_sells(self, sell_signals: list[SellSignal]) -> None:
        """Process sell signals before buys to free up capital."""
        if not sell_signals:
            return
        positions = {p["symbol"]: p for p in self._alpaca.get_positions()}
        for sig in sell_signals:
            pos = positions.get(sig.symbol)
            if not pos:
                logger.warning("Sell signal for %s but no open position found", sig.symbol)
                continue
            qty_held = float(pos["qty"])
            qty_to_sell = math.floor(qty_held * sig.trim_pct)
            if qty_to_sell <= 0:
                continue
            try:
                order = self._alpaca.submit_order(
                    symbol=sig.symbol,
                    qty=qty_to_sell,
                    side="sell",
                    reason=sig.reason,
                )
                store.log_order(
                    order_id=order["id"],
                    symbol=sig.symbol,
                    side="sell",
                    qty=qty_to_sell,
                    reason=sig.reason,
                )
                logger.info(
                    "SELL %s qty=%.0f reason=%s",
                    sig.symbol, qty_to_sell, sig.reason,
                )
            except Exception as exc:
                logger.error("Failed to submit sell for %s: %s", sig.symbol, exc)

    def execute_rebalance(
        self,
        target_weights: dict[str, float],
        current_positions: list[dict],
    ) -> None:
        """
        Bring portfolio to target_weights.

        Sells first (excess positions + removed names), then buys.
        """
        if not target_weights:
            self._exit_all(current_positions, reason="macro_derisking")
            return

        account = self._alpaca.get_account()
        portfolio_value = account["portfolio_value"]
        if portfolio_value <= 0:
            logger.warning("Portfolio value is 0 — skipping rebalance")
            return

        cfg = get_settings()
        current_map = {p["symbol"]: p for p in current_positions}
        target_symbols = set(target_weights.keys())
        current_symbols = set(current_map.keys())

        # Current weights by market value
        current_weights: dict[str, float] = {
            sym: float(pos["market_value"]) / portfolio_value
            for sym, pos in current_map.items()
        } if portfolio_value > 0 else {}

        # Sells: symbols no longer in target (clear opportunity cost — always exit)
        for sym in current_symbols - target_symbols:
            qty = float(current_map[sym]["qty"])
            if qty > 0:
                self._submit_order(sym, qty, "sell", "removed_from_portfolio")

        # Compute target shares for each position
        target_shares: dict[str, float] = {}
        for sym, weight in target_weights.items():
            price = (
                float(current_map[sym]["current_price"])
                if sym in current_map
                else self._get_last_price(sym)
            )
            if price and price > 0:
                target_shares[sym] = math.floor(portfolio_value * weight / price)

        # Sells: trim oversized carry-overs — only if drift exceeds threshold
        for sym in target_symbols & current_symbols:
            drift = current_weights.get(sym, 0) - target_weights.get(sym, 0)
            if drift < cfg.drift_threshold:
                continue
            delta = float(current_map[sym]["qty"]) - target_shares.get(sym, 0)
            if delta >= 1:
                self._submit_order(sym, delta, "sell", "rebalance_trim")

        # Buys: new positions and meaningfully undersized ones
        for sym in sorted(target_symbols, key=lambda s: target_weights.get(s, 0), reverse=True):
            current_qty = float(current_map[sym]["qty"]) if sym in current_map else 0.0
            drift = target_weights.get(sym, 0) - current_weights.get(sym, 0)
            if sym in current_symbols and drift < cfg.drift_threshold:
                continue
            delta = target_shares.get(sym, 0) - current_qty
            if delta >= 1:
                self._submit_order(sym, delta, "buy", "rebalance_add")

    def _exit_all(self, current_positions: list[dict], reason: str) -> None:
        for pos in current_positions:
            qty = float(pos["qty"])
            if qty > 0:
                self._submit_order(pos["symbol"], qty, "sell", reason)

    def _submit_order(self, symbol: str, qty: float, side: str, reason: str) -> None:
        try:
            order = self._alpaca.submit_order(symbol, qty, side, reason)
            store.log_order(order["id"], symbol, side, qty, reason)
        except Exception as exc:
            logger.error("Order failed %s %s %.0f: %s", side, symbol, qty, exc)

    def _get_last_price(self, symbol: str) -> float | None:
        """Fetch current price via Alpaca latest bar."""
        from datetime import date, timedelta
        try:
            bars = self._alpaca.get_bars([symbol], start=date.today() - timedelta(days=5))
            if symbol in bars.index.get_level_values(0):
                return float(bars.loc[symbol]["close"].iloc[-1])
        except Exception as exc:
            logger.warning("Could not get price for %s: %s", symbol, exc)
        return None
