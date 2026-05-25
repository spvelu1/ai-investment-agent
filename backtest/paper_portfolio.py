from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date


@dataclass
class PaperPortfolio:
    initial_cash: float = 100_000.0
    slippage_pct: float = 0.001  # 0.1% market impact per side

    cash: float = field(init=False)
    positions: dict[str, dict] = field(init=False, default_factory=dict)
    nav_history: list[dict] = field(init=False, default_factory=list)
    trade_log: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    # ── order execution ───────────────────────────────────────────────────────

    def fill(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        trade_date: date,
        reason: str = "",
    ) -> None:
        if qty <= 0 or price <= 0:
            return
        slippage = self.slippage_pct if side == "buy" else -self.slippage_pct
        fill_price = price * (1.0 + slippage)

        if side == "buy":
            cost = qty * fill_price
            if cost > self.cash:
                qty = math.floor(self.cash / fill_price)
                cost = qty * fill_price
            if qty <= 0:
                return
            self.cash -= cost
            if symbol in self.positions:
                old = self.positions[symbol]
                total = old["qty"] + qty
                self.positions[symbol] = {
                    **old,
                    "qty": total,
                    "entry_price": (old["qty"] * old["entry_price"] + qty * fill_price) / total,
                }
            else:
                self.positions[symbol] = {
                    "qty": qty,
                    "entry_price": fill_price,
                    "entry_date": trade_date.isoformat(),
                    "below_50dma_days": 0,
                }
        else:  # sell
            if symbol not in self.positions:
                return
            actual = min(qty, int(self.positions[symbol]["qty"]))
            if actual <= 0:
                return
            self.cash += actual * fill_price
            remaining = self.positions[symbol]["qty"] - actual
            if remaining < 1:
                del self.positions[symbol]
            else:
                self.positions[symbol]["qty"] = remaining

        self.trade_log.append({
            "date": trade_date,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "fill_price": round(fill_price, 4),
            "reason": reason,
        })

    # ── valuation ─────────────────────────────────────────────────────────────

    def nav(self, prices: dict[str, float]) -> float:
        equity = sum(
            pos["qty"] * prices.get(sym, pos["entry_price"])
            for sym, pos in self.positions.items()
        )
        return self.cash + equity

    def snapshot(self, d: date, prices: dict[str, float]) -> float:
        n = self.nav(prices)
        self.nav_history.append({
            "date": d,
            "nav": n,
            "cash": self.cash,
            "num_positions": len(self.positions),
        })
        return n

    # ── position format adapters ──────────────────────────────────────────────

    def get_positions_for_risk_manager(self) -> list[dict]:
        return [
            {
                "symbol": sym,
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "qty": pos["qty"],
                "below_50dma_days": pos.get("below_50dma_days", 0),
            }
            for sym, pos in self.positions.items()
        ]

    def get_positions_alpaca_format(self, prices: dict[str, float] | None = None) -> list[dict]:
        prices = prices or {}
        result = []
        for sym, pos in self.positions.items():
            cur = prices.get(sym, pos["entry_price"])
            result.append({
                "symbol": sym,
                "qty": pos["qty"],
                "avg_entry_price": pos["entry_price"],
                "current_price": cur,
                "market_value": pos["qty"] * cur,
                "unrealized_pl": (cur - pos["entry_price"]) * pos["qty"],
                "unrealized_plpc": (cur - pos["entry_price"]) / pos["entry_price"]
                    if pos["entry_price"] else 0.0,
            })
        return result
