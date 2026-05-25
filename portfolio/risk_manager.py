from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from config.settings import get_settings
from data import store

logger = logging.getLogger(__name__)


@dataclass
class SellSignal:
    symbol: str
    action: str          # "EXIT" or "TRIM"
    trim_pct: float      # 1.0 = full exit, 0.5 = trim half
    reason: str


def evaluate(
    positions: list[dict],
    bars: pd.DataFrame,
    signals_df: pd.DataFrame | None = None,
    as_of: date | None = None,
) -> list[SellSignal]:
    """
    Evaluate all sell rules for every open position.

    positions: list of dicts with keys: symbol, entry_date, entry_price, qty, below_50dma_days
    bars: MultiIndex (symbol, date) DataFrame with OHLCV columns
    signals_df: latest signals DataFrame (for revision + RS rank checks)

    Returns list of SellSignal — caller submits the orders.
    """
    cfg = get_settings()
    sell_signals: list[SellSignal] = []

    for pos in positions:
        sym = pos["symbol"]
        signals = _get_symbol_signals(sym, signals_df)
        sym_bars = _get_sym_bars(sym, bars)

        if sym_bars is None or sym_bars.empty:
            logger.warning("No bar data for position %s — skipping sell check", sym)
            continue

        last_close = float(sym_bars["close"].iloc[-1])
        entry_price = float(pos.get("entry_price", last_close))
        entry_date = _parse_date(pos.get("entry_date"))
        below_50dma_days = int(pos.get("below_50dma_days", 0))
        reference_date = as_of if as_of is not None else date.today()
        holding_days = (reference_date - entry_date).days if entry_date else 0

        ma50 = float(sym_bars["close"].iloc[-50:].mean()) if len(sym_bars) >= 50 else None
        ma10 = float(sym_bars["close"].iloc[-10:].mean()) if len(sym_bars) >= 10 else None
        gain = (last_close - entry_price) / entry_price if entry_price else 0.0

        sell = _check_rules(
            sym=sym,
            last_close=last_close,
            entry_price=entry_price,
            gain=gain,
            holding_days=holding_days,
            below_50dma_days=below_50dma_days,
            ma50=ma50,
            ma10=ma10,
            signals=signals,
            cfg=cfg,
        )

        # Update consecutive 50DMA breach counter in store
        if ma50 is not None:
            new_below_days = below_50dma_days + 1 if last_close < ma50 else 0
            if new_below_days != below_50dma_days:
                store.update_below_50dma_days(sym, new_below_days)

        if sell:
            sell_signals.append(sell)

    return sell_signals


def _check_rules(
    sym, last_close, entry_price, gain, holding_days,
    below_50dma_days, ma50, ma10, signals, cfg,
) -> SellSignal | None:

    # Rule 1: Close < 50DMA for 2 consecutive days
    if ma50 and last_close < ma50 and below_50dma_days >= cfg.below_50dma_days - 1:
        return SellSignal(sym, "EXIT", 1.0, "close_below_50dma_2d")

    # Rule 2: EPS revision score fell below 40th percentile
    if signals and signals.get("earnings_revision_score") is not None:
        if signals["earnings_revision_score"] < cfg.eps_rev_sell_pct:
            return SellSignal(sym, "EXIT", 1.0, "eps_revision_deteriorated")

    # Rule 3: Relative strength rank < 30th percentile
    if signals and signals.get("rs_63d") is not None:
        rs_rank = signals.get("rs_pct")
        if rs_rank is not None and rs_rank < cfg.rs_sell_pct:
            return SellSignal(sym, "EXIT", 1.0, "relative_strength_weak")

    # Rule 4: Time stop — holding > max_holding_days
    if holding_days > cfg.max_holding_days:
        return SellSignal(sym, "EXIT", 1.0, f"time_stop_{holding_days}d")

    # Rule 5: -8% trailing stop
    if gain < -cfg.stop_loss_pct:
        return SellSignal(sym, "EXIT", 1.0, f"stop_loss_{gain:.1%}")

    # Rule 6: Profit protection — gain > 15% but price fell below 10DMA → trim 50%
    if gain > cfg.profit_trim_threshold and ma10 and last_close < ma10:
        return SellSignal(sym, "TRIM", 0.5, f"profit_protect_{gain:.1%}")

    return None


def _get_sym_bars(sym: str, bars: pd.DataFrame) -> pd.DataFrame | None:
    if bars is None or bars.empty:
        return None
    try:
        return bars.loc[sym].sort_index() if sym in bars.index.get_level_values(0) else None
    except Exception:
        return None


def _get_symbol_signals(sym: str, signals_df: pd.DataFrame | None) -> dict | None:
    if signals_df is None or signals_df.empty:
        return None
    if sym in signals_df.index:
        row = signals_df.loc[sym]
        return row.to_dict() if hasattr(row, "to_dict") else None
    return None


def _parse_date(val) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None
