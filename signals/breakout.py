from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100.0


def compute(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute BreakoutScore for each symbol in bars.

    bars: MultiIndex (symbol, date) with columns open, high, low, close, volume.

    Returns DataFrame indexed by symbol with columns:
        breakout_20, vol_expansion, dist_50dma, breakout_score,
        above_50dma, above_200dma
    """
    cfg = get_settings()
    symbols = bars.index.get_level_values("symbol").unique().tolist()
    rows: list[dict] = []

    for sym in symbols:
        try:
            sym_bars = bars.loc[sym].sort_index()
            if len(sym_bars) < 21:
                continue
            row = _compute_symbol(sym, sym_bars)
            rows.append(row)
        except Exception as exc:
            logger.debug("Breakout skipped for %s: %s", sym, exc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")

    df["vol_expansion_pct"] = _pct_rank(df["vol_expansion"].fillna(df["vol_expansion"].median()))
    df["dist_50dma_pct"] = _pct_rank(df["dist_50dma"].fillna(df["dist_50dma"].median()))

    df["breakout_score"] = (
        cfg.w_breakout_binary * df["breakout_20"] * 100.0
        + cfg.w_vol_expansion * df["vol_expansion_pct"]
        + cfg.w_dist_50dma * df["dist_50dma_pct"]
    )

    return df


def _compute_symbol(symbol: str, sym_bars: pd.DataFrame) -> dict:
    close = sym_bars["close"]
    high = sym_bars["high"]
    volume = sym_bars["volume"]

    last_close = float(close.iloc[-1])
    last_vol = float(volume.iloc[-1])

    # BREAKOUT_20: today's close > max high over prior 20 trading days
    prior_20_high = float(high.iloc[-21:-1].max()) if len(high) >= 21 else np.nan
    breakout_20 = 1.0 if (prior_20_high and last_close > prior_20_high) else 0.0

    # VOL_EXPANSION: today's volume / 20-day avg volume
    avg_vol_20 = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else np.nan
    vol_expansion = (last_vol / avg_vol_20) if avg_vol_20 else np.nan

    # DIST_50DMA
    ma50 = float(close.iloc[-50:].mean()) if len(close) >= 50 else np.nan
    dist_50dma = ((last_close - ma50) / ma50) if ma50 else np.nan

    # Hard filter flags (used in master_score)
    ma200 = float(close.iloc[-200:].mean()) if len(close) >= 200 else np.nan
    above_50dma = bool(ma50 and last_close > ma50)
    above_200dma = bool(ma200 and last_close > ma200)

    return {
        "symbol": symbol,
        "breakout_20": breakout_20,
        "vol_expansion": vol_expansion,
        "dist_50dma": dist_50dma,
        "above_50dma": above_50dma,
        "above_200dma": above_200dma,
    }
