from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100.0


def compute(bars: pd.DataFrame, spy_bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute MomentumScore for each symbol in bars.

    bars: MultiIndex (symbol, date) DataFrame with 'close' column.
    spy_bars: Single-symbol DataFrame with 'close' column for SPY.

    Returns DataFrame indexed by symbol with columns:
        mom_6, mom_3, rs_63d, momentum_score,
        close, ma50, ma200, ma10, vol_20d
    """
    cfg = get_settings()
    symbols = bars.index.get_level_values("symbol").unique().tolist()
    rows: list[dict] = []

    # Pre-compute SPY 63-day return for RS calculation
    spy_return_63 = _spy_return_63d(spy_bars)

    for sym in symbols:
        try:
            s = bars.loc[sym]["close"].sort_index()
            if len(s) < 10:
                continue
            row = _compute_symbol(sym, s, spy_return_63)
            rows.append(row)
        except Exception as exc:
            logger.debug("Momentum skipped for %s: %s", sym, exc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol")

    df["mom_6_pct"] = _pct_rank(df["mom_6"].fillna(df["mom_6"].median()))
    df["mom_3_pct"] = _pct_rank(df["mom_3"].fillna(df["mom_3"].median()))
    df["rs_pct"] = _pct_rank(df["rs_63d"].fillna(df["rs_63d"].median()))

    df["momentum_score"] = (
        cfg.w_mom_6 * df["mom_6_pct"]
        + cfg.w_mom_3 * df["mom_3_pct"]
        + cfg.w_rs * df["rs_pct"]
    )

    return df


def _compute_symbol(symbol: str, close: pd.Series, spy_return_63: float) -> dict:
    n = len(close)
    last = float(close.iloc[-1])

    # MOM_6: Price(t-21) / Price(t-126) - 1  (skip most recent month)
    mom_6 = np.nan
    if n >= 127:
        p_t21 = float(close.iloc[-22])   # ~21 trading days ago
        p_t126 = float(close.iloc[-127]) # ~126 trading days ago
        if p_t126 != 0:
            mom_6 = p_t21 / p_t126 - 1.0

    # MOM_3: Price(t) / Price(t-63) - 1
    mom_3 = np.nan
    if n >= 64:
        p_t63 = float(close.iloc[-64])
        if p_t63 != 0:
            mom_3 = last / p_t63 - 1.0

    # RS vs SPY (63d)
    stock_return_63 = np.nan
    if n >= 64:
        p_t63 = float(close.iloc[-64])
        if p_t63 != 0:
            stock_return_63 = last / p_t63 - 1.0
    rs_63d = (stock_return_63 - spy_return_63) if (not np.isnan(stock_return_63) and not np.isnan(spy_return_63)) else np.nan

    # Moving averages and vol (used downstream by breakout + risk manager)
    ma50 = float(close.iloc[-50:].mean()) if n >= 50 else np.nan
    ma200 = float(close.iloc[-200:].mean()) if n >= 200 else np.nan
    ma10 = float(close.iloc[-10:].mean()) if n >= 10 else np.nan
    vol_20d = float(close.pct_change().iloc[-20:].std()) if n >= 21 else np.nan

    return {
        "symbol": symbol,
        "close": last,
        "mom_6": mom_6,
        "mom_3": mom_3,
        "rs_63d": rs_63d,
        "ma50": ma50,
        "ma200": ma200,
        "ma10": ma10,
        "vol_20d": vol_20d,
    }


def _spy_return_63d(spy_bars: pd.DataFrame | None) -> float:
    if spy_bars is None or spy_bars.empty:
        return np.nan
    close = spy_bars["close"].sort_index() if "close" in spy_bars.columns else spy_bars.sort_index()
    if len(close) < 64:
        return np.nan
    return float(close.iloc[-1]) / float(close.iloc[-64]) - 1.0
