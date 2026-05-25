from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
# Hysteresis band prevents whipsawing at the boundary.
# Signal must cross ENTER to flip to concentrated, fall below EXIT to flip back.
ENTER_CONCENTRATED = 4.0   # composite score → concentrated mode
EXIT_CONCENTRATED = 0.0    # composite score falls here → back to factor

# Lookbacks: short is the *leading* component; long is the *confirming* anchor.
_SHORT = 21    # ~1 month: fast, responsive — detects regime shifts early
_LONG = 63     # ~3 months: slow, stable — confirms the shift is real

_REGIME_TICKERS = ["SPY", "RSP", "QQQ", "IWM"]


def compute_from_bars(
    all_bars: pd.DataFrame,
    as_of: date,
    previous_regime: str = "factor",
) -> dict:
    """
    Compute regime from pre-loaded bar data (used in backtest).
    RSP, QQQ, IWM must be included in all_bars.
    previous_regime: last known regime — used for hysteresis.
    """
    def _ret(sym: str, window: int) -> float:
        try:
            s = all_bars.loc[sym]["close"]
            s = s[s.index <= pd.Timestamp(as_of)].dropna()
            if len(s) < window // 2:
                return 0.0
            return (float(s.iloc[-1]) / float(s.iloc[max(0, len(s) - window)]) - 1) * 100
        except (KeyError, IndexError):
            return 0.0

    return _classify(
        spy_short=_ret("SPY", _SHORT), rsp_short=_ret("RSP", _SHORT),
        qqq_short=_ret("QQQ", _SHORT), iwm_short=_ret("IWM", _SHORT),
        spy_long=_ret("SPY", _LONG),  rsp_long=_ret("RSP", _LONG),
        qqq_long=_ret("QQQ", _LONG),  iwm_long=_ret("IWM", _LONG),
        previous_regime=previous_regime,
        as_of=as_of,
    )


def compute_live(previous_regime: str = "factor") -> dict:
    """
    Fetch RSP/QQQ/IWM from yfinance and compute regime.
    Used in the live scheduler.
    """
    start = (date.today() - timedelta(days=_LONG * 2)).isoformat()
    data = yf.download(
        _REGIME_TICKERS, start=start, progress=False, auto_adjust=True
    )
    if data.empty:
        logger.warning("Regime data download failed — defaulting to %s", previous_regime)
        return _default(previous_regime)

    close = data["Close"] if "Close" in data.columns else data

    def _ret(sym: str, window: int) -> float:
        try:
            s = close[sym].dropna()
            if len(s) < window // 2:
                return 0.0
            return (float(s.iloc[-1]) / float(s.iloc[max(0, len(s) - window)]) - 1) * 100
        except (KeyError, IndexError):
            return 0.0

    return _classify(
        spy_short=_ret("SPY", _SHORT), rsp_short=_ret("RSP", _SHORT),
        qqq_short=_ret("QQQ", _SHORT), iwm_short=_ret("IWM", _SHORT),
        spy_long=_ret("SPY", _LONG),  rsp_long=_ret("RSP", _LONG),
        qqq_long=_ret("QQQ", _LONG),  iwm_long=_ret("IWM", _LONG),
        previous_regime=previous_regime,
        as_of=date.today(),
    )


def _classify(
    spy_short, rsp_short, qqq_short, iwm_short,
    spy_long, rsp_long, qqq_long, iwm_long,
    previous_regime: str,
    as_of,
) -> dict:
    """
    Two-component composite with hysteresis.

    Short signal (leading, 21d):
      Measures whether mega-cap dominance is *intensifying right now*.
      Reacts within weeks — fires early as a regime shift begins.

    Long signal (confirming, 63d):
      Anchors the signal — prevents a 3-week blip from triggering a full switch.

    Composite = 0.65 * short + 0.35 * long
      More weight on the leading component so the signal shifts ahead of
      a full regime establishment, not after.

    Hysteresis band [EXIT_CONCENTRATED, ENTER_CONCENTRATED]:
      Once in concentrated mode, stay until composite < EXIT (0.0).
      Once in factor mode, stay until composite > ENTER (4.0).
      Prevents monthly thrashing at the threshold.
    """
    # Breadth: SPY - RSP (positive = cap-weight winning = mega-cap dominant)
    breadth_short = spy_short - rsp_short
    breadth_long  = spy_long  - rsp_long

    # Tech: QQQ - IWM (positive = tech outperforming small-caps)
    tech_short = qqq_short - iwm_short
    tech_long  = qqq_long  - iwm_long

    short_signal = breadth_short + 0.4 * tech_short
    long_signal  = breadth_long  + 0.4 * tech_long

    composite = 0.65 * short_signal + 0.35 * long_signal

    # Hysteresis: only switch regime if composite crosses the appropriate threshold
    if previous_regime == "concentrated":
        regime = "concentrated" if composite > EXIT_CONCENTRATED else "factor"
    else:
        regime = "concentrated" if composite > ENTER_CONCENTRATED else "factor"

    logger.info(
        "Regime [%s]: short=%.1f (breadth=%.1f tech=%.1f) | "
        "long=%.1f (breadth=%.1f tech=%.1f) | "
        "composite=%.1f | prev=%s → %s",
        as_of,
        short_signal, breadth_short, tech_short,
        long_signal, breadth_long, tech_long,
        composite, previous_regime.upper(), regime.upper(),
    )

    return {
        "regime": regime,
        "composite_score": round(composite, 2),
        "short_signal": round(short_signal, 2),
        "long_signal": round(long_signal, 2),
        "breadth_spread_21d": round(breadth_short, 2),
        "breadth_spread_63d": round(breadth_long, 2),
        "tech_spread_21d": round(tech_short, 2),
        "tech_spread_63d": round(tech_long, 2),
    }


def _default(previous_regime: str = "factor") -> dict:
    return {
        "regime": previous_regime,
        "composite_score": 0.0,
        "short_signal": 0.0,
        "long_signal": 0.0,
        "breadth_spread_21d": 0.0,
        "breadth_spread_63d": 0.0,
        "tech_spread_21d": 0.0,
        "tech_spread_63d": 0.0,
    }
