from __future__ import annotations

import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

# AI/tech mega-cap concentrated universe — names that outperform in cap-weight regimes.
# Structured in tiers so the momentum score naturally picks the current leaders.
AI_UNIVERSE = [
    # AI infrastructure / semiconductors
    "NVDA", "AMD", "AVGO", "QCOM", "MRVL",
    # Hyperscalers / AI platforms
    "MSFT", "GOOGL", "META", "AMZN", "AAPL",
    # AI-native software
    "NOW", "ADBE", "CRM", "PANW", "SNPS", "CDNS",
    # High-momentum tech
    "TSLA", "ORCL", "INTU",
]

_LOOKBACK_SHORT = 21   # matches regime signal short window
_LOOKBACK_LONG = 126   # 6-month momentum — standard for cross-sectional momentum


def score(all_bars: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """
    Score AI universe names by momentum from pre-loaded bar data.
    Uses 21d (short) + 126d (long) momentum — no FMP required.
    Skips names with insufficient history.
    Returns DataFrame indexed by symbol, sorted by ai_score descending.
    """
    rows = []
    for sym in AI_UNIVERSE:
        try:
            s = all_bars.loc[sym]["close"]
            s = s[s.index <= pd.Timestamp(as_of)].dropna()
            if len(s) < _LOOKBACK_SHORT:
                continue

            mom_short = (
                float(s.iloc[-1]) / float(s.iloc[max(0, len(s) - _LOOKBACK_SHORT)]) - 1
            ) * 100

            mom_long = (
                float(s.iloc[-1]) / float(s.iloc[max(0, len(s) - _LOOKBACK_LONG)]) - 1
            ) * 100 if len(s) >= _LOOKBACK_LONG else mom_short

            # Weight short-term momentum heavier — consistent with leading regime signal
            ai_score = 0.60 * mom_short + 0.40 * mom_long

            rows.append({
                "symbol": sym,
                "mom_21d": round(mom_short, 2),
                "mom_126d": round(mom_long, 2),
                "ai_score": round(ai_score, 2),
            })
        except (KeyError, IndexError):
            continue

    if not rows:
        logger.warning("AI universe score: no valid bars found for any symbol")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("symbol").sort_values("ai_score", ascending=False)
    logger.info(
        "AI universe scored: %d/%d names | top=%s (score=%.1f)",
        len(df), len(AI_UNIVERSE),
        df.index[0] if not df.empty else "N/A",
        df["ai_score"].iloc[0] if not df.empty else 0,
    )
    return df


def score_live(alpaca, n_days: int = 200) -> pd.DataFrame:
    """
    Score AI universe using live Alpaca bar data.
    Used in the live scheduler.
    """
    try:
        bars = alpaca.get_bars_lookback(AI_UNIVERSE, days=n_days)
    except Exception as exc:
        logger.error("AI universe bar fetch failed: %s", exc)
        return pd.DataFrame()

    return score(bars, date.today())
