from __future__ import annotations

import logging

from config.settings import get_settings
from data.clients.macro_client import MacroClient

logger = logging.getLogger(__name__)


def compute(macro_client: MacroClient | None = None) -> dict:
    """
    Compute MacroAlignmentScore and individual regime flags.

    Returns dict with:
        macro_alignment_score  float 0–100
        spy_trend              100 or 0
        qqq_trend              100 or 0
        credit_trend           100 or 0
        low_vol_regime         100 or 0
        vix                    float
        spy_close              float
        spy_ma200              float
    """
    cfg = get_settings()
    client = macro_client or MacroClient()
    macro_data = client.get_macro_data()
    result = client.compute_macro_regime(macro_data, vix_threshold=cfg.vix_threshold)
    logger.info(
        "MacroAlignmentScore=%.1f | SPY_trend=%.0f | VIX=%.1f",
        result["macro_alignment_score"],
        result["spy_trend"],
        result.get("vix") or 0,
    )
    return result
