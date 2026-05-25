from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from config.settings import get_settings
from data.clients.alpaca_client import AlpacaClient
from data.clients.fmp_client import FMPClient
from data.clients.macro_client import MacroClient
from data import store
from signals import earnings_revision, momentum, breakout, quality, macro_overlay

logger = logging.getLogger(__name__)


def compute(
    universe: list[str],
    alpaca: AlpacaClient | None = None,
    fmp: FMPClient | None = None,
    macro_client: MacroClient | None = None,
    cached_ers_df: pd.DataFrame | None = None,
    cached_qlt_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute MASTER_SCORE for every symbol in universe.

    Returns DataFrame indexed by symbol, sorted by master_score desc, with columns:
        earnings_revision_score, momentum_score, breakout_score,
        quality_score, macro_alignment_score, master_score,
        passes_filters, close, ma50, ma200, ma10, vol_20d,
        above_50dma, above_200dma, adv_20d
    """
    cfg = get_settings()
    alpaca = alpaca or AlpacaClient()
    fmp = fmp or FMPClient()
    macro_client = macro_client or MacroClient()

    logger.info("Computing MASTER_SCORE for %d symbols", len(universe))

    # --- Macro (scalar — same for all stocks) ---
    macro = macro_overlay.compute(macro_client)
    macro_score = macro["macro_alignment_score"]

    # --- Price bars (single pull for momentum + breakout + ADV) ---
    logger.info("Fetching price bars...")
    bars = _fetch_bars(universe, alpaca)
    spy_bars = _fetch_spy_bars(alpaca)

    # --- Signal modules ---
    if cached_ers_df is not None:
        ers_df = cached_ers_df
    else:
        logger.info("Computing earnings revision scores...")
        ers_df = earnings_revision.compute(universe, fmp)

    logger.info("Computing momentum scores...")
    mom_df = momentum.compute(bars, spy_bars)

    logger.info("Computing breakout scores...")
    brk_df = breakout.compute(bars)

    if cached_qlt_df is not None:
        qlt_df = cached_qlt_df
    else:
        logger.info("Computing quality scores...")
        qlt_df = quality.compute(universe, fmp)

    # --- Merge all signals ---
    df = _merge_signals(universe, ers_df, mom_df, brk_df, qlt_df, macro_score, bars, cfg)

    # --- MASTER_SCORE ---
    df["master_score"] = (
        cfg.w_earnings_revision * df["earnings_revision_score"]
        + cfg.w_momentum * df["momentum_score"]
        + cfg.w_breakout * df["breakout_score"]
        + cfg.w_quality * df["quality_score"]
        + cfg.w_macro * df["macro_alignment_score"]
    )

    # --- Hard filters ---
    master_p85 = df["master_score"].quantile(0.85)
    ers_p70 = df["earnings_revision_score"].quantile(0.70)
    mom_p70 = df["momentum_score"].quantile(0.70)

    df["passes_filters"] = (
        (df["master_score"] >= master_p85)
        & (df["earnings_revision_score"] >= ers_p70)
        & (df["momentum_score"] >= mom_p70)
        & (df["above_50dma"].fillna(False))
        & (df["above_200dma"].fillna(False))
        & (df["adv_20d"] >= cfg.min_adv)
    )

    df = df.sort_values("master_score", ascending=False)
    logger.info(
        "MASTER_SCORE done. %d pass all filters. Macro=%.1f",
        df["passes_filters"].sum(),
        macro_score,
    )
    return df


def _fetch_bars(universe: list[str], alpaca: AlpacaClient) -> pd.DataFrame:
    from datetime import date, timedelta
    start = date.today() - timedelta(days=300)
    try:
        return alpaca.get_bars(universe, start=start)
    except Exception as exc:
        logger.error("Failed to fetch bars: %s", exc)
        return pd.DataFrame()


def _fetch_spy_bars(alpaca: AlpacaClient) -> pd.DataFrame:
    from datetime import date, timedelta
    start = date.today() - timedelta(days=300)
    try:
        raw = alpaca.get_bars(["SPY"], start=start)
        return raw.loc["SPY"] if "SPY" in raw.index.get_level_values(0) else pd.DataFrame()
    except Exception as exc:
        logger.warning("Failed to fetch SPY bars: %s", exc)
        return pd.DataFrame()


def _merge_signals(
    universe: list[str],
    ers_df: pd.DataFrame,
    mom_df: pd.DataFrame,
    brk_df: pd.DataFrame,
    qlt_df: pd.DataFrame,
    macro_score: float,
    bars: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    base = pd.DataFrame(index=pd.Index(universe, name="symbol"))

    def _merge(df, cols):
        available = [c for c in cols if c in df.columns]
        if available:
            base[available] = df[available].reindex(base.index)

    _merge(ers_df, ["earnings_revision_score"])
    _merge(mom_df, ["momentum_score", "close", "ma50", "ma200", "ma10", "vol_20d"])
    _merge(brk_df, ["breakout_score", "above_50dma", "above_200dma"])
    _merge(qlt_df, ["quality_score"])

    base["macro_alignment_score"] = macro_score

    # Fill missing scores with universe median; fall back to 50.0 when entire column is NaN
    for col in ["earnings_revision_score", "momentum_score", "breakout_score", "quality_score"]:
        if col in base.columns:
            median_val = base[col].median()
            base[col] = base[col].fillna(median_val if pd.notna(median_val) else 50.0)
        else:
            base[col] = 50.0

    # ADV: compute from bars
    adv_map = _compute_adv(bars)
    base["adv_20d"] = base.index.map(adv_map).astype(float)

    return base


def _compute_adv(bars: pd.DataFrame) -> dict[str, float]:
    if bars.empty:
        return {}
    adv: dict[str, float] = {}
    for sym in bars.index.get_level_values("symbol").unique():
        try:
            s = bars.loc[sym].tail(20)
            adv[sym] = float((s["close"] * s["volume"]).mean())
        except Exception:
            adv[sym] = 0.0
    return adv
