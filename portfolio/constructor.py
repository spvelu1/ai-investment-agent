from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


def build_sector_map(universe: list[str], fmp) -> dict[str, str]:
    """Fetch GICS sector for each symbol via FMP profile (24h cached)."""
    sector_map: dict[str, str] = {}
    for sym in universe:
        try:
            profile = fmp.get_profile(sym)
            sector = profile.get("sector", "")
            if sector:
                sector_map[sym] = sector
        except Exception:
            pass
    logger.info("Sector map built: %d / %d symbols have sector data", len(sector_map), len(universe))
    return sector_map


def construct(
    ranked_df: pd.DataFrame,
    current_positions: list[dict],
    macro_score: float,
    sector_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Build target portfolio as {symbol: target_weight}.

    Args:
        ranked_df:          Output from master_score.compute(), sorted by master_score desc.
        current_positions:  List of current Alpaca position dicts.
        macro_score:        MacroAlignmentScore (0–100). Below threshold → all cash.
        sector_map:         {symbol: sector} for sector cap enforcement.

    Returns:
        {symbol: weight} where weights sum to ~1.0, or {} (all cash).
    """
    cfg = get_settings()

    if macro_score < cfg.macro_derisking_threshold:
        logger.warning(
            "Macro score %.1f below threshold %.1f — moving to cash",
            macro_score,
            cfg.macro_derisking_threshold,
        )
        return {}

    sector_map = sector_map or {}
    eligible = ranked_df[ranked_df["passes_filters"] == True].copy()

    if eligible.empty:
        logger.warning("No stocks pass all hard filters — staying in cash")
        return {}

    # Continuity bonus: incumbent positions aren't replaced unless the new
    # candidate's score is clearly superior (not just marginally better).
    if current_positions:
        current_syms = {p["symbol"] for p in current_positions}
        incumbent_mask = eligible.index.isin(current_syms)
        eligible.loc[incumbent_mask, "master_score"] += cfg.continuity_bonus
        eligible = eligible.sort_values("master_score", ascending=False)

    # Enforce sector cap: remove lowest-ranked names in overweight sectors
    eligible = _enforce_sector_cap(eligible, sector_map, cfg.max_sector_pct, cfg.max_positions)

    n = min(len(eligible), cfg.max_positions)
    n = max(n, min(cfg.min_positions, len(eligible)))
    selected = eligible.head(n)

    symbols = selected.index.tolist()
    if not symbols:
        return {}

    # Opportunity-proportional sizing
    weights = _opportunity_weighted(selected, symbols, cfg)

    logger.info(
        "Portfolio constructed: %d positions | macro=%.1f",
        len(weights),
        macro_score,
    )
    for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
        logger.info("  %s  %.2f%%", sym, w * 100)

    return weights


def construct_concentrated(
    ai_scores_df: pd.DataFrame,
    current_positions: list[dict],
    cfg,
) -> dict[str, float]:
    """
    Build a concentrated AI/mega-cap portfolio weighted by momentum score.
    Used when regime detector signals mega-cap dominance.
    """
    if ai_scores_df is None or ai_scores_df.empty:
        logger.warning("AI scores empty — staying in cash")
        return {}

    selected = ai_scores_df.head(cfg.concentrated_n_positions)
    symbols = selected.index.tolist()

    scores = selected["ai_score"].clip(lower=0)
    total = scores.sum()
    w = (scores / total).to_dict() if total > 0 else {s: 1.0 / len(symbols) for s in symbols}

    w = _apply_position_cap(w, cfg.concentrated_max_position_pct)

    logger.info(
        "CONCENTRATED portfolio: %d AI positions | top=%s (%.1f%%)",
        len(w),
        symbols[0] if symbols else "N/A",
        w.get(symbols[0], 0) * 100 if symbols else 0,
    )
    for sym, wt in sorted(w.items(), key=lambda x: -x[1]):
        logger.info("  %s  %.2f%%", sym, wt * 100)
    return w


def _opportunity_weighted(df: pd.DataFrame, symbols: list[str], cfg) -> dict[str, float]:
    """
    Weight by (excess score above the weakest pick)^1.5.

    Raw master_scores are often bunched (e.g. 72–88), making simple
    proportional weighting near-equal.  Shifting by the minimum and raising
    to the 1.5 power amplifies the gap: the top-scoring stock gets
    meaningfully more capital than the marginal 15th pick, without needing
    an arbitrary rank scheme.
    """
    scores = df.loc[symbols, "master_score"] if "master_score" in df.columns else pd.Series(dtype=float)

    if scores.isna().all() or scores.empty:
        w = {sym: 1.0 / len(symbols) for sym in symbols}
    else:
        scores = scores.reindex(symbols).fillna(scores.median())
        spread = scores.max() - scores.min()
        # Floor = 20% of score spread so lowest-ranked stock still gets meaningful weight
        floor = spread * 0.20 if spread > 0 else 1.0
        excess = (scores - scores.min() + floor).clip(lower=floor) ** 1.5
        total = excess.sum()
        if total <= 0:
            w = {sym: 1.0 / len(symbols) for sym in symbols}
        else:
            w = (excess / total).to_dict()

    w = _apply_position_cap(w, cfg.max_position_pct)
    return w


def _apply_position_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Iteratively redistribute weight above the cap to uncapped positions."""
    w = dict(weights)
    max_iter = 20
    for _ in range(max_iter):
        overflow = {sym: wt - cap for sym, wt in w.items() if wt > cap}
        if not overflow:
            break
        total_overflow = sum(overflow.values())
        uncapped = [sym for sym, wt in w.items() if wt <= cap]
        for sym in overflow:
            w[sym] = cap
        if uncapped:
            extra_each = total_overflow / len(uncapped)
            for sym in uncapped:
                w[sym] += extra_each
    # Normalize
    total = sum(w.values())
    return {sym: wt / total for sym, wt in w.items()}


def _enforce_sector_cap(
    df: pd.DataFrame,
    sector_map: dict[str, str],
    max_sector_pct: float,
    max_positions: int,
) -> pd.DataFrame:
    """
    Drop lowest-ranked names in any sector that would exceed max_sector_pct
    if all candidates are included equally weighted.
    """
    if not sector_map:
        return df

    # Work through symbols ranked best to worst
    selected: list[str] = []
    sector_counts: dict[str, int] = {}
    max_per_sector = max(1, int(max_sector_pct * max_positions))

    for sym in df.index:
        sector = sector_map.get(sym, "Unknown")
        count = sector_counts.get(sector, 0)
        if count >= max_per_sector:
            logger.debug("Sector cap: dropping %s (sector=%s, count=%d)", sym, sector, count)
            continue
        selected.append(sym)
        sector_counts[sector] = count + 1

    return df.loc[selected]
