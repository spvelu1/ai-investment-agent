from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import get_settings
from data.clients.fmp_client import FMPClient

logger = logging.getLogger(__name__)


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100.0


def compute(universe: list[str], fmp: FMPClient | None = None) -> pd.DataFrame:
    """
    Compute EarningsRevisionScore for each symbol in universe.

    Returns DataFrame indexed by symbol with columns:
        eps_rev_30, eps_rev_breadth, rev_rev_30, earnings_revision_score
    """
    cfg = get_settings()
    fmp = fmp or FMPClient()
    rows: list[dict] = []

    for symbol in universe:
        try:
            estimates = fmp.get_analyst_estimates(symbol, limit=8)
            row = _compute_symbol(symbol, estimates)
            rows.append(row)
        except Exception as exc:
            logger.debug("Earnings revision skipped for %s: %s", symbol, exc)
            rows.append({"symbol": symbol, "eps_rev_30": np.nan, "eps_rev_breadth": np.nan, "rev_rev_30": np.nan})

    df = pd.DataFrame(rows).set_index("symbol")

    def _fill(s: pd.Series) -> pd.Series:
        m = s.median()
        return s.fillna(m if pd.notna(m) else 0.0)

    df["eps_rev_30_pct"] = _pct_rank(_fill(df["eps_rev_30"]))
    df["eps_rev_breadth_pct"] = _pct_rank(_fill(df["eps_rev_breadth"]))
    df["rev_rev_30_pct"] = _pct_rank(_fill(df["rev_rev_30"]))

    df["earnings_revision_score"] = (
        cfg.w_eps_rev_30 * df["eps_rev_30_pct"]
        + cfg.w_eps_rev_breadth * df["eps_rev_breadth_pct"]
        + cfg.w_rev_rev_30 * df["rev_rev_30_pct"]
    )

    return df


def _compute_symbol(symbol: str, estimates: list[dict]) -> dict:
    """Parse FMP analyst-estimates into revision metrics."""
    if not estimates or len(estimates) < 2:
        return {"symbol": symbol, "eps_rev_30": np.nan, "eps_rev_breadth": np.nan, "rev_rev_30": np.nan}

    # Sort descending by date (most recent first)
    estimates = sorted(estimates, key=lambda x: x.get("date", ""), reverse=True)
    latest = estimates[0]
    prior = estimates[1] if len(estimates) > 1 else {}

    # EPS_REV_30: change in FY1 EPS consensus over ~30 days (two quarters apart)
    eps_now = _safe_float(latest.get("estimatedEpsAvg"))
    eps_prior = _safe_float(prior.get("estimatedEpsAvg"))
    eps_rev_30 = _pct_change(eps_now, eps_prior)

    # Revenue revision
    rev_now = _safe_float(latest.get("estimatedRevenueAvg"))
    rev_prior = _safe_float(prior.get("estimatedRevenueAvg"))
    rev_rev_30 = _pct_change(rev_now, rev_prior)

    # EPS_REV_BREADTH: using high/low spread as proxy for analyst disagreement/direction
    eps_high = _safe_float(latest.get("estimatedEpsHigh"))
    eps_low = _safe_float(latest.get("estimatedEpsLow"))
    eps_prior_high = _safe_float(prior.get("estimatedEpsHigh"))
    eps_prior_low = _safe_float(prior.get("estimatedEpsLow"))

    eps_rev_breadth = np.nan
    if all(v is not None for v in [eps_high, eps_low, eps_prior_high, eps_prior_low]):
        # Positive breadth: both high and low moved up
        high_moved_up = 1 if eps_high > eps_prior_high else -1
        low_moved_up = 1 if eps_low > eps_prior_low else -1
        eps_rev_breadth = (high_moved_up + low_moved_up) / 2.0  # -1, 0, or +1

    return {
        "symbol": symbol,
        "eps_rev_30": eps_rev_30,
        "eps_rev_breadth": eps_rev_breadth,
        "rev_rev_30": rev_rev_30,
    }


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _pct_change(now: float | None, prior: float | None) -> float:
    if now is None or prior is None or prior == 0:
        return np.nan
    return (now - prior) / abs(prior)
