from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import get_settings
from data.clients.fmp_client import FMPClient

logger = logging.getLogger(__name__)


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100.0


def compute(universe: list[str], fmp: FMPClient | None = None) -> pd.DataFrame:
    """
    Compute QualityScore for each symbol in universe.

    Returns DataFrame indexed by symbol with columns:
        roic, rev_growth, fcf_margin, gm_stability, quality_score
    """
    cfg = get_settings()
    fmp = fmp or FMPClient()
    rows: list[dict] = []

    for symbol in universe:
        try:
            key_metrics = fmp.get_key_metrics(symbol, period="annual", limit=4)
            income = fmp.get_income_statement(symbol, period="quarter", limit=12)
            cash_flow = fmp.get_cash_flow(symbol, period="annual", limit=4)
            income_annual = fmp.get_income_statement(symbol, period="annual", limit=4)
            row = _compute_symbol(symbol, key_metrics, income, cash_flow, income_annual)
            rows.append(row)
        except Exception as exc:
            logger.debug("Quality skipped for %s: %s", symbol, exc)
            rows.append({"symbol": symbol, "roic": np.nan, "rev_growth": np.nan, "fcf_margin": np.nan, "gm_stability": np.nan})

    df = pd.DataFrame(rows).set_index("symbol")

    def _fill(s: pd.Series) -> pd.Series:
        m = s.median()
        return s.fillna(m if pd.notna(m) else 0.0)

    df["roic_pct"] = _pct_rank(_fill(df["roic"]))
    df["rev_growth_pct"] = _pct_rank(_fill(df["rev_growth"]))
    df["fcf_margin_pct"] = _pct_rank(_fill(df["fcf_margin"]))
    df["gm_stability_pct"] = _pct_rank(_fill(df["gm_stability"]))

    df["quality_score"] = (
        cfg.w_roic * df["roic_pct"]
        + cfg.w_rev_growth * df["rev_growth_pct"]
        + cfg.w_fcf_margin * df["fcf_margin_pct"]
        + cfg.w_gm_stability * df["gm_stability_pct"]
    )

    return df


def _compute_symbol(
    symbol: str,
    key_metrics: list[dict],
    income_quarterly: list[dict],
    cash_flow: list[dict],
    income_annual: list[dict],
) -> dict:
    # ROIC from FMP key_metrics
    roic = np.nan
    if key_metrics:
        roic = _safe_float(key_metrics[0].get("roic"))

    # Revenue YoY growth (annual)
    rev_growth = np.nan
    if len(income_annual) >= 2:
        rev_now = _safe_float(income_annual[0].get("revenue"))
        rev_prior = _safe_float(income_annual[1].get("revenue"))
        if rev_now and rev_prior and rev_prior != 0:
            rev_growth = (rev_now - rev_prior) / abs(rev_prior)

    # FCF margin (annual): FCF / Revenue
    fcf_margin = np.nan
    if cash_flow and income_annual:
        fcf = _safe_float(cash_flow[0].get("freeCashFlow"))
        rev = _safe_float(income_annual[0].get("revenue"))
        if fcf is not None and rev and rev != 0:
            fcf_margin = fcf / rev

    # GM_STABILITY: 1 / stddev of gross margin over last 12 quarters
    gm_stability = np.nan
    if len(income_quarterly) >= 4:
        gm_list = []
        for q in income_quarterly:
            rev = _safe_float(q.get("revenue"))
            gp = _safe_float(q.get("grossProfit"))
            if rev and rev != 0 and gp is not None:
                gm_list.append(gp / rev)
        if len(gm_list) >= 4:
            std = float(np.std(gm_list))
            gm_stability = 1.0 / std if std > 0 else 100.0

    return {
        "symbol": symbol,
        "roic": roic,
        "rev_growth": rev_growth,
        "fcf_margin": fcf_margin,
        "gm_stability": gm_stability,
    }


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None
