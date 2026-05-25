from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_MACRO_TICKERS = ["SPY", "QQQ", "HYG", "IEF", "^VIX"]


class MacroClient:
    def get_macro_data(self, lookback_days: int = 252) -> dict[str, pd.DataFrame]:
        """
        Download daily close prices for macro instruments.
        Returns {ticker: DataFrame(date, close)} for each instrument.
        """
        start = date.today() - timedelta(days=lookback_days + 30)
        raw = yf.download(
            _MACRO_TICKERS,
            start=start.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            logger.warning("yfinance returned empty macro data")
            return {}

        close = raw["Close"] if "Close" in raw.columns else raw
        result: dict[str, pd.DataFrame] = {}
        for ticker in _MACRO_TICKERS:
            col = ticker
            if col not in close.columns:
                logger.warning("Macro ticker %s not found in yfinance response", ticker)
                continue
            df = close[[col]].rename(columns={col: "close"}).dropna()
            df.index = pd.to_datetime(df.index).normalize()
            result[ticker] = df

        return result

    def compute_macro_regime(self, macro_data: dict[str, pd.DataFrame], vix_threshold: float = 25.0) -> dict:
        """
        Compute binary regime flags from macro data.
        Returns dict with individual flags and composite MacroAlignmentScore (0–100).
        """
        def last_close(ticker: str) -> float | None:
            df = macro_data.get(ticker)
            return float(df["close"].iloc[-1]) if df is not None and not df.empty else None

        def ma(ticker: str, window: int) -> float | None:
            df = macro_data.get(ticker)
            if df is None or len(df) < window:
                return None
            return float(df["close"].rolling(window).mean().iloc[-1])

        spy_close = last_close("SPY")
        spy_ma200 = ma("SPY", 200)
        qqq_close = last_close("QQQ")
        qqq_ma200 = ma("QQQ", 200)
        hyg_close = last_close("HYG")
        ief_close = last_close("IEF")
        vix_close = last_close("^VIX")

        spy_trend = 100.0 if (spy_close and spy_ma200 and spy_close > spy_ma200) else 0.0
        qqq_trend = 100.0 if (qqq_close and qqq_ma200 and qqq_close > qqq_ma200) else 0.0
        avg_equity_trend = (spy_trend + qqq_trend) / 2.0

        # Credit: HYG/IEF ratio > 20d moving average
        credit_trend = 0.0
        if hyg_close and ief_close:
            hyg_df = macro_data.get("HYG")
            ief_df = macro_data.get("IEF")
            if hyg_df is not None and ief_df is not None and len(hyg_df) >= 20:
                ratio_now = hyg_close / ief_close
                hyg_roll = hyg_df["close"].iloc[-20:]
                ief_roll = ief_df["close"].iloc[-20:]
                ratio_ma20 = (hyg_roll.values / ief_roll.values).mean()
                credit_trend = 100.0 if ratio_now > ratio_ma20 else 0.0

        low_vol = 100.0 if (vix_close is not None and vix_close < vix_threshold) else 0.0

        macro_score = (
            0.40 * avg_equity_trend
            + 0.30 * credit_trend
            + 0.30 * low_vol
        )

        return {
            "spy_trend": spy_trend,
            "qqq_trend": qqq_trend,
            "credit_trend": credit_trend,
            "low_vol_regime": low_vol,
            "macro_alignment_score": round(macro_score, 2),
            "spy_close": spy_close,
            "spy_ma200": spy_ma200,
            "vix": vix_close,
        }
