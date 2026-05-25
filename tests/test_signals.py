"""Tests for signal computation modules — no live API calls."""
import numpy as np
import pandas as pd
import pytest

from signals import momentum, breakout, earnings_revision, quality, master_score


def _make_bars(symbols: list[str], n: int = 260) -> pd.DataFrame:
    """Synthetic MultiIndex (symbol, date) bar DataFrame."""
    import numpy as np
    from datetime import date, timedelta

    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    rows = []
    for sym in symbols:
        np.random.seed(hash(sym) % (2**31))
        price = 100.0
        for d in dates:
            price *= 1 + np.random.normal(0.0003, 0.015)
            rows.append({
                "symbol": sym,
                "date": d,
                "open": price * 0.998,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "volume": np.random.uniform(5e6, 20e6),
            })
    df = pd.DataFrame(rows).set_index(["symbol", "date"])
    return df


class TestMomentum:
    def test_returns_correct_columns(self):
        bars = _make_bars(["AAPL", "MSFT", "NVDA"])
        result = momentum.compute(bars)
        assert "momentum_score" in result.columns
        assert "mom_6" in result.columns
        assert "mom_3" in result.columns
        assert "rs_63d" in result.columns
        assert set(result.index) == {"AAPL", "MSFT", "NVDA"}

    def test_scores_between_0_and_100(self):
        bars = _make_bars(["AAPL", "MSFT", "NVDA", "AMZN", "GOOG"])
        result = momentum.compute(bars)
        scores = result["momentum_score"].dropna()
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_handles_short_history(self):
        bars = _make_bars(["AAPL"], n=30)
        result = momentum.compute(bars)
        # Should still return a row even with short history
        assert "AAPL" in result.index

    def test_moving_averages_computed(self):
        bars = _make_bars(["AAPL"])
        result = momentum.compute(bars)
        assert not pd.isna(result.loc["AAPL", "ma50"])
        assert not pd.isna(result.loc["AAPL", "ma200"])


class TestBreakout:
    def test_returns_correct_columns(self):
        bars = _make_bars(["AAPL", "MSFT"])
        result = breakout.compute(bars)
        assert "breakout_score" in result.columns
        assert "breakout_20" in result.columns
        assert "vol_expansion" in result.columns
        assert "above_50dma" in result.columns
        assert "above_200dma" in result.columns

    def test_breakout_binary_is_0_or_1(self):
        bars = _make_bars(["AAPL", "MSFT", "NVDA"])
        result = breakout.compute(bars)
        assert result["breakout_20"].isin([0.0, 1.0]).all()

    def test_scores_between_0_and_100(self):
        bars = _make_bars(["AAPL", "MSFT", "NVDA", "AMZN"])
        result = breakout.compute(bars)
        scores = result["breakout_score"].dropna()
        assert (scores >= 0).all() and (scores <= 100).all()


class TestEarningsRevision:
    def _make_estimates(self, eps_now: float, eps_prior: float, rev_now: float, rev_prior: float) -> list[dict]:
        return [
            {
                "date": "2024-03-31",
                "estimatedEpsAvg": eps_now,
                "estimatedEpsHigh": eps_now * 1.05,
                "estimatedEpsLow": eps_now * 0.95,
                "estimatedRevenueAvg": rev_now,
            },
            {
                "date": "2023-12-31",
                "estimatedEpsAvg": eps_prior,
                "estimatedEpsHigh": eps_prior * 1.03,
                "estimatedEpsLow": eps_prior * 0.97,
                "estimatedRevenueAvg": rev_prior,
            },
        ]

    def test_positive_revision(self):
        estimates = self._make_estimates(eps_now=2.5, eps_prior=2.0, rev_now=10e9, rev_prior=9e9)
        row = earnings_revision._compute_symbol("AAPL", estimates)
        assert row["eps_rev_30"] > 0
        assert row["rev_rev_30"] > 0

    def test_negative_revision(self):
        estimates = self._make_estimates(eps_now=1.5, eps_prior=2.0, rev_now=8e9, rev_prior=9e9)
        row = earnings_revision._compute_symbol("AAPL", estimates)
        assert row["eps_rev_30"] < 0

    def test_empty_estimates_returns_nan(self):
        row = earnings_revision._compute_symbol("AAPL", [])
        assert np.isnan(row["eps_rev_30"])

    def test_compute_full_universe(self, monkeypatch):
        """Integration-style: patch FMP, verify score computation runs end-to-end."""
        from unittest.mock import MagicMock
        fmp_mock = MagicMock()
        fmp_mock.get_analyst_estimates.return_value = self._make_estimates(2.5, 2.0, 10e9, 9e9)

        result = earnings_revision.compute(["AAPL", "MSFT"], fmp=fmp_mock)
        assert "earnings_revision_score" in result.columns
        assert len(result) == 2
        scores = result["earnings_revision_score"].dropna()
        assert (scores >= 0).all() and (scores <= 100).all()


class TestPortfolioConstructor:
    def _make_ranked_df(self, symbols: list[str]) -> pd.DataFrame:
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame(index=pd.Index(symbols, name="symbol"))
        df["master_score"] = np.random.uniform(50, 100, len(symbols))
        df["earnings_revision_score"] = np.random.uniform(50, 100, len(symbols))
        df["momentum_score"] = np.random.uniform(50, 100, len(symbols))
        df["breakout_score"] = np.random.uniform(50, 100, len(symbols))
        df["quality_score"] = np.random.uniform(50, 100, len(symbols))
        df["passes_filters"] = True
        df["vol_20d"] = np.random.uniform(0.01, 0.04, len(symbols))
        df["above_50dma"] = True
        df["above_200dma"] = True
        df["adv_20d"] = 50e6
        return df.sort_values("master_score", ascending=False)

    def test_weights_sum_to_one(self):
        from portfolio.constructor import construct
        df = self._make_ranked_df(["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA", "JPM", "V", "UNH"])
        weights = construct(df, current_positions=[], macro_score=80.0)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_no_position_exceeds_cap(self):
        from portfolio.constructor import construct
        df = self._make_ranked_df([f"STOCK{i}" for i in range(20)])
        weights = construct(df, current_positions=[], macro_score=80.0)
        for sym, w in weights.items():
            assert w <= 0.15 + 1e-9, f"{sym} weight {w:.4f} exceeds 15%"

    def test_macro_derisking_returns_empty(self):
        from portfolio.constructor import construct
        df = self._make_ranked_df(["AAPL", "MSFT", "NVDA"])
        weights = construct(df, current_positions=[], macro_score=20.0)
        assert weights == {}

    def test_position_count_in_range(self):
        from portfolio.constructor import construct
        df = self._make_ranked_df([f"STOCK{i}" for i in range(25)])
        weights = construct(df, current_positions=[], macro_score=80.0)
        assert 10 <= len(weights) <= 15
