from __future__ import annotations

import logging
import math
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

import data.store as _store
from backtest import metrics as bt_metrics
from backtest.historical_clients import HistoricalAlpacaClient, HistoricalMacroClient
from backtest.paper_portfolio import PaperPortfolio
from config.settings import get_settings
from data.clients.alpaca_client import AlpacaClient
from data.clients.fmp_client import FMPClient
from data.clients.macro_client import _MACRO_TICKERS
from portfolio.constructor import construct, construct_concentrated, build_sector_map
from portfolio.risk_manager import evaluate as evaluate_risk
from signals import master_score, earnings_revision as _ers_signal, quality as _qlt_signal
from signals.regime import compute_from_bars as compute_regime
from universe.screener import _FALLBACK_UNIVERSE
from universe.ai_universe import AI_UNIVERSE, score as ai_score

logger = logging.getLogger(__name__)


@contextmanager
def _no_store_writes():
    """Replace store side-effects with no-ops for the duration of the block."""
    orig_log = _store.log_order
    orig_50dma = _store.update_below_50dma_days
    _store.log_order = lambda *a, **kw: None
    _store.update_below_50dma_days = lambda *a, **kw: None
    try:
        yield
    finally:
        _store.log_order = orig_log
        _store.update_below_50dma_days = orig_50dma


class BacktestEngine:
    """
    Walk-forward backtest engine.

    Parameters
    ----------
    rebalance_freq : "weekly" | "biweekly" | "monthly"
        How often to run the full signal→construct→execute cycle.
        Weekly = every Friday (default, most churn).
        Monthly = first Friday of each calendar month (least churn).

    below_50dma_days : int | None
        Override the production setting for consecutive days below 50-DMA
        before an exit is triggered. None = use settings default (2).
        Set to 10 to require two full weeks; set to 999 to disable the rule.

    min_hold_days : int
        Minimum calendar days to hold a position before any sell rule can
        exit it (stop-loss always overrides this floor).

    disable_time_stop : bool
        Suppress the max_holding_days time-stop sell rule. Good for
        trend-following mode where you want to let winners run.
    """

    def __init__(
        self,
        start_date: date,
        end_date: date,
        initial_cash: float = 100_000.0,
        slippage_pct: float = 0.001,
        rebalance_freq: str = "weekly",
        below_50dma_days: int | None = None,
        min_hold_days: int = 0,
        disable_time_stop: bool = False,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.rebalance_freq = rebalance_freq
        self.below_50dma_days = below_50dma_days
        self.min_hold_days = min_hold_days
        self.disable_time_stop = disable_time_stop

    def run(self) -> dict:
        logger.info(
            "=== BACKTEST %s → %s | cash=$%.0f | slippage=%.1f%% | "
            "freq=%s | 50dma_days=%s | min_hold=%d | no_time_stop=%s ===",
            self.start_date, self.end_date, self.initial_cash,
            self.slippage_pct * 100, self.rebalance_freq,
            self.below_50dma_days if self.below_50dma_days is not None else "default",
            self.min_hold_days, self.disable_time_stop,
        )

        universe = list(_FALLBACK_UNIVERSE)

        # ── 1. Pre-load all bar data ──────────────────────────────────────────
        lookback_start = self.start_date - timedelta(days=400)
        logger.info(
            "Downloading bars for %d symbols + SPY (%s → %s)...",
            len(universe), lookback_start, self.end_date,
        )
        # AI names in the fallback universe already; RSP/QQQ/IWM needed for regime detection
        _REGIME_EXTRAS = ["RSP", "QQQ", "IWM"]
        alpaca = AlpacaClient()
        all_bars = alpaca.get_bars(
            universe + ["SPY"] + _REGIME_EXTRAS, start=lookback_start, end=self.end_date
        )
        logger.info("Downloaded %d bar rows", len(all_bars))

        # ── 2. Pre-load macro data (yfinance) ─────────────────────────────────
        logger.info("Downloading macro data (%s → %s)...", lookback_start, self.end_date)
        macro_data_full = _load_macro_data(lookback_start, self.end_date)

        # ── 3. FMP signals — compute once (frozen at current date, reused every rebalance) ──
        logger.warning(
            "FMP signals (earnings revision, quality) use current fundamental data "
            "for all backtest dates — price-based signals are fully historical."
        )
        fmp = FMPClient()
        logger.info("Pre-computing FMP signals once for all rebalance dates...")
        cached_ers_df = _ers_signal.compute(universe, fmp)
        cached_qlt_df = _qlt_signal.compute(universe, fmp)
        cached_sector_map = build_sector_map(universe, fmp)
        logger.info("FMP signals + sector map cached — skipping on all future rebalances.")

        # ── 4. Portfolio + historical client setup ────────────────────────────
        portfolio = PaperPortfolio(
            initial_cash=self.initial_cash,
            slippage_pct=self.slippage_pct,
        )
        hist_alpaca = HistoricalAlpacaClient(all_bars, portfolio)
        hist_macro = HistoricalMacroClient(macro_data_full)

        # ── 5. Walk forward ───────────────────────────────────────────────────
        trading_dates = _trading_dates(all_bars, self.start_date, self.end_date)
        logger.info(
            "Walking %d trading days (%s → %s)",
            len(trading_dates), trading_dates[0], trading_dates[-1],
        )

        last_signals_df: pd.DataFrame | None = None
        rebalance_dates: list[date] = []
        positions_at_rebalance: list[int] = []
        sell_reason_counts: dict[str, int] = defaultdict(int)
        current_regime: str = "factor"   # regime state carried across rebalances
        regime_log: list[tuple] = []

        for i, d in enumerate(trading_dates):
            prices = _close_prices(all_bars, universe, d)
            hist_alpaca.set_date(d, prices)
            hist_macro.set_date(d)

            # Daily sell-rule check (skip day 0 — no positions yet)
            if last_signals_df is not None:
                triggered = self._daily_check(
                    d, portfolio, all_bars, last_signals_df, prices,
                    regime=current_regime,
                )
                for reason in triggered:
                    sell_reason_counts[reason] += 1

            # Rebalance gate
            if _should_rebalance(d, i, trading_dates, self.rebalance_freq):
                regime_data = compute_regime(all_bars, d, previous_regime=current_regime)
                current_regime = regime_data["regime"]
                regime_log.append((d, current_regime, regime_data["composite_score"]))

                signals = self._rebalance(
                    d, universe, portfolio, hist_alpaca, fmp, hist_macro, prices,
                    cached_ers_df=cached_ers_df, cached_qlt_df=cached_qlt_df,
                    cached_sector_map=cached_sector_map,
                    regime=current_regime,
                    all_bars=all_bars,
                )
                if signals is not None:
                    last_signals_df = signals
                    rebalance_dates.append(d)
                    positions_at_rebalance.append(len(portfolio.positions))

            portfolio.snapshot(d, prices)

            if (i + 1) % 50 == 0:
                nav = portfolio.nav(prices)
                logger.info(
                    "Progress: %d/%d | date=%s | NAV=$%.0f | positions=%d",
                    i + 1, len(trading_dates), d, nav, len(portfolio.positions),
                )

        # ── 6. Metrics ────────────────────────────────────────────────────────
        spy_df = _spy_slice(all_bars, self.start_date, self.end_date)
        result = bt_metrics.compute(portfolio.nav_history, spy_df, portfolio.trade_log)
        result["nav_history"] = portfolio.nav_history
        result["trade_log"] = portfolio.trade_log
        result["sell_reason_counts"] = dict(sell_reason_counts)
        result["rebalance_count"] = len(rebalance_dates)
        result["avg_positions_post_rebalance"] = (
            round(sum(positions_at_rebalance) / len(positions_at_rebalance), 1)
            if positions_at_rebalance else 0
        )
        n_concentrated = sum(1 for _, r, _ in regime_log if r == "concentrated")
        n_factor = len(regime_log) - n_concentrated
        result["regime_log"] = [(str(d), r, s) for d, r, s in regime_log]
        logger.info(
            "Regime summary: %d concentrated rebalances / %d factor rebalances",
            n_concentrated, n_factor,
        )
        return result

    # ── private helpers ───────────────────────────────────────────────────────

    def _daily_check(
        self,
        d: date,
        portfolio: PaperPortfolio,
        all_bars: pd.DataFrame,
        signals_df: pd.DataFrame,
        prices: dict[str, float],
        regime: str = "factor",
    ) -> list[str]:
        """Run sell rules. Returns list of triggered sell reasons."""
        positions = portfolio.get_positions_for_risk_manager()
        if not positions:
            return []

        cfg = get_settings()
        orig_50dma_threshold = cfg.below_50dma_days
        orig_time_stop = cfg.max_holding_days
        orig_stop_loss = cfg.stop_loss_pct
        if self.below_50dma_days is not None:
            cfg.below_50dma_days = self.below_50dma_days
        if self.disable_time_stop:
            cfg.max_holding_days = 9999
        if regime == "concentrated":
            cfg.stop_loss_pct = cfg.concentrated_stop_loss_pct

        try:
            symbols = [p["symbol"] for p in positions]
            bars = _slice_bars(all_bars, symbols, d, lookback_days=260)
            with _no_store_writes():
                sell_signals = evaluate_risk(positions, bars, signals_df, as_of=d)
        finally:
            cfg.below_50dma_days = orig_50dma_threshold
            cfg.max_holding_days = orig_time_stop
            cfg.stop_loss_pct = orig_stop_loss

        triggered: list[str] = []
        for sig in sell_signals:
            if sig.symbol not in portfolio.positions:
                continue

            # min_hold_days: only stop-loss bypasses the minimum hold floor
            if self.min_hold_days > 0:
                entry = pd.to_datetime(
                    portfolio.positions[sig.symbol].get("entry_date", d.isoformat())
                ).date()
                held = (d - entry).days
                is_stop_loss = sig.reason.startswith("stop_loss")
                if held < self.min_hold_days and not is_stop_loss:
                    continue

            qty_held = portfolio.positions[sig.symbol]["qty"]
            qty_to_sell = math.floor(qty_held * sig.trim_pct)
            if qty_to_sell > 0 and prices.get(sig.symbol, 0) > 0:
                portfolio.fill(
                    sig.symbol, "sell", qty_to_sell, prices[sig.symbol], d, sig.reason
                )
                triggered.append(sig.reason)
                logger.debug(
                    "SELL %s qty=%d reason=%s [%s]", sig.symbol, qty_to_sell, sig.reason, d
                )

        # Track consecutive days below 50DMA in portfolio positions
        for pos in positions:
            sym = pos["symbol"]
            if sym not in portfolio.positions:
                continue
            sym_bars = _get_sym_bars(all_bars, sym, d)
            if sym_bars is None or len(sym_bars) < 50:
                continue
            ma50 = float(sym_bars["close"].iloc[-50:].mean())
            last_close = float(sym_bars["close"].iloc[-1])
            current_below = portfolio.positions[sym].get("below_50dma_days", 0)
            portfolio.positions[sym]["below_50dma_days"] = (
                current_below + 1 if last_close < ma50 else 0
            )

        return triggered

    def _rebalance(
        self,
        d: date,
        universe: list[str],
        portfolio: PaperPortfolio,
        hist_alpaca: HistoricalAlpacaClient,
        fmp: FMPClient,
        hist_macro: HistoricalMacroClient,
        prices: dict[str, float],
        cached_ers_df: pd.DataFrame | None = None,
        cached_qlt_df: pd.DataFrame | None = None,
        cached_sector_map: dict[str, str] | None = None,
        regime: str = "factor",
        all_bars: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        cfg = get_settings()
        try:
            if regime == "concentrated" and all_bars is not None:
                # ── Concentrated AI mode ─────────────────────────────────────
                ai_df = ai_score(all_bars, d)
                current_positions = hist_alpaca.get_positions()
                target_weights = construct_concentrated(ai_df, current_positions, cfg)
                _execute_rebalance(target_weights, portfolio, prices, d)
                logger.info("REBALANCE %s | CONCENTRATED | target=%d", d, len(target_weights))
                # Build a minimal signals_df so daily sell rules still work
                return _minimal_signals_df(ai_df)

            else:
                # ── Factor mode ──────────────────────────────────────────────
                with _no_store_writes():
                    ranked_df = master_score.compute(
                        universe, alpaca=hist_alpaca, fmp=fmp, macro_client=hist_macro,
                        cached_ers_df=cached_ers_df, cached_qlt_df=cached_qlt_df,
                    )

                macro_val = (
                    float(ranked_df["macro_alignment_score"].iloc[0])
                    if not ranked_df.empty else 0.0
                )
                current_positions = hist_alpaca.get_positions()
                target_weights = construct(
                    ranked_df, current_positions,
                    macro_score=macro_val, sector_map=cached_sector_map or {},
                )
                _execute_rebalance(target_weights, portfolio, prices, d)

                n_pass = int(ranked_df["passes_filters"].sum()) if "passes_filters" in ranked_df.columns else 0
                logger.info(
                    "REBALANCE %s | FACTOR | target=%d | pass_filters=%d | macro=%.1f",
                    d, len(target_weights), n_pass, macro_val,
                )
                return ranked_df

        except Exception:
            logger.exception("Rebalance failed on %s", d)
            return None


# ── module-level helpers ──────────────────────────────────────────────────────

def _should_rebalance(d: date, i: int, trading_dates: list[date], freq: str) -> bool:
    if i == 0:
        return True
    if freq == "weekly":
        return d.weekday() == 4  # Friday
    if freq == "biweekly":
        # Every other Friday — count Fridays seen so far
        friday_count = sum(1 for td in trading_dates[:i] if td.weekday() == 4)
        return d.weekday() == 4 and friday_count % 2 == 0
    if freq == "monthly":
        # First Friday of each calendar month
        if d.weekday() != 4:
            return False
        prev_friday = next(
            (td for td in reversed(trading_dates[:i]) if td.weekday() == 4), None
        )
        return prev_friday is None or prev_friday.month != d.month
    return False


def _execute_rebalance(
    target_weights: dict[str, float],
    portfolio: PaperPortfolio,
    prices: dict[str, float],
    trade_date: date,
) -> None:
    current_syms = set(portfolio.positions.keys())
    target_syms = set(target_weights.keys())

    if not target_weights:
        for sym in list(current_syms):
            qty = int(portfolio.positions[sym]["qty"])
            portfolio.fill(sym, "sell", qty, prices.get(sym, 0), trade_date, "macro_derisking")
        return

    # ── Step 1: Exit positions not in new target ──────────────────────────────
    for sym in current_syms - target_syms:
        qty = int(portfolio.positions[sym]["qty"])
        if qty > 0 and prices.get(sym, 0) > 0:
            portfolio.fill(sym, "sell", qty, prices[sym], trade_date, "rebalance_exit")

    # ── Step 2: Compute target shares from post-exit portfolio value ──────────
    portfolio_value = portfolio.nav(prices)
    target_shares: dict[str, int] = {
        sym: math.floor(portfolio_value * w / prices[sym])
        for sym, w in target_weights.items()
        if prices.get(sym, 0) > 0
    }

    # ── Step 3: Trim carry-over positions that are oversized ─────────────────
    for sym in current_syms & target_syms:
        if sym not in portfolio.positions:
            continue
        delta = int(portfolio.positions[sym]["qty"]) - target_shares.get(sym, 0)
        if delta >= 1 and prices.get(sym, 0) > 0:
            portfolio.fill(sym, "sell", delta, prices[sym], trade_date, "rebalance_trim")

    # ── Step 4: Buy new or undersized positions (highest weight first) ────────
    for sym in sorted(target_syms, key=lambda s: target_weights.get(s, 0), reverse=True):
        current_qty = int(portfolio.positions.get(sym, {}).get("qty", 0))
        delta = target_shares.get(sym, 0) - current_qty
        if delta >= 1 and prices.get(sym, 0) > 0:
            portfolio.fill(sym, "buy", delta, prices[sym], trade_date, "rebalance_add")


def _trading_dates(bars: pd.DataFrame, start: date, end: date) -> list[date]:
    dates = bars.index.get_level_values("date").unique()
    return sorted(d.date() for d in dates if start <= d.date() <= end)


def _close_prices(bars: pd.DataFrame, universe: list[str], as_of: date) -> dict[str, float]:
    ts = pd.Timestamp(as_of)
    result: dict[str, float] = {}
    for sym in universe:
        try:
            sym_bars = bars.loc[sym]
            filtered = sym_bars[sym_bars.index <= ts]
            if not filtered.empty:
                result[sym] = float(filtered["close"].iloc[-1])
        except KeyError:
            pass
    return result


def _slice_bars(
    bars: pd.DataFrame,
    symbols: list[str],
    as_of: date,
    lookback_days: int = 260,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(as_of - timedelta(days=lookback_days + 60))
    end_ts = pd.Timestamp(as_of)
    sym_idx = bars.index.get_level_values("symbol")
    date_idx = bars.index.get_level_values("date")
    mask = sym_idx.isin(symbols) & (date_idx >= start_ts) & (date_idx <= end_ts)
    return bars[mask]


def _get_sym_bars(bars: pd.DataFrame, sym: str, as_of: date) -> pd.DataFrame | None:
    try:
        s = bars.loc[sym]
        return s[s.index <= pd.Timestamp(as_of)]
    except KeyError:
        return None


def _spy_slice(bars: pd.DataFrame, start: date, end: date) -> pd.DataFrame | None:
    try:
        spy = bars.loc["SPY"]
        return spy[
            (spy.index >= pd.Timestamp(start)) & (spy.index <= pd.Timestamp(end))
        ]
    except KeyError:
        return None


def _minimal_signals_df(ai_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a minimal signals DataFrame from AI scores so daily sell rules
    have something to look up (RS/EPS revision checks will get None and skip).
    """
    if ai_df.empty:
        return pd.DataFrame()
    return ai_df.rename(columns={"ai_score": "master_score"})


def _load_macro_data(start: date, end: date) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        _MACRO_TICKERS,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return {}
    close = raw["Close"] if "Close" in raw.columns else raw
    result: dict[str, pd.DataFrame] = {}
    for ticker in _MACRO_TICKERS:
        if ticker not in close.columns:
            continue
        df = close[[ticker]].rename(columns={ticker: "close"}).dropna()
        df.index = pd.to_datetime(df.index).normalize()
        result[ticker] = df
    return result
