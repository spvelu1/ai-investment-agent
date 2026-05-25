from __future__ import annotations

import logging
from datetime import date, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import get_settings
from data.clients.alpaca_client import AlpacaClient
from data.clients.fmp_client import FMPClient
from data.clients.macro_client import MacroClient
from data import store
from portfolio.constructor import build_sector_map, construct_concentrated
from portfolio.executor import Executor
from portfolio.risk_manager import evaluate as evaluate_risk
from signals import master_score
from signals.regime import compute_live as compute_regime
from universe.ai_universe import score_live as ai_score_live
from universe.screener import get_eligible_universe

logger = logging.getLogger(__name__)


def run_daily_check() -> None:
    """
    Daily pre-market job (6:30 AM ET).
    - Detect current regime → tighter stop-loss when concentrated
    - Pull latest bars for open positions
    - Evaluate and execute sell rules
    """
    logger.info("=== DAILY CHECK starting ===")
    alpaca = AlpacaClient()
    fmp = FMPClient()
    macro_client = MacroClient()

    positions = alpaca.get_positions()
    if not positions:
        logger.info("No open positions — daily check complete")
        return

    symbols = [p["symbol"] for p in positions]
    bars = alpaca.get_bars_lookback(symbols, days=260)
    stored_positions = store.read_positions()

    # Latest signals for RS + revision checks
    signals_df = store.read_signals()

    # Apply tighter stop-loss when in concentrated AI mode
    cfg = get_settings()
    regime_data = compute_regime()
    regime = regime_data["regime"]
    logger.info("Daily check regime: %s (composite=%.1f)", regime.upper(), regime_data["composite_score"])

    orig_stop_loss = cfg.stop_loss_pct
    if regime == "concentrated":
        cfg.stop_loss_pct = cfg.concentrated_stop_loss_pct
        logger.info("Stop-loss tightened to %.0f%% (concentrated mode)", cfg.stop_loss_pct * 100)

    try:
        sell_signals = evaluate_risk(stored_positions, bars, signals_df)
    finally:
        cfg.stop_loss_pct = orig_stop_loss

    if sell_signals:
        logger.info("Sell signals: %s", [(s.symbol, s.action, s.reason) for s in sell_signals])
        executor = Executor(alpaca)
        executor.execute_sells(sell_signals)
    else:
        logger.info("No sell signals triggered")

    logger.info("=== DAILY CHECK complete ===")


def run_weekly_rebalance() -> None:
    """
    Weekly rebalance job (Friday 6:00 AM ET).
    - Refresh full universe
    - Compute all signals → MASTER_SCORE
    - Construct target portfolio
    - Execute trades
    """
    logger.info("=== WEEKLY REBALANCE starting ===")
    cfg = get_settings()
    alpaca = AlpacaClient()
    fmp = FMPClient()
    macro_client = MacroClient()

    # Step 1: Build universe
    logger.info("Building eligible universe...")
    universe = get_eligible_universe(alpaca=alpaca, fmp=fmp)
    logger.info("Universe size: %d", len(universe))
    if not universe:
        logger.error("Empty universe — aborting rebalance")
        return

    # Step 2: Detect market regime (leading signal: 21d breadth + tech spread)
    regime_data = compute_regime()
    regime = regime_data["regime"]
    logger.info(
        "Regime: %s | composite=%.1f (breadth_21d=%.1f, tech_21d=%.1f)",
        regime.upper(),
        regime_data["composite_score"],
        regime_data["breadth_spread_21d"],
        regime_data["tech_spread_21d"],
    )

    # Step 4: Get current positions
    current_positions = alpaca.get_positions()

    if regime == "concentrated":
        # ── Concentrated AI mode ─────────────────────────────────────────────
        logger.info("Entering CONCENTRATED AI mode — scoring mega-cap tech universe")
        ai_df = ai_score_live(alpaca)
        if ai_df.empty:
            logger.error("AI scoring failed — aborting rebalance")
            return
        target_weights = construct_concentrated(ai_df, current_positions, cfg)
        ranked_df = ai_df.rename(columns={"ai_score": "master_score"})
        macro_score_val = 100.0  # concentrated mode doesn't use macro gating
    else:
        # ── Factor mode ──────────────────────────────────────────────────────
        logger.info("Entering FACTOR mode — running full MASTER_SCORE pipeline")
        # Step 2b: Compute MASTER_SCORE
        ranked_df = master_score.compute(universe, alpaca=alpaca, fmp=fmp, macro_client=macro_client)
        macro_score_val = float(ranked_df["macro_alignment_score"].iloc[0]) if not ranked_df.empty else 0.0

        # Step 3: Persist signals
        store.write_signals(ranked_df.reset_index(), signal_date=date.today())

        from portfolio.constructor import construct
        sector_map = build_sector_map(universe, fmp)
        target_weights = construct(ranked_df, current_positions, macro_score=macro_score_val, sector_map=sector_map)

    # Step 6: Execute
    executor = Executor(alpaca)
    executor.execute_rebalance(target_weights, current_positions)

    # Step 7: Sync position store
    updated_positions = alpaca.get_positions()
    _sync_position_store(updated_positions, stored_positions=store.read_positions())

    # Reset FMP counter for the day
    FMPClient.reset_daily_counter()

    logger.info(
        "=== WEEKLY REBALANCE complete | regime=%s | %d target positions | macro=%.1f ===",
        regime.upper(),
        len(target_weights),
        macro_score_val,
    )


def _sync_position_store(alpaca_positions: list[dict], stored_positions: list[dict]) -> None:
    """Merge Alpaca live positions with stored entry metadata."""
    stored_map = {p["symbol"]: p for p in stored_positions}
    merged = []
    for pos in alpaca_positions:
        sym = pos["symbol"]
        stored = stored_map.get(sym, {})
        merged.append({
            "symbol": sym,
            "entry_date": stored.get("entry_date", date.today().isoformat()),
            "entry_price": stored.get("entry_price", pos["avg_entry_price"]),
            "qty": pos["qty"],
            "below_50dma_days": stored.get("below_50dma_days", 0),
        })
    store.write_positions(merged)


def start_scheduler() -> None:
    """Start the blocking APScheduler with daily + weekly jobs."""
    scheduler = BlockingScheduler(timezone="America/New_York")

    # Daily check: Monday–Friday at 6:30 AM ET
    scheduler.add_job(
        run_daily_check,
        trigger=CronTrigger(day_of_week="mon-fri", hour=6, minute=30),
        id="daily_check",
        name="Daily Sell-Rule Check",
        misfire_grace_time=600,
    )

    # Weekly rebalance: Friday at 6:00 AM ET
    scheduler.add_job(
        run_weekly_rebalance,
        trigger=CronTrigger(day_of_week="fri", hour=6, minute=0),
        id="weekly_rebalance",
        name="Weekly Rebalance",
        misfire_grace_time=600,
    )

    logger.info("Scheduler started — daily check 6:30 AM ET Mon–Fri, rebalance 6:00 AM ET Fri")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")
