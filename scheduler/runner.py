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
from portfolio.constructor import build_sector_map
from portfolio.executor import Executor
from portfolio.risk_manager import evaluate as evaluate_risk
from signals import master_score
from universe.screener import get_eligible_universe

logger = logging.getLogger(__name__)


def run_daily_check() -> None:
    """
    Daily pre-market job (6:30 AM ET).
    - Refresh macro regime
    - Pull latest bars for open positions
    - Evaluate and execute sell rules
    - Persist updated signal scores
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

    sell_signals = evaluate_risk(stored_positions, bars, signals_df)
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

    # Step 2: Compute MASTER_SCORE
    ranked_df = master_score.compute(universe, alpaca=alpaca, fmp=fmp, macro_client=macro_client)
    macro_score_val = float(ranked_df["macro_alignment_score"].iloc[0]) if not ranked_df.empty else 0.0

    # Step 3: Persist signals
    store.write_signals(ranked_df.reset_index(), signal_date=date.today())

    # Step 4: Get current positions
    current_positions = alpaca.get_positions()

    # Step 5: Construct target portfolio
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
        "=== WEEKLY REBALANCE complete | %d target positions | macro=%.1f ===",
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
