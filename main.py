#!/usr/bin/env python3
"""
AI Investment Agent — CLI entry point.

Usage:
    python main.py run          Start the scheduler (blocking)
    python main.py rebalance    Run one full rebalance immediately
    python main.py daily        Run daily sell-rule check immediately
    python main.py signals      Print MASTER_SCORE ranking table
    python main.py status       Print current positions
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_run(_args) -> None:
    from scheduler.runner import start_scheduler
    start_scheduler()


def cmd_rebalance(_args) -> None:
    from scheduler.runner import run_weekly_rebalance
    run_weekly_rebalance()


def cmd_daily(_args) -> None:
    from scheduler.runner import run_daily_check
    run_daily_check()


def cmd_signals(args) -> None:
    from data import store
    from tabulate import tabulate

    df = store.read_signals()
    if df.empty:
        print("No signals in store. Run 'python main.py rebalance' first.")
        return

    cols = [
        "symbol", "master_score", "earnings_revision_score",
        "momentum_score", "breakout_score", "quality_score",
        "macro_alignment_score", "passes_filters",
    ]
    available = [c for c in cols if c in df.columns]
    display = df[available].copy()

    score_cols = [c for c in available if "score" in c]
    display[score_cols] = display[score_cols].round(1)

    n = args.top if hasattr(args, "top") else 30
    print(f"\n=== MASTER_SCORE Ranking ({date.today()}) — top {n} ===\n")
    print(tabulate(display.head(n), headers="keys", tablefmt="rounded_outline", showindex=False))

    passing = display[display["passes_filters"] == True] if "passes_filters" in display.columns else display.iloc[0:0]
    print(f"\n{len(passing)} symbols pass all hard filters\n")


_RESULTS_DIR = Path(__file__).parent / "backtest" / "results"


def _save_backtest_result(args, result: dict) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    stem = (
        f"{run_ts}"
        f"_{args.start}_{args.end}"
        f"_{args.freq}"
        f"_50dma{args.below_50dma_days if args.below_50dma_days is not None else 'default'}"
        f"_hold{args.min_hold_days}"
        f"{'_noTS' if args.no_time_stop else ''}"
    )

    payload = {
        "run_id": run_ts,
        "config": {
            "start": args.start,
            "end": args.end,
            "cash": args.cash,
            "slippage_pct": args.slippage,
            "freq": args.freq,
            "below_50dma_days": args.below_50dma_days,
            "min_hold_days": args.min_hold_days,
            "no_time_stop": args.no_time_stop,
        },
        "metrics": {
            k: result.get(k)
            for k in (
                "total_return_pct", "cagr_pct", "sharpe_ratio", "sortino_ratio",
                "calmar_ratio", "max_drawdown_pct",
                "spy_total_return_pct", "spy_cagr_pct", "alpha_pct",
                "avg_positions", "avg_positions_post_rebalance",
                "rebalance_count", "n_trades", "n_buys", "n_sells",
                "n_trading_days", "initial_nav", "final_nav",
            )
        },
        "sell_reason_counts": result.get("sell_reason_counts", {}),
        "nav_history": result.get("nav_history", []),
    }

    json_path = _RESULTS_DIR / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    csv_path = _RESULTS_DIR / f"{stem}_nav.csv"
    import csv
    nav_rows = result.get("nav_history", [])
    if nav_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=nav_rows[0].keys())
            writer.writeheader()
            writer.writerows(nav_rows)

    return json_path


def cmd_backtest(args) -> None:
    from datetime import date
    from backtest.engine import BacktestEngine
    from tabulate import tabulate

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    engine = BacktestEngine(
        start_date=start,
        end_date=end,
        initial_cash=float(args.cash),
        slippage_pct=args.slippage / 100.0,
        rebalance_freq=args.freq,
        below_50dma_days=args.below_50dma_days,
        min_hold_days=args.min_hold_days,
        disable_time_stop=args.no_time_stop,
    )
    result = engine.run()

    saved_path = _save_backtest_result(args, result)
    print(f"\n[Saved] {saved_path}")

    def _fmt(val, fmt=".2f"):
        return f"{val:{fmt}}" if val is not None else "N/A"

    # ── Main metrics table ────────────────────────────────────────────────────
    n_trades = result.get("n_trades", 0)
    n_days   = result.get("n_trading_days", 1)
    n_rb     = result.get("rebalance_count", 0)
    avg_rb   = result.get("avg_positions_post_rebalance", 0)

    config_line = (
        f"freq={args.freq}  50dma_days="
        f"{'default(2)' if args.below_50dma_days is None else args.below_50dma_days}"
        f"  min_hold={args.min_hold_days}d"
        f"{'  no_time_stop' if args.no_time_stop else ''}"
    )
    print(f"\n=== BACKTEST RESULTS  {start} → {end}  [{config_line}] ===\n")

    rows = [
        ["Total Return",              f"{_fmt(result.get('total_return_pct'))}%"],
        ["CAGR",                      f"{_fmt(result.get('cagr_pct'))}%"],
        ["Sharpe Ratio",              _fmt(result.get('sharpe_ratio'), ".3f")],
        ["Sortino Ratio",             _fmt(result.get('sortino_ratio'), ".3f")],
        ["Calmar Ratio",              _fmt(result.get('calmar_ratio'), ".3f")],
        ["Max Drawdown",              f"{_fmt(result.get('max_drawdown_pct'))}%"],
        ["─" * 20,                    "─" * 12],
        ["SPY Total Return",          f"{_fmt(result.get('spy_total_return_pct'))}%"],
        ["SPY CAGR",                  f"{_fmt(result.get('spy_cagr_pct'))}%"],
        ["Alpha (vs SPY)",            f"{_fmt(result.get('alpha_pct'))}%"],
        ["─" * 20,                    "─" * 12],
        ["Avg Positions (overall)",   str(result.get('avg_positions', "N/A"))],
        ["Avg Positions (post-rebln)",str(avg_rb)],
        ["Rebalances executed",       str(n_rb)],
        ["Total Trades",              f"{n_trades}  ({n_trades/n_days:.1f}/day)"],
        ["  Buys",                    str(result.get('n_buys', 0))],
        ["  Sells",                   str(result.get('n_sells', 0))],
        ["Trading Days",              str(n_days)],
        ["Initial NAV",               f"${result.get('initial_nav', 0):,.2f}"],
        ["Final NAV",                 f"${result.get('final_nav', 0):,.2f}"],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_outline"))

    # ── Sell-reason breakdown ─────────────────────────────────────────────────
    sell_reasons = result.get("sell_reason_counts", {})
    if sell_reasons:
        print("\n--- Exit Reasons (daily check) ---")
        for reason, count in sorted(sell_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:<35} {count:>4}")

    if args.trades:
        log = result.get("trade_log", [])
        if log:
            print(f"\n=== TRADE LOG (last {args.trades}) ===\n")
            recent = log[-args.trades:]
            print(tabulate(recent, headers="keys", tablefmt="rounded_outline"))

    if args.nav_csv:
        import csv
        with open(args.nav_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "nav", "cash", "num_positions"])
            writer.writeheader()
            writer.writerows(result.get("nav_history", []))
        print(f"\nNAV history saved to {args.nav_csv}")


def cmd_status(_args) -> None:
    from data.clients.alpaca_client import AlpacaClient
    from tabulate import tabulate

    alpaca = AlpacaClient()
    account = alpaca.get_account()
    positions = alpaca.get_positions()

    print(f"\n=== Account ({date.today()}) ===")
    for k, v in account.items():
        print(f"  {k}: ${v:,.2f}")

    if not positions:
        print("\nNo open positions.\n")
        return

    print(f"\n=== Open Positions ({len(positions)}) ===\n")
    rows = [
        {
            "Symbol": p["symbol"],
            "Qty": p["qty"],
            "Entry $": p["avg_entry_price"],
            "Current $": p["current_price"],
            "Mkt Val $": p["market_value"],
            "P&L %": f"{p['unrealized_plpc']*100:.2f}%",
        }
        for p in sorted(positions, key=lambda x: -x["market_value"])
    ]
    print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Investment Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Start the scheduler (blocking)")
    sub.add_parser("rebalance", help="Run one full rebalance now")
    sub.add_parser("daily", help="Run daily sell-rule check now")

    sig_parser = sub.add_parser("signals", help="Print MASTER_SCORE ranking table")
    sig_parser.add_argument("--top", type=int, default=30, help="Rows to display")

    sub.add_parser("status", help="Print current positions and account")

    bt_parser = sub.add_parser("backtest", help="Run walk-forward backtest on historical data")
    bt_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    bt_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    bt_parser.add_argument("--cash", type=float, default=100_000, help="Starting cash (default 100000)")
    bt_parser.add_argument("--slippage", type=float, default=0.1, help="Slippage %% per trade (default 0.1)")
    bt_parser.add_argument(
        "--freq", default="weekly", choices=["weekly", "biweekly", "monthly"],
        help="Rebalance frequency (default: weekly). Monthly = ~12 rebalances/yr vs 52.",
    )
    bt_parser.add_argument(
        "--below-50dma-days", dest="below_50dma_days", type=int, default=None,
        metavar="N",
        help="Consecutive days below 50-DMA before exit (default: 2). "
             "Set 10 for 2-week buffer; 999 to disable.",
    )
    bt_parser.add_argument(
        "--min-hold-days", dest="min_hold_days", type=int, default=0,
        metavar="N",
        help="Min calendar days to hold before any exit fires (stop-loss bypasses). "
             "Recommended: 21.",
    )
    bt_parser.add_argument(
        "--no-time-stop", dest="no_time_stop", action="store_true",
        help="Disable the 40-day time-stop so winners can run freely.",
    )
    bt_parser.add_argument("--trades", type=int, default=0, help="Print last N trades (0 = none)")
    bt_parser.add_argument("--nav-csv", dest="nav_csv", default="", help="Save daily NAV to CSV file")

    args = parser.parse_args()
    commands = {
        "run": cmd_run,
        "rebalance": cmd_rebalance,
        "daily": cmd_daily,
        "signals": cmd_signals,
        "status": cmd_status,
        "backtest": cmd_backtest,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
