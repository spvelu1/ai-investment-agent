from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _conn() -> duckdb.DuckDBPyConnection:
    cfg = get_settings()
    Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(cfg.db_path)


def _ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            symbol      VARCHAR,
            date        DATE,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            trade_count DOUBLE,
            vwap        DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            symbol                  VARCHAR,
            signal_date             DATE,
            earnings_revision_score DOUBLE,
            momentum_score          DOUBLE,
            breakout_score          DOUBLE,
            quality_score           DOUBLE,
            macro_alignment_score   DOUBLE,
            master_score            DOUBLE,
            passes_filters          BOOLEAN,
            PRIMARY KEY (symbol, signal_date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol          VARCHAR PRIMARY KEY,
            entry_date      DATE,
            entry_price     DOUBLE,
            qty             DOUBLE,
            below_50dma_days INT DEFAULT 0
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id    VARCHAR PRIMARY KEY,
            symbol      VARCHAR,
            side        VARCHAR,
            qty         DOUBLE,
            reason      VARCHAR,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


# ── Bars ──────────────────────────────────────────────────────────────────────

def write_bars(df: pd.DataFrame) -> None:
    """Upsert bars. df must have MultiIndex (symbol, date) or columns [symbol, date]."""
    con = _conn()
    _ensure_tables(con)
    flat = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()
    flat.columns = [c.lower() for c in flat.columns]
    if "date" not in flat.columns and "timestamp" in flat.columns:
        flat = flat.rename(columns={"timestamp": "date"})
    flat["date"] = pd.to_datetime(flat["date"]).dt.date
    for col in ["trade_count", "vwap"]:
        if col not in flat.columns:
            flat[col] = None
    cols = ["symbol", "date", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    flat = flat[[c for c in cols if c in flat.columns]]
    con.execute("DELETE FROM bars WHERE symbol = ANY(?)", [flat["symbol"].unique().tolist()])
    con.execute("INSERT INTO bars SELECT * FROM flat")
    con.close()
    logger.debug("Wrote %d bar rows to store", len(flat))


def read_bars(symbols: list[str], start: date | None = None, end: date | None = None) -> pd.DataFrame:
    con = _conn()
    _ensure_tables(con)
    where_parts = ["symbol = ANY(?)"]
    params: list = [symbols]
    if start:
        where_parts.append("date >= ?")
        params.append(start)
    if end:
        where_parts.append("date <= ?")
        params.append(end)
    where = " AND ".join(where_parts)
    df = con.execute(f"SELECT * FROM bars WHERE {where} ORDER BY symbol, date", params).df()
    con.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["symbol", "date"])


# ── Signals ───────────────────────────────────────────────────────────────────

def write_signals(df: pd.DataFrame, signal_date: date | None = None) -> None:
    con = _conn()
    _ensure_tables(con)
    flat = df.reset_index() if "symbol" not in df.columns else df.copy()
    if signal_date and "signal_date" not in flat.columns:
        flat["signal_date"] = signal_date
    flat["signal_date"] = pd.to_datetime(flat["signal_date"]).dt.date
    cols = [
        "symbol", "signal_date", "earnings_revision_score", "momentum_score",
        "breakout_score", "quality_score", "macro_alignment_score", "master_score",
        "passes_filters",
    ]
    flat = flat[[c for c in cols if c in flat.columns]]
    con.execute(
        "DELETE FROM signals WHERE signal_date = ?",
        [flat["signal_date"].iloc[0]],
    )
    con.execute("INSERT INTO signals SELECT * FROM flat")
    con.close()


def read_signals(signal_date: date | None = None) -> pd.DataFrame:
    con = _conn()
    _ensure_tables(con)
    if signal_date:
        df = con.execute(
            "SELECT * FROM signals WHERE signal_date = ? ORDER BY master_score DESC",
            [signal_date],
        ).df()
    else:
        df = con.execute(
            "SELECT * FROM signals WHERE signal_date = (SELECT MAX(signal_date) FROM signals) ORDER BY master_score DESC"
        ).df()
    con.close()
    return df


# ── Positions ────────────────────────────────────────────────────────────────

def write_positions(positions: list[dict]) -> None:
    con = _conn()
    _ensure_tables(con)
    if not positions:
        con.execute("DELETE FROM positions")
        con.close()
        return
    df = pd.DataFrame(positions)
    df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
    con.execute("DELETE FROM positions")
    con.execute("INSERT INTO positions SELECT * FROM df")
    con.close()


def read_positions() -> list[dict]:
    con = _conn()
    _ensure_tables(con)
    df = con.execute("SELECT * FROM positions").df()
    con.close()
    return df.to_dict("records")


def update_below_50dma_days(symbol: str, days: int) -> None:
    con = _conn()
    _ensure_tables(con)
    con.execute("UPDATE positions SET below_50dma_days = ? WHERE symbol = ?", [days, symbol])
    con.close()


# ── Orders ────────────────────────────────────────────────────────────────────

def log_order(order_id: str, symbol: str, side: str, qty: float, reason: str) -> None:
    con = _conn()
    _ensure_tables(con)
    con.execute(
        "INSERT OR IGNORE INTO orders (order_id, symbol, side, qty, reason) VALUES (?, ?, ?, ?, ?)",
        [order_id, symbol, side, qty, reason],
    )
    con.close()
