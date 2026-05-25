from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from config.settings import get_settings
from data.clients.alpaca_client import AlpacaClient
from data.clients.fmp_client import FMPClient

logger = logging.getLogger(__name__)

# Static fallback — top 100 liquid S&P 500 names in case FMP call fails
_FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK.B", "LLY", "AVGO",
    "TSLA", "JPM", "UNH", "V", "XOM", "MA", "COST", "HD", "PG", "JNJ",
    "WMT", "ABBV", "BAC", "MRK", "NFLX", "CRM", "CVX", "ORCL", "KO", "AMD",
    "PEP", "TMO", "ACN", "MCD", "ADBE", "LIN", "CSCO", "ABT", "DHR", "INTC",
    "TXN", "PM", "NEE", "INTU", "WFC", "AMGN", "UPS", "RTX", "COP", "LOW",
    "SPGI", "HON", "IBM", "AMAT", "GE", "BKNG", "CAT", "SYK", "ISRG", "QCOM",
    "BLK", "MS", "GS", "ELV", "MU", "PLD", "AXP", "MDT", "VRTX", "LRCX",
    "ADI", "DE", "SBUX", "GILD", "MMC", "TJX", "CI", "MDLZ", "SO", "DUK",
    "PGR", "BSX", "SHW", "ETN", "ZTS", "REGN", "NOW", "APH", "PANW", "KLAC",
    "SNPS", "CDNS", "CME", "EOG", "MCK", "PSA", "HCA", "ITW", "ADP", "EW",
]

# Exchanges considered eligible (NYSE and Nasdaq main boards)
_ELIGIBLE_EXCHANGES = {"NYSE", "NASDAQ", "BATS"}

# Asset types to exclude
_EXCLUDED_TYPES = {"ETF", "ETN", "FUND", "INDEX"}


def get_eligible_universe(
    alpaca: AlpacaClient | None = None,
    fmp: FMPClient | None = None,
) -> list[str]:
    """
    Return list of tickers that pass the minimum liquidity / quality filters:
      - Listed on NYSE or Nasdaq
      - Not an ETF, ADR, or pre-revenue biotech proxy
      - Price > $10
      - Market cap > $2B (via FMP profile)
      - 20-day avg dollar volume > $20M (computed from Alpaca bars)
    """
    cfg = get_settings()
    fmp = fmp or FMPClient()
    alpaca = alpaca or AlpacaClient()

    # --- Step 1: base universe from FMP S&P 500 + tradeable list ---
    candidates = _get_base_candidates(fmp)

    # --- Step 2: filter by exchange and instrument type via FMP tradeable list ---
    tradeable = _build_tradeable_index(fmp)
    candidates = [
        sym for sym in candidates
        if _is_exchange_eligible(sym, tradeable)
    ]
    logger.info("After exchange filter: %d candidates", len(candidates))

    # --- Step 3: price + ADV filter via Alpaca bars ---
    candidates = _filter_by_adv(candidates, alpaca, cfg)
    logger.info("After ADV/price filter: %d candidates", len(candidates))

    return sorted(candidates)


def _get_base_candidates(fmp: FMPClient) -> list[str]:
    try:
        sp500 = fmp.get_sp500_constituents()
        if sp500:
            logger.info("FMP returned %d S&P 500 constituents", len(sp500))
            return sp500
    except Exception as exc:
        logger.warning("FMP S&P 500 fetch failed (%s), using fallback universe", exc)
    return list(_FALLBACK_UNIVERSE)


def _build_tradeable_index(fmp: FMPClient) -> dict[str, dict]:
    """Map symbol → {exchange, type} from FMP's available-traded list."""
    try:
        items = fmp.get_tradeable_symbols()
        return {
            item.get("symbol", ""): {
                "exchange": item.get("exchangeShortName", ""),
                "type": item.get("type", ""),
            }
            for item in items
            if item.get("symbol")
        }
    except Exception as exc:
        logger.warning("FMP tradeable list fetch failed (%s), skipping exchange filter", exc)
        return {}


def _is_exchange_eligible(symbol: str, tradeable_index: dict[str, dict]) -> bool:
    if not tradeable_index:
        return True
    info = tradeable_index.get(symbol, {})
    exchange = (info.get("exchange") or "").upper()
    asset_type = (info.get("type") or "").upper()
    if asset_type in _EXCLUDED_TYPES:
        return False
    if exchange and exchange not in _ELIGIBLE_EXCHANGES:
        return False
    return True


def _filter_by_adv(
    symbols: list[str],
    alpaca: AlpacaClient,
    cfg,
) -> list[str]:
    """Filter symbols by price > $10 and 20-day avg dollar volume > $20M."""
    from datetime import date, timedelta

    end = date.today()
    start = end - timedelta(days=60)  # fetch extra buffer

    # Batch fetch bars (Alpaca accepts up to ~100 symbols per request cleanly)
    eligible: list[str] = []
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            bars = alpaca.get_bars(batch, start=start, end=end)
        except Exception as exc:
            logger.warning("Alpaca bars fetch failed for batch %d (%s), skipping", i, exc)
            continue

        for sym in batch:
            try:
                sym_bars = bars.loc[sym] if sym in bars.index.get_level_values(0) else None
                if sym_bars is None or sym_bars.empty:
                    continue
                last_price = float(sym_bars["close"].iloc[-1])
                if last_price < cfg.min_price:
                    continue
                # 20-day avg dollar volume
                last_20 = sym_bars.tail(20)
                adv = (last_20["close"] * last_20["volume"]).mean()
                if adv < cfg.min_adv:
                    continue
                eligible.append(sym)
            except Exception as exc:
                logger.debug("Skipping %s during ADV filter: %s", sym, exc)

    return eligible
