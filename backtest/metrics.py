from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    nav_history: list[dict],
    spy_bars: pd.DataFrame | None = None,
    trade_log: list[dict] | None = None,
) -> dict:
    if not nav_history:
        return {}

    df = pd.DataFrame(nav_history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    initial = float(df["nav"].iloc[0])
    final = float(df["nav"].iloc[-1])
    total_return = final / initial - 1.0

    n_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (final / initial) ** (1.0 / max(n_years, 1e-9)) - 1.0

    daily_ret = df["nav"].pct_change().dropna()
    std = float(daily_ret.std())
    sharpe = float(daily_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    down_std = float(daily_ret[daily_ret < 0].std())
    sortino = float(daily_ret.mean() / down_std * np.sqrt(252)) if down_std > 0 else 0.0

    cummax = df["nav"].cummax()
    max_dd = float(((df["nav"] - cummax) / cummax).min())

    # Calmar: CAGR / |max_drawdown|
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # SPY benchmark
    spy_return = spy_cagr = None
    if spy_bars is not None and not spy_bars.empty:
        spy_close = spy_bars["close"].sort_index()
        spy_start = float(spy_close.iloc[0])
        spy_end = float(spy_close.iloc[-1])
        spy_return = spy_end / spy_start - 1.0
        spy_cagr = (spy_end / spy_start) ** (1.0 / max(n_years, 1e-9)) - 1.0

    trade_log = trade_log or []
    n_buys = sum(1 for t in trade_log if t.get("side") == "buy")
    n_sells = sum(1 for t in trade_log if t.get("side") == "sell")

    return {
        "initial_nav": round(initial, 2),
        "final_nav": round(final, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "spy_total_return_pct": round(spy_return * 100, 2) if spy_return is not None else None,
        "spy_cagr_pct": round(spy_cagr * 100, 2) if spy_cagr is not None else None,
        "alpha_pct": round((total_return - spy_return) * 100, 2) if spy_return is not None else None,
        "n_trading_days": len(df),
        "n_trades": len(trade_log),
        "n_buys": n_buys,
        "n_sells": n_sells,
        "avg_positions": round(float(df["num_positions"].mean()), 1),
    }
