from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Alpaca
    alpaca_api_key: str = Field(..., description="Alpaca API key")
    alpaca_secret_key: str = Field(..., description="Alpaca secret key")
    alpaca_base_url: str = Field(
        "https://paper-api.alpaca.markets",
        description="Alpaca base URL (paper or live)",
    )

    # Financial Modeling Prep
    fmp_api_key: str = Field(..., description="FMP API key")
    fmp_base_url: str = "https://financialmodelingprep.com/api/v3"

    # Universe filters
    min_market_cap: float = 2e9       # $2B
    min_adv: float = 20e6             # $20M avg daily dollar volume
    min_price: float = 10.0           # $10

    # Portfolio construction
    max_positions: int = 15
    min_positions: int = 10
    max_position_pct: float = 0.15    # 15% max single position
    max_sector_pct: float = 0.30      # 30% max sector exposure

    # Signal thresholds (percentile cutoffs)
    master_score_min_pct: float = 85.0
    earnings_revision_min_pct: float = 70.0
    momentum_min_pct: float = 70.0
    rs_sell_pct: float = 30.0         # sell if RS rank drops below this
    eps_rev_sell_pct: float = 40.0    # sell if EPS rev drops below this

    # Sell rules
    stop_loss_pct: float = 0.08       # 8% trailing stop
    profit_trim_threshold: float = 0.15  # trim at 15% gain
    max_holding_days: int = 40
    below_50dma_days: int = 2         # consecutive days below 50DMA → exit

    # Rebalancing discipline
    continuity_bonus: float = 3.0   # score-point bonus for incumbents — only displace if clearly better
    drift_threshold: float = 0.04   # min weight drift (4%) needed to trigger a rebalance trade

    # Concentrated AI regime portfolio
    concentrated_n_positions: int = 7        # top N AI names when in concentrated mode
    concentrated_max_position_pct: float = 0.22  # higher concentration allowed vs factor mode
    concentrated_stop_loss_pct: float = 0.05  # tighter stop in AI mode (5% vs 8% in factor)

    # Macro de-risking
    macro_derisking_threshold: float = 40.0  # go to cash if macro score < 40
    vix_threshold: float = 25.0

    # Email reporting (Gmail SMTP)
    # Create an App Password at: https://myaccount.google.com/apppasswords
    email_from: str = Field("", description="Gmail address to send from")
    email_to: str = Field("spvelu@gmail.com", description="Recipient email address")
    gmail_app_password: str = Field("", description="Gmail App Password (16-char, no spaces)")

    # DuckDB storage
    db_path: str = "data/market.db"

    # Signal weights — MASTER_SCORE
    w_earnings_revision: float = 0.30
    w_momentum: float = 0.25
    w_breakout: float = 0.20
    w_quality: float = 0.15
    w_macro: float = 0.10

    # Earnings revision sub-weights
    w_eps_rev_30: float = 0.50
    w_eps_rev_breadth: float = 0.30
    w_rev_rev_30: float = 0.20

    # Momentum sub-weights
    w_mom_6: float = 0.50
    w_mom_3: float = 0.30
    w_rs: float = 0.20

    # Breakout sub-weights
    w_breakout_binary: float = 0.40
    w_vol_expansion: float = 0.30
    w_dist_50dma: float = 0.30

    # Quality sub-weights
    w_roic: float = 0.30
    w_rev_growth: float = 0.30
    w_fcf_margin: float = 0.20
    w_gm_stability: float = 0.20

    # Macro sub-weights
    w_spy_trend: float = 0.40
    w_credit_trend: float = 0.30
    w_low_vol: float = 0.30


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
