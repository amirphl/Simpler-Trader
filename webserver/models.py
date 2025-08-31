from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal

from backtest.engulfing_strategy import StopLossMode
from pydantic import BaseModel, ConfigDict, Field, field_validator  # type: ignore[import-not-found]


class PinbarStrategyParams(BaseModel):
    """User-provided parameters for the Bullish Pinbar strategy."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="ETHUSDT")
    timeframe: str = Field(default="15m")
    leverage: float = Field(default=1.0, gt=0)
    take_profit_pct: float = Field(default=0.02, gt=0)

    stop_loss_mode: StopLossMode = Field(default=StopLossMode.PERCENT)
    stop_loss_pct: float = Field(default=0.005, gt=0)
    exchange_fee_pct: float = Field(default=0.0004, ge=0)

    min_shadow_body_ratio: float = Field(default=0.5, gt=0)
    shadow_dominance_ratio: float = Field(default=2.0, gt=0)

    http_proxy: str | None = None
    https_proxy: str | None = None
    risk_free_rate: float = 0.0


class EngulfingStrategyParams(BaseModel):
    """User-provided parameters for the Engulfing strategy."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="ETHUSDT", description="Trading pair to backtest")
    timeframe: str = Field(default="15m", description="Binance kline interval")
    window_size: int = Field(default=5, ge=1)
    leverage: float = Field(default=1.0, gt=0)
    take_profit_pct: float = Field(default=0.02, gt=0)
    doji_size: float = Field(default=0.05, gt=0)

    stop_loss_mode: StopLossMode = Field(default=StopLossMode.PERCENT)
    stop_loss_pct: float = Field(default=0.005, gt=0)

    skip_large_upper_wick: bool = False
    skip_bollinger_cross: bool = False
    bollinger_period: int = Field(default=20, ge=2)
    bollinger_stddev: float = Field(default=2.0, gt=0)

    enable_volume_pressure_filter: bool = True
    volume_window: int = Field(default=20, ge=2)
    max_volume_pressure_score: float = Field(default=3.0, gt=0)

    enable_stochastic_filter: bool = True
    stochastic_first_line: str = Field(default="k")
    stochastic_first_period: int = Field(default=20, ge=2)
    stochastic_first_threshold: float | None = None
    stochastic_second_line: str = Field(default="k")
    stochastic_second_period: int = Field(default=100, ge=2)
    stochastic_second_threshold: float | None = None
    stochastic_comparison: str = Field(default="gt")
    stochastic_d_smoothing: int = Field(default=3, ge=1)

    http_proxy: str | None = None
    https_proxy: str | None = None
    risk_free_rate: float = 0.0
    exchange_fee_pct: float = Field(default=0.0004, ge=0.0)

    @field_validator("stochastic_first_line", "stochastic_second_line")
    @classmethod
    def _validate_line(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"k", "d"}:
            raise ValueError("stochastic line must be 'k' or 'd'")
        return normalized

    @field_validator("stochastic_comparison")
    @classmethod
    def _validate_comparison(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"gt", "lt"}:
            raise ValueError("stochastic comparison must be 'gt' or 'lt'")
        return normalized


class BacktestSubmission(BaseModel):
    """Payload used by the web UI to request a backtest run."""

    model_config = ConfigDict(extra="ignore")

    strategy: Literal["engulfing", "pinbar"] = Field(default="engulfing")
    start: datetime
    end: datetime
    initial_capital: float = Field(default=10_000.0, gt=0)
    override_download: bool = True
    params: EngulfingStrategyParams | PinbarStrategyParams


class JobResponse(BaseModel):
    job_id: str
    status: str
    submitted_at: datetime
    completed_at: datetime | None = None
    error: str | None = None


class BacktestResultPayload(BaseModel):
    job_id: str
    status: str
    submitted_at: datetime
    completed_at: datetime | None = None
    error: str | None = None
    result: Dict[str, Any] | None = None
    request: Dict[str, Any]

