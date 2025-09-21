from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal

from backtest.engulfing_strategy import StopLossMode
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator  # type: ignore[import-not-found]
from typing import List


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


class StochasticFsmParams(BaseModel):
    """Parameters for the dual-timeframe stochastic FSM strategy."""

    model_config = ConfigDict(extra="ignore")

    symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT"])
    base_timeframe: str = Field(default="1h")
    higher_timeframe: str = Field(default="4h")
    higher_timeframe_2: str | None = Field(default=None)

    k_period: int = Field(default=8, ge=2)
    k_slowing: int = Field(default=8, ge=1)
    d_period: int = Field(default=1, ge=1)
    use_d_line: bool = False
    oversold: float = Field(default=20.0, ge=0.0, le=100.0)
    overbought: float = Field(default=80.0, ge=0.0, le=100.0)

    initial_order_usdt: float = Field(default=100.0, gt=0.0)
    initial_leverage: float = Field(default=3.0, gt=0.0)
    martingale_multiplier: float = Field(default=1.1, gt=0.0)
    martingale_multipliers: List[float] = Field(
        default_factory=lambda: [1.5, 2.0, 2.5, 3.0]
    )
    martingale_leverages: List[float] = Field(
        default_factory=lambda: [3.0, 3.0, 3.0, 3.0]
    )
    max_concurrent_positions: int = Field(default=5, ge=1)

    take_profit_pct: float = Field(default=0.02, gt=0.0)
    slippage_pct: float = Field(default=0.0002, ge=0.0)
    maker_fee_pct: float = Field(default=0.0002, ge=0.0)
    taker_fee_pct: float = Field(default=0.0006, ge=0.0)
    funding_rate_per_day_pct: float = Field(default=0.0)

    trailing_activation_pct: float = Field(default=1.5, ge=0.0)
    trailing_gap_pct: float = Field(default=1.0, ge=0.0)
    trailing_check_interval_seconds: float = Field(default=10.0, gt=0.0)
    max_position_days: float = Field(default=30.0, gt=0.0)

    margin_mode: str = Field(default="cross")
    aligned_high_stoch_mode: str = Field(default="v3", pattern="^v[123]$")
    signal_offset: int = Field(default=0, ge=0)
    enable_take_profit_check: bool = False
    enable_high_exit_cross: bool = False
    use_midsold_filter: bool = False
    enable_reversal_logic: bool = False
    enable_reversal_reentry: bool = False
    trailing_use_first_entry_price: bool = True
    trailing_use_close_for_stop_activation: bool = True
    take_profit_use_first_entry_price: bool = True
    enable_grid_martingales: bool = True
    grid_martingales_percent: float = Field(default=3.0, ge=0.0)

    http_proxy: str | None = None
    https_proxy: str | None = None
    risk_free_rate: float = 0.0


class PinbarMagicStrategyParams(BaseModel):
    """User-provided parameters for the Pin Bar Magic strategy."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="ETHUSDT")
    timeframe: str = Field(default="1h")
    leverage: float = Field(default=1.0, gt=0)

    equity_risk_pct: float = Field(default=3.0, gt=0)
    atr_multiple: float = Field(default=0.5, gt=0)
    trail_points: float = Field(default=1.0, gt=0)
    trail_offset: float = Field(default=1.0, ge=0)

    slow_sma_period: int = Field(default=50, ge=1)
    medium_ema_period: int = Field(default=18, ge=1)
    fast_ema_period: int = Field(default=6, ge=1)
    atr_period: int = Field(default=14, ge=1)
    entry_cancel_bars: int = Field(default=3, ge=1)

    http_proxy: str | None = None
    https_proxy: str | None = None
    risk_free_rate: float = 0.0


class PinbarMagicStrategyParamsV2(BaseModel):
    """Parameters for Pin Bar Magic v2 strategy."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="ETHUSDT")
    timeframe: str = Field(default="1h")
    leverage: float = Field(default=1.0, gt=0)

    equity_risk_pct: float = Field(default=3.0, gt=0)
    atr_multiple: float = Field(default=0.5, gt=0)
    trail_points: float = Field(default=1.0, gt=0)
    trail_offset: float = Field(default=1.0, ge=0)

    slow_sma_period: int = Field(default=50, ge=1)
    medium_ema_period: int = Field(default=18, ge=1)
    fast_ema_period: int = Field(default=6, ge=1)
    atr_period: int = Field(default=14, ge=1)
    entry_cancel_bars: int = Field(default=3, ge=1)
    trailing_tick_timeframe: str = Field(default="15m")
    use_trailing_tick_emulation: bool = False
    use_stop_fill_open_gap: bool = True
    entry_activation_mode: Literal["next_bar", "same_bar"] = "next_bar"
    enable_friday_close: bool = True
    friday_close_hour_utc: int = Field(default=16, ge=0, le=23)
    enable_ema_cross_close: bool = True
    risk_equity_include_unrealized: bool = True
    risk_equity_mark_source: Literal["close", "open", "hl2", "ohlc4"] = "close"

    http_proxy: str | None = None
    https_proxy: str | None = None
    risk_free_rate: float = 0.0

    @field_validator("entry_activation_mode", mode="before")
    @classmethod
    def _normalize_entry_activation_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("risk_equity_mark_source", mode="before")
    @classmethod
    def _normalize_risk_equity_mark_source(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


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

    strategy: Literal[
        "engulfing",
        "pinbar",
        "pinbar_magic",
        "pinbar_magic_v2",
        "stochastic_fsm",
    ] = Field(default="engulfing")
    start: datetime
    end: datetime
    initial_capital: float = Field(default=10_000.0, gt=0)
    override_download: bool = True
    warmup_days: int = Field(default=30, ge=0)
    params: (
        EngulfingStrategyParams
        | PinbarStrategyParams
        | PinbarMagicStrategyParams
        | PinbarMagicStrategyParamsV2
        | StochasticFsmParams
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_params(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        params = data.get("params")
        strategy = data.get("strategy")
        if params is None or strategy is None:
            return data
        if isinstance(
            params,
            (
                EngulfingStrategyParams,
                PinbarStrategyParams,
                PinbarMagicStrategyParams,
                PinbarMagicStrategyParamsV2,
                StochasticFsmParams,
            ),
        ):
            return data
        if not isinstance(params, dict):
            return data

        strategy_map = {
            "engulfing": EngulfingStrategyParams,
            "pinbar": PinbarStrategyParams,
            "pinbar_magic": PinbarMagicStrategyParams,
            "pinbar_magic_v2": PinbarMagicStrategyParamsV2,
            "stochastic_fsm": StochasticFsmParams,
        }
        model = strategy_map.get(strategy)
        if model is None:
            return data
        data["params"] = model.model_validate(params)
        return data


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
