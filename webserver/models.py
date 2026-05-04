from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal

from backtest.engulfing_strategy import StopLossMode
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator  # type: ignore[import-not-found]
from typing import List


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _require_non_empty_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


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

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized = [
            _require_non_empty_text(str(symbol), "symbols entry").upper()
            for symbol in value
        ]
        if not normalized:
            raise ValueError("symbols must not be empty")
        return normalized

    @field_validator("base_timeframe", "higher_timeframe", "higher_timeframe_2", mode="before")
    @classmethod
    def _normalize_timeframe_fields(cls, value: Any) -> Any:
        if value is None:
            return None
        return _require_non_empty_text(str(value), "timeframe")

    @field_validator("martingale_multipliers", "martingale_leverages")
    @classmethod
    def _validate_non_empty_sequences(cls, value: List[float], info: Any) -> List[float]:
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value


class PinbarMagicStrategyParamsV3(BaseModel):
    """Parameters for Pin Bar Magic v3 strategy."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="ETHUSDT")
    timeframe: str = Field(default="1h")
    leverage: float = Field(default=1.0, gt=0)

    equity_risk_pct: float = Field(default=3.0, gt=0)
    atr_multiple: float = Field(default=0.5, gt=0)
    trail_points: float = Field(default=100.0, gt=0)
    trail_offset: float = Field(default=50.0, ge=0)
    symbol_mintick: float = Field(default=0.01, gt=0)

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


class StrongTrendStairParams(BaseModel):
    """Parameters for strong trend stair strategy."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="BTCUSDT")
    timeframe: str = Field(default="1m")
    leverage: float = Field(default=100.0, gt=0.0)
    position_size_pct: float = Field(default=2.0, gt=0.0)
    starting_balance_usd: float = Field(default=10000.0, gt=0.0)
    hard_stop_loss_pct: float = Field(default=5.0, gt=0.0)
    trail_start_pct: float = Field(default=2.0, gt=0.0)
    trail_offset_pct: float = Field(default=1.0, gt=0.0)
    ema_fast_len: int = Field(default=50, ge=1)
    ema_mid_len: int = Field(default=100, ge=1)
    ema_slow_len: int = Field(default=200, ge=1)
    slope_lookback: int = Field(default=10, ge=1)
    st_atr_len: int = Field(default=10, ge=1)
    st_factor: float = Field(default=3.0, gt=0.0)
    di_len: int = Field(default=14, ge=1)
    adx_smooth: int = Field(default=14, ge=1)
    adx_min: float = Field(default=20.0, ge=0.0)
    reverse_on_opposite_signal: bool = False

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


class PivotRequest(BaseModel):
    """Parameters for pivot detection experiment."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="ETHUSDT")
    timeframe: str = Field(default="1h")
    start: datetime
    end: datetime
    scan_length: int = Field(default=500, ge=1)
    source: Literal["binance", "csv"] = Field(default="binance")
    csv_path: str | None = None
    http_proxy: str | None = None
    https_proxy: str | None = None

    @field_validator("symbol", "timeframe", mode="before")
    @classmethod
    def _normalize_pivot_text_fields(cls, value: Any, info: Any) -> Any:
        return _require_non_empty_text(str(value), info.field_name) if value is not None else value

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_pivot_source(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("csv_path", "http_proxy", "https_proxy", mode="before")
    @classmethod
    def _normalize_pivot_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _normalize_optional_text(value)
        return value

    @model_validator(mode="after")
    def _validate_pivot_range(self) -> "PivotRequest":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.source == "csv" and not self.csv_path:
            raise ValueError("csv_path is required when source='csv'")
        return self


class PivotPoint(BaseModel):
    index: int
    type: Literal["bullish", "bearish"]
    high: float
    low: float
    haunted: bool
    time: datetime
    reference_index: int
    reference_time: datetime
    trigger_index: int
    trigger_time: datetime
    invalidation_index: int | None = None
    invalidation_time: datetime | None = None
    next_bullish_index: int | None = None
    next_bullish_time: datetime | None = None
    next_bearish_index: int | None = None
    next_bearish_time: datetime | None = None
    previous_bullish_index: int | None = None
    previous_bullish_time: datetime | None = None
    previous_bearish_index: int | None = None
    previous_bearish_time: datetime | None = None


class CandleForPivot(BaseModel):
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class PivotResponse(BaseModel):
    pivots: List[PivotPoint]
    candles: List[CandleForPivot]


class BOSCHoCHRequest(BaseModel):
    """Parameters for BOS/CHoCH detection experiment."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="ETHUSDT")
    timeframe: str = Field(default="1h")
    start: datetime
    end: datetime
    direction_window: int = Field(default=3, ge=1)
    hunt_mode: Literal["wick", "close"] = Field(default="wick")
    include_bos_in_choch_range: bool = False
    include_hunt_candle_in_choch_range: bool = True
    source: Literal["binance", "csv"] = Field(default="binance")
    csv_path: str | None = None
    http_proxy: str | None = None
    https_proxy: str | None = None

    @field_validator("symbol", "timeframe", mode="before")
    @classmethod
    def _normalize_bos_choch_text_fields(cls, value: Any, info: Any) -> Any:
        return _require_non_empty_text(str(value), info.field_name) if value is not None else value

    @field_validator("hunt_mode", "source", mode="before")
    @classmethod
    def _normalize_bos_choch_enums(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("csv_path", "http_proxy", "https_proxy", mode="before")
    @classmethod
    def _normalize_bos_choch_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _normalize_optional_text(value)
        return value

    @model_validator(mode="after")
    def _validate_bos_choch_request(self) -> "BOSCHoCHRequest":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.source == "csv" and not self.csv_path:
            raise ValueError("csv_path is required when source='csv'")
        return self


class BOSCHoCHMarker(BaseModel):
    type: Literal["BOS", "CHoCH"]
    index: int
    direction: Literal["UPWARD", "DOWNWARD"]
    candle_index: int
    time: datetime
    price: float
    high: float
    low: float
    label: str


class BOSCHoCHDirectionState(BaseModel):
    direction: Literal["UPWARD", "DOWNWARD"]
    since_index: int


class BOSCHoCHResponse(BaseModel):
    candles: List[CandleForPivot]
    markers: List[BOSCHoCHMarker]
    direction_state: BOSCHoCHDirectionState


class LiquidityZoneRequest(BaseModel):
    """Parameters for liquidity-zone detection experiment."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(default="ETHUSDT")
    timeframe: str = Field(default="1h")
    start: datetime
    end: datetime
    scan_length: int = Field(default=500, ge=1)
    direction_window: int = Field(default=3, ge=1)
    hunt_mode: Literal["wick", "close"] = Field(default="wick")
    include_bos_in_choch_range: bool = False
    include_hunt_candle_in_choch_range: bool = True
    up_pivot_filter: Literal["BULLISH", "BEARISH", "ALL"] = Field(default="BULLISH")
    down_pivot_filter: Literal["BULLISH", "BEARISH", "ALL"] = Field(default="BEARISH")
    include_hunted_pivots: bool = False
    representative_include_hunted: bool = False
    maximum_pivot_distance: int | None = Field(default=None, ge=1)
    minimum_overlap: float = Field(default=0.0, ge=0.0)
    minimum_overlap_ratio: float = Field(default=0.0, ge=0.0)
    allow_reuse: bool = False
    relaxed_slope: bool = False
    slope_epsilon: float = Field(default=0.0, ge=0.0)
    epsilon: float = Field(default=1e-9, gt=0.0)
    source: Literal["binance", "csv"] = Field(default="binance")
    csv_path: str | None = None
    http_proxy: str | None = None
    https_proxy: str | None = None

    @field_validator("symbol", "timeframe", mode="before")
    @classmethod
    def _normalize_liquidity_text_fields(cls, value: Any, info: Any) -> Any:
        return _require_non_empty_text(str(value), info.field_name) if value is not None else value

    @field_validator("hunt_mode", "source", mode="before")
    @classmethod
    def _normalize_liquidity_lowercase_enums(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("up_pivot_filter", "down_pivot_filter", mode="before")
    @classmethod
    def _normalize_liquidity_pivot_filters(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("csv_path", "http_proxy", "https_proxy", mode="before")
    @classmethod
    def _normalize_liquidity_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _normalize_optional_text(value)
        return value

    @model_validator(mode="after")
    def _validate_liquidity_request(self) -> "LiquidityZoneRequest":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        if self.source == "csv" and not self.csv_path:
            raise ValueError("csv_path is required when source='csv'")
        if self.minimum_overlap_ratio > 1:
            raise ValueError("minimum_overlap_ratio must be in the range [0, 1]")
        return self


class LiquidityZonePivot(BaseModel):
    index: int
    type: Literal["bullish", "bearish"]
    high: float
    low: float
    haunted: bool
    time: datetime


class LiquidityDirectionSegment(BaseModel):
    index: int
    direction: Literal["UPWARD", "DOWNWARD"]
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    pivot_count: int
    representative_pivot_index: int | None = None
    representative_pivot_time: datetime | None = None


class LiquidityZonePayload(BaseModel):
    id: str
    direction: Literal["UPWARD", "DOWNWARD"]
    level: Literal[1, 2]
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    price_low: float
    price_high: float
    is_hunted: bool
    left_pivot_index: int
    right_pivot_index: int
    left_pivot_time: datetime
    right_pivot_time: datetime
    left_pivot_type: Literal["bullish", "bearish"]
    right_pivot_type: Literal["bullish", "bearish"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LiquidityZoneResponse(BaseModel):
    candles: List[CandleForPivot]
    pivots: List[LiquidityZonePivot]
    segments: List[LiquidityDirectionSegment]
    zones: List[LiquidityZonePayload]


class BacktestSubmission(BaseModel):
    """Payload used by the web UI to request a backtest run."""

    model_config = ConfigDict(extra="ignore")

    strategy: Literal[
        "engulfing",
        "pinbar",
        "pinbar_magic_v3",
        "stochastic_fsm",
        "strong_trend_stair",
    ] = Field(default="engulfing")
    start: datetime
    end: datetime
    initial_capital: float = Field(default=10_000.0, gt=0)
    override_download: bool = True
    warmup_days: int = Field(default=30, ge=0)
    params: (
        EngulfingStrategyParams
        | PinbarStrategyParams
        | PinbarMagicStrategyParamsV3
        | StochasticFsmParams
        | StrongTrendStairParams
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
                PinbarMagicStrategyParamsV3,
                StochasticFsmParams,
                StrongTrendStairParams,
            ),
        ):
            return data
        if not isinstance(params, dict):
            return data

        strategy_map = {
            "engulfing": EngulfingStrategyParams,
            "pinbar": PinbarStrategyParams,
            "pinbar_magic_v3": PinbarMagicStrategyParamsV3,
            "stochastic_fsm": StochasticFsmParams,
            "strong_trend_stair": StrongTrendStairParams,
        }
        model = strategy_map.get(strategy)
        if model is None:
            return data
        data["params"] = model.model_validate(params)
        return data

    @model_validator(mode="after")
    def _validate_time_range(self) -> "BacktestSubmission":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self


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
