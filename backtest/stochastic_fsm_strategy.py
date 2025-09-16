from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any

from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import Candle, to_milliseconds, to_datetime

from .base import BacktestContext, BacktestStrategy, TradePerformance


class PositionDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class StochasticRsiFsmConfig:
    """Configuration for the FSM-based multi-timeframe stochastic strategy."""

    symbols: Sequence[str]
    tf_1: str = "1h"
    tf_2: str = "4h"
    tf_3: Optional[str] = None

    k_period: int = 8
    k_slowing: int = 1
    d_period: int = 1
    use_d_line: bool = False
    oversold: float = 20.0
    overbought: float = 80.0

    initial_order_usdt: float = 100.0
    initial_leverage: float = 1.0
    martingale_multiplier: Optional[float] = 1.1
    martingale_multipliers: Sequence[float] = (1.5, 2.25, 3.375, 5.0625)
    martingale_leverages: Sequence[float] = (2.0, 3.0, 4.0, 5.0)
    max_martingale_steps: int = field(init=False)
    max_concurrent_positions: int = 5

    take_profit_pct: float = 0.02
    slippage_pct: float = 0.0002  # 0.02%
    maker_fee_pct: float = 0.0002  # 0.02%
    taker_fee_pct: float = 0.0006  # 0.06%
    funding_rate_per_day_pct: float = 0.0
    apply_funding: bool = False

    trailing_activation_pct: float = 1.5
    trailing_gap_pct: float = 1.0
    trailing_check_interval_seconds: float = 10.0
    max_position_days: Optional[float] = 30.0
    signal_offset: int = 0
    enable_grid_martingales: bool = True
    grid_martingales_percent: float = 3.0

    margin_mode: str = "cross"
    aligned_high_stoch_mode: str = "v3"  # one of: "v1", "v2", "v3"
    enable_take_profit_check: bool = False
    enable_high_exit_cross: bool = False
    use_midsold_filter: bool = False
    enable_reversal_logic: bool = False
    enable_reversal_reentry: bool = False
    trailing_use_first_entry_price: bool = True
    trailing_use_close_for_stop_activation: bool = True
    take_profit_use_first_entry_price: bool = True

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        if self.k_period <= 1:
            raise ValueError("k_period must be greater than 1")
        if self.k_slowing <= 0:
            raise ValueError("k_slowing must be positive")
        if self.d_period <= 0:
            raise ValueError("d_period must be positive")
        if self.initial_order_usdt <= 0:
            raise ValueError("initial_order_usdt must be positive")
        if self.initial_leverage <= 0:
            raise ValueError("initial_leverage must be positive")
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.overbought <= self.oversold:
            raise ValueError("overbought must be greater than oversold")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        if self.maker_fee_pct < 0 or self.taker_fee_pct < 0:
            raise ValueError("fees must be non-negative")
        if self.martingale_multiplier is not None and self.martingale_multiplier <= 0:
            raise ValueError("martingale_multiplier must be positive when provided")
        if self.trailing_activation_pct < 0 or self.trailing_gap_pct < 0:
            raise ValueError("trailing percentages must be non-negative")
        if self.trailing_check_interval_seconds <= 0:
            raise ValueError("trailing_check_interval_seconds must be positive")
        if self.max_position_days is not None and self.max_position_days <= 0:
            raise ValueError("max_position_days must be positive when provided")
        if self.signal_offset < 0:
            raise ValueError("signal_offset must be non-negative")
        if self.grid_martingales_percent < 0:
            raise ValueError("grid_martingales_percent must be non-negative")
        if self.aligned_high_stoch_mode not in {"v1", "v2", "v3"}:
            raise ValueError("aligned_high_stoch_mode must be one of: v1, v2, v3")
        # Validate intervals early to fail fast on typos.
        interval_to_milliseconds(self.tf_1)
        interval_to_milliseconds(self.tf_2)
        if self.tf_3:
            interval_to_milliseconds(self.tf_3)
        if self.martingale_multiplier is not None:
            steps = len(self.martingale_leverages)
        else:
            steps = min(
                len(self.martingale_leverages), len(self.martingale_multipliers)
            )
        if steps < 0:
            raise ValueError("computed max_martingale_steps must be non-negative")
        object.__setattr__(self, "max_martingale_steps", steps)


@dataclass
class OpenPositionState:
    symbol: str
    side: PositionDirection
    entry_time: datetime
    first_entry_price: float
    avg_entry_price: float
    quantity: float
    notional: float
    margin: float
    take_profit: float
    leverage: float
    last_add_price: float
    martingale_step: int = 0
    trailing_stop: Optional[float] = None
    trailing_active: bool = False
    liquidation_price: float = 0.0
    margin_call_price: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    last_update_time: Optional[datetime] = None
    notes: str = ""
    metadata: Dict[Any, Any] = None

    def effective_leverage(self) -> float:
        if self.margin <= 0:
            return 0.0
        return self.notional / self.margin

    def to_dict(self) -> Dict[str, float | int | str | None | Dict | List]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_time": self.entry_time.isoformat(),
            "first_entry_price": self.first_entry_price,
            "avg_entry_price": self.avg_entry_price,
            "quantity": self.quantity,
            "notional": self.notional,
            "margin": self.margin,
            "take_profit": self.take_profit,
            "leverage": self.leverage,
            "last_add_price": self.last_add_price,
            "martingale_step": self.martingale_step,
            "trailing_stop": self.trailing_stop,
            "trailing_active": self.trailing_active,
            "liquidation_price": self.liquidation_price,
            "margin_call_price": self.margin_call_price,
            "fees_paid": self.fees_paid,
            "funding_paid": self.funding_paid,
            "last_update_time": self.last_update_time.isoformat()
            if self.last_update_time
            else None,
            "notes": self.notes,
            # TODO:
            # "metadata": self.metadata,
        }


def _sma(values: Sequence[Optional[float]], length: int, index: int) -> Optional[float]:
    if length <= 0 or index < length - 1:
        return None
    window = [v for v in values[index - length + 1 : index + 1] if v is not None]
    if len(window) < length:
        return None
    return sum(window) / float(length)


def _rsi(closes: Sequence[float], period: int) -> List[Optional[float]]:
    """Standard Wilder RSI, returning None until enough history is available."""
    if period <= 0:
        raise ValueError("period must be positive")
    n = len(closes)
    result: List[Optional[float]] = [None] * n
    if n < period + 1:
        return result

    gains: List[float] = [0.0] * n
    losses: List[float] = [0.0] * n
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = -change

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


def _stochastic_rsi(
    closes: Sequence[float],
    *,
    rsi_period: int,
    stoch_period: int,
    k_smoothing: int,
    d_period: int,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Full Stochastic RSI (%K and %D)."""
    rsi_vals = _rsi(closes, rsi_period)
    raw_k: List[Optional[float]] = [None] * len(closes)
    slow_k: List[Optional[float]] = [None] * len(closes)
    slow_d: List[Optional[float]] = [None] * len(closes)

    for idx in range(len(closes)):
        rsi_val = rsi_vals[idx]
        if rsi_val is None or idx < stoch_period - 1:
            continue
        window = rsi_vals[idx - stoch_period + 1 : idx + 1]
        if any(v is None for v in window):
            continue
        highest = max(window)  # type: ignore[arg-type]
        lowest = min(window)  # type: ignore[arg-type]
        if highest == lowest:
            raw_k[idx] = 50.0
        else:
            raw_k[idx] = ((rsi_val - lowest) / (highest - lowest)) * 100.0
        slow_k[idx] = _sma(raw_k, k_smoothing, idx) if k_smoothing > 1 else raw_k[idx]
        slow_d[idx] = _sma(slow_k, d_period, idx) if d_period > 1 else slow_k[idx]

    return slow_k, slow_d


def _select_stochastic_rsi_line(
    k_values: Sequence[Optional[float]],
    d_values: Sequence[Optional[float]],
    use_d_line: bool,
) -> List[Optional[float]]:
    return [d if use_d_line else k for k, d in zip(k_values, d_values)]


def _aligned_higher_stoch_rsi_for_base(
    *,
    tf_1_candles: Sequence[Candle],
    tf_2_candles: Sequence[Candle],
    tf_2_ind: Sequence[Optional[float]],
    tf_2: str,
    cfg: StochasticRsiFsmConfig,
) -> Tuple[List[Optional[float]], List[Optional[Candle]]]:
    """Align higher-timeframe StochRSI to each base candle close.

    Behavior:
    - If the higher candle that contains the base candle is already closed, use
      the precomputed (closed) StochRSI value from `closed_high_series`.
    - Otherwise, compute a provisional StochRSI by taking all closed higher
      closes plus the current base close as the in-flight higher close.
    - Also returns a provisional higher candle built by aggregating base OHLCV
      within the current higher bucket (useful for debugging/plotting).

    Returns:
    - aligned_values: per-base StochRSI values (may be None until enough data).
    - aligned_candles: higher candles aligned to each base close (closed or provisional).
    """
    if not tf_2_candles:
        return [None] * len(tf_1_candles), [None] * len(tf_1_candles)

    high_ms = interval_to_milliseconds(tf_2)
    anchor_ms = tf_2_candles[0].open_time_ms
    high_closes = [c.close for c in tf_2_candles]
    high_open_to_index = {c.open_time_ms: idx for idx, c in enumerate(tf_2_candles)}
    high_open_to_close_ms = {
        c.open_time_ms: to_milliseconds(c.close_time) for c in tf_2_candles
    }

    aligned_values: List[Optional[float]] = [None] * len(tf_1_candles)
    aligned_candles: List[Optional[Candle]] = [None] * len(tf_1_candles)
    current_bucket_start_ms: Optional[int] = None
    agg_open = agg_high = agg_low = agg_close = agg_volume = None  # type: ignore[assignment]
    for idx, candle in enumerate(tf_1_candles):
        base_close_ms = to_milliseconds(candle.close_time)
        bucket_offset = max(0, base_close_ms - anchor_ms)
        bucket_start_ms = anchor_ms + (bucket_offset // high_ms) * high_ms
        bucket_changed = bucket_start_ms != current_bucket_start_ms

        if bucket_changed:
            current_bucket_start_ms = bucket_start_ms
            agg_open = candle.open
            agg_high = candle.high
            agg_low = candle.low
            agg_volume = candle.volume
        else:
            agg_high = max(agg_high, candle.high)  # type: ignore[arg-type]
            agg_low = min(agg_low, candle.low)  # type: ignore[arg-type]
            agg_volume = agg_volume + candle.volume  # type: ignore[operator]
        agg_close = candle.close

        high_idx = high_open_to_index.get(bucket_start_ms)

        # If the higher candle is already closed, reuse the finished value.
        if high_idx is not None:
            high_close_ms = high_open_to_close_ms[bucket_start_ms]
            if base_close_ms >= high_close_ms:
                aligned_values[idx] = tf_2_ind[high_idx]
                aligned_candles[idx] = tf_2_candles[high_idx]
                continue
            closed_closes = high_closes[:high_idx]
        else:
            closed_closes = list(high_closes)

        provisional_close = candle.close
        provisional_closes = closed_closes + [provisional_close]
        k_vals, d_vals = _stochastic_rsi(
            provisional_closes,
            rsi_period=cfg.k_period,
            stoch_period=cfg.k_period,
            k_smoothing=cfg.k_slowing,
            d_period=cfg.d_period,
        )
        provisional_series = _select_stochastic_rsi_line(k_vals, d_vals, cfg.use_d_line)
        aligned_values[idx] = provisional_series[-1]
        aligned_candles[idx] = Candle(
            symbol=candle.symbol,
            interval=tf_2,
            open_time=to_datetime(bucket_start_ms),
            close_time=candle.close_time,
            open=agg_open,  # type: ignore[arg-type]
            high=agg_high,  # type: ignore[arg-type]
            low=agg_low,  # type: ignore[arg-type]
            close=agg_close,  # type: ignore[arg-type]
            volume=agg_volume,  # type: ignore[arg-type]
        )

    return aligned_values, aligned_candles


def _aligned_higher_stoch_rsi_for_base_v2(
    *,
    tf_1_candles: Sequence[Candle],
    tf_2_candles: Sequence[Candle],
    tf_2_ind: Sequence[Optional[float]],
    tf_2: str,
    cfg: StochasticRsiFsmConfig,
) -> Tuple[List[Optional[float]], List[Optional[Candle]]]:
    """Align higher-timeframe StochRSI to base candles using carry-forward only.

    Behavior:
    - Tracks the latest CLOSED higher candle for each base close.
    - Returns that last-closed StochRSI value for every base candle in the
      current higher bucket (no provisional/in-flight calculation).

    This is the most conservative alignment: it never uses in-progress data.
    """
    if not tf_2_candles:
        return [None] * len(tf_1_candles), [None] * len(tf_1_candles)

    high_ms = interval_to_milliseconds(tf_2)
    anchor_ms = tf_2_candles[0].open_time_ms
    high_open_to_index = {c.open_time_ms: idx for idx, c in enumerate(tf_2_candles)}
    high_open_to_close_ms = {
        c.open_time_ms: to_milliseconds(c.close_time) for c in tf_2_candles
    }

    aligned_values: List[Optional[float]] = [None] * len(tf_1_candles)
    aligned_candles: List[Optional[Candle]] = [None] * len(tf_1_candles)
    last_closed_value: Optional[float] = None

    for idx, candle in enumerate(tf_1_candles):
        base_close_ms = to_milliseconds(candle.close_time)
        bucket_offset = max(0, base_close_ms - anchor_ms)
        bucket_start_ms = anchor_ms + (bucket_offset // high_ms) * high_ms
        high_idx = high_open_to_index.get(bucket_start_ms)

        if high_idx is not None:
            aligned_candles[idx] = tf_2_candles[high_idx]
            high_close_ms = high_open_to_close_ms[bucket_start_ms]
            if base_close_ms >= high_close_ms:
                last_closed_value = tf_2_ind[high_idx]
        aligned_values[idx] = last_closed_value

    return aligned_values, aligned_candles


def _aligned_higher_stoch_rsi_for_base_v3(
    *,
    tf_1_candles: Sequence[Candle],  # 1h candles
    tf_2_candles: Sequence[
        Candle
    ],  # 4h candles (may include in-flight or only closed; both OK)
    tf_2_ind: Sequence[Optional[float]],  # StochRSI computed on CLOSED 4h candles
    tf_2: str,
    cfg: StochasticRsiFsmConfig,
) -> Tuple[List[Optional[float]], List[Optional[Candle]]]:
    """Align higher-timeframe StochRSI to each base candle close (hybrid mode).

    Behavior:
    - If the higher candle that just closed ends exactly at the current base
      close, use its CLOSED StochRSI value.
    - Otherwise, compute a provisional StochRSI using:
      (all CLOSED higher closes) + (current base close as the in-flight higher close).
    - Returns a provisional higher candle (aggregated from base OHLCV) for
      plotting/inspection alongside the aligned values.

    This mode blends closed-candle correctness with in-flight responsiveness.
    """
    n = len(tf_1_candles)
    if not tf_2_candles:
        return [None] * n, [None] * n

    high_ms = interval_to_milliseconds(tf_2)
    anchor_ms = tf_2_candles[0].open_time_ms

    # Precompute close times for pointer walk over higher candles
    higher_close_ms: List[int] = [to_milliseconds(c.close_time) for c in tf_2_candles]
    higher_closes: List[float] = [c.close for c in tf_2_candles]

    aligned_values: List[Optional[float]] = [None] * n
    aligned_candles: List[Optional[Candle]] = [None] * n

    # Pointer: number of higher candles that are CLOSED as-of current base_close
    # closed_count means indices [0 .. closed_count-1] are closed
    closed_count = 0

    current_bucket_start_ms: Optional[int] = None
    agg_open = agg_high = agg_low = agg_close = agg_volume = None  # type: ignore[assignment]

    for i, bc in enumerate(tf_1_candles):
        base_close_ms = to_milliseconds(bc.close_time)

        # Advance closed_count while higher candle close <= base_close
        while (
            closed_count < len(tf_2_candles)
            and higher_close_ms[closed_count] <= base_close_ms
        ):
            closed_count += 1

        # Identify current 4h bucket start based on anchor/grid
        bucket_offset = max(0, base_close_ms - anchor_ms)
        bucket_start_ms = anchor_ms + (bucket_offset // high_ms) * high_ms
        bucket_changed = bucket_start_ms != current_bucket_start_ms

        # Aggregate base candles inside this bucket (for provisional candle construction)
        if bucket_changed:
            current_bucket_start_ms = bucket_start_ms
            agg_open = bc.open
            agg_high = bc.high
            agg_low = bc.low
            agg_volume = bc.volume
        else:
            agg_high = max(agg_high, bc.high)  # type: ignore[arg-type]
            agg_low = min(agg_low, bc.low)  # type: ignore[arg-type]
            agg_volume = agg_volume + bc.volume  # type: ignore[operator]
        agg_close = bc.close

        # If there is a closed 4h candle whose close == base_close_ms, use its CLOSED value.
        # The most recent closed candle index is closed_count-1 (if closed_count>0).
        if closed_count > 0:
            last_closed_idx = closed_count - 1
            # If this base close is at/after the close of that candle AND that candle is the bucket that ended now,
            # we should use the closed series value directly. In general, if base_close_ms >= that close,
            # the last_closed_idx is correct to use.
            if base_close_ms >= higher_close_ms[last_closed_idx]:
                # If we're exactly at a higher close boundary, using the closed series is desired (b2 at 03:59).
                # Even if we're later than the close, last_closed_idx is still "last closed candle".
                # But we only want to "lock in" the closed value when the current higher candle is indeed closed.
                # That is already ensured by how last_closed_idx is defined.
                # If you ALWAYS want last-closed value (carry-forward), you can return it here.
                # For your requested behavior: at 03:59 we must use b2 (which is last_closed_idx then).
                if base_close_ms == higher_close_ms[last_closed_idx]:
                    aligned_values[i] = tf_2_ind[last_closed_idx]
                    aligned_candles[i] = tf_2_candles[last_closed_idx]
                    continue

        # Otherwise compute provisional:
        # - closed higher closes up to last CLOSED candle (closed_count items)
        # - plus one provisional close (current base close, representing in-flight higher candle close so far)
        closed_closes = higher_closes[:closed_count]
        provisional_closes = closed_closes + [bc.close]

        k_vals, d_vals = _stochastic_rsi(
            provisional_closes,
            rsi_period=cfg.k_period,
            stoch_period=cfg.k_period,
            k_smoothing=cfg.k_slowing,
            d_period=cfg.d_period,
        )
        provisional_series = _select_stochastic_rsi_line(k_vals, d_vals, cfg.use_d_line)

        aligned_values[i] = provisional_series[-1]
        aligned_candles[i] = Candle(
            symbol=bc.symbol,
            interval=cfg.tf_2,
            open_time=to_datetime(bucket_start_ms),
            close_time=bc.close_time,  # in-flight close "so far"
            open=agg_open,  # type: ignore[arg-type]
            high=agg_high,  # type: ignore[arg-type]
            low=agg_low,  # type: ignore[arg-type]
            close=agg_close,  # type: ignore[arg-type]
            volume=agg_volume,  # type: ignore[arg-type]
        )

    return aligned_values, aligned_candles


def _apply_slippage(
    price: float, side: PositionDirection, is_entry: bool, slippage_pct: float
) -> float:
    if slippage_pct <= 0:
        return price
    direction = (
        1.0
        if (side is PositionDirection.LONG and is_entry)
        or (side is PositionDirection.SHORT and not is_entry)
        else -1.0
    )
    return price * (1.0 + direction * slippage_pct)


def _calc_exit_price_with_slippage(
    price: float, side: PositionDirection, slippage_pct: float
) -> float:
    return _apply_slippage(price, side, False, slippage_pct)


def _calc_fee(notional: float, fee_pct: float) -> float:
    return abs(notional) * fee_pct


def _calc_risk_prices(
    avg_entry_price: float,
    margin: float,
    quantity: float,
    side: PositionDirection,
) -> Tuple[float, float]:
    """Return (liquidation_price, margin_call_price)."""
    if avg_entry_price <= 0 or margin <= 0 or quantity <= 0:
        return 0.0, 0.0
    move = margin / quantity
    call_move = move * 0.8
    if side is PositionDirection.LONG:
        liquidation = avg_entry_price - move
        margin_call = avg_entry_price - call_move
    else:
        liquidation = avg_entry_price + move
        margin_call = avg_entry_price + call_move
    return max(liquidation, 0.0), max(margin_call, 0.0)


class StochasticRsiFsmStrategy(BacktestStrategy):
    """Finite-state-machine strategy driven by dual-timeframe stochastic rsi signals."""

    def __init__(self, config: StochasticRsiFsmConfig) -> None:
        self._config = config
        self._log = logging.getLogger(self.__class__.__name__)

    def name(self) -> str:
        return "StochasticRsiFsmStrategy"

    def symbols(self) -> Sequence[str]:
        return self._config.symbols

    def timeframes(self) -> Sequence[str]:
        tfs = [self._config.tf_1, self._config.tf_2]
        if self._config.tf_3:
            tfs.append(self._config.tf_3)
        return tfs

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        cfg = self._config
        trades: List[TradePerformance] = []

        tf_1_candles: Dict[str, Sequence[Candle]] = {}
        tf_2_candles: Dict[str, Sequence[Candle]] = {}
        tf_3_candles: Dict[str, Sequence[Candle]] = {}

        for symbol in cfg.symbols:
            c1 = context.data.get(symbol, {}).get(cfg.tf_1, [])
            c2 = context.data.get(symbol, {}).get(cfg.tf_2, [])
            c3 = context.data.get(symbol, {}).get(cfg.tf_3, []) if cfg.tf_3 else []
            if not c1 or not c2:
                self._log.warning(
                    "Missing candles for %s (base or primary higher timeframe)", symbol
                )
                continue
            tf_1_candles[symbol] = c1
            tf_2_candles[symbol] = c2
            tf_3_candles[symbol] = c3

        if not tf_1_candles:
            return trades, None

        tf_1_ind: Dict[str, List[Optional[float]]] = {}
        tf_2_ind: Dict[str, List[Optional[float]]] = {}
        tf_3_ind: Dict[str, List[Optional[float]]] = {}
        al_tf_2_ind: Dict[str, List[Optional[float]]] = {}
        al_tf_2_candles: Dict[str, List[Optional[Candle]]] = {}
        al_tf_3_ind: Dict[str, List[Optional[float]]] = {}
        al_tf_3_candles: Dict[str, List[Optional[Candle]]] = {}

        for symbol, candles in tf_1_candles.items():
            closes = [c.close for c in candles]
            k_vals, d_vals = _stochastic_rsi(
                closes,
                rsi_period=cfg.k_period,
                stoch_period=cfg.k_period,
                k_smoothing=cfg.k_slowing,
                d_period=cfg.d_period,
            )
            tf_1_ind[symbol] = _select_stochastic_rsi_line(
                k_vals, d_vals, cfg.use_d_line
            )

        for symbol, candles in tf_2_candles.items():
            closes = [c.close for c in candles]
            k_vals, d_vals = _stochastic_rsi(
                closes,
                rsi_period=cfg.k_period,
                stoch_period=cfg.k_period,
                k_smoothing=cfg.k_slowing,
                d_period=cfg.d_period,
            )
            tf_2_ind[symbol] = _select_stochastic_rsi_line(
                k_vals, d_vals, cfg.use_d_line
            )

        if cfg.tf_3:
            for symbol, candles in tf_3_candles.items():
                if not candles:
                    continue
                closes = [c.close for c in candles]
                k_vals, d_vals = _stochastic_rsi(
                    closes,
                    rsi_period=cfg.k_period,
                    stoch_period=cfg.k_period,
                    k_smoothing=cfg.k_slowing,
                    d_period=cfg.d_period,
                )
                tf_3_ind[symbol] = _select_stochastic_rsi_line(
                    k_vals, d_vals, cfg.use_d_line
                )

        for symbol, candles in tf_1_candles.items():
            # NOTE:
            if cfg.aligned_high_stoch_mode == "v1":
                al_ind, al_candles = _aligned_higher_stoch_rsi_for_base(
                    tf_1_candles=candles,
                    tf_2_candles=tf_2_candles[symbol],
                    tf_2_ind=tf_2_ind[symbol],
                    tf_2=cfg.tf_2,
                    cfg=cfg,
                )
            elif cfg.aligned_high_stoch_mode == "v2":
                al_ind, al_candles = _aligned_higher_stoch_rsi_for_base_v2(
                    tf_1_candles=candles,
                    tf_2_candles=tf_2_candles[symbol],
                    tf_2_ind=tf_2_ind[symbol],
                    tf_2=cfg.tf_2,
                    cfg=cfg,
                )
            else:
                al_ind, al_candles = _aligned_higher_stoch_rsi_for_base_v3(
                    tf_1_candles=candles,
                    tf_2_candles=tf_2_candles[symbol],
                    tf_2_ind=tf_2_ind[symbol],
                    tf_2=cfg.tf_2,
                    cfg=cfg,
                )
            al_tf_2_ind[symbol] = al_ind
            al_tf_2_candles[symbol] = al_candles

            if cfg.tf_3 and symbol in tf_3_ind:
                if cfg.aligned_high_stoch_mode == "v1":
                    al_ind, al_candles = _aligned_higher_stoch_rsi_for_base(
                        tf_1_candles=candles,
                        tf_2_candles=tf_3_candles.get(symbol, []),
                        tf_2_ind=tf_3_ind[symbol],
                        tf_2=cfg.tf_3,
                        cfg=cfg,
                    )
                elif cfg.aligned_high_stoch_mode == "v2":
                    al_ind, al_candles = _aligned_higher_stoch_rsi_for_base_v2(
                        tf_1_candles=candles,
                        tf_2_candles=tf_3_candles.get(symbol, []),
                        tf_2_ind=tf_3_ind[symbol],
                        tf_2=cfg.tf_3,
                        cfg=cfg,
                    )
                else:
                    al_ind, al_candles = _aligned_higher_stoch_rsi_for_base_v3(
                        tf_1_candles=candles,
                        tf_2_candles=tf_3_candles.get(symbol, []),
                        tf_2_ind=tf_3_ind[symbol],
                        tf_2=cfg.tf_3,
                        cfg=cfg,
                    )
                al_tf_3_ind[symbol] = al_ind
                al_tf_3_candles[symbol] = al_candles

        # def _format_zipped(
        #     base_times: Sequence[datetime],
        #     base_vals: Sequence[Optional[float]],
        #     aligned_vals: Sequence[Optional[float]],
        # ) -> str:
        #     rows: List[str] = []
        #     for bt, bv, hv in zip(base_times, base_vals, aligned_vals):
        #         if bv is None or hv is None:
        #             continue
        #         rows.append(
        #             f"({bt.strftime('%m-%d %H')}, base={bv:.4f}; high={hv:.4f})"
        #         )
        #     return "[" + ", ".join(rows) + "]"

        # for symbol in tf_1_candles_by_symbol:
        #     try:
        #         tf_1_times = [c.open_time for c in tf_1_candles_by_symbol[symbol]]
        #         tf_2_times = [c.open_time for c in tf_2_candles_by_symbol[symbol]]
        #         aligned_tf_2_times = [c.open_time for c in aligned_tf_2_candles[symbol]]
        #         tf_1_formatted = self._format_series(tf_1_times, tf_1_stoch_rsi[symbol])
        #         tf_2_formatted = self._format_series(tf_2_times, tf_2_stoch_rsi[symbol])
        #         aligned_tf_2_formatted = self._format_series(
        #             aligned_tf_2_times, aligned_tf_2_stoch_rsi[symbol]
        #         )
        #         zipped_formatted = _format_zipped(
        #             tf_1_times,
        #             tf_1_stoch_rsi[symbol],
        #             aligned_tf_2_stoch_rsi[symbol],
        #         )
        #         self._log.info(
        #             "StochRSI series computed\n\n\nBase (%s): %s\nHigher (%s): %s\n\n\nAligned Higher (%s): %s\n\n\nZipped (base + aligned higher): %s\n\n\n",
        #             cfg.tf_1,
        #             tf_1_formatted,
        #             cfg.tf_2,
        #             tf_2_formatted,
        #             cfg.tf_2,
        #             aligned_tf_2_formatted,
        #             zipped_formatted,
        #         )
        #         self._log.info(
        #             "StochRSI series computed\n\n\nZipped (base + aligned higher): %s\n\n\n",
        #             zipped_formatted,
        #         )
        #     except Exception as exc:  # pragma: no cover - defensive logging
        #         self._log.warning(
        #             "Failed to pretty print StochRSI series for %s: %s", symbol, exc
        #         )

        states: Dict[str, Optional[OpenPositionState]] = {s: None for s in tf_1_candles}
        high_ptrs: Dict[str, Tuple[int, Optional[float], Optional[float]]] = {}
        last_symbol_time: Dict[str, datetime] = {}
        for symbol in tf_1_candles:
            high_ptrs[symbol] = (-1, None, None)

        events: List[Tuple[datetime, str, int]] = []
        for symbol, candles in tf_1_candles.items():
            skip = (
                context.ignore_candles.get(symbol, {}).get(cfg.tf_1, 0)
                if hasattr(context, "ignore_candles")
                else 0
            )
            events.extend(
                (candle.close_time, symbol, idx)
                for idx, candle in enumerate(candles)
                if idx >= skip
            )
        events.sort(key=lambda tpl: (tpl[0], tpl[1]))

        open_positions_count = 0

        for _, symbol, idx in events:
            candle = tf_1_candles[symbol][idx]
            ind = tf_1_ind[symbol][idx]
            prev_ind = tf_1_ind[symbol][idx - 1] if idx > 0 else None
            # high_idx, prev_high, current_high = high_ptrs[symbol]
            # high_candles = higher_candles_by_symbol[symbol]
            # high_values = high_stoch_rsi[symbol]
            # high_closed = False
            # while (
            #     high_idx + 1 < len(high_candles)
            #     and high_candles[high_idx + 1].close_time <= candle.close_time # TODO: Millisecond resolution might cause trouble.
            # ):
            #     high_idx += 1
            #     prev_high = current_high
            #     current_high = high_values[high_idx]
            #     high_closed = True
            # high_ptrs[symbol] = (high_idx, prev_high, current_high)

            position = states[symbol]

            # Funding accrual based on time elapsed since last update for this symbol.
            last_seen = last_symbol_time.get(symbol)
            if cfg.apply_funding and position is not None and last_seen is not None:
                delta_seconds = (candle.close_time - last_seen).total_seconds()
                funding_rate_per_sec = (cfg.funding_rate_per_day_pct / 100.0) / 86_400.0
                funding = position.notional * funding_rate_per_sec * delta_seconds
                position.funding_paid += funding
            last_symbol_time[symbol] = candle.close_time

            al_ind2 = al_tf_2_ind[symbol][idx]
            prev_al_ind2 = al_tf_2_ind[symbol][idx - 1] if idx > 0 else None

            al_ind3 = None
            prev_al_ind3 = None

            if al_tf_3_ind.get(symbol):
                al_ind3 = al_tf_3_ind[symbol][idx]
                prev_al_ind3 = al_tf_3_ind[symbol][idx - 1] if idx > 0 else None

            exit_metadata = {
                "exit[candle]": str(candle),
                "exit[ind]": ind,
                "exit[prev_ind]": prev_ind,
                "exit[al_ind2]": al_ind2,
                "exit[prev_al_ind2]": prev_al_ind2,
                "exit[al_ind3]": al_ind3,
                "exit[prev_al_ind3]": prev_al_ind3,
            }

            if position is not None:
                if cfg.max_position_days is not None:
                    held_seconds = (
                        candle.close_time - position.entry_time
                    ).total_seconds()
                    if held_seconds >= cfg.max_position_days * 86_400:
                        timed_meta = {
                            "max_position_days": cfg.max_position_days,
                            "holding_days": held_seconds / 86_400,
                        }
                        timed_meta.update(exit_metadata)
                        exit_price = _calc_exit_price_with_slippage(
                            candle.close, position.side, cfg.slippage_pct
                        )
                        trades.append(
                            self._close_position(
                                position=position,
                                exit_price=exit_price,
                                exit_time=candle.close_time,
                                reason="Max holding period",
                                use_taker_fee=True,
                                cfg=cfg,
                                metadata=timed_meta,
                            )
                        )
                        states[symbol] = None
                        open_positions_count = max(0, open_positions_count - 1)
                        continue
                liq_hit, liq_meta = self._check_liquidation(
                    position,
                    candle,
                )
                if liq_meta:
                    liq_meta.update(exit_metadata)
                if liq_hit:
                    exit_price = max(position.liquidation_price, 0.0)
                    exit_price = _calc_exit_price_with_slippage(
                        exit_price, position.side, cfg.slippage_pct
                    )
                    trades.append(
                        self._close_position(
                            position=position,
                            exit_price=exit_price,
                            exit_time=candle.close_time,
                            reason="Liquidation",
                            use_taker_fee=True,
                            cfg=cfg,
                            metadata=liq_meta,
                        )
                    )
                    states[symbol] = None
                    open_positions_count = max(0, open_positions_count - 1)
                    continue

                trailing_exit = self._maybe_update_trailing(
                    position,
                    candle,
                    cfg,
                )
                if trailing_exit is not None:
                    exit_price, note, trail_meta = trailing_exit
                    trail_meta.update(exit_metadata)
                    trades.append(
                        self._close_position(
                            position=position,
                            exit_price=exit_price,
                            exit_time=candle.close_time,
                            reason=note,
                            use_taker_fee=True,
                            cfg=cfg,
                            metadata=trail_meta,
                        )
                    )
                    states[symbol] = None
                    open_positions_count = max(0, open_positions_count - 1)
                    continue

                if cfg.enable_take_profit_check:
                    tp_hit, tp_meta = self._check_take_profit(
                        position,
                        candle,
                    )
                    if tp_meta:
                        tp_meta.update(exit_metadata)
                    if tp_hit is not None:
                        exit_price = _calc_exit_price_with_slippage(
                            tp_hit, position.side, cfg.slippage_pct
                        )
                        trades.append(
                            self._close_position(
                                position=position,
                                exit_price=exit_price,
                                exit_time=candle.close_time,
                                reason="Take profit",
                                use_taker_fee=False,
                                cfg=cfg,
                                metadata=tp_meta,
                            )
                        )
                        states[symbol] = None
                        open_positions_count = max(0, open_positions_count - 1)
                        continue

                if cfg.enable_high_exit_cross:
                    if (
                        prev_al_ind2 is not None
                        and al_ind2 is not None
                        and self._crossed_opposite(
                            position.side,
                            prev_al_ind2,
                            al_ind2,
                            cfg,
                        )
                    ):
                        exit_price = _calc_exit_price_with_slippage(
                            candle.close, position.side, cfg.slippage_pct
                        )
                        trades.append(
                            self._close_position(
                                position=position,
                                exit_price=exit_price,
                                exit_time=candle.close_time,
                                reason="Higher timeframe exit cross",
                                use_taker_fee=True,
                                cfg=cfg,
                                metadata=exit_metadata,
                            )
                        )
                        states[symbol] = None
                        open_positions_count = max(0, open_positions_count - 1)
                        position = None

            # Skip until all stochastic RSI series have valid values.
            if (
                ind is None
                or prev_ind is None
                or al_ind2 is None
                or prev_al_ind2 is None
            ):
                continue

            # NOTE:
            mid = (cfg.overbought + cfg.oversold) / 2

            if cfg.use_midsold_filter:
                long_signal = (
                    al_ind2 < cfg.oversold
                    and prev_ind < cfg.oversold
                    and cfg.oversold < ind < mid
                )
                short_signal = (
                    al_ind2 > cfg.overbought
                    and prev_ind > cfg.overbought
                    and mid < ind < cfg.overbought
                )
            else:
                long_signal = (
                    al_ind2 < cfg.oversold
                    and prev_ind < cfg.oversold
                    and ind > cfg.oversold
                )
                short_signal = (
                    al_ind2 > cfg.overbought
                    and prev_ind > cfg.overbought
                    and ind < cfg.overbought
                )

            if al_ind3:
                long_signal = long_signal and al_ind3 < cfg.oversold
                short_signal = short_signal and al_ind3 > cfg.overbought

            if position is None:
                if open_positions_count >= cfg.max_concurrent_positions:
                    self._log.debug(
                        "Max concurrent positions reached for symbol %s, skipping new entry",
                        symbol,
                    )
                    continue
                if long_signal or short_signal:
                    side = (
                        PositionDirection.LONG
                        if long_signal
                        else PositionDirection.SHORT
                    )

                    entry_metadata = {
                        "entry[candle]": str(candle),
                        "entry[ind]": ind,
                        "entry[prev_ind]": prev_ind,
                        "entry[al_ind2]": al_ind2,
                        "entry[prev_al_ind2]": prev_al_ind2,
                        "entry[al_ind3]": al_ind3,
                        "entry[prev_al_ind3]": prev_al_ind3,
                    }

                    new_position = self._open_position(
                        symbol=symbol,
                        side=side,
                        fill_price=candle.close,
                        timestamp=candle.close_time,
                        size_usdt=cfg.initial_order_usdt,
                        leverage=cfg.initial_leverage,
                        cfg=cfg,
                        reason="Initial entry",
                        metadata=entry_metadata,
                    )
                    if new_position:
                        states[symbol] = new_position
                        open_positions_count += 1
            else:
                if (long_signal and position.side is PositionDirection.LONG) or (
                    short_signal and position.side is PositionDirection.SHORT
                ):
                    if position.martingale_step >= cfg.max_martingale_steps:
                        self._log.debug(
                            "Max martingale steps reached for symbol %s, skipping add to position",
                            symbol,
                        )
                        continue
                    if cfg.enable_grid_martingales:
                        last_add = position.last_add_price
                        if last_add and last_add > 0:
                            move_pct = abs((candle.close - last_add) / last_add) * 100.0
                            if move_pct < cfg.grid_martingales_percent:
                                self._log.debug(
                                    "Price move %.2f%% since last add is less than grid martingale threshold %.2f%% for symbol %s, skipping add to position",
                                    move_pct,
                                    cfg.grid_martingales_percent,
                                    symbol,
                                )
                                continue
                    step_index = position.martingale_step
                    if cfg.martingale_multiplier is not None:
                        multiplier = cfg.martingale_multiplier ** (step_index + 1)
                    else:
                        multiplier = cfg.martingale_multipliers[
                            min(step_index, len(cfg.martingale_multipliers) - 1)
                        ]
                    leverage = cfg.martingale_leverages[
                        min(step_index, len(cfg.martingale_leverages) - 1)
                    ]
                    add_size = cfg.initial_order_usdt * multiplier
                    martingale_metadata = {
                        "martingale[candle]": str(candle),
                        "martingale[ind]": ind,
                        "martingale[prev_ind]": prev_ind,
                        "martingale[al_ind2]": al_ind2,
                        "martingale[prev_al_ind2]": prev_al_ind2,
                        "martingale[al_ind3]": al_ind3,
                        "martingale[prev_al_ind3]": prev_al_ind3,
                    }
                    self._add_to_position(
                        position=position,
                        fill_price=candle.close,
                        size_usdt=add_size,
                        leverage=leverage,
                        cfg=cfg,
                        timestamp=candle.close_time,
                        metadata=martingale_metadata,
                    )
                elif cfg.enable_reversal_logic and (long_signal or short_signal):
                    # Reverse: close then open in opposite direction if slots available.
                    exit_price = _calc_exit_price_with_slippage(
                        candle.close, position.side, cfg.slippage_pct
                    )
                    trades.append(
                        self._close_position(
                            position=position,
                            exit_price=exit_price,
                            exit_time=candle.close_time,
                            reason="Reversal signal",
                            use_taker_fee=True,
                            cfg=cfg,
                            metadata=exit_metadata,
                        )
                    )
                    open_positions_count = max(0, open_positions_count - 1)
                    states[symbol] = None

                    if cfg.enable_reversal_reentry:
                        if open_positions_count >= cfg.max_concurrent_positions:
                            continue
                        new_side = (
                            PositionDirection.LONG
                            if long_signal
                            else PositionDirection.SHORT
                        )
                        entry_metadata = {
                            "entry[candle]": str(candle),
                            "entry[ind]": ind,
                            "entry[prev_ind]": prev_ind,
                            "entry[al_ind2]": al_ind2,
                            "entry[prev_al_ind2]": prev_al_ind2,
                            "entry[al_ind3]": al_ind3,
                            "entry[prev_al_ind3]": prev_al_ind3,
                        }
                        new_state = self._open_position(
                            symbol=symbol,
                            side=new_side,
                            fill_price=candle.close,
                            timestamp=candle.close_time,
                            size_usdt=cfg.initial_order_usdt,
                            leverage=cfg.initial_leverage,
                            cfg=cfg,
                            reason="Reversal entry",
                            metadata=entry_metadata,
                        )
                        if new_state:
                            states[symbol] = new_state
                            open_positions_count += 1

        adds: List[float] = []
        for trade in trades:
            adds_val = None
            if trade.metadata:
                adds_val = trade.metadata.get("adds")
            if isinstance(adds_val, (int, float)):
                adds.append(float(adds_val))
        num_martingales = sum(adds)
        avg_martingales = (sum(adds) / len(adds)) if adds else 0.0
        stddev_martingales = 0.0
        if len(adds) > 1:
            mean_adds = avg_martingales
            variance = sum((a - mean_adds) ** 2 for a in adds) / len(adds)
            stddev_martingales = math.sqrt(variance)
        extra_stats = {
            "num_martingales": num_martingales,
            "avg_martingales": avg_martingales,
            "stddev_martingales": stddev_martingales,
        }
        return trades, extra_stats

    # ------------------------------------------------------------------ helpers

    def _crossed_opposite(
        self,
        side: PositionDirection,
        prev_value: float,
        current_value: float,
        cfg: StochasticRsiFsmConfig,
    ) -> bool:
        if side is PositionDirection.LONG:
            return current_value < cfg.overbought < prev_value
        return prev_value < cfg.oversold < current_value

    def _check_take_profit(
        self,
        position: OpenPositionState,
        candle: Candle,
    ) -> Tuple[Optional[float], Optional[Dict[str, float | int | str | None]]]:
        if position.side is PositionDirection.LONG:
            if candle.high >= position.take_profit:
                return position.take_profit, {}
        else:
            if candle.low <= position.take_profit:
                return position.take_profit, {}
        return None, None

    def _maybe_update_trailing(
        self,
        position: OpenPositionState,
        candle: Candle,
        cfg: StochasticRsiFsmConfig,
    ) -> Optional[Tuple[float, str, Dict[str, float | int | str | None]]]:
        if position.last_update_time is not None:
            delta = (candle.close_time - position.last_update_time).total_seconds()
            if delta < cfg.trailing_check_interval_seconds:
                return None

        # NOTE:
        entry = (
            position.first_entry_price
            if cfg.trailing_use_first_entry_price
            else position.avg_entry_price
        )
        if entry <= 0:
            return None

        if position.side is PositionDirection.LONG:
            ref_price = candle.high
            move_pct = ((ref_price - entry) / entry) * 100.0
        else:
            ref_price = candle.low
            move_pct = ((entry - ref_price) / entry) * 100.0

        if move_pct < cfg.trailing_activation_pct:
            position.last_update_time = candle.close_time
            return None

        lock_in_pct = move_pct - cfg.trailing_gap_pct
        if lock_in_pct < 0:
            lock_in_pct = 0.0

        desired_stop = (
            entry * (1 + lock_in_pct / 100.0)
            if position.side is PositionDirection.LONG
            else entry * (1 - lock_in_pct / 100.0)
        )

        if (
            position.trailing_stop is None
            or (
                position.side is PositionDirection.LONG
                and desired_stop > position.trailing_stop
            )
            or (
                position.side is PositionDirection.SHORT
                and desired_stop < position.trailing_stop
            )
        ):
            position.trailing_stop = desired_stop
            position.trailing_active = True

        position.last_update_time = candle.close_time

        # NOTE:
        long_stop_hit = (
            candle.close <= position.trailing_stop
            if cfg.trailing_use_close_for_stop_activation
            else candle.low <= position.trailing_stop
        )
        if position.side is PositionDirection.LONG and long_stop_hit:
            exit_price = _calc_exit_price_with_slippage(
                position.trailing_stop, position.side, cfg.slippage_pct
            )
            return (
                exit_price,
                "Trailing stop",
                {
                    "trailing_stop": position.trailing_stop,
                    "trailing_move_pct": move_pct,
                    "trailing_lock_pct": lock_in_pct,
                },
            )
        # NOTE:
        short_stop_hit = (
            candle.close >= position.trailing_stop
            if cfg.trailing_use_close_for_stop_activation
            else candle.high >= position.trailing_stop
        )
        if position.side is PositionDirection.SHORT and short_stop_hit:
            exit_price = _calc_exit_price_with_slippage(
                position.trailing_stop, position.side, cfg.slippage_pct
            )
            return (
                exit_price,
                "Trailing stop",
                {
                    "trailing_stop": position.trailing_stop,
                    "trailing_move_pct": move_pct,
                    "trailing_lock_pct": lock_in_pct,
                },
            )
        return None

    def _check_liquidation(
        self,
        position: OpenPositionState,
        candle: Candle,
    ) -> Tuple[bool, Optional[Dict[Any, Any]]]:
        if position.margin <= 0 or position.quantity <= 0:
            return False, None
        if position.liquidation_price <= 0 or position.margin_call_price <= 0:
            (
                position.liquidation_price,
                position.margin_call_price,
            ) = _calc_risk_prices(
                position.avg_entry_price,
                position.margin,
                position.quantity,
                position.side,
            )
        adverse_price = (
            candle.low if position.side is PositionDirection.LONG else candle.high
        )
        if adverse_price <= 0:
            return False, None
        if position.side is PositionDirection.LONG:
            unrealized = (adverse_price - position.avg_entry_price) * position.quantity
        else:
            unrealized = (position.avg_entry_price - adverse_price) * position.quantity
        unrealized_loss = -unrealized if unrealized < 0 else 0.0
        if position.side is PositionDirection.LONG:
            hit = adverse_price <= position.liquidation_price
        else:
            hit = adverse_price >= position.liquidation_price
        meta = {
            "unrealized_loss": unrealized_loss,
            "margin": position.margin,
            "liquidation_price": position.liquidation_price,
            "margin_call_price": position.margin_call_price,
        }
        return hit, meta if hit else None

    def _open_position(
        self,
        *,
        symbol: str,
        side: PositionDirection,
        fill_price: float,
        timestamp: datetime,
        size_usdt: float,
        leverage: float,
        cfg: StochasticRsiFsmConfig,
        reason: str,
        metadata: Optional[Dict[Any, Any]] = None,
    ) -> Optional[OpenPositionState]:
        fill = _apply_slippage(fill_price, side, True, cfg.slippage_pct)
        if fill <= 0 or size_usdt <= 0:
            self._log.warning("Invalid fill encountered for %s, skipping entry", symbol)
            return None
        qty = size_usdt / fill if fill > 0 else 0.0
        if qty <= 0:
            self._log.warning("Non-positive quantity for %s, skipping entry", symbol)
            return None
        fee = _calc_fee(size_usdt, cfg.maker_fee_pct)
        margin = size_usdt / leverage if leverage > 0 else size_usdt
        tp_basis = fill if cfg.take_profit_use_first_entry_price else fill
        take_profit = (
            tp_basis * (1 + cfg.take_profit_pct)
            if side is PositionDirection.LONG
            else tp_basis * (1 - cfg.take_profit_pct)
        )
        liquidation_price, margin_call_price = _calc_risk_prices(
            fill, margin, qty, side
        )
        return OpenPositionState(
            symbol=symbol,
            side=side,
            entry_time=timestamp,
            first_entry_price=fill,
            avg_entry_price=fill,
            quantity=qty,
            notional=size_usdt,
            margin=margin,
            take_profit=take_profit,
            leverage=leverage,
            last_add_price=fill,
            martingale_step=0,
            trailing_stop=None,
            trailing_active=False,
            liquidation_price=liquidation_price,
            margin_call_price=margin_call_price,
            fees_paid=fee,
            funding_paid=0.0,
            last_update_time=timestamp,
            notes=reason,
            metadata=metadata or {},
        )

    def _add_to_position(
        self,
        *,
        position: OpenPositionState,
        fill_price: float,
        size_usdt: float,
        leverage: float,
        cfg: StochasticRsiFsmConfig,
        timestamp: datetime,
        metadata: Optional[Dict[Any, Any]] = None,
    ) -> None:
        fill = _apply_slippage(fill_price, position.side, True, cfg.slippage_pct)
        if fill <= 0 or size_usdt <= 0:
            self._log.warning("Invalid add fill for %s, skipping", position.symbol)
            return

        pre_state = position.to_dict()

        qty_add = size_usdt / fill if fill > 0 else 0.0
        new_quantity = position.quantity + qty_add
        if new_quantity <= 0:
            self._log.warning(
                "Non-positive new quantity for %s, skipping add", position.symbol
            )
            return
        notional_add = size_usdt
        new_notional = position.notional + notional_add
        new_avg_entry_price = (
            (position.avg_entry_price * position.quantity) + (fill * qty_add)
        ) / new_quantity
        margin_add = size_usdt / leverage if leverage > 0 else size_usdt
        new_margin = position.margin + margin_add
        new_leverage = (
            (new_avg_entry_price * new_quantity) / new_margin if new_margin > 0 else 0.0
        )
        position.avg_entry_price = new_avg_entry_price
        position.quantity = new_quantity
        position.notional = new_notional
        position.margin = new_margin
        """
        Example:
        Overall Margin = 30 + 7 = 37 USD
        Combined Quantity = 4 + 1.448 = 5.448 units
        Weighted Entry Price = ((4 × 150) + (1.448 × 145)) / 5.448 = (600 + 209) / 5.448 = 149.0 (approx)
        Total position size = 600 + 210 = 810
        Total margin = 37
        Effective leverage = 810 / 37 ≈ 21.89x
        Liquidation estimate = 149 - 149 / 21.89 ≈ 149 − 6.81 = 142.2 USD (approx)
        """
        position.leverage = new_leverage
        position.last_add_price = fill
        position.martingale_step += 1
        tp_basis = (
            position.first_entry_price
            if cfg.take_profit_use_first_entry_price
            else position.avg_entry_price
        )
        position.take_profit = (
            tp_basis * (1 + cfg.take_profit_pct)
            if position.side is PositionDirection.LONG
            else tp_basis * (1 - cfg.take_profit_pct)
        )
        position.liquidation_price, position.margin_call_price = _calc_risk_prices(
            position.avg_entry_price, position.margin, position.quantity, position.side
        )
        fees_add = _calc_fee(size_usdt, cfg.maker_fee_pct)
        position.fees_paid += fees_add
        position.last_update_time = timestamp
        position.notes = "Martingale add"
        step_number = position.martingale_step

        martingale_state = {
            "step_number": step_number,
            "fill_price": fill_price,
            "fill": fill,
            "size_usdt": size_usdt,
            "leverage": leverage,
            "qty_add": qty_add,
            "notional_add": notional_add,
            "margin_add": margin_add,
            "fees_add": fees_add,
            "timestamp": timestamp.isoformat(),
        }
        if metadata is not None:
            martingale_state.update(metadata)

        post_state = position.to_dict()

        states = position.metadata.get("states")
        if not isinstance(states, list):
            states = []
            position.metadata["states"] = states
        overall_state = {
            "pre_state": pre_state,
            "martingale_state": martingale_state,
            "post_state": post_state,
        }
        states.append(overall_state)

    def _close_position(
        self,
        *,
        position: OpenPositionState,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        use_taker_fee: bool,
        cfg: StochasticRsiFsmConfig,
        metadata: Optional[Dict[Any, Any]] = None,
    ) -> TradePerformance:
        if position.side is PositionDirection.LONG:
            gross = (exit_price - position.avg_entry_price) * position.quantity
        else:
            gross = (position.avg_entry_price - exit_price) * position.quantity

        exit_notional = exit_price * position.quantity
        fee_pct = cfg.taker_fee_pct if use_taker_fee else cfg.maker_fee_pct
        fee = _calc_fee(exit_notional, fee_pct)
        total_fees = position.fees_paid + fee
        pnl = gross - total_fees - position.funding_paid
        ret_pct = (pnl / position.notional) * 100.0 if position.notional else 0.0

        metadata.update(
            {
                "symbol": position.symbol,
                "side": position.side.value,
                "avg_entry_price": position.avg_entry_price,
                "exit_price": exit_price,
                "exit_notional": exit_notional,
                "take_profit": position.take_profit,
                "leverage": position.leverage,
                "adds": position.martingale_step,
                "taker_fee": fee,
                "fees_paid": total_fees,
                "funding_paid": position.funding_paid,
                "effective_leverage": position.effective_leverage(),
                "trailing_stop": position.trailing_stop,
                "liquidation_price": position.liquidation_price,
                "margin_call_price": position.margin_call_price,
                "position_meta": position.metadata,
            }
        )

        return TradePerformance(
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            return_pct=ret_pct,
            notes=reason,
            metadata=metadata,
        )

    def _format_series(
        self, timestamps: Sequence[datetime], series: Sequence[Optional[float]]
    ) -> str:
        parts: List[str] = []
        for ts, value in zip(timestamps, series):
            ts_str = ts.strftime("%Y-%m-%d %H:%M")
            val_str = "None" if value is None else f"{value:.4f}"
            parts.append(f"({ts_str}, {val_str})")
        return "[" + ", ".join(parts) + "]"
