from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .indicators import ema as calc_ema

Direction = Literal["long", "short"]
EmaValidationMode = Literal["body", "wick"]
SetupWaitingReplacementMode = Literal["keep_waiting", "replace_waiting"]
PositionSizingMode = Literal["risk_distance", "risk_amount_per_price"]


@dataclass(frozen=True)
class EmaAvwapPullbackStrategyConfig:
    symbol: str
    timeframe: str

    initial_equity: float = 100.0
    leverage: float = 1.0
    equity_risk_pct: float = 1.0

    ema_length: int = 55
    consecutive_count: int = 4
    ema_validation_mode: EmaValidationMode = "body"
    setup_waiting_replacement_mode: SetupWaitingReplacementMode = "keep_waiting"
    position_sizing_mode: PositionSizingMode = "risk_distance"

    avwap_multiplier_1: float = 1.0
    avwap_multiplier_2: float = 2.0
    avwap_multiplier_3: float = 3.0

    rigid_stop_loss_pct: float = 0.0
    trailing_activation_threshold_pct: float = 0.0
    trailing_gap_pct: float = 1.0

    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0006
    entry_slippage_pct: float = 0.0
    exit_slippage_pct: float = 0.0
    use_gap_cross_detection: bool = True
    max_decision_log_entries: int = 20000

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        timeframe = self.timeframe.strip()
        if not symbol:
            raise ValueError("symbol must not be empty")
        if not timeframe:
            raise ValueError("timeframe must not be empty")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.equity_risk_pct <= 0:
            raise ValueError("equity_risk_pct must be positive")
        if self.ema_length <= 0:
            raise ValueError("ema_length must be positive")
        if self.consecutive_count <= 0:
            raise ValueError("consecutive_count must be positive")
        if self.ema_validation_mode not in {"body", "wick"}:
            raise ValueError("ema_validation_mode must be one of: body, wick")
        if self.setup_waiting_replacement_mode not in {
            "keep_waiting",
            "replace_waiting",
        }:
            raise ValueError(
                "setup_waiting_replacement_mode must be one of: "
                "keep_waiting, replace_waiting"
            )
        if self.position_sizing_mode not in {"risk_distance", "risk_amount_per_price"}:
            raise ValueError(
                "position_sizing_mode must be one of: "
                "risk_distance, risk_amount_per_price"
            )
        if min(
            self.avwap_multiplier_1,
            self.avwap_multiplier_2,
            self.avwap_multiplier_3,
        ) <= 0:
            raise ValueError("AVWAP multipliers must be positive")
        if self.rigid_stop_loss_pct < 0:
            raise ValueError("rigid_stop_loss_pct must be non-negative")
        if self.trailing_activation_threshold_pct < 0:
            raise ValueError("trailing_activation_threshold_pct must be non-negative")
        if self.trailing_gap_pct < 0:
            raise ValueError("trailing_gap_pct must be non-negative")
        if min(self.maker_fee_pct, self.taker_fee_pct) < 0:
            raise ValueError("fee values must be non-negative")
        if min(self.entry_slippage_pct, self.exit_slippage_pct) < 0:
            raise ValueError("slippage values must be non-negative")
        if self.max_decision_log_entries <= 0:
            raise ValueError("max_decision_log_entries must be positive")
        interval_to_milliseconds(timeframe)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)


@dataclass(frozen=True)
class _AvwapSnapshot:
    anchor_index: int
    anchor_time: datetime
    candle_index: int
    vwap: float
    stdev: float
    upper1: float
    lower1: float
    upper2: float
    lower2: float
    upper3: float
    lower3: float


@dataclass(frozen=True)
class _SetupState:
    direction: Direction
    anchor_index: int
    detected_index: int
    detected_time: datetime
    consecutive_count: int
    is_waiting_for_cross: bool = False


@dataclass
class _PositionState:
    direction: Direction
    anchor_index: int
    setup_detected_index: int
    setup_detected_time: datetime
    entry_time: datetime
    entry_index: int
    raw_entry_price: float
    entry_price: float
    qty: float
    risk_amount: float
    risk_amount_interpretation: str
    entry_fee: float
    stop_level_at_entry: float
    rigid_stop_level_at_entry: float | None
    trailing_activation_level_at_entry: float
    entry_trigger_mode: str
    position_sizing_mode: PositionSizingMode
    trailing_active: bool = False
    trailing_stop: float | None = None
    extreme_price: float | None = None


@dataclass(frozen=True)
class _SizingDecision:
    qty: float
    distance: float
    entry_price: float
    estimated_exit_price: float
    risk_amount_interpretation: str
    base_qty_before_costs: float
    qty_reduction_from_costs: float
    sizing_reference_price: float
    effective_price_for_sizing: float
    entry_slippage_per_unit: float
    exit_slippage_per_unit: float
    entry_fee_per_unit: float
    exit_fee_per_unit: float
    total_cost_per_unit: float


@dataclass(frozen=True)
class _CrossDecision:
    crossed: bool
    mode: str | None = None


@dataclass(frozen=True)
class _ExitDecision:
    reason: str
    raw_exit_price: float
    stop_level: float
    activation_level: float


class EmaAvwapPullbackStrategy(BacktestStrategy):
    def __init__(self, config: EmaAvwapPullbackStrategyConfig) -> None:
        self._config = config
        self._log = logging.getLogger(self.__class__.__name__)

    def name(self) -> str:
        return "EmaAvwapPullbackStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        cfg = self._config
        candles = context.data.get(cfg.symbol, {}).get(cfg.timeframe, [])
        if not candles:
            return [], {"note": "no_data"}
        if len(candles) < max(cfg.ema_length, cfg.consecutive_count):
            return [], {"note": "insufficient_data", "candles": len(candles)}

        closes = [candle.close for candle in candles]
        ema_values = calc_ema(closes, cfg.ema_length)
        tpv_prefix, vol_prefix, tpv2_prefix = self._build_avwap_prefixes(candles)

        ignore_count = context.ignore_candles.get(cfg.symbol, {}).get(cfg.timeframe, 0)
        start_index = max(ignore_count, cfg.ema_length - 1, cfg.consecutive_count - 1)

        trades: List[TradePerformance] = []
        decision_log: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "config": self._config_as_dict(),
            "execution_assumptions": {
                "entry_fill_model": "intrabar_intersection",
                "exit_fill_model": "intrabar_intersection",
                "intrabar_price_path": "open -> nearest extreme -> far extreme -> close",
                "gap_cross_detection": cfg.use_gap_cross_detection,
                "avwap_value_source": "completed_bar_snapshot",
            },
            "setups_detected_long": 0,
            "setups_detected_short": 0,
            "setups_replaced_long": 0,
            "setups_replaced_short": 0,
            "setups_invalidated_long": 0,
            "setups_invalidated_short": 0,
            "setups_kept_waiting_long": 0,
            "setups_kept_waiting_short": 0,
            "waiting_setups_replaced_long": 0,
            "waiting_setups_replaced_short": 0,
            "entries_long": 0,
            "entries_short": 0,
            "entries_skipped_invalid_risk": 0,
            "entries_skipped_non_positive_equity": 0,
            "entries_skipped_zero_qty": 0,
            "stop_exits": 0,
            "rigid_stop_exits": 0,
            "trailing_exits": 0,
            "end_of_backtest_exits": 0,
            "trailing_activations": 0,
            "trailing_updates": 0,
            "decision_log_truncated_count": 0,
            "exit_reason_counts": {},
            "total_entry_fees": 0.0,
            "total_exit_fees": 0.0,
            "max_margin_required": 0.0,
            "decision_log": decision_log,
            "initial_equity": cfg.initial_equity,
        }

        active_long_setup: _SetupState | None = None
        active_short_setup: _SetupState | None = None
        position: _PositionState | None = None
        realized_equity = cfg.initial_equity
        last_in_range_index: int | None = None

        for idx in range(start_index, len(candles)):
            candle = candles[idx]
            ema_value = ema_values[idx]
            if ema_value is None:
                continue
            if candle.close_time < context.config.start:
                continue
            if candle.open_time > context.config.end:
                break

            ema_value = float(ema_value)
            last_in_range_index = idx
            prev_close = candles[idx - 1].close if idx > 0 else candle.open

            if position is not None:
                avwap = self._build_avwap_snapshot(
                    candles=candles,
                    anchor_index=position.anchor_index,
                    candle_index=idx,
                    tpv_prefix=tpv_prefix,
                    vol_prefix=vol_prefix,
                    tpv2_prefix=tpv2_prefix,
                )
                exit_decision = self._process_position_for_candle(
                    position=position,
                    candle=candle,
                    candle_index=idx,
                    prev_close=prev_close,
                    avwap=avwap,
                    stats=stats,
                    decision_log=decision_log,
                )
                if exit_decision is not None:
                    pnl = self._close_position(
                        position=position,
                        candle=candle,
                        candle_index=idx,
                        exit_time=candle.close_time,
                        exit_reason=exit_decision.reason,
                        raw_exit_price=exit_decision.raw_exit_price,
                        stop_level=exit_decision.stop_level,
                        activation_level=exit_decision.activation_level,
                        avwap=avwap,
                        trades=trades,
                        stats=stats,
                    )
                    realized_equity += pnl
                    position = None

            if position is None:
                active_long_setup, position = self._process_pending_setup(
                    setup=active_long_setup,
                    candle=candle,
                    candle_index=idx,
                    prev_close=prev_close,
                    realized_equity=realized_equity,
                    candles=candles,
                    tpv_prefix=tpv_prefix,
                    vol_prefix=vol_prefix,
                    tpv2_prefix=tpv2_prefix,
                    stats=stats,
                    decision_log=decision_log,
                )
                if position is not None:
                    active_long_setup = None
                    active_short_setup = None

            if position is None:
                active_short_setup, position = self._process_pending_setup(
                    setup=active_short_setup,
                    candle=candle,
                    candle_index=idx,
                    prev_close=prev_close,
                    realized_equity=realized_equity,
                    candles=candles,
                    tpv_prefix=tpv_prefix,
                    vol_prefix=vol_prefix,
                    tpv2_prefix=tpv2_prefix,
                    stats=stats,
                    decision_log=decision_log,
                )
                if position is not None:
                    active_long_setup = None
                    active_short_setup = None

            if position is None:
                maybe_long_setup = self._detect_setup(
                    direction="long",
                    candles=candles,
                    candle_index=idx,
                    ema_value=ema_value,
                )
                if maybe_long_setup is not None:
                    active_long_setup = self._replace_or_store_setup(
                        current_setup=active_long_setup,
                        new_setup=maybe_long_setup,
                        candles=candles,
                        candle_index=idx,
                        tpv_prefix=tpv_prefix,
                        vol_prefix=vol_prefix,
                        tpv2_prefix=tpv2_prefix,
                        stats=stats,
                        decision_log=decision_log,
                    )

                maybe_short_setup = self._detect_setup(
                    direction="short",
                    candles=candles,
                    candle_index=idx,
                    ema_value=ema_value,
                )
                if maybe_short_setup is not None:
                    active_short_setup = self._replace_or_store_setup(
                        current_setup=active_short_setup,
                        new_setup=maybe_short_setup,
                        candles=candles,
                        candle_index=idx,
                        tpv_prefix=tpv_prefix,
                        vol_prefix=vol_prefix,
                        tpv2_prefix=tpv2_prefix,
                        stats=stats,
                        decision_log=decision_log,
                    )

        if position is not None and last_in_range_index is not None:
            last_candle = candles[last_in_range_index]
            avwap = self._build_avwap_snapshot(
                candles=candles,
                anchor_index=position.anchor_index,
                candle_index=last_in_range_index,
                tpv_prefix=tpv_prefix,
                vol_prefix=vol_prefix,
                tpv2_prefix=tpv2_prefix,
            )
            pnl = self._close_position(
                position=position,
                candle=last_candle,
                candle_index=last_in_range_index,
                exit_time=last_candle.close_time,
                exit_reason="End of backtest",
                raw_exit_price=last_candle.close,
                stop_level=avwap.lower2 if position.direction == "long" else avwap.upper2,
                activation_level=self._trailing_activation_level(position.direction, avwap),
                avwap=avwap,
                trades=trades,
                stats=stats,
            )
            realized_equity += pnl
            stats["end_of_backtest_exits"] += 1

        stats["final_equity"] = realized_equity
        stats["pending_setups_at_end"] = {
            "long": active_long_setup is not None,
            "short": active_short_setup is not None,
        }
        stats["open_position_at_end"] = position.direction if position is not None else None
        stats.update(self._summarize_trade_stats(trades))
        return trades, stats

    def _detect_setup(
        self,
        *,
        direction: Direction,
        candles: Sequence[Candle],
        candle_index: int,
        ema_value: float,
    ) -> _SetupState | None:
        cfg = self._config
        anchor_index = candle_index - cfg.consecutive_count + 1
        if anchor_index < 0:
            return None

        window = candles[anchor_index : candle_index + 1]
        if direction == "long":
            if not all(candle.is_bullish() for candle in window):
                return None
        else:
            if not all(candle.is_bearish() for candle in window):
                return None

        current_candle = candles[candle_index]
        if not self._validate_ema_position(
            candle=current_candle,
            ema_value=ema_value,
            direction=direction,
        ):
            return None

        return _SetupState(
            direction=direction,
            anchor_index=anchor_index,
            detected_index=candle_index,
            detected_time=current_candle.close_time,
            consecutive_count=cfg.consecutive_count,
        )

    def _validate_ema_position(
        self,
        *,
        candle: Candle,
        ema_value: float,
        direction: Direction,
    ) -> bool:
        mode = self._config.ema_validation_mode
        if direction == "long":
            if mode == "wick":
                return candle.low > ema_value
            return min(candle.open, candle.close) > ema_value
        if mode == "wick":
            return candle.high < ema_value
        return max(candle.open, candle.close) < ema_value

    def _replace_or_store_setup(
        self,
        *,
        current_setup: _SetupState | None,
        new_setup: _SetupState,
        candles: Sequence[Candle],
        candle_index: int,
        tpv_prefix: Sequence[float],
        vol_prefix: Sequence[float],
        tpv2_prefix: Sequence[float],
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> _SetupState:
        direction = new_setup.direction
        if current_setup is not None:
            if current_setup.is_waiting_for_cross:
                if self._config.setup_waiting_replacement_mode == "keep_waiting":
                    stats[f"setups_kept_waiting_{direction}"] += 1
                    avwap = self._build_avwap_snapshot(
                        candles=candles,
                        anchor_index=new_setup.anchor_index,
                        candle_index=candle_index,
                        tpv_prefix=tpv_prefix,
                        vol_prefix=vol_prefix,
                        tpv2_prefix=tpv2_prefix,
                    )
                    self._record_event(
                        decision_log=decision_log,
                        stats=stats,
                        event="setup_detected_ignored",
                        payload={
                            "timestamp": candles[candle_index].close_time.isoformat(),
                            "candle_index": candle_index,
                            "setup_type": direction,
                            "reason": "existing setup_waiting kept active",
                            "waiting_anchor_index": current_setup.anchor_index,
                            "waiting_anchor_time": candles[
                                current_setup.anchor_index
                            ].open_time.isoformat(),
                            "ignored_anchor_index": new_setup.anchor_index,
                            "ignored_anchor_time": candles[
                                new_setup.anchor_index
                            ].open_time.isoformat(),
                            "replacement_mode": self._config.setup_waiting_replacement_mode,
                            "vwap_middle_line": avwap.vwap,
                            "upper_band_1": avwap.upper1,
                            "lower_band_1": avwap.lower1,
                            "upper_band_2": avwap.upper2,
                            "lower_band_2": avwap.lower2,
                        },
                    )
                    return current_setup
                stats[f"waiting_setups_replaced_{direction}"] += 1
            stats[f"setups_replaced_{direction}"] += 1

        stats[f"setups_detected_{direction}"] += 1
        avwap = self._build_avwap_snapshot(
            candles=candles,
            anchor_index=new_setup.anchor_index,
            candle_index=candle_index,
            tpv_prefix=tpv_prefix,
            vol_prefix=vol_prefix,
            tpv2_prefix=tpv2_prefix,
        )
        self._record_event(
            decision_log=decision_log,
            stats=stats,
            event="setup_detected",
            payload={
                "timestamp": candles[candle_index].close_time.isoformat(),
                "candle_index": candle_index,
                "setup_type": direction,
                "consecutive_count": new_setup.consecutive_count,
                "ema_validation_mode": self._config.ema_validation_mode,
                "anchor_index": new_setup.anchor_index,
                "anchor_time": candles[new_setup.anchor_index].open_time.isoformat(),
                "vwap_middle_line": avwap.vwap,
                "upper_band_1": avwap.upper1,
                "lower_band_1": avwap.lower1,
                "upper_band_2": avwap.upper2,
                "lower_band_2": avwap.lower2,
            },
        )
        self._log.info(
            "Detected %s setup on candle %s anchored at %s",
            direction,
            candle_index,
            new_setup.anchor_index,
        )
        return new_setup

    def _process_pending_setup(
        self,
        *,
        setup: _SetupState | None,
        candle: Candle,
        candle_index: int,
        prev_close: float,
        realized_equity: float,
        candles: Sequence[Candle],
        tpv_prefix: Sequence[float],
        vol_prefix: Sequence[float],
        tpv2_prefix: Sequence[float],
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> Tuple[_SetupState | None, _PositionState | None]:
        if setup is None or candle_index <= setup.detected_index:
            return setup, None

        expects_pullback = (
            candle.is_bearish() if setup.direction == "long" else candle.is_bullish()
        )
        if not expects_pullback:
            return setup, None

        avwap = self._build_avwap_snapshot(
            candles=candles,
            anchor_index=setup.anchor_index,
            candle_index=candle_index,
            tpv_prefix=tpv_prefix,
            vol_prefix=vol_prefix,
            tpv2_prefix=tpv2_prefix,
        )
        cross_direction = "down" if setup.direction == "long" else "up"
        cross = self._detect_level_cross(
            candle=candle,
            prev_close=prev_close,
            level=avwap.vwap,
            direction=cross_direction,
        )

        if not cross.crossed:
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="setup_waiting",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "consecutive_count": setup.consecutive_count,
                    "anchor_index": setup.anchor_index,
                    "anchor_time": candles[setup.anchor_index].open_time.isoformat(),
                    "entry_signal_details": "pullback candle did not cross AVWAP middle line yet",
                    "vwap_middle_line": avwap.vwap,
                    "upper_band_1": avwap.upper1,
                    "lower_band_1": avwap.lower1,
                    "upper_band_2": avwap.upper2,
                    "lower_band_2": avwap.lower2,
                },
            )
            return replace(setup, is_waiting_for_cross=True), None

        risk_amount = realized_equity * (self._config.equity_risk_pct / 100.0)
        if risk_amount <= 0:
            stats["entries_skipped_non_positive_equity"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="entry_skipped",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_signal_details": "non-positive risk amount",
                    "realized_equity": realized_equity,
                },
            )
            return None, None

        raw_entry_price = avwap.vwap
        stop_level = avwap.lower2 if setup.direction == "long" else avwap.upper2
        sizing = self._build_sizing_decision(
            direction=setup.direction,
            raw_entry_price=raw_entry_price,
            stop_level=stop_level,
            risk_amount=risk_amount,
        )
        if sizing is None:
            stats["entries_skipped_invalid_risk"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="entry_skipped",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_signal_details": "stop distance was not positive",
                    "entry_intersection_price": raw_entry_price,
                    "stop_loss_level": stop_level,
                },
            )
            return None, None

        qty = sizing.qty
        if qty <= 0:
            stats["entries_skipped_zero_qty"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="entry_skipped",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_signal_details": "computed quantity was not positive",
                },
            )
            return None, None

        entry_price = sizing.entry_price
        entry_fee = entry_price * qty * self._config.maker_fee_pct
        rigid_stop_level = self._rigid_stop_level(setup.direction, entry_price)
        trailing_activation_level = self._trailing_activation_level(setup.direction, avwap)
        estimated_entry_slippage_cost = sizing.entry_slippage_per_unit * qty
        estimated_exit_slippage_cost = sizing.exit_slippage_per_unit * qty
        estimated_exit_fee = sizing.exit_fee_per_unit * qty
        estimated_total_cost_buffer = sizing.total_cost_per_unit * qty

        position = _PositionState(
            direction=setup.direction,
            anchor_index=setup.anchor_index,
            setup_detected_index=setup.detected_index,
            setup_detected_time=setup.detected_time,
            entry_time=candle.close_time,
            entry_index=candle_index,
            raw_entry_price=raw_entry_price,
            entry_price=entry_price,
            qty=qty,
            risk_amount=risk_amount,
            risk_amount_interpretation=sizing.risk_amount_interpretation,
            entry_fee=entry_fee,
            stop_level_at_entry=stop_level,
            rigid_stop_level_at_entry=rigid_stop_level,
            trailing_activation_level_at_entry=trailing_activation_level,
            entry_trigger_mode=cross.mode or "intrabar",
            position_sizing_mode=self._config.position_sizing_mode,
        )

        stats[f"entries_{setup.direction}"] += 1
        stats["total_entry_fees"] += entry_fee
        stats["max_margin_required"] = max(
            stats["max_margin_required"],
            (entry_price * qty) / self._config.leverage,
        )

        self._record_event(
            decision_log=decision_log,
            stats=stats,
            event="entry_triggered",
            payload={
                "timestamp": candle.close_time.isoformat(),
                "candle_index": candle_index,
                "setup_type": setup.direction,
                "consecutive_count": setup.consecutive_count,
                "anchor_index": setup.anchor_index,
                "anchor_time": candles[setup.anchor_index].open_time.isoformat(),
                "entry_signal_details": "first pullback candle crossed AVWAP middle line",
                "entry_trigger_mode": cross.mode,
                "entry_intersection_price": raw_entry_price,
                "executed_entry_price": entry_price,
                "position_qty": qty,
                "risk_amount": risk_amount,
                "risk_amount_interpretation": sizing.risk_amount_interpretation,
                "position_sizing_mode": self._config.position_sizing_mode,
                "sizing_reference_price": sizing.sizing_reference_price,
                "effective_price_for_sizing": sizing.effective_price_for_sizing,
                "stop_distance": sizing.distance,
                "base_position_qty_before_costs": sizing.base_qty_before_costs,
                "qty_reduction_from_costs": sizing.qty_reduction_from_costs,
                "estimated_entry_slippage_per_unit": sizing.entry_slippage_per_unit,
                "estimated_exit_slippage_per_unit": sizing.exit_slippage_per_unit,
                "estimated_entry_fee_per_unit": sizing.entry_fee_per_unit,
                "estimated_exit_fee_per_unit": sizing.exit_fee_per_unit,
                "estimated_total_cost_per_unit": sizing.total_cost_per_unit,
                "estimated_entry_slippage_cost": estimated_entry_slippage_cost,
                "estimated_exit_slippage_cost": estimated_exit_slippage_cost,
                "estimated_entry_fee": entry_fee,
                "estimated_exit_fee": estimated_exit_fee,
                "estimated_total_cost_buffer": estimated_total_cost_buffer,
                "stop_loss_level": stop_level,
                "rigid_stop_loss_pct": self._config.rigid_stop_loss_pct,
                "rigid_stop_loss_level": rigid_stop_level,
                "trailing_activation_level": trailing_activation_level,
                "vwap_middle_line": avwap.vwap,
                "upper_band_1": avwap.upper1,
                "lower_band_1": avwap.lower1,
                "upper_band_2": avwap.upper2,
                "lower_band_2": avwap.lower2,
            },
        )
        self._log.info(
            "Opened %s position on candle %s at %.6f",
            setup.direction,
            candle_index,
            entry_price,
        )
        return None, position

    def _process_position_for_candle(
        self,
        *,
        position: _PositionState,
        candle: Candle,
        candle_index: int,
        prev_close: float,
        avwap: _AvwapSnapshot,
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> _ExitDecision | None:
        stop_level = avwap.lower2 if position.direction == "long" else avwap.upper2
        rigid_stop_level = position.rigid_stop_level_at_entry
        activation_level = self._trailing_activation_level(position.direction, avwap)

        if position.direction == "long":
            gap_exit = self._check_long_gap_exit(
                position=position,
                prev_close=prev_close,
                open_price=candle.open,
                stop_level=stop_level,
                rigid_stop_level=rigid_stop_level,
            )
            if gap_exit is not None:
                return _ExitDecision(
                    reason=gap_exit[0],
                    raw_exit_price=gap_exit[1],
                    stop_level=stop_level,
                    activation_level=activation_level,
                )

            if not position.trailing_active and candle.open >= activation_level:
                self._activate_long_trailing(
                    position=position,
                    extreme_price=candle.open,
                    candle=candle,
                    candle_index=candle_index,
                    activation_level=activation_level,
                    avwap=avwap,
                    mode="gap",
                    stats=stats,
                    decision_log=decision_log,
                )

            start_price = candle.open
            for end_price in self._price_path(candle):
                if end_price >= start_price:
                    if not position.trailing_active and start_price <= activation_level <= end_price:
                        self._activate_long_trailing(
                            position=position,
                            extreme_price=end_price,
                            candle=candle,
                            candle_index=candle_index,
                            activation_level=activation_level,
                            avwap=avwap,
                            mode="intrabar",
                            stats=stats,
                            decision_log=decision_log,
                        )
                    elif position.trailing_active:
                        self._update_long_trailing(
                            position=position,
                            extreme_price=end_price,
                            candle=candle,
                            candle_index=candle_index,
                            avwap=avwap,
                            stats=stats,
                            decision_log=decision_log,
                        )
                else:
                    adverse_exit = self._first_long_downside_exit(
                        position=position,
                        start_price=start_price,
                        end_price=end_price,
                        stop_level=stop_level,
                        rigid_stop_level=rigid_stop_level,
                    )
                    if adverse_exit is not None:
                        return _ExitDecision(
                            reason=adverse_exit[0],
                            raw_exit_price=adverse_exit[1],
                            stop_level=stop_level,
                            activation_level=activation_level,
                        )
                start_price = end_price
            return None

        gap_exit = self._check_short_gap_exit(
            position=position,
            prev_close=prev_close,
            open_price=candle.open,
            stop_level=stop_level,
            rigid_stop_level=rigid_stop_level,
        )
        if gap_exit is not None:
            return _ExitDecision(
                reason=gap_exit[0],
                raw_exit_price=gap_exit[1],
                stop_level=stop_level,
                activation_level=activation_level,
            )

        if not position.trailing_active and candle.open <= activation_level:
            self._activate_short_trailing(
                position=position,
                extreme_price=candle.open,
                candle=candle,
                candle_index=candle_index,
                activation_level=activation_level,
                avwap=avwap,
                mode="gap",
                stats=stats,
                decision_log=decision_log,
            )

        start_price = candle.open
        for end_price in self._price_path(candle):
            if end_price <= start_price:
                if not position.trailing_active and start_price >= activation_level >= end_price:
                    self._activate_short_trailing(
                        position=position,
                        extreme_price=end_price,
                        candle=candle,
                        candle_index=candle_index,
                        activation_level=activation_level,
                        avwap=avwap,
                        mode="intrabar",
                        stats=stats,
                        decision_log=decision_log,
                    )
                elif position.trailing_active:
                    self._update_short_trailing(
                        position=position,
                        extreme_price=end_price,
                        candle=candle,
                        candle_index=candle_index,
                        avwap=avwap,
                        stats=stats,
                        decision_log=decision_log,
                    )
            else:
                adverse_exit = self._first_short_upside_exit(
                    position=position,
                    start_price=start_price,
                    end_price=end_price,
                    stop_level=stop_level,
                    rigid_stop_level=rigid_stop_level,
                )
                if adverse_exit is not None:
                    return _ExitDecision(
                        reason=adverse_exit[0],
                        raw_exit_price=adverse_exit[1],
                        stop_level=stop_level,
                        activation_level=activation_level,
                    )
            start_price = end_price
        return None

    def _check_long_gap_exit(
        self,
        *,
        position: _PositionState,
        prev_close: float,
        open_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if (
            position.trailing_active
            and position.trailing_stop is not None
            and prev_close >= position.trailing_stop >= open_price
        ):
            candidates.append(("Trailing stop", position.trailing_stop))
        if rigid_stop_level is not None and prev_close >= rigid_stop_level >= open_price:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if prev_close >= stop_level >= open_price:
            candidates.append(("Stop loss", stop_level))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])

    def _check_short_gap_exit(
        self,
        *,
        position: _PositionState,
        prev_close: float,
        open_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if (
            position.trailing_active
            and position.trailing_stop is not None
            and prev_close <= position.trailing_stop <= open_price
        ):
            candidates.append(("Trailing stop", position.trailing_stop))
        if rigid_stop_level is not None and prev_close <= rigid_stop_level <= open_price:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if prev_close <= stop_level <= open_price:
            candidates.append(("Stop loss", stop_level))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])

    def _first_long_downside_exit(
        self,
        *,
        position: _PositionState,
        start_price: float,
        end_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if (
            position.trailing_active
            and position.trailing_stop is not None
            and start_price >= position.trailing_stop >= end_price
        ):
            candidates.append(("Trailing stop", position.trailing_stop))
        if rigid_stop_level is not None and start_price >= rigid_stop_level >= end_price:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if start_price >= stop_level >= end_price:
            candidates.append(("Stop loss", stop_level))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])

    def _first_short_upside_exit(
        self,
        *,
        position: _PositionState,
        start_price: float,
        end_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if (
            position.trailing_active
            and position.trailing_stop is not None
            and start_price <= position.trailing_stop <= end_price
        ):
            candidates.append(("Trailing stop", position.trailing_stop))
        if rigid_stop_level is not None and start_price <= rigid_stop_level <= end_price:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if start_price <= stop_level <= end_price:
            candidates.append(("Stop loss", stop_level))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1])

    def _activate_long_trailing(
        self,
        *,
        position: _PositionState,
        extreme_price: float,
        candle: Candle,
        candle_index: int,
        activation_level: float,
        avwap: _AvwapSnapshot,
        mode: str,
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> None:
        position.trailing_active = True
        position.extreme_price = extreme_price
        position.trailing_stop = extreme_price * (1.0 - self._config.trailing_gap_pct / 100.0)
        stats["trailing_activations"] += 1
        self._record_event(
            decision_log=decision_log,
            stats=stats,
            event="trailing_activated",
            payload={
                "timestamp": candle.close_time.isoformat(),
                "candle_index": candle_index,
                "setup_type": position.direction,
                "entry_trigger_mode": mode,
                "trailing_activation_level": activation_level,
                "trailing_stop": position.trailing_stop,
                "upper_band_1": avwap.upper1,
                "lower_band_2": avwap.lower2,
                "extreme_price": extreme_price,
            },
        )
        self._log.info(
            "Activated long trailing stop on candle %s at %.6f",
            candle_index,
            position.trailing_stop,
        )

    def _activate_short_trailing(
        self,
        *,
        position: _PositionState,
        extreme_price: float,
        candle: Candle,
        candle_index: int,
        activation_level: float,
        avwap: _AvwapSnapshot,
        mode: str,
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> None:
        position.trailing_active = True
        position.extreme_price = extreme_price
        position.trailing_stop = extreme_price * (1.0 + self._config.trailing_gap_pct / 100.0)
        stats["trailing_activations"] += 1
        self._record_event(
            decision_log=decision_log,
            stats=stats,
            event="trailing_activated",
            payload={
                "timestamp": candle.close_time.isoformat(),
                "candle_index": candle_index,
                "setup_type": position.direction,
                "entry_trigger_mode": mode,
                "trailing_activation_level": activation_level,
                "trailing_stop": position.trailing_stop,
                "lower_band_1": avwap.lower1,
                "upper_band_2": avwap.upper2,
                "extreme_price": extreme_price,
            },
        )
        self._log.info(
            "Activated short trailing stop on candle %s at %.6f",
            candle_index,
            position.trailing_stop,
        )

    def _update_long_trailing(
        self,
        *,
        position: _PositionState,
        extreme_price: float,
        candle: Candle,
        candle_index: int,
        avwap: _AvwapSnapshot,
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> None:
        if not position.trailing_active:
            return
        current_extreme = position.extreme_price if position.extreme_price is not None else extreme_price
        if extreme_price <= current_extreme:
            return
        previous_stop = position.trailing_stop
        position.extreme_price = extreme_price
        position.trailing_stop = extreme_price * (1.0 - self._config.trailing_gap_pct / 100.0)
        if previous_stop == position.trailing_stop:
            return
        stats["trailing_updates"] += 1
        self._record_event(
            decision_log=decision_log,
            stats=stats,
            event="trailing_updated",
            payload={
                "timestamp": candle.close_time.isoformat(),
                "candle_index": candle_index,
                "setup_type": position.direction,
                "trailing_stop": position.trailing_stop,
                "previous_trailing_stop": previous_stop,
                "extreme_price": extreme_price,
                "upper_band_1": avwap.upper1,
                "lower_band_2": avwap.lower2,
            },
        )

    def _update_short_trailing(
        self,
        *,
        position: _PositionState,
        extreme_price: float,
        candle: Candle,
        candle_index: int,
        avwap: _AvwapSnapshot,
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> None:
        if not position.trailing_active:
            return
        current_extreme = position.extreme_price if position.extreme_price is not None else extreme_price
        if extreme_price >= current_extreme:
            return
        previous_stop = position.trailing_stop
        position.extreme_price = extreme_price
        position.trailing_stop = extreme_price * (1.0 + self._config.trailing_gap_pct / 100.0)
        if previous_stop == position.trailing_stop:
            return
        stats["trailing_updates"] += 1
        self._record_event(
            decision_log=decision_log,
            stats=stats,
            event="trailing_updated",
            payload={
                "timestamp": candle.close_time.isoformat(),
                "candle_index": candle_index,
                "setup_type": position.direction,
                "trailing_stop": position.trailing_stop,
                "previous_trailing_stop": previous_stop,
                "extreme_price": extreme_price,
                "lower_band_1": avwap.lower1,
                "upper_band_2": avwap.upper2,
            },
        )

    def _close_position(
        self,
        *,
        position: _PositionState,
        candle: Candle,
        candle_index: int,
        exit_time: datetime,
        exit_reason: str,
        raw_exit_price: float,
        stop_level: float,
        activation_level: float,
        avwap: _AvwapSnapshot,
        trades: List[TradePerformance],
        stats: Dict[str, Any],
    ) -> float:
        exit_price = self._apply_exit_slippage(position.direction, raw_exit_price)
        exit_fee = exit_price * position.qty * self._config.taker_fee_pct
        stats["total_exit_fees"] += exit_fee

        if position.direction == "long":
            gross_pnl = (exit_price - position.entry_price) * position.qty
            return_pct = ((exit_price - position.entry_price) / position.entry_price) * 100.0
        else:
            gross_pnl = (position.entry_price - exit_price) * position.qty
            return_pct = ((position.entry_price - exit_price) / position.entry_price) * 100.0

        net_pnl = gross_pnl - position.entry_fee - exit_fee
        position_notional = position.entry_price * position.qty
        position_pnl_pct = (net_pnl / position_notional) * 100.0 if position_notional > 0 else 0.0
        r_multiple = net_pnl / position.risk_amount if position.risk_amount > 0 else 0.0
        holding_bars = max(candle_index - position.entry_index, 0)

        metadata: Dict[str, str | float | int | None] = {
            "direction": position.direction,
            "anchor_index": position.anchor_index,
            "setup_detected_index": position.setup_detected_index,
            "entry_raw_price": position.raw_entry_price,
            "entry_price": position.entry_price,
            "exit_raw_price": raw_exit_price,
            "exit_price": exit_price,
            "qty": position.qty,
            "risk_amount": position.risk_amount,
            "entry_fee": position.entry_fee,
            "exit_fee": exit_fee,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "position_pnl_pct": position_pnl_pct,
            "price_return_pct": return_pct,
            "r_multiple": r_multiple,
            "holding_bars": holding_bars,
            "stop_level_at_entry": position.stop_level_at_entry,
            "stop_level_at_exit": stop_level,
            "rigid_stop_level_at_entry": position.rigid_stop_level_at_entry,
            "trailing_activation_level_at_entry": position.trailing_activation_level_at_entry,
            "trailing_activation_level_at_exit": activation_level,
            "trailing_stop": position.trailing_stop,
            "reason": exit_reason,
            "position_sizing_mode": position.position_sizing_mode,
            "risk_amount_interpretation": position.risk_amount_interpretation,
        }

        trades.append(
            TradePerformance(
                entry_time=position.entry_time,
                exit_time=exit_time,
                pnl=net_pnl,
                return_pct=return_pct,
                notes=exit_reason,
                metadata=metadata,
            )
        )

        if exit_reason == "Stop loss":
            stats["stop_exits"] += 1
        elif exit_reason == "Rigid stop loss":
            stats["rigid_stop_exits"] += 1
        elif exit_reason == "Trailing stop":
            stats["trailing_exits"] += 1

        reason_counts = stats["exit_reason_counts"]
        reason_counts[exit_reason] = int(reason_counts.get(exit_reason, 0)) + 1

        self._record_event(
            decision_log=stats["decision_log"],
            stats=stats,
            event="position_closed",
            payload={
                "timestamp": candle.close_time.isoformat(),
                "candle_index": candle_index,
                "setup_type": position.direction,
                "exit_reason": exit_reason,
                "exit_price": exit_price,
                "raw_exit_price": raw_exit_price,
                "position_pnl": net_pnl,
                "position_pnl_pct": position_pnl_pct,
                "price_return_pct": return_pct,
                "gross_pnl": gross_pnl,
                "entry_fee": position.entry_fee,
                "exit_fee": exit_fee,
                "risk_amount": position.risk_amount,
                "risk_amount_interpretation": position.risk_amount_interpretation,
                "position_sizing_mode": position.position_sizing_mode,
                "trailing_stop": position.trailing_stop,
                "stop_loss_level": stop_level,
                "rigid_stop_loss_level": position.rigid_stop_level_at_entry,
                "trailing_activation_level": activation_level,
                "vwap_middle_line": avwap.vwap,
                "upper_band_1": avwap.upper1,
                "lower_band_1": avwap.lower1,
                "upper_band_2": avwap.upper2,
                "lower_band_2": avwap.lower2,
            },
        )
        self._log.info(
            "Closed %s position on candle %s via %s at %.6f",
            position.direction,
            candle_index,
            exit_reason,
            exit_price,
        )
        return net_pnl

    def _detect_level_cross(
        self,
        *,
        candle: Candle,
        prev_close: float,
        level: float,
        direction: Literal["up", "down"],
    ) -> _CrossDecision:
        if self._config.use_gap_cross_detection:
            if direction == "down" and prev_close >= level >= candle.open:
                return _CrossDecision(True, "gap")
            if direction == "up" and prev_close <= level <= candle.open:
                return _CrossDecision(True, "gap")

        start_price = candle.open
        for end_price in self._price_path(candle):
            if direction == "down" and start_price >= level >= end_price:
                return _CrossDecision(True, "intrabar")
            if direction == "up" and start_price <= level <= end_price:
                return _CrossDecision(True, "intrabar")
            start_price = end_price
        return _CrossDecision(False)

    def _price_path(self, candle: Candle) -> Tuple[float, float, float]:
        if abs(candle.open - candle.high) < abs(candle.open - candle.low):
            return candle.high, candle.low, candle.close
        return candle.low, candle.high, candle.close

    def _build_avwap_prefixes(
        self, candles: Sequence[Candle]
    ) -> Tuple[List[float], List[float], List[float]]:
        tpv_prefix = [0.0]
        vol_prefix = [0.0]
        tpv2_prefix = [0.0]
        for candle in candles:
            typical_price = (candle.high + candle.low + candle.close) / 3.0
            tpv_prefix.append(tpv_prefix[-1] + typical_price * candle.volume)
            vol_prefix.append(vol_prefix[-1] + candle.volume)
            tpv2_prefix.append(tpv2_prefix[-1] + (typical_price**2) * candle.volume)
        return tpv_prefix, vol_prefix, tpv2_prefix

    def _build_avwap_snapshot(
        self,
        *,
        candles: Sequence[Candle],
        anchor_index: int,
        candle_index: int,
        tpv_prefix: Sequence[float],
        vol_prefix: Sequence[float],
        tpv2_prefix: Sequence[float],
    ) -> _AvwapSnapshot:
        weighted_sum = tpv_prefix[candle_index + 1] - tpv_prefix[anchor_index]
        volume_sum = vol_prefix[candle_index + 1] - vol_prefix[anchor_index]
        weighted_sq_sum = tpv2_prefix[candle_index + 1] - tpv2_prefix[anchor_index]
        if volume_sum <= 0:
            raise ValueError("AVWAP requires positive cumulative volume")

        vwap = weighted_sum / volume_sum
        variance = max((weighted_sq_sum / volume_sum) - (vwap**2), 0.0)
        stdev = math.sqrt(variance)
        cfg = self._config
        return _AvwapSnapshot(
            anchor_index=anchor_index,
            anchor_time=candles[anchor_index].open_time,
            candle_index=candle_index,
            vwap=vwap,
            stdev=stdev,
            upper1=vwap + cfg.avwap_multiplier_1 * stdev,
            lower1=vwap - cfg.avwap_multiplier_1 * stdev,
            upper2=vwap + cfg.avwap_multiplier_2 * stdev,
            lower2=vwap - cfg.avwap_multiplier_2 * stdev,
            upper3=vwap + cfg.avwap_multiplier_3 * stdev,
            lower3=vwap - cfg.avwap_multiplier_3 * stdev,
        )

    def _trailing_activation_level(
        self, direction: Direction, avwap: _AvwapSnapshot
    ) -> float:
        threshold = self._config.trailing_activation_threshold_pct / 100.0
        if direction == "long":
            return avwap.upper1 * (1.0 + threshold)
        return avwap.lower1 * (1.0 - threshold)

    def _rigid_stop_level(
        self, direction: Direction, entry_price: float
    ) -> float | None:
        pct = self._config.rigid_stop_loss_pct / 100.0
        if pct <= 0:
            return None
        if direction == "long":
            return entry_price * (1.0 - pct)
        return entry_price * (1.0 + pct)

    def _build_sizing_decision(
        self,
        *,
        direction: Direction,
        raw_entry_price: float,
        stop_level: float,
        risk_amount: float,
    ) -> _SizingDecision | None:
        distance = (
            raw_entry_price - stop_level if direction == "long" else stop_level - raw_entry_price
        )
        if distance <= 0:
            return None

        entry_price = self._apply_entry_slippage(direction, raw_entry_price)
        estimated_exit_price = self._apply_exit_slippage(direction, raw_entry_price)
        entry_slippage_per_unit = abs(entry_price - raw_entry_price)
        exit_slippage_per_unit = abs(estimated_exit_price - raw_entry_price)
        entry_fee_per_unit = entry_price * self._config.maker_fee_pct
        exit_fee_per_unit = estimated_exit_price * self._config.taker_fee_pct
        total_cost_per_unit = (
            entry_slippage_per_unit
            + exit_slippage_per_unit
            + entry_fee_per_unit
            + exit_fee_per_unit
        )

        if self._config.position_sizing_mode == "risk_amount_per_price":
            base_qty_before_costs = risk_amount / raw_entry_price
            effective_price_for_sizing = raw_entry_price + total_cost_per_unit
            qty = risk_amount / effective_price_for_sizing
            risk_amount_interpretation = "position_notional_budget"
        else:
            qty = risk_amount / distance
            base_qty_before_costs = qty
            effective_price_for_sizing = distance
            risk_amount_interpretation = "stop_loss_risk"

        return _SizingDecision(
            qty=qty,
            distance=distance,
            entry_price=entry_price,
            estimated_exit_price=estimated_exit_price,
            risk_amount_interpretation=risk_amount_interpretation,
            base_qty_before_costs=base_qty_before_costs,
            qty_reduction_from_costs=max(base_qty_before_costs - qty, 0.0),
            sizing_reference_price=raw_entry_price,
            effective_price_for_sizing=effective_price_for_sizing,
            entry_slippage_per_unit=entry_slippage_per_unit,
            exit_slippage_per_unit=exit_slippage_per_unit,
            entry_fee_per_unit=entry_fee_per_unit,
            exit_fee_per_unit=exit_fee_per_unit,
            total_cost_per_unit=total_cost_per_unit,
        )

    def _apply_entry_slippage(self, direction: Direction, price: float) -> float:
        slip = self._config.entry_slippage_pct
        if direction == "long":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _apply_exit_slippage(self, direction: Direction, price: float) -> float:
        slip = self._config.exit_slippage_pct
        if direction == "long":
            return price * (1.0 - slip)
        return price * (1.0 + slip)

    def _record_event(
        self,
        *,
        decision_log: List[Dict[str, Any]],
        stats: Dict[str, Any],
        event: str,
        payload: Dict[str, Any],
    ) -> None:
        if len(decision_log) >= self._config.max_decision_log_entries:
            stats["decision_log_truncated_count"] += 1
            return
        item = {"event": event}
        item.update(payload)
        decision_log.append(item)

    def _summarize_trade_stats(
        self, trades: Sequence[TradePerformance]
    ) -> Dict[str, Any]:
        if not trades:
            return {
                "trade_count": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_r_multiple": 0.0,
                "avg_return_pct": 0.0,
                "avg_holding_bars": 0.0,
                "long_trade_count": 0,
                "short_trade_count": 0,
                "gross_profit_long": 0.0,
                "gross_profit_short": 0.0,
                "gross_loss_long": 0.0,
                "gross_loss_short": 0.0,
                "total_net_pnl": 0.0,
            }

        wins = sum(1 for trade in trades if trade.pnl > 0)
        losses = sum(1 for trade in trades if trade.pnl < 0)
        long_trades: List[TradePerformance] = []
        short_trades: List[TradePerformance] = []
        r_values: List[float] = []
        hold_bars: List[int] = []
        gross_profit_long = 0.0
        gross_profit_short = 0.0
        gross_loss_long = 0.0
        gross_loss_short = 0.0

        for trade in trades:
            metadata = dict(trade.metadata or {})
            direction = str(metadata.get("direction", ""))
            r_values.append(float(metadata.get("r_multiple", 0.0)))
            hold_bars.append(int(metadata.get("holding_bars", 0)))

            if direction == "long":
                long_trades.append(trade)
                if trade.pnl >= 0:
                    gross_profit_long += trade.pnl
                else:
                    gross_loss_long += abs(trade.pnl)
            elif direction == "short":
                short_trades.append(trade)
                if trade.pnl >= 0:
                    gross_profit_short += trade.pnl
                else:
                    gross_loss_short += abs(trade.pnl)

        return {
            "trade_count": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(trades),
            "avg_r_multiple": mean(r_values) if r_values else 0.0,
            "avg_return_pct": mean(trade.return_pct for trade in trades),
            "avg_holding_bars": mean(hold_bars) if hold_bars else 0.0,
            "best_trade_pnl": max(trade.pnl for trade in trades),
            "worst_trade_pnl": min(trade.pnl for trade in trades),
            "long_trade_count": len(long_trades),
            "short_trade_count": len(short_trades),
            "gross_profit_long": gross_profit_long,
            "gross_profit_short": gross_profit_short,
            "gross_loss_long": gross_loss_long,
            "gross_loss_short": gross_loss_short,
            "total_net_pnl": sum(trade.pnl for trade in trades),
        }

    def _config_as_dict(self) -> Dict[str, Any]:
        cfg = self._config
        return {
            "symbol": cfg.symbol,
            "timeframe": cfg.timeframe,
            "initial_equity": cfg.initial_equity,
            "leverage": cfg.leverage,
            "equity_risk_pct": cfg.equity_risk_pct,
            "ema_length": cfg.ema_length,
            "consecutive_count": cfg.consecutive_count,
            "ema_validation_mode": cfg.ema_validation_mode,
            "setup_waiting_replacement_mode": cfg.setup_waiting_replacement_mode,
            "position_sizing_mode": cfg.position_sizing_mode,
            "avwap_multiplier_1": cfg.avwap_multiplier_1,
            "avwap_multiplier_2": cfg.avwap_multiplier_2,
            "avwap_multiplier_3": cfg.avwap_multiplier_3,
            "rigid_stop_loss_pct": cfg.rigid_stop_loss_pct,
            "trailing_activation_threshold_pct": cfg.trailing_activation_threshold_pct,
            "trailing_gap_pct": cfg.trailing_gap_pct,
            "maker_fee_pct": cfg.maker_fee_pct,
            "taker_fee_pct": cfg.taker_fee_pct,
            "entry_slippage_pct": cfg.entry_slippage_pct,
            "exit_slippage_pct": cfg.exit_slippage_pct,
            "use_gap_cross_detection": cfg.use_gap_cross_detection,
            "max_decision_log_entries": cfg.max_decision_log_entries,
        }
