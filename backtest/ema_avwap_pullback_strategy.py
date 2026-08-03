from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .ema_avwap_pullback.calculations import EmaAvwapCalculationsMixin
from .ema_avwap_pullback.config import (
    Direction,
    EmaAvwapPullbackStrategyConfig,
    EntryMode,
    ExitBand,
    ExitMode,
)
from .ema_avwap_pullback.models import (
    _AvwapSnapshot,
    _CrossDecision,
    _ExitDecision,
    _PositionState,
    _SetupState,
)
from .ema_avwap_pullback.reporting import EmaAvwapReportingMixin
from .indicators import ema as calc_ema


__all__ = [
    "EmaAvwapPullbackStrategy",
    "EmaAvwapPullbackStrategyConfig",
    "EntryMode",
    "ExitMode",
    "ExitBand",
]


class EmaAvwapPullbackStrategy(
    EmaAvwapCalculationsMixin, EmaAvwapReportingMixin, BacktestStrategy
):
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
        is_closed_candle_compatible = (
            cfg.entry_mode is EntryMode.CLOSE and cfg.exit_mode is ExitMode.CLOSE
        )
        stats: Dict[str, Any] = {
            "config": self._config_as_dict(),
            "execution_assumptions": {
                "historical_data_source": "completed_ohlcv_candles_only",
                "live_strategy_compatibility": (
                    "closed_candle_equivalent"
                    if is_closed_candle_compatible
                    else "live_tick_ohlc_approximation"
                ),
                "live_tick_approximation_warning": (
                    None
                    if is_closed_candle_compatible
                    else "Live entry/exit behavior uses forming-candle AVWAP and "
                    "individual quotes. Historical OHLCV cannot replay that path "
                    "without lower-timeframe or tick data."
                ),
                "entry_mode": cfg.entry_mode.value,
                "exit_mode": cfg.exit_mode.value,
                "exit_band": cfg.exit_band.value,
                "target_band_number": cfg.exit_band.number,
                "entry_fill_model": (
                    "closed_candle_market_price_proxy"
                    if cfg.entry_mode is EntryMode.CLOSE
                    else "next_candle_open_first_live_quote_proxy"
                ),
                "exit_fill_model": (
                    "closed_candle_avwap_target_emulation"
                    if cfg.exit_mode is ExitMode.CLOSE
                    else "ohlc_live_avwap_target_emulation"
                ),
                "gap_fill_model": "next_observed_open_proxy",
                "intrabar_price_path": "open -> nearest extreme -> far extreme -> close",
                "gap_cross_detection": cfg.use_gap_cross_detection,
                "entry_avwap_value_source": (
                    "closed_candle_snapshot"
                    if cfg.entry_mode is EntryMode.CLOSE
                    else "completed_bar_proxy_for_forming_candle"
                ),
                "target_avwap_value_source": (
                    "closed_candle_snapshot"
                    if cfg.exit_mode is ExitMode.CLOSE
                    else "completed_bar_proxy_for_forming_candle"
                ),
                "live_kline_limitation": (
                    "Historical OHLCV has no per-poll forming-candle snapshots; "
                    "live modes use the next candle open as the first quote and "
                    "that bar's final AVWAP snapshot as a deterministic proxy."
                ),
                "dynamic_trailing_management": (
                    "enabled: live-equivalent activation and ratchet rules over "
                    "the deterministic OHLC path proxy"
                ),
                "protective_stop_model": "rigid_stop_plus_ratcheting_trailing_stop",
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
            "entries_skipped_minimum_balance": 0,
            "entries_skipped_zero_qty": 0,
            "entries_capped_by_live_notional_limits": 0,
            "setups_expired": 0,
            "setups_invalidated_by_ema": 0,
            "setups_discarded_unfavorable_first_observation": 0,
            "entries_skipped_excessive_deviation": 0,
            "entries_skipped_unmarketable": 0,
            "entries_skipped_stop_already_breached": 0,
            "stop_exits": 0,
            "rigid_stop_exits": 0,
            "trailing_exits": 0,
            "target_exits_band_1": 0,
            "target_exits_band_2": 0,
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

        if not is_closed_candle_compatible:
            self._log.warning(
                "EMA+AVWAP backtest uses a live-tick OHLC approximation "
                "(entry_mode=%s, exit_mode=%s). Use close/close for a "
                "closed-candle-equivalent comparison with live trading.",
                cfg.entry_mode.value,
                cfg.exit_mode.value,
            )

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
                    trailing_avwap=position.trailing_avwap,
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
                        target_level=exit_decision.target_level,
                        avwap=avwap,
                        trades=trades,
                        stats=stats,
                    )
                    realized_equity += pnl
                    position = None
                else:
                    # The live coordinator refreshes the trailing reference
                    # only when this candle has closed.  Targets remain based
                    # on the forming-candle AVWAP proxy passed above.
                    position.trailing_avwap = avwap

            entered_this_candle = False
            if position is None:
                active_long_setup, position = self._process_pending_setup(
                    setup=active_long_setup,
                    candle=candle,
                    candle_index=idx,
                    prev_close=prev_close,
                    ema_value=ema_value,
                    realized_equity=realized_equity,
                    candles=candles,
                    tpv_prefix=tpv_prefix,
                    vol_prefix=vol_prefix,
                    tpv2_prefix=tpv2_prefix,
                    stats=stats,
                    decision_log=decision_log,
                )
                if position is not None:
                    entered_this_candle = True
                    active_long_setup = None
                    active_short_setup = None

            if position is None:
                active_short_setup, position = self._process_pending_setup(
                    setup=active_short_setup,
                    candle=candle,
                    candle_index=idx,
                    prev_close=prev_close,
                    ema_value=ema_value,
                    realized_equity=realized_equity,
                    candles=candles,
                    tpv_prefix=tpv_prefix,
                    vol_prefix=vol_prefix,
                    tpv2_prefix=tpv2_prefix,
                    stats=stats,
                    decision_log=decision_log,
                )
                if position is not None:
                    entered_this_candle = True
                    active_long_setup = None
                    active_short_setup = None

            if position is not None and entered_this_candle and position.entry_path_remaining:
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
                    prev_close=position.raw_entry_price,
                    avwap=avwap,
                    trailing_avwap=position.trailing_avwap,
                    stats=stats,
                    decision_log=decision_log,
                    path_override=position.entry_path_remaining,
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
                        target_level=exit_decision.target_level,
                        avwap=avwap,
                        trades=trades,
                        stats=stats,
                    )
                    realized_equity += pnl
                    position = None

            if position is None and not entered_this_candle:
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
                stop_level=avwap.lower1 if position.direction == "long" else avwap.upper1,
                activation_level=self._trailing_activation_level(position.direction, avwap),
                target_level=self._target_band_level(
                    position.direction, avwap, position.exit_band
                ),
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

    @staticmethod
    def _price_respects_ema(
        *, direction: Direction, price: float, ema_value: float
    ) -> bool:
        return price > ema_value if direction == "long" else price < ema_value

    @staticmethod
    def _entry_price_is_marketable(
        *, direction: Direction, current_price: float, entry_price: float
    ) -> bool:
        return current_price <= entry_price if direction == "long" else current_price >= entry_price

    @staticmethod
    def _entry_deviation_pct(price: float, avwap: float) -> float:
        if avwap <= 0:
            return float("inf")
        return abs(price - avwap) / avwap * 100.0

    @staticmethod
    def _setup_after_skipped_entry(
        setup: _SetupState, entry_mode: EntryMode
    ) -> _SetupState:
        """Preserve a setup when the live coordinator can retry it on a tick."""
        if entry_mode is EntryMode.LIVE:
            return replace(setup, is_waiting_for_cross=True)
        return setup

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
        ema_value: float,
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

        setup_age_bars = candle_index - setup.detected_index
        if setup_age_bars > self._config.max_setup_age_bars:
            stats["setups_expired"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="setup_discarded",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "reason": "maximum setup age exceeded",
                    "setup_age_bars": setup_age_bars,
                    "max_setup_age_bars": self._config.max_setup_age_bars,
                },
            )
            return None, None

        entry_mode = self._config.entry_mode
        has_persisted_live_observation = (
            setup.last_observed_price is not None
            and setup.last_observed_middle is not None
        )
        is_first_live_observation = (
            entry_mode is EntryMode.LIVE
            and not has_persisted_live_observation
            and candle_index == setup.detected_index + 1
        )
        # A live entry receives its first forming-candle quote immediately
        # after a setup is detected.  Its candle has not closed yet, so using
        # the final close here would add a look-ahead EMA filter that does not
        # exist in the coordinator. Every later evaluation, including a retry
        # after a rejected live entry candidate, does see this completed-bar
        # gate before the next forming-candle tick.
        if not is_first_live_observation and not self._price_respects_ema(
            direction=setup.direction,
            price=candle.close,
            ema_value=ema_value,
        ):
            stats["setups_invalidated_by_ema"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="setup_discarded",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "reason": "closed price crossed the EMA",
                    "closed_price": candle.close,
                    "ema_value": ema_value,
                },
            )
            return None, None

        avwap = self._build_avwap_snapshot(
            candles=candles,
            anchor_index=setup.anchor_index,
            candle_index=candle_index,
            tpv_prefix=tpv_prefix,
            vol_prefix=vol_prefix,
            tpv2_prefix=tpv2_prefix,
        )

        exit_mode = self._config.exit_mode
        exit_band = self._config.exit_band
        decision_price: float
        raw_entry_price: float
        if entry_mode is EntryMode.CLOSE:
            expects_pullback = (
                candle.is_bearish() if setup.direction == "long" else candle.is_bullish()
            )
            if not expects_pullback:
                return setup, None
            crossed_middle = (
                candle.close < avwap.vwap
                if setup.direction == "long"
                else candle.close > avwap.vwap
            )
            cross = _CrossDecision(crossed_middle, "candle_close")
            decision_price = candle.close
            # Live mode 2 submits its order at the current market price after the
            # close signal. The closing price is the best historical-bar proxy.
            raw_entry_price = candle.close
        else:
            # The live coordinator has no previous tick for a new setup.  Its
            # first quote must already be on the favourable side of AVWAP or it
            # discards the setup; it does not infer an intrabar cross later
            # from OHLC extremes.  The next candle open is the only unbiased
            # historical proxy for that first live quote.
            decision_price = candle.open
            raw_entry_price = avwap.vwap
            if (
                setup.last_observed_price is None
                or setup.last_observed_middle is None
            ):
                crossed = self._price_is_past_entry_line(
                    direction=setup.direction,
                    price=decision_price,
                    entry_price=raw_entry_price,
                )
            else:
                crossed = self._price_crossed_entry_line(
                    direction=setup.direction,
                    previous_price=setup.last_observed_price,
                    previous_entry_price=setup.last_observed_middle,
                    current_price=decision_price,
                    current_entry_price=raw_entry_price,
                )
            cross = _CrossDecision(
                crossed,
                "live_tick",
                tuple((candle.open, *self._price_path(candle))),
            )

        if not cross.crossed:
            has_previous_live_observation = (
                setup.last_observed_price is not None
                and setup.last_observed_middle is not None
            )
            if entry_mode is EntryMode.LIVE and not has_previous_live_observation:
                stats["setups_discarded_unfavorable_first_observation"] += 1
                self._record_event(
                    decision_log=decision_log,
                    stats=stats,
                    event="setup_discarded",
                    payload={
                        "timestamp": candle.close_time.isoformat(),
                        "candle_index": candle_index,
                        "setup_type": setup.direction,
                        "entry_mode": entry_mode.value,
                        "exit_mode": exit_mode.value,
                        "exit_band": exit_band.value,
                        "reason": "first live observation was not at a favorable AVWAP cross",
                        "decision_price": decision_price,
                        "vwap_middle_line": avwap.vwap,
                    },
                )
                return None, None
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="setup_waiting",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_mode": entry_mode.value,
                    "exit_mode": exit_mode.value,
                    "exit_band": exit_band.value,
                    "consecutive_count": setup.consecutive_count,
                    "anchor_index": setup.anchor_index,
                    "anchor_time": candles[setup.anchor_index].open_time.isoformat(),
                    "entry_signal_details": (
                        "opposite candle did not close past AVWAP middle"
                        if entry_mode is EntryMode.CLOSE
                        else "first live observation did not touch AVWAP middle"
                    ),
                    "vwap_middle_line": avwap.vwap,
                    "upper_band_1": avwap.upper1,
                    "lower_band_1": avwap.lower1,
                    "upper_band_2": avwap.upper2,
                    "lower_band_2": avwap.lower2,
                },
            )
            return (
                replace(
                    setup,
                    is_waiting_for_cross=True,
                    last_observed_price=(
                        decision_price if entry_mode is EntryMode.LIVE else None
                    ),
                    last_observed_middle=(
                        avwap.vwap if entry_mode is EntryMode.LIVE else None
                    ),
                ),
                None,
            )

        if realized_equity <= self._config.minimum_balance_usdt:
            stats["entries_skipped_minimum_balance"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="entry_skipped",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_signal_details": "available-equity proxy was at or below the minimum balance",
                    "realized_equity": realized_equity,
                    "minimum_balance_usdt": self._config.minimum_balance_usdt,
                },
            )
            return self._setup_after_skipped_entry(setup, entry_mode), None

        requested_notional_budget = realized_equity * (
            self._config.position_notional_pct / 100.0
        )
        percentage_cap = realized_equity * (
            self._config.max_position_size_pct / 100.0
        )
        position_notional_budget = min(
            requested_notional_budget,
            percentage_cap,
            self._config.max_entry_notional_usdt,
        )
        if position_notional_budget <= 0:
            stats["entries_skipped_non_positive_equity"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="entry_skipped",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_signal_details": "non-positive position-notional budget",
                    "realized_equity": realized_equity,
                },
            )
            return self._setup_after_skipped_entry(setup, entry_mode), None

        if position_notional_budget < requested_notional_budget:
            stats["entries_capped_by_live_notional_limits"] += 1

        dynamic_stop_level = avwap.lower1 if setup.direction == "long" else avwap.upper1
        sizing = self._build_sizing_decision(
            direction=setup.direction,
            raw_entry_price=raw_entry_price,
            position_notional_budget=position_notional_budget,
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
                    "entry_mode": entry_mode.value,
                    "exit_mode": exit_mode.value,
                    "exit_band": exit_band.value,
                    "entry_signal_details": "notional-budget sizing was unavailable",
                    "entry_intersection_price": raw_entry_price,
                },
            )
            return self._setup_after_skipped_entry(setup, entry_mode), None

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
            return self._setup_after_skipped_entry(setup, entry_mode), None

        entry_price = sizing.entry_price
        rigid_stop_level = self._rigid_stop_level(setup.direction, entry_price)
        if entry_mode is EntryMode.LIVE and not self._entry_price_is_marketable(
            direction=setup.direction,
            current_price=decision_price,
            entry_price=entry_price,
        ):
            stats["entries_skipped_unmarketable"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="entry_skipped",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_signal_details": "live AVWAP middle order was no longer marketable",
                    "decision_price": decision_price,
                    "entry_price": entry_price,
                },
            )
            return self._setup_after_skipped_entry(setup, entry_mode), None
        if rigid_stop_level is not None and self._rigid_stop_is_touched(
            direction=setup.direction,
            price=decision_price,
            rigid_stop=rigid_stop_level,
        ):
            stats["entries_skipped_stop_already_breached"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="entry_skipped",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_signal_details": "price was already beyond the protective stop",
                    "decision_price": decision_price,
                    "rigid_stop_loss_level": rigid_stop_level,
                },
            )
            return self._setup_after_skipped_entry(setup, entry_mode), None
        deviation_pct = self._entry_deviation_pct(decision_price, avwap.vwap)
        if deviation_pct > self._config.max_entry_deviation_pct:
            stats["entries_skipped_excessive_deviation"] += 1
            self._record_event(
                decision_log=decision_log,
                stats=stats,
                event="entry_skipped",
                payload={
                    "timestamp": candle.close_time.isoformat(),
                    "candle_index": candle_index,
                    "setup_type": setup.direction,
                    "entry_signal_details": "live price exceeded the AVWAP deviation limit",
                    "decision_price": decision_price,
                    "vwap_middle_line": avwap.vwap,
                    "entry_deviation_pct": deviation_pct,
                    "max_entry_deviation_pct": self._config.max_entry_deviation_pct,
                },
            )
            return self._setup_after_skipped_entry(setup, entry_mode), None

        entry_fee = entry_price * qty * self._config.taker_fee_pct
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
            entry_time=(
                candle.close_time if entry_mode is EntryMode.CLOSE else candle.open_time
            ),
            entry_index=candle_index,
            raw_entry_price=raw_entry_price,
            entry_price=entry_price,
            qty=qty,
            position_notional_budget=position_notional_budget,
            entry_fee=entry_fee,
            stop_level_at_entry=dynamic_stop_level,
            rigid_stop_level_at_entry=rigid_stop_level,
            trailing_activation_level_at_entry=trailing_activation_level,
            entry_trigger_mode=cross.mode or "live_tick",
            position_sizing_mode=self._config.position_sizing_mode,
            entry_mode=entry_mode,
            exit_mode=exit_mode,
            exit_band=exit_band,
            decision_price=decision_price,
            ema_value_at_entry=ema_value,
            entry_path_remaining=(
                () if entry_mode is EntryMode.CLOSE else cross.remaining_path
            ),
            trailing_avwap=avwap,
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
                "entry_mode": entry_mode.value,
                "exit_mode": exit_mode.value,
                "exit_band": exit_band.value,
                "exit_band_number": exit_band.number,
                "consecutive_count": setup.consecutive_count,
                "anchor_index": setup.anchor_index,
                "anchor_time": candles[setup.anchor_index].open_time.isoformat(),
                "entry_signal_details": (
                    "opposite candle closed past AVWAP middle"
                    if entry_mode is EntryMode.CLOSE
                    else "first live observation was at or past AVWAP middle"
                ),
                "entry_trigger_mode": cross.mode,
                "decision_price": decision_price,
                "entry_intersection_price": raw_entry_price,
                "executed_entry_price": entry_price,
                "position_qty": qty,
                "position_notional_budget": position_notional_budget,
                "requested_position_notional_budget": requested_notional_budget,
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
                "dynamic_stop_level_at_entry": dynamic_stop_level,
                "dynamic_stop_management_enabled": True,
                "rigid_stop_loss_pct": self._config.rigid_stop_loss_pct,
                "rigid_stop_loss_level": rigid_stop_level,
                "trailing_activation_level": trailing_activation_level,
                "target_level": self._target_band_level(setup.direction, avwap, exit_band),
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
        trailing_avwap: _AvwapSnapshot | None = None,
        path_override: Sequence[float] | None = None,
    ) -> _ExitDecision | None:
        """Emulate live target, rigid-stop, and trailing decisions over OHLC.

        The live coordinator observes individual prices, whereas historical
        candles only provide OHLC.  We therefore process the deterministic
        ``open -> nearest extreme -> far extreme -> close`` proxy already used
        by the strategy.  At every observed price the ordering is the same as
        live trading: target first, then protective stops; a trailing stop only
        activates or ratchets on a favourable move.  A zero-percent trailing
        gap is the intentional exception: its inclusive stop condition fires
        at the activating price in both live and backtest execution.
        """
        stop_level = avwap.lower1 if position.direction == "long" else avwap.upper1
        trailing_reference = trailing_avwap or avwap
        rigid_stop_level = position.rigid_stop_level_at_entry
        activation_level = self._trailing_activation_level(
            position.direction, trailing_reference
        )
        exit_mode = position.exit_mode
        exit_band = position.exit_band
        target_level = self._target_band_level(
            position.direction, avwap, exit_band
        )
        target_reason = f"AVWAP band {exit_band.number} target"
        close_target_reason = f"{target_reason} on candle close"
        target_is_live = exit_mode is ExitMode.LIVE

        if path_override is None:
            gap_exit = self._first_exit_on_price_move(
                direction=position.direction,
                start_price=prev_close,
                end_price=candle.open,
                target_level=target_level,
                target_reason=target_reason,
                rigid_stop_level=rigid_stop_level,
                trailing_stop=(
                    position.trailing_stop if position.trailing_active else None
                ),
                include_target=target_is_live,
            )
            if gap_exit is not None:
                return _ExitDecision(
                    reason=gap_exit[0],
                    # Live target/stop checks run on the currently observed
                    # quote.  A historical bar gap therefore exits at the
                    # next observed open proxy, not at an unavailable exact
                    # intersection within the gap.
                    raw_exit_price=candle.open,
                    stop_level=stop_level,
                    activation_level=activation_level,
                    target_level=target_level,
                )
            path = (candle.open, *self._price_path(candle))
        else:
            path = tuple(path_override)

        if not path:
            return None
        first_price = path[0]
        observed_exit = self._exit_at_observed_price(
            position=position,
            price=first_price,
            target_level=target_level,
            target_reason=(
                target_reason
                if target_is_live or len(path) == 1
                else close_target_reason
            ),
            rigid_stop_level=rigid_stop_level,
            include_target=target_is_live or len(path) == 1,
        )
        if observed_exit is not None:
            return _ExitDecision(
                reason=observed_exit[0],
                raw_exit_price=first_price,
                stop_level=stop_level,
                activation_level=activation_level,
                target_level=target_level,
            )
        trailing_exit = self._advance_trailing_at_price(
            position=position,
            price=first_price,
            candle=candle,
            candle_index=candle_index,
            activation_level=activation_level,
            avwap=avwap,
            stats=stats,
            decision_log=decision_log,
        )
        if trailing_exit is not None:
            return _ExitDecision(
                reason=trailing_exit[0],
                raw_exit_price=trailing_exit[1],
                stop_level=stop_level,
                activation_level=activation_level,
                target_level=target_level,
            )

        start_price = first_price
        for path_index, end_price in enumerate(path[1:], start=1):
            crossed = self._first_exit_on_price_move(
                direction=position.direction,
                start_price=start_price,
                end_price=end_price,
                target_level=target_level,
                target_reason=target_reason,
                rigid_stop_level=rigid_stop_level,
                trailing_stop=(
                    position.trailing_stop if position.trailing_active else None
                ),
                include_target=target_is_live,
            )
            if crossed is not None:
                return _ExitDecision(
                    reason=crossed[0],
                    raw_exit_price=crossed[1],
                    stop_level=stop_level,
                    activation_level=activation_level,
                    target_level=target_level,
                )
            is_close = path_index == len(path) - 1
            if (
                is_close
                and not target_is_live
                and self._target_is_touched(
                    direction=position.direction,
                    price=end_price,
                    target=target_level,
                )
            ):
                return _ExitDecision(
                    reason=close_target_reason,
                    raw_exit_price=end_price,
                    stop_level=stop_level,
                    activation_level=activation_level,
                    target_level=target_level,
                )

            trailing_exit = self._advance_trailing_on_price_move(
                position=position,
                start_price=start_price,
                end_price=end_price,
                candle=candle,
                candle_index=candle_index,
                activation_level=activation_level,
                avwap=avwap,
                stats=stats,
                decision_log=decision_log,
            )
            if trailing_exit is not None:
                return _ExitDecision(
                    reason=trailing_exit[0],
                    raw_exit_price=trailing_exit[1],
                    stop_level=stop_level,
                    activation_level=activation_level,
                    target_level=target_level,
                )
            start_price = end_price
        return None

    def _exit_at_observed_price(
        self,
        *,
        position: _PositionState,
        price: float,
        target_level: float,
        target_reason: str,
        rigid_stop_level: float | None,
        include_target: bool,
    ) -> Tuple[str, float] | None:
        # Live target handling precedes the trailing manager on each tick.
        # Close-mode targets are evaluated only at the completed-bar close.
        if include_target and self._target_is_touched(
            direction=position.direction, price=price, target=target_level
        ):
            return target_reason, target_level

        candidates: List[Tuple[str, float]] = []
        trailing_stop = position.trailing_stop if position.trailing_active else None
        if position.direction == "long":
            if trailing_stop is not None and price <= trailing_stop:
                candidates.append(("Trailing stop", trailing_stop))
            if rigid_stop_level is not None and price <= rigid_stop_level:
                candidates.append(("Rigid stop loss", rigid_stop_level))
            return max(candidates, key=lambda item: item[1]) if candidates else None

        if trailing_stop is not None and price >= trailing_stop:
            candidates.append(("Trailing stop", trailing_stop))
        if rigid_stop_level is not None and price >= rigid_stop_level:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        return min(candidates, key=lambda item: item[1]) if candidates else None

    def _first_exit_on_price_move(
        self,
        *,
        direction: Direction,
        start_price: float,
        end_price: float,
        target_level: float,
        target_reason: str,
        rigid_stop_level: float | None,
        trailing_stop: float | None,
        include_target: bool,
    ) -> Tuple[str, float] | None:
        """Return the first executable level reached on one price segment."""
        if end_price == start_price:
            return None

        candidates: List[Tuple[str, float]] = []
        if direction == "long":
            if (
                include_target
                and end_price > start_price
                and start_price < target_level <= end_price
            ):
                candidates.append((target_reason, target_level))
            if end_price < start_price:
                if trailing_stop is not None and start_price > trailing_stop >= end_price:
                    candidates.append(("Trailing stop", trailing_stop))
                if (
                    rigid_stop_level is not None
                    and start_price > rigid_stop_level >= end_price
                ):
                    candidates.append(("Rigid stop loss", rigid_stop_level))
        else:
            if (
                include_target
                and end_price < start_price
                and start_price > target_level >= end_price
            ):
                candidates.append((target_reason, target_level))
            if end_price > start_price:
                if trailing_stop is not None and start_price < trailing_stop <= end_price:
                    candidates.append(("Trailing stop", trailing_stop))
                if (
                    rigid_stop_level is not None
                    and start_price < rigid_stop_level <= end_price
                ):
                    candidates.append(("Rigid stop loss", rigid_stop_level))

        if not candidates:
            return None
        return (
            min(candidates, key=lambda item: item[1])
            if end_price > start_price
            else max(candidates, key=lambda item: item[1])
        )

    def _advance_trailing_at_price(
        self,
        *,
        position: _PositionState,
        price: float,
        candle: Candle,
        candle_index: int,
        activation_level: float,
        avwap: _AvwapSnapshot,
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> Tuple[str, float] | None:
        if position.direction == "long":
            if not position.trailing_active and price >= activation_level:
                self._activate_long_trailing(
                    position=position,
                    extreme_price=price,
                    candle=candle,
                    candle_index=candle_index,
                    activation_level=activation_level,
                    avwap=avwap,
                    mode="ohlc_path",
                    stats=stats,
                    decision_log=decision_log,
                )
            elif position.trailing_active:
                self._update_long_trailing(
                    position=position,
                    extreme_price=price,
                    candle=candle,
                    candle_index=candle_index,
                    avwap=avwap,
                    stats=stats,
                    decision_log=decision_log,
                )
            return self._trailing_exit_at_price(position=position, price=price)

        if not position.trailing_active and price <= activation_level:
            self._activate_short_trailing(
                position=position,
                extreme_price=price,
                candle=candle,
                candle_index=candle_index,
                activation_level=activation_level,
                avwap=avwap,
                mode="ohlc_path",
                stats=stats,
                decision_log=decision_log,
            )
        elif position.trailing_active:
            self._update_short_trailing(
                position=position,
                extreme_price=price,
                candle=candle,
                candle_index=candle_index,
                avwap=avwap,
                stats=stats,
                decision_log=decision_log,
            )
        return self._trailing_exit_at_price(position=position, price=price)

    def _advance_trailing_on_price_move(
        self,
        *,
        position: _PositionState,
        start_price: float,
        end_price: float,
        candle: Candle,
        candle_index: int,
        activation_level: float,
        avwap: _AvwapSnapshot,
        stats: Dict[str, Any],
        decision_log: List[Dict[str, Any]],
    ) -> Tuple[str, float] | None:
        if position.direction == "long" and end_price >= start_price:
            return self._advance_trailing_at_price(
                position=position,
                price=end_price,
                candle=candle,
                candle_index=candle_index,
                activation_level=activation_level,
                avwap=avwap,
                stats=stats,
                decision_log=decision_log,
            )
        if position.direction == "short" and end_price <= start_price:
            return self._advance_trailing_at_price(
                position=position,
                price=end_price,
                candle=candle,
                candle_index=candle_index,
                activation_level=activation_level,
                avwap=avwap,
                stats=stats,
                decision_log=decision_log,
            )
        return None

    def _trailing_exit_at_price(
        self, *, position: _PositionState, price: float
    ) -> Tuple[str, float] | None:
        """Mirror the live tick's inclusive trailing-stop trigger."""
        if not position.trailing_active or position.trailing_stop is None:
            return None
        if position.direction == "long" and price <= position.trailing_stop:
            return "Trailing stop", position.trailing_stop
        if position.direction == "short" and price >= position.trailing_stop:
            return "Trailing stop", position.trailing_stop
        return None

    def _target_is_touched(self, *, direction: Direction, price: float, target: float) -> bool:
        if direction == "long":
            return price >= target
        return price <= target

    def _rigid_stop_is_touched(
        self, *, direction: Direction, price: float, rigid_stop: float
    ) -> bool:
        if direction == "long":
            return price <= rigid_stop
        return price >= rigid_stop

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

    def _check_long_open_exit(
        self,
        *,
        position: _PositionState,
        open_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if (
            position.trailing_active
            and position.trailing_stop is not None
            and open_price <= position.trailing_stop
        ):
            candidates.append(("Trailing stop", position.trailing_stop))
        if rigid_stop_level is not None and open_price <= rigid_stop_level:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if open_price <= stop_level:
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

    def _check_short_open_exit(
        self,
        *,
        position: _PositionState,
        open_price: float,
        stop_level: float,
        rigid_stop_level: float | None,
    ) -> Tuple[str, float] | None:
        candidates: List[Tuple[str, float]] = []
        if (
            position.trailing_active
            and position.trailing_stop is not None
            and open_price >= position.trailing_stop
        ):
            candidates.append(("Trailing stop", position.trailing_stop))
        if rigid_stop_level is not None and open_price >= rigid_stop_level:
            candidates.append(("Rigid stop loss", rigid_stop_level))
        if open_price >= stop_level:
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
        trailing_stop = extreme_price * (1.0 - self._config.trailing_gap_pct / 100.0)
        position.trailing_stop = self._constrain_trailing_stop(position, trailing_stop)
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
        trailing_stop = extreme_price * (1.0 + self._config.trailing_gap_pct / 100.0)
        position.trailing_stop = self._constrain_trailing_stop(position, trailing_stop)
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
        current_extreme = (
            position.extreme_price if position.extreme_price is not None else extreme_price
        )
        if extreme_price <= current_extreme:
            return
        previous_stop = position.trailing_stop
        position.extreme_price = extreme_price
        trailing_stop = extreme_price * (1.0 - self._config.trailing_gap_pct / 100.0)
        position.trailing_stop = self._constrain_trailing_stop(position, trailing_stop)
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
        current_extreme = (
            position.extreme_price if position.extreme_price is not None else extreme_price
        )
        if extreme_price >= current_extreme:
            return
        previous_stop = position.trailing_stop
        position.extreme_price = extreme_price
        trailing_stop = extreme_price * (1.0 + self._config.trailing_gap_pct / 100.0)
        position.trailing_stop = self._constrain_trailing_stop(position, trailing_stop)
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

    def _constrain_trailing_stop(self, position: _PositionState, trailing_stop: float) -> float:
        rigid_stop = position.rigid_stop_level_at_entry
        if rigid_stop is None:
            return trailing_stop
        if position.direction == "long":
            return max(trailing_stop, rigid_stop)
        return min(trailing_stop, rigid_stop)

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
        target_level: float | None,
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
        notional_multiple = (
            net_pnl / position.position_notional_budget
            if position.position_notional_budget > 0
            else 0.0
        )
        holding_bars = max(candle_index - position.entry_index, 0)

        metadata: Dict[str, str | float | int | None] = {
            "direction": position.direction,
            "anchor_index": position.anchor_index,
            "setup_detected_index": position.setup_detected_index,
            "entry_raw_price": position.raw_entry_price,
            "entry_price": position.entry_price,
            "entry_mode": position.entry_mode.value,
            "exit_mode": position.exit_mode.value,
            "exit_band": position.exit_band.value,
            "exit_band_number": position.exit_band.number,
            "entry_trigger_mode": position.entry_trigger_mode,
            "entry_decision_price": position.decision_price,
            "ema_value_at_entry": position.ema_value_at_entry,
            "exit_raw_price": raw_exit_price,
            "exit_price": exit_price,
            "qty": position.qty,
            "position_notional_budget": position.position_notional_budget,
            "entry_fee": position.entry_fee,
            "exit_fee": exit_fee,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "position_pnl_pct": position_pnl_pct,
            "price_return_pct": return_pct,
            "notional_multiple": notional_multiple,
            "holding_bars": holding_bars,
            "stop_level_at_entry": position.stop_level_at_entry,
            "stop_level_at_exit": stop_level,
            "dynamic_stop_management_enabled": True,
            "rigid_stop_level_at_entry": position.rigid_stop_level_at_entry,
            "trailing_activation_level_at_entry": position.trailing_activation_level_at_entry,
            "trailing_activation_level_at_exit": activation_level,
            "trailing_stop": position.trailing_stop,
            "reason": exit_reason,
            "target_level_at_exit": target_level,
            "position_sizing_mode": position.position_sizing_mode,
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
        elif exit_reason.startswith("AVWAP band 1 target"):
            stats["target_exits_band_1"] += 1
        elif exit_reason.startswith("AVWAP band 2 target"):
            stats["target_exits_band_2"] += 1

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
                "entry_mode": position.entry_mode.value,
                "exit_mode": position.exit_mode.value,
                "exit_band": position.exit_band.value,
                "exit_band_number": position.exit_band.number,
                "exit_reason": exit_reason,
                "exit_price": exit_price,
                "raw_exit_price": raw_exit_price,
                "position_pnl": net_pnl,
                "position_pnl_pct": position_pnl_pct,
                "price_return_pct": return_pct,
                "gross_pnl": gross_pnl,
                "entry_fee": position.entry_fee,
                "exit_fee": exit_fee,
                "position_notional_budget": position.position_notional_budget,
                "position_sizing_mode": position.position_sizing_mode,
                "trailing_stop": position.trailing_stop,
                "stop_loss_level": stop_level,
                "dynamic_stop_management_enabled": True,
                "rigid_stop_loss_level": position.rigid_stop_level_at_entry,
                "trailing_activation_level": activation_level,
                "target_level": target_level,
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
