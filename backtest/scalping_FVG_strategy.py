from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance
from .indicators import atr, ema, rsi, sma

_EPSILON = 1e-9


@dataclass
class FVGZone:
    direction: str
    lower: float
    upper: float
    created_idx: int
    active: bool = True
    used: bool = False


@dataclass(frozen=True)
class ScalpingFVGStrategyConfig:
    symbol: str
    timeframe: str

    leverage: float = 5.0
    starting_capital: float = 100.0

    ema_fast_period: int = 20
    ema_slow_period: int = 50
    rsi_period: int = 14

    atr_period: int = 14
    atr_tp_mult: float = 1.0
    atr_sl_mult: float = 0.7

    risk_per_trade_pct: float = 0.01
    max_position_risk_fraction_of_price: float = 0.05

    volume_sma_period: int = 20
    min_volume_ratio: float = 0.8

    max_fvg_age: int = 50
    max_open_trades: int = 1

    long_rsi_min: float = 30.0
    long_rsi_max: float = 55.0
    short_rsi_min: float = 45.0
    short_rsi_max: float = 70.0

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        timeframe = self.timeframe.strip()

        if not symbol:
            raise ValueError("symbol must not be empty")
        if not timeframe:
            raise ValueError("timeframe must not be empty")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct >= 1:
            raise ValueError("risk_per_trade_pct should be in (0, 1)")
        if self.max_position_risk_fraction_of_price <= 0:
            raise ValueError("max_position_risk_fraction_of_price must be positive")
        if self.min_volume_ratio < 0:
            raise ValueError("min_volume_ratio must be non-negative")
        if self.atr_tp_mult <= 0 or self.atr_sl_mult <= 0:
            raise ValueError("ATR multipliers must be positive")
        if self.max_fvg_age <= 0:
            raise ValueError("max_fvg_age must be positive")
        if self.max_open_trades <= 0:
            raise ValueError("max_open_trades must be positive")
        if self.long_rsi_min > self.long_rsi_max:
            raise ValueError("long RSI bounds are invalid")
        if self.short_rsi_min > self.short_rsi_max:
            raise ValueError("short RSI bounds are invalid")
        if (
            min(
                self.ema_fast_period,
                self.ema_slow_period,
                self.rsi_period,
                self.atr_period,
                self.volume_sma_period,
            )
            <= 0
        ):
            raise ValueError("indicator periods must be positive")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)


@dataclass
class Position:
    direction: int
    entry_time: datetime
    entry_index: int
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    qty: float
    capital_at_entry: float
    risk_amount: float
    fvg_created_idx: int


class ScalpingFVGStrategy(BacktestStrategy):
    def __init__(self, config: ScalpingFVGStrategyConfig) -> None:
        self._config = config

    def name(self) -> str:
        return "ScalpingFVGStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        cfg = self._config
        candles = context.data.get(cfg.symbol, {}).get(cfg.timeframe, [])
        ignore_count = context.ignore_candles.get(cfg.symbol, {}).get(cfg.timeframe, 0)
        start_idx = max(ignore_count, self._required_history())

        if len(candles) <= start_idx:
            return [], {"note": "insufficient_data", "candles": len(candles)}

        closes = [c.close for c in candles]
        volumes = [float(getattr(c, "volume", 0.0) or 0.0) for c in candles]

        indicators = _IndicatorSnapshot(
            ema_fast=ema(closes, cfg.ema_fast_period),
            ema_slow=ema(closes, cfg.ema_slow_period),
            rsi=rsi(closes, cfg.rsi_period),
            atr=atr(candles, cfg.atr_period),
            volume_sma=sma(volumes, cfg.volume_sma_period),
        )

        trades: List[TradePerformance] = []
        zones: List[FVGZone] = []
        position: Optional[Position] = None
        capital = cfg.starting_capital
        stats: dict[str, int | float | str] = {
            "starting_capital": cfg.starting_capital,
            "ending_capital": cfg.starting_capital,
            "signals_detected": 0,
            "entries_opened": 0,
            "zones_detected": 0,
            "zones_expired": 0,
            "zones_invalidated": 0,
            "entries_skipped_volume": 0,
            "entries_skipped_indicator_data": 0,
            "entries_skipped_invalid_levels": 0,
            "entries_skipped_invalid_risk": 0,
            "entries_skipped_no_capacity": 0,
            "forced_exit_end_of_backtest": 0,
        }

        for idx in range(2, len(candles)):
            candle = candles[idx]
            if candle.open_time > context.config.end:
                break

            new_zones = self._detect_fvg(candles, idx)
            zones.extend(new_zones)
            stats["zones_detected"] += len(new_zones)
            stats["zones_expired"] += self._expire_zones(zones, idx)
            stats["zones_invalidated"] += self._invalidate_zones(zones, candle, idx)

            if idx < start_idx or candle.close_time < context.config.start:
                continue

            if position is not None:
                exit_fill = self._check_exit(candle, position)
                if exit_fill is not None:
                    trades.append(self._close_position(position, candle, *exit_fill))
                    capital = max(capital + trades[-1].pnl, 0.0)
                    position = None

            if position is not None or capital <= 0:
                continue

            signal, skip_reason = self._build_signal(
                idx, candles, volumes, indicators, zones
            )
            if skip_reason is not None:
                stats[skip_reason] += 1
                continue
            if signal is None:
                continue

            stats["signals_detected"] += 1
            position = self._build_position(
                candle=candle,
                entry_index=idx,
                signal=signal,
                current_capital=capital,
            )
            if position is None:
                stats[signal.skip_reason] += 1
                continue

            signal.zone.used = True
            stats["entries_opened"] += 1

            same_bar_exit = self._check_exit(candle, position)
            if same_bar_exit is not None:
                trades.append(self._close_position(position, candle, *same_bar_exit))
                capital = max(capital + trades[-1].pnl, 0.0)
                position = None

        if position is not None and candles:
            last_candle = candles[-1]
            trades.append(
                self._close_position(
                    position,
                    last_candle,
                    last_candle.close,
                    "End of backtest",
                    exit_time=last_candle.close_time,
                )
            )
            capital = max(capital + trades[-1].pnl, 0.0)
            stats["forced_exit_end_of_backtest"] += 1

        stats["ending_capital"] = capital
        return trades, stats

    def _required_history(self) -> int:
        return max(
            2,
            self._config.ema_fast_period,
            self._config.ema_slow_period,
            self._config.rsi_period + 1,
            self._config.atr_period,
            self._config.volume_sma_period,
        )

    @staticmethod
    def _detect_fvg(candles: Sequence[Candle], idx: int) -> List[FVGZone]:
        if idx < 2:
            return []

        first = candles[idx - 2]
        middle = candles[idx - 1]
        third = candles[idx]
        zones: List[FVGZone] = []

        bullish_gap = third.low - first.high
        if bullish_gap > _EPSILON and middle.low > first.high:
            zones.append(
                FVGZone(
                    direction="bullish",
                    lower=first.high,
                    upper=third.low,
                    created_idx=idx,
                )
            )

        bearish_gap = first.low - third.high
        if bearish_gap > _EPSILON and middle.high < first.low:
            zones.append(
                FVGZone(
                    direction="bearish",
                    lower=third.high,
                    upper=first.low,
                    created_idx=idx,
                )
            )

        return zones

    @staticmethod
    def _candle_intersects_zone(candle: Candle, zone: FVGZone) -> bool:
        return candle.low <= zone.upper and candle.high >= zone.lower

    def _expire_zones(self, zones: Sequence[FVGZone], idx: int) -> int:
        expired = 0
        for zone in zones:
            if zone.active and (idx - zone.created_idx) > self._config.max_fvg_age:
                zone.active = False
                expired += 1
        return expired

    @staticmethod
    def _invalidate_zones(zones: Sequence[FVGZone], candle: Candle, idx: int) -> int:
        invalidated = 0
        for zone in zones:
            if not zone.active or zone.used or zone.created_idx >= idx:
                continue
            if zone.direction == "bullish" and candle.close < zone.lower:
                zone.active = False
                invalidated += 1
            elif zone.direction == "bearish" and candle.close > zone.upper:
                zone.active = False
                invalidated += 1
        return invalidated

    def _build_signal(
        self,
        idx: int,
        candles: Sequence[Candle],
        volumes: Sequence[float],
        indicators: "_IndicatorSnapshot",
        zones: Sequence[FVGZone],
    ) -> Tuple[Optional["_PendingSignal"], Optional[str]]:
        prev_idx = idx - 1
        if prev_idx < 1:
            return None, None

        ema_fast_value = indicators.ema_fast[prev_idx]
        ema_slow_value = indicators.ema_slow[prev_idx]
        rsi_value = indicators.rsi[prev_idx]
        atr_value = indicators.atr[prev_idx]
        volume_sma_value = indicators.volume_sma[prev_idx]
        prev_volume = volumes[prev_idx]

        if any(
            value is None
            for value in (
                ema_fast_value,
                ema_slow_value,
                rsi_value,
                atr_value,
                volume_sma_value,
            )
        ):
            return None, "entries_skipped_indicator_data"

        if prev_volume < float(volume_sma_value) * self._config.min_volume_ratio:
            return None, "entries_skipped_volume"

        prev_candle = candles[prev_idx]
        long_zone = self._find_candidate_zone(
            direction="bullish",
            zones=zones,
            candle=prev_candle,
        )
        short_zone = self._find_candidate_zone(
            direction="bearish",
            zones=zones,
            candle=prev_candle,
        )

        if (
            float(ema_fast_value) > float(ema_slow_value)
            and self._config.long_rsi_min <= float(rsi_value) <= self._config.long_rsi_max
            and long_zone is not None
        ):
            return (
                _PendingSignal(
                    direction=1,
                    zone=long_zone,
                    atr_value=float(atr_value),
                    rsi_value=float(rsi_value),
                    ema_fast=float(ema_fast_value),
                    ema_slow=float(ema_slow_value),
                    skip_reason="entries_skipped_invalid_levels",
                ),
                None,
            )

        if (
            float(ema_fast_value) < float(ema_slow_value)
            and self._config.short_rsi_min <= float(rsi_value) <= self._config.short_rsi_max
            and short_zone is not None
        ):
            return (
                _PendingSignal(
                    direction=-1,
                    zone=short_zone,
                    atr_value=float(atr_value),
                    rsi_value=float(rsi_value),
                    ema_fast=float(ema_fast_value),
                    ema_slow=float(ema_slow_value),
                    skip_reason="entries_skipped_invalid_levels",
                ),
                None,
            )

        return None, None

    @staticmethod
    def _find_candidate_zone(
        *,
        direction: str,
        zones: Sequence[FVGZone],
        candle: Candle,
    ) -> Optional[FVGZone]:
        for zone in reversed(zones):
            if (
                zone.active
                and not zone.used
                and zone.direction == direction
                and ScalpingFVGStrategy._candle_intersects_zone(candle, zone)
            ):
                return zone
        return None

    def _build_position(
        self,
        *,
        candle: Candle,
        entry_index: int,
        signal: "_PendingSignal",
        current_capital: float,
    ) -> Optional[Position]:
        entry_price = float(candle.open)
        atr_value = signal.atr_value
        if entry_price <= 0 or atr_value <= 0:
            signal.skip_reason = "entries_skipped_invalid_levels"
            return None

        if signal.direction == 1:
            stop_loss = entry_price - self._config.atr_sl_mult * atr_value
            take_profit = entry_price + self._config.atr_tp_mult * atr_value
            risk_per_unit = entry_price - stop_loss
        else:
            stop_loss = entry_price + self._config.atr_sl_mult * atr_value
            take_profit = entry_price - self._config.atr_tp_mult * atr_value
            risk_per_unit = stop_loss - entry_price

        if (
            stop_loss <= 0
            or take_profit <= 0
            or risk_per_unit <= _EPSILON
            or (risk_per_unit / entry_price)
            > self._config.max_position_risk_fraction_of_price
        ):
            signal.skip_reason = "entries_skipped_invalid_levels"
            return None

        risk_amount = current_capital * self._config.risk_per_trade_pct
        risk_qty = risk_amount / risk_per_unit
        max_notional = current_capital * self._config.leverage
        max_qty = max_notional / entry_price if entry_price > 0 else 0.0
        qty = min(risk_qty, max_qty)

        if qty <= _EPSILON:
            signal.skip_reason = "entries_skipped_no_capacity"
            return None
        if risk_amount <= _EPSILON:
            signal.skip_reason = "entries_skipped_invalid_risk"
            return None

        return Position(
            direction=signal.direction,
            entry_time=candle.open_time,
            entry_index=entry_index,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self._config.leverage,
            qty=qty,
            capital_at_entry=current_capital,
            risk_amount=risk_per_unit * qty,
            fvg_created_idx=signal.zone.created_idx,
        )

    @staticmethod
    def _check_exit(candle: Candle, position: Position) -> Optional[Tuple[float, str]]:
        if position.direction == 1:
            if candle.low <= position.stop_loss:
                return position.stop_loss, "Stop Loss"
            if candle.high >= position.take_profit:
                return position.take_profit, "Take Profit"
            return None

        if candle.high >= position.stop_loss:
            return position.stop_loss, "Stop Loss"
        if candle.low <= position.take_profit:
            return position.take_profit, "Take Profit"
        return None

    def _close_position(
        self,
        position: Position,
        candle: Candle,
        exit_price: float,
        reason: str,
        *,
        exit_time: Optional[datetime] = None,
    ) -> TradePerformance:
        price_delta = exit_price - position.entry_price
        pnl = price_delta * position.qty * position.direction
        return_pct = (
            (pnl / position.capital_at_entry) * 100.0
            if position.capital_at_entry > _EPSILON
            else 0.0
        )
        notional = position.entry_price * position.qty
        margin = notional / position.leverage if position.leverage > _EPSILON else 0.0
        r_multiple = pnl / position.risk_amount if position.risk_amount > _EPSILON else 0.0

        return TradePerformance(
            entry_time=position.entry_time,
            exit_time=exit_time or candle.close_time,
            pnl=pnl,
            return_pct=return_pct,
            notes=f"{reason} at {exit_price:.4f}",
            metadata={
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "direction": position.direction,
                "qty": position.qty,
                "notional": notional,
                "margin": margin,
                "capital_at_entry": position.capital_at_entry,
                "risk_amount": position.risk_amount,
                "r_multiple": r_multiple,
                "leverage": position.leverage,
                "fvg_created_idx": position.fvg_created_idx,
            },
        )


@dataclass(frozen=True)
class _IndicatorSnapshot:
    ema_fast: List[Optional[float]]
    ema_slow: List[Optional[float]]
    rsi: List[Optional[float]]
    atr: List[Optional[float]]
    volume_sma: List[Optional[float]]


@dataclass
class _PendingSignal:
    direction: int
    zone: FVGZone
    atr_value: float
    rsi_value: float
    ema_fast: float
    ema_slow: float
    skip_reason: str
