from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from candle_downloader.models import Candle

from backtest.engulfing_strategy import (
    EngulfingStrategyConfig,
    are_bearish,
    calculate_stochastic_k,
    calculate_volume_pressure_score,
)
from backtest.patterns import detect_candle_patterns
from backtest.scalping_FVG_strategy import (
    FVGZone,
    ScalpingFVGStrategyConfig,
    atr,
    ema,
    rsi,
    sma,
)

from .signals import SignalSpec


class SignalGenerator(ABC):
    """Strategy-agnostic signal generator interface."""

    @abstractmethod
    def evaluate(self, candles: Sequence[Candle]) -> Optional[SignalSpec]:
        """Return a signal when the latest candles satisfy the strategy conditions."""


@dataclass(frozen=True)
class EngulfingSignalGeneratorConfig:
    window_size: int = 5
    take_profit_pct: float = 0.02
    doji_size: float = 0.05
    volume_window: int = 20
    max_volume_pressure_score: float = 3.0


class EngulfingSignalGenerator(SignalGenerator):
    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        config: EngulfingSignalGeneratorConfig | None = None,
    ) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._config = config or EngulfingSignalGeneratorConfig()

    def evaluate(self, candles: Sequence[Candle]) -> Optional[SignalSpec]:
        if len(candles) < max(100, self._config.window_size + 2):
            return None

        idx = len(candles) - 1
        prev_idx = idx - 1
        if prev_idx < 1:
            return None

        patterns = detect_candle_patterns(candles, doji_size=self._config.doji_size)
        prev_pattern = patterns[prev_idx]
        if not prev_pattern.bullish_engulfing:
            return None

        check_start = prev_idx - self._config.window_size
        if check_start < 0 or not are_bearish(candles, check_start, self._config.window_size):
            return None

        stoch_k20 = calculate_stochastic_k(candles, 20, prev_idx)
        stoch_k100 = calculate_stochastic_k(candles, 100, prev_idx)
        if (
            stoch_k20 is None
            or stoch_k100 is None
            or stoch_k20 <= stoch_k100
        ):
            return None

        volume_score = calculate_volume_pressure_score(
            candles,
            prev_idx,
            self._config.volume_window,
        )
        if (
            volume_score is not None
            and volume_score > self._config.max_volume_pressure_score
        ):
            # Exhaustion engulfing, skip
            return None

        entry_candle = candles[idx]
        prev_candle = candles[prev_idx]

        entry_price = entry_candle.open
        stop_loss = prev_candle.open
        take_profit = entry_price * (1.0 + self._config.take_profit_pct)

        metadata = {
            "stochastic_20": stoch_k20,
            "stochastic_100": stoch_k100,
            "volume_score": volume_score,
        }
        return SignalSpec(
            timestamp=entry_candle.open_time,
            symbol=self._symbol,
            timeframe=self._timeframe,
            direction="LONG",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
            notes="Bullish engulfing signal",
        )


class ScalpingFVGSignalGenerator(SignalGenerator):
    def __init__(
        self,
        *,
        config: ScalpingFVGStrategyConfig,
    ) -> None:
        self._config = config

    def evaluate(self, candles: Sequence[Candle]) -> Optional[SignalSpec]:
        if len(candles) < 200:
            return None

        closes = [c.close for c in candles]
        volumes = [float(getattr(c, "volume", 0.0) or 0.0) for c in candles]

        ema_fast_vals = ema(closes, self._config.ema_fast_period)
        ema_slow_vals = ema(closes, self._config.ema_slow_period)
        rsi_vals = rsi(closes, self._config.rsi_period)
        atr_vals = atr(candles, self._config.atr_period)
        vol_sma_vals = sma(volumes, self._config.volume_sma_period)

        fvg_zones: list[FVGZone] = []
        for idx in range(2, len(candles)):
            fvg_zones.extend(self._detect_fvg(candles, idx))
            for zone in fvg_zones:
                if zone.active and (idx - zone.created_idx) > self._config.max_fvg_age:
                    zone.active = False

        idx = len(candles) - 1
        prev_idx = idx - 1
        if prev_idx < 1:
            return None

        ema_fast_val = ema_fast_vals[prev_idx]
        ema_slow_val = ema_slow_vals[prev_idx]
        rsi_val = rsi_vals[prev_idx]
        atr_val = atr_vals[prev_idx]
        vol_ma = vol_sma_vals[prev_idx]
        prev_vol = volumes[prev_idx]

        if (
            ema_fast_val is None
            or ema_slow_val is None
            or rsi_val is None
            or atr_val is None
            or vol_ma is None
        ):
            return None
        if prev_vol < vol_ma * self._config.min_volume_ratio:
            return None

        long_trend = ema_fast_val > ema_slow_val
        long_rsi_ok = 30.0 <= rsi_val <= 55.0
        short_trend = ema_fast_val < ema_slow_val
        short_rsi_ok = 45.0 <= rsi_val <= 70.0

        prev_candle = candles[prev_idx]
        chosen_zone: Optional[FVGZone] = None
        direction: Optional[int] = None

        if long_trend and long_rsi_ok:
            for zone in fvg_zones:
                if (
                    zone.active
                    and not zone.used
                    and zone.direction == "bullish"
                    and self._candle_intersects_zone(prev_candle, zone)
                ):
                    chosen_zone = zone
                    direction = +1
                    break

        if chosen_zone is None and short_trend and short_rsi_ok:
            for zone in fvg_zones:
                if (
                    zone.active
                    and not zone.used
                    and zone.direction == "bearish"
                    and self._candle_intersects_zone(prev_candle, zone)
                ):
                    chosen_zone = zone
                    direction = -1
                    break

        if chosen_zone is None or direction is None:
            return None

        entry_candle = candles[idx]
        entry_price = entry_candle.open

        if direction == +1:
            stop_loss = entry_price - self._config.atr_sl_mult * atr_val
            take_profit = entry_price + self._config.atr_tp_mult * atr_val
            risk_per_unit = entry_price - stop_loss
            direction_label = "LONG"
        else:
            stop_loss = entry_price + self._config.atr_sl_mult * atr_val
            take_profit = entry_price - self._config.atr_tp_mult * atr_val
            risk_per_unit = stop_loss - entry_price
            direction_label = "SHORT"

        if risk_per_unit <= 0:
            return None

        metadata = {
            "direction_value": direction,
            "atr": atr_val,
            "rsi": rsi_val,
            "ema_fast": ema_fast_val,
            "ema_slow": ema_slow_val,
            "zone_created_idx": chosen_zone.created_idx,
        }
        chosen_zone.used = True
        return SignalSpec(
            timestamp=entry_candle.open_time,
            symbol=self._config.symbol,
            timeframe=self._config.timeframe,
            direction=direction_label,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
            notes="Scalping FVG signal",
        )

    @staticmethod
    def _detect_fvg(candles: Sequence[Candle], idx: int) -> list[FVGZone]:
        zones: list[FVGZone] = []
        if idx < 2:
            return zones

        c1 = candles[idx - 2]
        c2 = candles[idx - 1]
        c3 = candles[idx]

        if c1.high < c3.low and c2.low > c1.high:
            zones.append(
                FVGZone(
                    direction="bullish",
                    lower=c1.high,
                    upper=c3.low,
                    created_idx=idx,
                )
            )
        if c1.low > c3.high and c2.high < c1.low:
            zones.append(
                FVGZone(
                    direction="bearish",
                    lower=c3.high,
                    upper=c1.low,
                    created_idx=idx,
                )
            )
        return zones

    @staticmethod
    def _candle_intersects_zone(candle: Candle, zone: FVGZone) -> bool:
        return candle.low <= zone.upper and candle.high >= zone.lower

