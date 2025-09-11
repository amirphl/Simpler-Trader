from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Optional, Tuple, Mapping, Any

from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance


# ======================
# Indicator helpers
# ======================

def ema(values: Sequence[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be positive")

    result: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return result

    # Simple MA for first EMA value
    sma = sum(values[:period]) / period
    result[period - 1] = sma
    k = 2.0 / (period + 1)

    ema_prev = sma
    for i in range(period, len(values)):
        v = values[i]
        ema_prev = v * k + ema_prev * (1 - k)
        result[i] = ema_prev

    return result


def rsi(values: Sequence[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be positive")

    result: List[Optional[float]] = [None] * len(values)
    if len(values) < period + 1:
        return result

    gains: List[float] = [0.0] * len(values)
    losses: List[float] = [0.0] * len(values)

    # First differences
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        if change > 0:
            gains[i] = change
            losses[i] = 0.0
        else:
            gains[i] = 0.0
            losses[i] = -change

    # First average gain/loss
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's smoothing
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


from typing import List, Optional, Sequence

def atr(candles: Sequence["Candle"], length: int) -> List[Optional[float]]:
    """
    ATR implementation that mimics TradingView's:
        pine_atr(length) =>
            trueRange = na(high[1]) ? high - low :
                        max(max(high - low, abs(high - close[1])),
                            abs(low - close[1]))
            rma(trueRange, length)

    Returns a list where values are None until enough history is available,
    exactly like Pine (first ATR at index length-1).
    """
    if length <= 0:
        raise ValueError("length must be positive")

    n = len(candles)
    result: List[Optional[float]] = [None] * n
    if n == 0 or n < length:
        return result

    # --- True Range, Pine-style ---
    tr: List[float] = [0.0] * n
    for i in range(n):
        high = candles[i].high
        low = candles[i].low
        if i == 0:
            # na(high[1]) ? high - low
            tr[i] = high - low
        else:
            prev_close = candles[i - 1].close
            tr_hl = high - low
            tr_hc = abs(high - prev_close)
            tr_lc = abs(low - prev_close)
            tr[i] = max(tr_hl, tr_hc, tr_lc)

    # --- RMA(trueRange, length) ---
    # First RMA value is SMA of first `length` TRs, placed at index length-1
    first_rma = sum(tr[0:length]) / float(length)
    result[length - 1] = first_rma
    prev_rma = first_rma

    # Wilder / RMA smoothing: rma[i] = (prev_rma * (length - 1) + tr[i]) / length
    for i in range(length, n):
        prev_rma = (prev_rma * (length - 1) + tr[i]) / float(length)
        result[i] = prev_rma

    return result

def sma(values: Sequence[float], period: int) -> List[Optional[float]]:
    if period <= 0:
        raise ValueError("period must be positive")

    result: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return result

    window_sum = sum(values[:period])
    result[period - 1] = window_sum / period

    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        result[i] = window_sum / period

    return result


# ======================
# FVG data structure
# ======================

@dataclass
class FVGZone:
    direction: str  # "bullish" or "bearish"
    lower: float
    upper: float
    created_idx: int
    active: bool = True
    used: bool = False


# ======================
# Config & Position
# ======================

@dataclass(frozen=True)
class ScalpingFVGStrategyConfig:
    """Config for FVG-based scalping strategy."""

    symbol: str
    timeframe: str

    leverage: float = 5.0

    # Trend filter
    ema_fast_period: int = 20
    ema_slow_period: int = 50

    # Oscillator
    rsi_period: int = 14

    # Volatility / TP / SL
    atr_period: int = 14
    atr_tp_mult: float = 1.0
    atr_sl_mult: float = 0.7  # tighter SL than TP for scalping

    # Risk management
    risk_per_trade_pct: float = 0.01  # 1% of equity per trade
    max_position_risk_fraction_of_price: float = 0.05  # safety

    # Volume filter
    volume_sma_period: int = 20
    min_volume_ratio: float = 0.8  # ignore ultra-low volume

    # FVG settings
    max_fvg_age: int = 50  # candles
    max_open_trades: int = 1  # per symbol; we use 1 to keep it simple

    def __post_init__(self) -> None:
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")
        if self.risk_per_trade_pct <= 0:
            raise ValueError("risk_per_trade_pct must be positive")
        if not (0 < self.risk_per_trade_pct < 1):
            raise ValueError("risk_per_trade_pct should be in (0, 1)")
        if self.ema_fast_period <= 0 or self.ema_slow_period <= 0:
            raise ValueError("EMA periods must be positive")
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.max_fvg_age <= 0:
            raise ValueError("max_fvg_age must be positive")


@dataclass
class Position:
    """Represents an open scalping position (long or short)."""

    direction: int  # +1 = long, -1 = short
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: float
    size: float  # position size in base currency
    fvg_created_idx: int


# ======================
# Strategy implementation
# ======================

class ScalpingFVGStrategy(BacktestStrategy):
    """Low-timeframe scalping strategy based on FVG + EMA trend + RSI + ATR."""

    def __init__(self, config: ScalpingFVGStrategyConfig) -> None:
        self._config = config

    def name(self) -> str:
        return "ScalpingFVGStrategy"

    def symbols(self) -> Sequence[str]:
        return [self._config.symbol]

    def timeframes(self) -> Sequence[str]:
        return [self._config.timeframe]

    # ---- FVG detection ----

    @staticmethod
    def _detect_fvg(candles: Sequence[Candle], idx: int) -> List[FVGZone]:
        """Detect bullish/bearish FVG ending at index idx (needs idx >= 2)."""
        zones: List[FVGZone] = []
        if idx < 2:
            return zones

        c1 = candles[idx - 2]
        c2 = candles[idx - 1]
        c3 = candles[idx]

        # Bullish FVG: gap between c1.high and c3.low
        if c1.high < c3.low:
            lower = c1.high
            upper = c3.low
            # Optional: require c2 to also respect gap (no overlap)
            if c2.low > c1.high:
                zones.append(
                    FVGZone(
                        direction="bullish",
                        lower=lower,
                        upper=upper,
                        created_idx=idx,
                    )
                )

        # Bearish FVG: gap between c3.high and c1.low
        if c1.low > c3.high:
            lower = c3.high
            upper = c1.low
            if c2.high < c1.low:
                zones.append(
                    FVGZone(
                        direction="bearish",
                        lower=lower,
                        upper=upper,
                        created_idx=idx,
                    )
                )

        return zones

    @staticmethod
    def _candle_intersects_zone(candle: Candle, zone: FVGZone) -> bool:
        return candle.low <= zone.upper and candle.high >= zone.lower

    def run(
        self, context: BacktestContext
    ) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]:
        symbol = self._config.symbol
        timeframe = self._config.timeframe

        candles = context.data.get(symbol, {}).get(timeframe, [])
        if len(candles) < 200:
            # Need enough history for indicators + FVG
            return [], None

        closes = [c.close for c in candles]
        volumes = [float(getattr(c, "volume", 0.0) or 0.0) for c in candles]

        ema_fast = ema(closes, self._config.ema_fast_period)
        ema_slow = ema(closes, self._config.ema_slow_period)
        rsis = rsi(closes, self._config.rsi_period)
        atr_vals = atr(candles, self._config.atr_period)
        vol_sma = sma(volumes, self._config.volume_sma_period)

        trades: List[TradePerformance] = []
        position: Optional[Position] = None
        fvg_zones: List[FVGZone] = []

        initial_capital = context.config.initial_capital
        current_capital = initial_capital

        for idx in range(2, len(candles)):
            candle = candles[idx]
            prev_candle = candles[idx - 1]

            # --- Update FVG zones (based on candles up to idx) ---
            # 1. Detect new FVG(s) ending at idx
            new_zones = self._detect_fvg(candles, idx)
            fvg_zones.extend(new_zones)

            # 2. Expire very old zones
            for z in fvg_zones:
                if z.active and (idx - z.created_idx) > self._config.max_fvg_age:
                    z.active = False

            # --- Check exit for open position ---
            if position is not None:
                exit_reason: Optional[str] = None
                exit_price: Optional[float] = None

                if position.direction == +1:  # Long
                    # SL first (conservative)
                    if candle.low <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "Stop Loss"
                    elif candle.high >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "Take Profit"
                else:  # Short
                    if candle.high >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "Stop Loss"
                    elif candle.low <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "Take Profit"

                if exit_price is not None:
                    price_change = (exit_price - position.entry_price) * position.direction
                    pnl = (price_change / position.entry_price) * position.size * position.leverage
                    return_pct = (price_change / position.entry_price) * 100.0 * position.leverage

                    trades.append(
                        TradePerformance(
                            entry_time=position.entry_time,
                            exit_time=candle.open_time,
                            pnl=pnl,
                            return_pct=return_pct,
                            notes=f"{exit_reason} at {exit_price:.4f}",
                            metadata={
                                "entry_price": position.entry_price,
                                "exit_price": exit_price,
                                "stop_loss": position.stop_loss,
                                "take_profit": position.take_profit,
                                "direction": position.direction,
                                "leverage": position.leverage,
                                "fvg_created_idx": position.fvg_created_idx,
                            },
                        )
                    )

                    current_capital += pnl
                    if current_capital < 0:
                        current_capital = 0.0

                    position = None

            # --- Entry logic: only if flat and we have enough indicator data ---
            if position is None and idx >= 2:
                # Enforce max FVG-related trades (optional, here we only allow 1 open position anyway)
                if current_capital <= 0:
                    continue

                ema_f = ema_fast[idx - 1]
                ema_s = ema_slow[idx - 1]
                rsi_val = rsis[idx - 1]
                atr_val = atr_vals[idx - 1]
                vol_ma = vol_sma[idx - 1]
                prev_vol = volumes[idx - 1]

                if (
                    ema_f is None
                    or ema_s is None
                    or rsi_val is None
                    or atr_val is None
                    or vol_ma is None
                ):
                    continue

                # Basic volume filter (avoid dead periods)
                if prev_vol < vol_ma * self._config.min_volume_ratio:
                    continue

                # ---- LONG candidate ----
                long_trend = ema_f > ema_s
                long_rsi_ok = 30.0 <= rsi_val <= 55.0

                # ---- SHORT candidate ----
                short_trend = ema_f < ema_s
                short_rsi_ok = 45.0 <= rsi_val <= 70.0

                chosen_zone: Optional[FVGZone] = None
                direction: Optional[int] = None

                if long_trend and long_rsi_ok:
                    # Look for bullish FVG retested by prev candle
                    for z in fvg_zones:
                        if (
                            z.active
                            and not z.used
                            and z.direction == "bullish"
                            and self._candle_intersects_zone(prev_candle, z)
                        ):
                            chosen_zone = z
                            direction = +1
                            break

                if chosen_zone is None and short_trend and short_rsi_ok:
                    # Look for bearish FVG retested by prev candle
                    for z in fvg_zones:
                        if (
                            z.active
                            and not z.used
                            and z.direction == "bearish"
                            and self._candle_intersects_zone(prev_candle, z)
                        ):
                            chosen_zone = z
                            direction = -1
                            break

                if chosen_zone is None or direction is None:
                    continue

                # ---- Build position ----
                entry_price = candle.open

                if direction == +1:
                    stop_loss = entry_price - self._config.atr_sl_mult * atr_val
                    take_profit = entry_price + self._config.atr_tp_mult * atr_val
                    risk_per_unit = entry_price - stop_loss
                else:
                    stop_loss = entry_price + self._config.atr_sl_mult * atr_val
                    take_profit = entry_price - self._config.atr_tp_mult * atr_val
                    risk_per_unit = stop_loss - entry_price

                # Safety checks
                if risk_per_unit <= 0 or stop_loss <= 0 or take_profit <= 0:
                    continue

                # Risk per trade in account currency
                risk_amount = current_capital * self._config.risk_per_trade_pct
                size = risk_amount / risk_per_unit

                # Safety: avoid insane leverage exposure
                if risk_per_unit / entry_price > self._config.max_position_risk_fraction_of_price:
                    # SL too far from entry for scalping → skip
                    continue

                if size <= 0:
                    continue

                # Mark zone as used
                chosen_zone.used = True

                position = Position(
                    direction=direction,
                    entry_time=candle.open_time,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=self._config.leverage,
                    size=size,
                    fvg_created_idx=chosen_zone.created_idx,
                )

        # --- Close any remaining position at the end of backtest ---
        if position is not None and candles:
            last_candle = candles[-1]
            exit_price = last_candle.close
            price_change = (exit_price - position.entry_price) * position.direction
            pnl = (price_change / position.entry_price) * position.size * position.leverage
            return_pct = (price_change / position.entry_price) * 100.0 * position.leverage

            trades.append(
                TradePerformance(
                    entry_time=position.entry_time,
                    exit_time=last_candle.close_time,
                    pnl=pnl,
                    return_pct=return_pct,
                    notes="End of backtest",
                    metadata={
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit,
                        "direction": position.direction,
                        "leverage": position.leverage,
                        "fvg_created_idx": position.fvg_created_idx,
                    },
                )
            )

        return trades, None
