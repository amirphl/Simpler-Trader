from .base import (
    BacktestContext,
    BacktestReport,
    BacktestRunConfig,
    BacktestStatistics,
    BacktestStrategy,
    BaseBacktester,
    TradePerformance,
)
from .engulfing_strategy import EngulfingStrategy, EngulfingStrategyConfig, StopLossMode
from .ema_avwap_pullback_strategy import (
    EmaAvwapPullbackStrategy,
    EmaAvwapPullbackStrategyConfig,
)
from .indicators import atr, ema, rsi, sma
from .pinbar_strategy import PinbarStrategy, PinbarStrategyConfig
from .pinbar_magic_strategy import PinBarMagicStrategy, PinBarMagicStrategyConfig

from .patterns import CandlePatternSignals, detect_candle_patterns
from .plotter import plot_backtest, plot_backtest_from_store, save_plot, show_plot
from .scalping_FVG_strategy import ScalpingFVGStrategy, ScalpingFVGStrategyConfig
from .stochastic_fsm_strategy import (
    PositionDirection,
    StochasticRsiFsmConfig,
    StochasticRsiFsmStrategy,
)
from .strong_trend_stair_strategy import (
    StrongTrendStairStrategy,
    StrongTrendStairStrategyConfig,
)

__all__ = [
    "BacktestContext",
    "BacktestReport",
    "BacktestRunConfig",
    "BacktestStatistics",
    "BacktestStrategy",
    "BaseBacktester",
    "TradePerformance",
    "CandlePatternSignals",
    "detect_candle_patterns",
    "EngulfingStrategy",
    "EngulfingStrategyConfig",
    "EmaAvwapPullbackStrategy",
    "EmaAvwapPullbackStrategyConfig",
    "PinbarStrategy",
    "PinbarStrategyConfig",
    "PinBarMagicStrategy",
    "PinBarMagicStrategyConfig",
    "PinBarMagicStrategyV3",
    "PinBarMagicStrategyConfigV3",
    "ScalpingFVGStrategy",
    "ScalpingFVGStrategyConfig",
    "StopLossMode",
    "PositionDirection",
    "StochasticRsiFsmConfig",
    "StochasticRsiFsmStrategy",
    "StrongTrendStairStrategy",
    "StrongTrendStairStrategyConfig",
    "atr",
    "ema",
    "rsi",
    "sma",
    "plot_backtest",
    "plot_backtest_from_store",
    "save_plot",
    "show_plot",
]

# Backward-compatible aliases preserved for callers still using the v3 names.
PinBarMagicStrategyV3 = PinBarMagicStrategy
PinBarMagicStrategyConfigV3 = PinBarMagicStrategyConfig
