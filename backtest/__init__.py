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
    EntryMode,
    ExitMode,
    ExitBand,
)
from .indicators import atr, ema, rsi, sma
from .performance import (
    build_performance_statistics,
    compute_deflated_sharpe_ratio,
)
from .pinbar_strategy import PinbarStrategy, PinbarStrategyConfig
from .pinbar_magic_strategy import PinBarMagicStrategy, PinBarMagicStrategyConfig

from .patterns import CandlePatternSignals, detect_candle_patterns
from .plotter import plot_backtest, plot_backtest_from_store, save_plot, show_plot
from .robustness import (
    EvaluationPeriod,
    MarketCase,
    MonteCarloConfig,
    OutOfSamplePlan,
    ParameterCandidate,
    ParameterPerturbationRule,
    RegimeSegment,
    WalkForwardConfig,
    build_three_way_oos_plan,
    build_walk_forward_windows,
    classify_regime_segments,
    default_ema_avwap_parameter_perturbation_rules,
    generate_parameter_perturbation_candidates,
    label_trades_by_regime,
    run_block_bootstrap_monte_carlo,
    run_market_universe,
    run_monte_carlo_suite,
    run_out_of_sample_evaluation,
    run_parameter_perturbation,
    run_regime_monte_carlo,
    run_remove_best_trades_analysis,
    run_slippage_monte_carlo,
    run_trade_reshuffling_monte_carlo,
    run_walk_forward,
    score_report,
    summarize_trades_by_regime,
)
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
    "build_performance_statistics",
    "compute_deflated_sharpe_ratio",
    "CandlePatternSignals",
    "detect_candle_patterns",
    "EngulfingStrategy",
    "EngulfingStrategyConfig",
    "EmaAvwapPullbackStrategy",
    "EmaAvwapPullbackStrategyConfig",
    "EntryMode",
    "ExitMode",
    "ExitBand",
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
    "EvaluationPeriod",
    "MarketCase",
    "MonteCarloConfig",
    "OutOfSamplePlan",
    "ParameterCandidate",
    "ParameterPerturbationRule",
    "RegimeSegment",
    "WalkForwardConfig",
    "build_three_way_oos_plan",
    "build_walk_forward_windows",
    "classify_regime_segments",
    "default_ema_avwap_parameter_perturbation_rules",
    "generate_parameter_perturbation_candidates",
    "label_trades_by_regime",
    "run_block_bootstrap_monte_carlo",
    "run_market_universe",
    "run_monte_carlo_suite",
    "run_out_of_sample_evaluation",
    "run_parameter_perturbation",
    "run_regime_monte_carlo",
    "run_remove_best_trades_analysis",
    "run_slippage_monte_carlo",
    "run_trade_reshuffling_monte_carlo",
    "run_walk_forward",
    "score_report",
    "summarize_trades_by_regime",
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
