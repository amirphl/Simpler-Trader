from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, cast

from backtest import (
    BacktestRunConfig,
    BaseBacktester,
    EngulfingStrategy,
    EngulfingStrategyConfig,
    EmaAvwapPullbackStrategy,
    EmaAvwapPullbackStrategyConfig,
    EntryMode,
    ExitBand,
    ExitMode,
    MonteCarloConfig,
    ParameterCandidate,
    PinBarMagicStrategyConfigV3,
    PinBarMagicStrategyV3,
    PinbarStrategy,
    PinbarStrategyConfig,
    StrongTrendStairStrategy,
    StrongTrendStairStrategyConfig,
    WalkForwardConfig,
    build_three_way_oos_plan,
    classify_regime_segments,
    default_ema_avwap_parameter_perturbation_rules,
    label_trades_by_regime,
    run_monte_carlo_suite,
    run_out_of_sample_evaluation,
    run_parameter_perturbation,
    run_walk_forward,
)

from backtest.stochastic_fsm_strategy import (
    StochasticRsiFsmConfig,
    StochasticRsiFsmStrategy,
)
from candle_downloader.binance import BinanceClient, BinanceClientConfig
from candle_downloader.downloader import CandleDownloader
from candle_downloader.storage import build_store

from .models import (
    BacktestSubmission,
    EngulfingStrategyParams,
    EmaAvwapPullbackParams,
    PinbarMagicStrategyParamsV3,
    PinbarStrategyParams,
    StrongTrendStairParams,
    StochasticFsmParams,
)


def run_backtest_job(
    job_id: str, submission: BacktestSubmission, *, cache_dir: Path
) -> Dict[str, Any]:
    """Execute the requested backtest synchronously and return its report."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    params = submission.params
    proxies = {}
    if params.http_proxy:
        proxies["http"] = params.http_proxy
    if params.https_proxy:
        proxies["https"] = params.https_proxy

    store = build_store("postgres", None)
    client_logger = logging.getLogger(f"web-backtest.{job_id}.binance")
    client = BinanceClient(
        BinanceClientConfig(proxies=proxies or None), logger=client_logger
    )
    downloader = CandleDownloader(client=client, store=store)

    if submission.strategy == "stochastic_fsm":
        stoch_params = cast(StochasticFsmParams, params)
        strategy = StochasticRsiFsmStrategy(
            StochasticRsiFsmConfig(
                symbols=stoch_params.symbols,
                tf_1=stoch_params.base_timeframe,
                tf_2=stoch_params.higher_timeframe,
                tf_3=stoch_params.higher_timeframe_2,
                k_period=stoch_params.k_period,
                k_slowing=stoch_params.k_slowing,
                d_period=stoch_params.d_period,
                use_d_line=stoch_params.use_d_line,
                oversold=stoch_params.oversold,
                overbought=stoch_params.overbought,
                initial_order_usdt=stoch_params.initial_order_usdt,
                initial_leverage=stoch_params.initial_leverage,
                martingale_multiplier=stoch_params.martingale_multiplier,
                martingale_multipliers=tuple(stoch_params.martingale_multipliers),
                martingale_leverages=tuple(stoch_params.martingale_leverages),
                max_concurrent_positions=stoch_params.max_concurrent_positions,
                take_profit_pct=stoch_params.take_profit_pct,
                slippage_pct=stoch_params.slippage_pct,
                maker_fee_pct=stoch_params.maker_fee_pct,
                taker_fee_pct=stoch_params.taker_fee_pct,
                funding_rate_per_day_pct=stoch_params.funding_rate_per_day_pct,
                trailing_activation_pct=stoch_params.trailing_activation_pct,
                trailing_gap_pct=stoch_params.trailing_gap_pct,
                trailing_check_interval_seconds=stoch_params.trailing_check_interval_seconds,
                max_position_days=stoch_params.max_position_days,
                margin_mode=stoch_params.margin_mode,
                aligned_high_stoch_mode=stoch_params.aligned_high_stoch_mode,
                signal_offset=stoch_params.signal_offset,
                enable_take_profit_check=stoch_params.enable_take_profit_check,
                enable_high_exit_cross=stoch_params.enable_high_exit_cross,
                use_midsold_filter=stoch_params.use_midsold_filter,
                enable_reversal_logic=stoch_params.enable_reversal_logic,
                enable_reversal_reentry=stoch_params.enable_reversal_reentry,
                trailing_use_first_entry_price=stoch_params.trailing_use_first_entry_price,
                trailing_use_close_for_stop_activation=stoch_params.trailing_use_close_for_stop_activation,
                take_profit_use_first_entry_price=stoch_params.take_profit_use_first_entry_price,
                enable_grid_martingales=stoch_params.enable_grid_martingales,
                grid_martingales_percent=stoch_params.grid_martingales_percent,
            )
        )
    elif submission.strategy == "pinbar_magic_v3":
        magic_params_v3 = cast(PinbarMagicStrategyParamsV3, params)
        strategy = PinBarMagicStrategyV3(
            PinBarMagicStrategyConfigV3(
                symbol=magic_params_v3.symbol,
                timeframe=magic_params_v3.timeframe,
                initial_equity=submission.initial_capital,
                leverage=magic_params_v3.leverage,
                equity_risk_pct=magic_params_v3.equity_risk_pct,
                atr_multiple=magic_params_v3.atr_multiple,
                trail_points=magic_params_v3.trail_points,
                trail_offset=magic_params_v3.trail_offset,
                symbol_mintick=magic_params_v3.symbol_mintick,
                slow_sma_period=magic_params_v3.slow_sma_period,
                medium_ema_period=magic_params_v3.medium_ema_period,
                fast_ema_period=magic_params_v3.fast_ema_period,
                atr_period=magic_params_v3.atr_period,
                entry_cancel_bars=magic_params_v3.entry_cancel_bars,
                trailing_tick_timeframe=magic_params_v3.trailing_tick_timeframe.strip(),
                use_trailing_tick_emulation=magic_params_v3.use_trailing_tick_emulation,
                use_stop_fill_open_gap=magic_params_v3.use_stop_fill_open_gap,
                entry_activation_mode=magic_params_v3.entry_activation_mode.strip().lower(),
                enable_friday_close=magic_params_v3.enable_friday_close,
                friday_close_hour_utc=magic_params_v3.friday_close_hour_utc,
                enable_ema_cross_close=magic_params_v3.enable_ema_cross_close,
                risk_equity_include_unrealized=magic_params_v3.risk_equity_include_unrealized,
                risk_equity_mark_source=magic_params_v3.risk_equity_mark_source.strip().lower(),
            )
        )
    elif submission.strategy == "ema_avwap_pullback":
        ema_avwap_params = cast(EmaAvwapPullbackParams, params)
        strategy = _build_ema_avwap_pullback_strategy(
            ema_avwap_params,
            initial_equity=submission.initial_capital,
        )
    elif submission.strategy == "pinbar":
        pinbar_params = cast(PinbarStrategyParams, params)
        strategy = PinbarStrategy(
            PinbarStrategyConfig(
                symbol=pinbar_params.symbol,
                timeframe=pinbar_params.timeframe,
                leverage=pinbar_params.leverage,
                take_profit_pct=pinbar_params.take_profit_pct,
                stop_loss_mode=pinbar_params.stop_loss_mode,
                stop_loss_pct=pinbar_params.stop_loss_pct,
                exchange_fee_pct=pinbar_params.exchange_fee_pct,
                min_shadow_body_ratio=pinbar_params.min_shadow_body_ratio,
                shadow_dominance_ratio=pinbar_params.shadow_dominance_ratio,
            )
        )
    elif submission.strategy == "engulfing":
        engulfing_params = cast(EngulfingStrategyParams, params)
        strategy = EngulfingStrategy(
            EngulfingStrategyConfig(
                symbol=engulfing_params.symbol,
                timeframe=engulfing_params.timeframe,
                window_size=engulfing_params.window_size,
                leverage=engulfing_params.leverage,
                take_profit_pct=engulfing_params.take_profit_pct,
                doji_size=engulfing_params.doji_size,
                stop_loss_mode=engulfing_params.stop_loss_mode,
                stop_loss_pct=engulfing_params.stop_loss_pct,
                skip_large_upper_wick=engulfing_params.skip_large_upper_wick,
                skip_bollinger_cross=engulfing_params.skip_bollinger_cross,
                bollinger_period=engulfing_params.bollinger_period,
                bollinger_stddev=engulfing_params.bollinger_stddev,
                enable_volume_pressure_filter=engulfing_params.enable_volume_pressure_filter,
                volume_window=engulfing_params.volume_window,
                max_volume_pressure_score=engulfing_params.max_volume_pressure_score,
                enable_stochastic_filter=engulfing_params.enable_stochastic_filter,
                stochastic_first_line=engulfing_params.stochastic_first_line,
                stochastic_first_period=engulfing_params.stochastic_first_period,
                stochastic_first_threshold=engulfing_params.stochastic_first_threshold,
                stochastic_second_line=engulfing_params.stochastic_second_line,
                stochastic_second_period=engulfing_params.stochastic_second_period,
                stochastic_second_threshold=engulfing_params.stochastic_second_threshold,
                stochastic_comparison=engulfing_params.stochastic_comparison,
                stochastic_d_smoothing=engulfing_params.stochastic_d_smoothing,
                exchange_fee_pct=engulfing_params.exchange_fee_pct,
            )
        )
    elif submission.strategy == "strong_trend_stair":
        strong_params = cast(StrongTrendStairParams, params)
        strategy = StrongTrendStairStrategy(
            StrongTrendStairStrategyConfig(
                symbol=strong_params.symbol,
                timeframe=strong_params.timeframe,
                leverage=strong_params.leverage,
                position_balance_pct=strong_params.position_size_pct,
                starting_balance_usd=strong_params.starting_balance_usd,
                hard_stop_loss_pct=strong_params.hard_stop_loss_pct,
                trail_start_pct=strong_params.trail_start_pct,
                trail_offset_pct=strong_params.trail_offset_pct,
                ema_fast_len=strong_params.ema_fast_len,
                ema_mid_len=strong_params.ema_mid_len,
                ema_slow_len=strong_params.ema_slow_len,
                slope_lookback=strong_params.slope_lookback,
                st_atr_len=strong_params.st_atr_len,
                st_factor=strong_params.st_factor,
                di_len=strong_params.di_len,
                adx_smooth=strong_params.adx_smooth,
                adx_min=strong_params.adx_min,
                reverse_on_opposite_signal=strong_params.reverse_on_opposite_signal,
            )
        )
    else:
        raise ValueError(f"Unknown strategy: {submission.strategy}")

    backtester = BaseBacktester(strategy=strategy, downloader=downloader, store=store)
    run_config = BacktestRunConfig(
        start=_ensure_utc(submission.start),
        end=_ensure_utc(submission.end),
        initial_capital=submission.initial_capital,
        override_download=submission.override_download,
        risk_free_rate=params.risk_free_rate,
        warmup_days=submission.warmup_days,
    )

    try:
        report = backtester.run(run_config)
        payload: Dict[str, Any] = {
            "job_id": job_id,
            "strategy": submission.strategy,
            "report": report.as_dict(),
        }
        if submission.strategy == "ema_avwap_pullback":
            payload["robust_analysis"] = _run_ema_avwap_robust_analysis(
                params=cast(EmaAvwapPullbackParams, params),
                analysis=submission.analysis,
                report=report,
                run_config=run_config,
                downloader=downloader,
                store=store,
            )
        return payload
    finally:
        store.close()
        client.close()


def _build_ema_avwap_pullback_strategy(
    params: EmaAvwapPullbackParams,
    *,
    initial_equity: float,
    overrides: Dict[str, Any] | None = None,
) -> EmaAvwapPullbackStrategy:
    values = params.model_dump()
    values.update(overrides or {})
    return EmaAvwapPullbackStrategy(
        EmaAvwapPullbackStrategyConfig(
            symbol=str(values["symbol"]),
            timeframe=str(values["timeframe"]),
            initial_equity=initial_equity,
            leverage=float(values["leverage"]),
            max_entry_notional_usdt=float(values["max_entry_notional_usdt"]),
            max_position_size_pct=float(values["max_position_size_pct"]),
            position_notional_pct=float(values["position_notional_pct"]),
            minimum_balance_usdt=float(values["minimum_balance_usdt"]),
            ema_length=int(values["ema_length"]),
            consecutive_count=int(values["consecutive_count"]),
            ema_validation_mode=str(values["ema_validation_mode"]),
            setup_waiting_replacement_mode=str(
                values["setup_waiting_replacement_mode"]
            ),
            max_setup_age_bars=int(values["max_setup_age_bars"]),
            max_entry_deviation_pct=float(values["max_entry_deviation_pct"]),
            position_sizing_mode=str(values["position_sizing_mode"]),
            entry_mode=EntryMode(str(values["entry_mode"])),
            exit_mode=ExitMode(str(values["exit_mode"])),
            exit_band=ExitBand(str(values["exit_band"])),
            avwap_multiplier_1=float(values["avwap_multiplier_1"]),
            avwap_multiplier_2=float(values["avwap_multiplier_2"]),
            avwap_multiplier_3=float(values["avwap_multiplier_3"]),
            rigid_stop_loss_pct=float(values["rigid_stop_loss_pct"]),
            trailing_activation_threshold_pct=float(
                values["trailing_activation_threshold_pct"]
            ),
            trailing_gap_pct=float(values["trailing_gap_pct"]),
            maker_fee_pct=float(values["maker_fee_pct"]),
            taker_fee_pct=float(values["taker_fee_pct"]),
            entry_slippage_pct=float(values["entry_slippage_pct"]),
            exit_slippage_pct=float(values["exit_slippage_pct"]),
            use_gap_cross_detection=bool(values["use_gap_cross_detection"]),
            max_decision_log_entries=int(values["max_decision_log_entries"]),
        )
    )


def _run_ema_avwap_robust_analysis(
    *,
    params: EmaAvwapPullbackParams,
    analysis: Any,
    report: Any,
    run_config: BacktestRunConfig,
    downloader: CandleDownloader,
    store: Any,
) -> Dict[str, Any]:
    robust: Dict[str, Any] = {}
    enabled = any(
        (
            analysis.include_monte_carlo,
            analysis.include_walk_forward,
            analysis.include_out_of_sample,
            analysis.include_parameter_perturbation,
        )
    )
    if not enabled:
        return robust

    analysis_run_config = replace(run_config, override_download=False)

    def build_backtester(
        parameter_overrides: Any,
        initial_capital: float,
    ) -> BaseBacktester:
        overrides = dict(parameter_overrides or {})
        strategy = _build_ema_avwap_pullback_strategy(
            params,
            initial_equity=initial_capital,
            overrides=overrides,
        )
        return BaseBacktester(strategy=strategy, downloader=downloader, store=store)

    if analysis.include_monte_carlo:
        mc_config = MonteCarloConfig(
            iterations=analysis.monte_carlo_iterations,
            seed=analysis.monte_carlo_seed,
            initial_capital=run_config.initial_capital,
            block_size=analysis.monte_carlo_block_size,
            drawdown_threshold_pct=analysis.monte_carlo_drawdown_threshold_pct,
            missed_fill_probability=analysis.monte_carlo_missed_fill_probability,
            extra_spread_slippage_pct_range=(
                analysis.monte_carlo_extra_spread_min_pct,
                analysis.monte_carlo_extra_spread_max_pct,
            ),
        )
        trade_regimes = None
        try:
            candles = store.load(
                params.symbol,
                params.timeframe,
                run_config.start,
                run_config.end,
            )
            segments = classify_regime_segments(candles)
            if segments:
                trade_regimes = label_trades_by_regime(report.trades, segments)
                robust["regime_segments"] = [segment.as_dict() for segment in segments]
        except Exception as exc:
            robust["regime_classification_error"] = str(exc)
        robust["monte_carlo"] = run_monte_carlo_suite(
            trades=report.trades,
            config=mc_config,
            start=run_config.start,
            end=run_config.end,
            risk_free_rate=run_config.risk_free_rate,
            trade_regimes=trade_regimes,
        )

    if analysis.include_out_of_sample:
        plan = build_three_way_oos_plan(
            start=run_config.start,
            end=run_config.end,
            training_fraction=analysis.oos_training_fraction,
            validation_fraction=analysis.oos_validation_fraction,
        )
        oos_result = run_out_of_sample_evaluation(
            build_backtester=build_backtester,
            run_config=analysis_run_config,
            plan=plan,
            candidates=[ParameterCandidate("current", {})],
            selection_metric="performance.net_profit_pct",
        )
        robust["out_of_sample"] = oos_result.as_dict(include_reports=False)

    if analysis.include_walk_forward:
        wf_config = WalkForwardConfig(
            start=run_config.start,
            end=run_config.end,
            train_window=timedelta(days=analysis.walk_forward_train_days),
            test_window=timedelta(days=analysis.walk_forward_test_days),
            step=(
                timedelta(days=analysis.walk_forward_step_days)
                if analysis.walk_forward_step_days
                else None
            ),
            anchored=analysis.walk_forward_anchored,
        )
        wf_result = run_walk_forward(
            build_backtester=build_backtester,
            run_config=analysis_run_config,
            walk_forward_config=wf_config,
            candidates=[ParameterCandidate("current", {})],
            selection_metric="performance.net_profit_pct",
        )
        robust["walk_forward"] = wf_result.as_dict(include_reports=False)

    if analysis.include_parameter_perturbation:
        perturbation = run_parameter_perturbation(
            build_backtester=build_backtester,
            run_config=analysis_run_config,
            base_parameters=params.model_dump(),
            rules=default_ema_avwap_parameter_perturbation_rules(),
            samples=analysis.parameter_perturbation_samples,
            seed=analysis.parameter_perturbation_seed,
            score_metric="performance.net_profit_pct",
        )
        robust["parameter_perturbation"] = perturbation

    return robust


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)
