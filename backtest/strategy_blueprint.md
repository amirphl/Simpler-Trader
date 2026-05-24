# Backtest Strategy Blueprint

Use this file as the checklist and starter template for adding or reviewing a
strategy under `backtest/`. It is based on the current architecture in
`backtest/base.py` and on the execution patterns used by
`backtest/pinbar_magic_strategy.py`, `backtest/strong_trend_stair_strategy.py`,
and the other strategy modules.

## The Contract

Every strategy must implement `BacktestStrategy` from `backtest/base.py`:

- `name() -> str`
- `symbols() -> Sequence[str]`
- `timeframes() -> Sequence[str]`
- `run(context: BacktestContext) -> Tuple[Sequence[TradePerformance], Mapping[str, Any] | None]`

`BaseBacktester` handles data download, warmup range expansion, report creation,
and global statistics. The strategy handles only strategy-specific execution.

The strategy must:

- Read candles from `context.data[symbol][timeframe]`.
- Respect `context.config.start` and `context.config.end`.
- Respect `context.ignore_candles` so warmup candles can initialize indicators
  but cannot generate trades.
- Return `TradePerformance` objects with finite `pnl` and `return_pct`.
- Return optional extra stats as a plain mapping.
- Avoid mutating the shared context or candle data.

## Backtester Semantics

Understand these base-layer behaviors before coding:

- `BacktestRunConfig.start` is normalized to the beginning of its UTC day.
- `BacktestRunConfig.end` is normalized to the end of its UTC day.
- `warmup_days` expands the downloaded data range backward, but
  `build_ignore_candles()` marks pre-start candles as ignored.
- Global report stats are built from returned trades and
  `BacktestRunConfig.initial_capital`.
- Strategy-specific capital or equity fields must use the same initial capital
  passed from shell/UI into the strategy config when sizing depends on equity.
- `validate_trade_sequence()` normalizes trade timestamps to UTC, rejects
  negative durations, rejects non-finite PnL/returns, and sorts trades by
  `(exit_time, entry_time)`.
- Extra strategy stats are merged into `BacktestStatistics.extra`; avoid keys
  that accidentally contradict global stats unless the distinction is explicit.

## Design Principles

Follow these rules unless a strategy has a documented reason to differ:

- Keep user-facing configuration in a frozen dataclass.
- Validate all numeric and enum inputs in `__post_init__`.
- Normalize user-facing strings such as symbols, timeframes, and mode names.
- Validate Binance intervals early with `interval_to_milliseconds()` when a
  strategy accepts timeframe strings.
- Keep shared indicators in `backtest/indicators.py` or another neutral helper,
  not inside a specific strategy module.
- Use explicit internal dataclasses for pending orders, open positions, signal
  snapshots, tick snapshots, or setup state.
- Keep candle-by-candle execution order readable and documented.
- Be explicit about fill ordering, gap behavior, same-bar behavior, reversal
  behavior, pyramiding, pending-order cancellation, and forced exits.
- Use only data available at or before the current decision point.
- Return rich trade metadata so a report can be debugged without rerunning the
  strategy.
- Return counters for skipped, replaced, cancelled, filled, and closed events.
- Keep shell, web model, web runner, UI form, docs, and strategy defaults aligned.

## Decisions To Write Down Before Coding

Before implementation, document these choices in code comments, config names, or
strategy docs:

- Inputs: single symbol or many symbols; one timeframe or multiple timeframes.
- Minimum history: indicator periods, candle lookbacks, warmup days, and whether
  warmup is configured by the runner or forced by the strategy.
- Entry signals: exact long and short conditions.
- Order type: market, next-bar open, stop order, limit order, close fill, or
  same-bar activation.
- Exit rules: hard stop, trailing stop, take profit, reversal, time/date exit,
  indicator close, forced end-of-backtest close.
- Position sizing: fixed quantity, percent of equity, risk per trade, margin
  percent, or notional cap.
- Leverage semantics: whether leverage changes quantity, margin, PnL, reporting
  only, or liquidation math.
- Equity source: realized equity only, or realized plus unrealized PnL.
- Mark price for unrealized equity: close, open, HL2, OHLC4, or another source.
- Intrabar model: high-first, low-first, nearest-wick-first, tick emulation, or
  close-only.
- Gap behavior: whether an open beyond a trigger fills at open or at trigger.
- Pyramiding: disabled, limited, or unlimited.
- Opposite signals: ignore, close only, reverse, or replace pending order.
- Pending orders: activation timing, cancellation age, replacement rules.
- Fees, slippage, funding, spread, and whether they are included in PnL.
- Stats and metadata required for post-run debugging.

## Strategy Config Checklist

Every config dataclass should include and validate:

- Required market inputs: `symbol`, `timeframe`, or `symbols`.
- Capital/risk fields that must be consistent with `BacktestRunConfig`.
- Indicator periods and lookback lengths.
- Execution-mode flags and enum fields.
- Stop, take-profit, trailing, fee, slippage, and sizing fields.
- Optional extra timeframe fields.
- Positive values for prices, periods, capital, leverage, and multipliers.
- Non-negative values for slippage, fees, warmup-like counts, and optional rates.
- Valid ranges for percentages and hours.
- Compatible period ordering when required, such as fast < mid < slow.

Use `object.__setattr__` in frozen dataclasses after normalization.

## Data And Warmup Checklist

In `run()`:

- Load candles with `context.data.get(symbol, {}).get(timeframe, [])`.
- Return `([], {"note": "no_data"})` or `([], {"note": "insufficient_data"})`
  rather than raising for normal empty-data cases.
- Compute indicator arrays before the main loop.
- Compute `start_index` or `min_idx` from both indicator requirements and
  `context.ignore_candles`.
- Skip any candle whose `close_time < context.config.start`.
- Break once a candle's `open_time > context.config.end`.
- Track the last in-range candle if forced end closes should respect the run
  window instead of closing on post-window data.
- For multi-timeframe strategies, declare every needed timeframe in
  `timeframes()` so `BaseBacktester` downloads it.
- If a secondary timeframe is optional and missing, document whether to fall
  back or abort.

## Execution Model Checklist

Keep the main loop ordered in the same sequence the strategy assumes. A good
loop usually has these phases:

1. Load current candle and previous candles.
2. Read indicator values and skip if any required value is unavailable.
3. Process open-position exits that should happen before entries.
4. Fill pending orders whose activation time has arrived.
5. Process bar-close exits, date exits, or close-all rules.
6. Cancel stale pending orders.
7. Generate new signals.
8. Place or replace pending orders.
9. Optionally fill same-bar orders if the model allows it.
10. Close any open position at the end if the strategy requires it.

If your order differs, write down why.

## PnL, Returns, And Equity

Be precise:

- `pnl` is in quote currency and should be signed.
- `return_pct` must state its basis: price return, margin return, or equity
  return. Do not mix meanings across one strategy.
- Base report `net_profit_pct` uses `BacktestRunConfig.initial_capital`.
- If strategy sizing uses `initial_equity`, shell and UI runners must pass the
  same value as `BacktestRunConfig.initial_capital`.
- If leverage is reporting-only, do not multiply quantity or PnL by leverage.
- If leverage changes margin or liquidation risk, store those values in metadata.
- Keep gross-loss stats either signed or absolute, and name them accordingly.
- Clamp or handle non-positive equity deliberately; do not let division by zero
  produce `inf` or `nan`.

## Trade Metadata Checklist

Minimum recommended metadata:

- `direction`
- `entry_price`
- `exit_price`
- `qty`
- `reason` or `exit_reason`

Recommended additions:

- `entry_index`
- `holding_bars`
- `risk_amount`
- `r_multiple`
- `leverage`
- `notional`
- `margin`
- `stop_loss` or `stop_at_exit`
- `take_profit`
- `trailing_active`
- `trailing_stop`
- signal/index fields needed to reconstruct the decision

## Extra Stats Checklist

Useful counters and summary fields:

- starting and ending equity/balance
- signal counts by direction
- entries by direction
- replaced order counts
- cancelled order counts
- filled order counts
- skipped signal counts and reasons
- stop, take-profit, trailing, reversal, time/date, and forced-close counts
- pending orders at end
- open position at end
- average return, average R multiple, average holding bars
- long/short trade counts and win rates
- gross profit/loss split by direction
- exit reason counts

## Pin Bar Magic Notes

`backtest/pinbar_magic_strategy.py` is the reference for a complex,
TradingView/Pine-compatible strategy. Carry these notes forward when reviewing
or extending it.

### Configuration

- `initial_equity` is configurable and must be wired from shell/UI
  `initial_capital`.
- `equity_risk_pct` defines risk amount as a percentage of risk equity.
- `leverage` is recorded for reporting only in Pin Bar Magic; it does not
  multiply position size.
- `atr_multiple` contributes to stop distance used for risk sizing.
- `trail_points` and `trail_offset` are expressed in ticks, matching Pine
  `slPoints` and `slOffset` inputs.
- `symbol_mintick` is the absolute price value of one tick.
- Convert ticks to price at use time with `trail_points_price` and
  `trail_offset_price`; do not bake the conversion into a replaced frozen config.
- `trail_offset` must be greater than zero. A zero offset makes the trailing
  stop equal to the peak price and can exit immediately on the first adverse
  tick.
- `entry_activation_mode` is either `next_bar` or `same_bar`.
- `use_stop_fill_open_gap` decides whether stop entries crossed by the open fill
  at the open.
- `enable_friday_close` and `friday_close_hour_utc` model Friday close-all
  behavior.
- `enable_ema_cross_close` models EMA crossover/crossunder close-all behavior.
- `risk_equity_include_unrealized` controls whether open PnL is included in risk
  equity for new signals.
- `risk_equity_mark_source` is one of `close`, `open`, `hl2`, or `ohlc4`.
- `use_trailing_tick_emulation` may add `trailing_tick_timeframe` to
  `timeframes()`.

### Indicators And Signals

- Indicators are precomputed from closes/candles:
  - fast EMA
  - medium EMA
  - slow SMA
  - ATR
- Start after the maximum indicator period plus one candle.
- Long signal:
  - fast EMA > medium EMA > slow SMA
  - bullish pinbar
  - candle pierces one of the moving averages from above
- Short signal:
  - fast EMA < medium EMA < slow SMA
  - bearish pinbar
  - candle pierces one of the moving averages from below
- Pinbar range must be positive.
- Bullish/bearish pinbar detection supports both green and red candles by using
  wick position within the full range.

### Exact Loop Order

Pin Bar Magic uses this order:

1. Process trailing stop on the current bar.
2. Fill pending stop-entry orders that are active for the current bar.
3. Process close-all rules: Friday close and EMA cross close.
4. Cancel stale pending orders after `entry_cancel_bars`.
5. Generate long/short signals and place or replace stop orders.
6. If `entry_activation_mode == "same_bar"` and no close-all fired, attempt
   same-bar activation.

This order matters. Trailing exits happen before entry fills so an exit on bar N
does not block an entry signal on bar N. Close-all rules cancel pending orders
and suppress same-bar re-entry.

### Pending Orders And Fills

- Pending orders store direction, entry price, quantity, risk amount, creation
  index, and activation index.
- `next_bar` activation uses signal index + 1.
- `same_bar` activation uses signal index.
- Long stop entries fill if high reaches the trigger, or if open gaps above the
  trigger and `use_stop_fill_open_gap` is enabled.
- Short stop entries fill if low reaches the trigger, or if open gaps below the
  trigger and `use_stop_fill_open_gap` is enabled.
- Only one pending order can fill per bar; the long side is checked first.
- This one-fill rule enforces Pine `pyramiding=0` semantics and prevents both
  long and short orders from filling in one candle.
- Same-direction fills while already in a same-direction position are no-ops.
- Opposite-direction fills close the existing position at the fill price and
  open the reversal.

### Risk Sizing

- Position size follows Pine's formula:
  `units = risk_amount / abs(entry_price - stop_price)`.
- `_compute_quantity` takes explicit `entry_price`, `stop_price`, and
  `risk_amount` arguments.
- Leverage does not multiply units in Pin Bar Magic. It determines exchange
  margin in live trading, not base-asset units for a fixed dollar risk.
- Long risk stop uses previous candle low minus ATR multiple.
- Short risk stop uses previous candle high plus ATR multiple.
- If distance or risk amount is non-positive, quantity is zero and no order is
  placed.

### Trailing Stops

- Trail distances are always converted from ticks to absolute price at use time.
- Bar-resolution trailing uses a TradingView broker-emulator heuristic:
  `open -> nearest extreme -> other extreme -> close`.
- For longs:
  - activation is `entry_price + trail_points_price`
  - extreme tracks the highest price since activation
  - trailing stop is extreme minus `trail_offset_price`
  - exit when the path trades at or below the trailing stop
- For shorts:
  - activation is `entry_price - trail_points_price`
  - extreme tracks the lowest price since activation
  - trailing stop is extreme plus `trail_offset_price`
  - exit when the path trades at or above the trailing stop
- Tick emulation uses close prices from the tick timeframe bucketed into primary
  bars.
- If tick data is missing, the strategy falls back to primary candles and logs a
  warning.
- Empty tick buckets fall back to the primary bar close.

### Close-All Behavior

- Friday close fires when `moment.weekday() == 4` and the configured UTC hour
  matches.
- EMA close fires on fast/medium EMA crossunder or crossover when enabled.
- Close-all closes at current candle close.
- Close-all cancels both pending long and short orders immediately.
- Same-bar re-entry after close-all is suppressed.
- Signals may still be generated on a close-all bar, but in same-bar mode they
  cannot fill on that same bar.

### PnL, Returns, And Stats

- Long PnL is `(exit - entry) * qty`.
- Short PnL is `(entry - exit) * qty`.
- `return_pct` is price return:
  - long: `(exit - entry) / entry * 100`
  - short: `(entry - exit) / entry * 100`
- `r_multiple` is `pnl / risk_amount`.
- `holding_bars` is `max(exit_index - entry_index, 0)`.
- `gross_loss_long` and `gross_loss_short` are positive magnitudes, not negative
  signed sums.
- Summary stats include `avg_return_pct`.
- Track `pending_orders_at_end` and `open_position_at_end` to reveal unfinished
  state.
- Store exit reason counts in `exit_reason_counts`.

### Pin Bar Magic Bug-Fix Log To Preserve

These are all current implementation notes from the strategy header:

1. `_compute_quantity` no longer multiplies by leverage. Pine's sizing formula
   is units = risk / distance; leverage determines exchange margin, not units.
2. `gross_loss_long` and `gross_loss_short` accumulate `abs(pnl)` so they are
   positive magnitudes as the names imply.
3. `return_pct` is percentage price return:
   `(exit - entry) / entry * 100` for longs and
   `(entry - exit) / entry * 100` for shorts.
4. Same-bar re-entry after close-all is suppressed. When an EMA cross or Friday
   close fires in step 3, any signal from step 5 of the same bar is not filled
   in same-bar activation mode. Pending orders are immediately cancelled when
   close-all fires, matching Pine semantics.
5. `_fill_pending_entries_for_bar` breaks after the first fill so both pending
   orders cannot execute in the same bar, matching Pine `pyramiding=0`.
6. `_compute_quantity` takes explicit `entry_price`, `stop_price`, and
   `risk_amount` arguments, removing the confusing positional
   numerator/denominator API.
7. `symbol_mintick` is stored in config as a tick size; conversion to absolute
   price happens at use time through `trail_points_price` and
   `trail_offset_price`, consistent with the live coordinator.
8. `trail_offset` requires `> 0`; zero offset makes the trailing stop equal to
   the peak price and causes immediate exits on adverse ticks.
9. `initial_equity` is configurable.
10. `avg_return_pct` is included in summary stats.

## Shell And UI Integration Checklist

When adding a strategy, update every entrypoint:

- `backtest/__init__.py` exports.
- `cmd/backtest/main.py`:
  - env loader
  - parser
  - config resolver
  - validator
  - strategy builder
  - strategy selection
- `webserver/models.py`:
  - params model
  - request union
  - strategy map
  - validators matching strategy config
- `webserver/runner.py` strategy construction.
- `web/ui/<strategy>.html` form controls.
- `web/ui/js/<strategy>.js` payload serialization.
- `web/ui/index.html` strategy link.
- README/config examples/scripts, if the strategy is user-facing.
- Plot metadata support if `backtest/plotter.py` should show entry, exit, stop,
  or take-profit levels.

For equity-based strategies, shell and UI must pass one consistent initial
capital into both:

- `BacktestRunConfig.initial_capital`
- the strategy config's equity field, such as `initial_equity` or
  `starting_balance_usd`

## Testing Checklist

At minimum, add focused tests for:

- config validation and normalization
- no data and insufficient data
- warmup/no-trade-before-start behavior
- long and short signal generation
- stop hit and take-profit/trailing hit behavior
- open gaps through entries or exits
- same-bar behavior
- opposite signal/reversal behavior
- end-of-backtest forced close
- PnL sign correctness for long and short
- return percentage basis
- metadata fields
- extra stats counters
- shell/web config construction for important defaults

Run practical checks:

- `python -m compileall backtest cmd/backtest/main.py webserver`
- focused unit tests for the strategy
- CLI `--help` for the strategy
- direct synthetic-candle run
- web model validation for the strategy params
- web runner construction with external IO mocked
- UI JS syntax check with `node --check`
- local FastAPI static-page smoke test when web dependencies are available

## Plot And Report Compatibility

For clean reports and charts:

- `BacktestReport.as_dict()` serializes `TradePerformance.metadata`, so only
  store JSON-friendly scalar values there.
- `backtest/plotter.py` can use `entry_price`, `exit_price`, `stop_loss`, and
  `take_profit` metadata when present.
- Keep `notes` short and stable; use metadata for verbose diagnostics.
- If a strategy uses custom exit names, keep them consistent across
  `notes`, `metadata["reason"]`, and `exit_reason_counts`.
- If a strategy has no natural take-profit or stop-loss, omit those metadata
  keys instead of inventing misleading values.

## Common Failure Modes

- Using future candle data by accident.
- Starting signals before indicators are warmed up.
- Ignoring `context.ignore_candles`.
- Closing end-of-backtest positions using data beyond `context.config.end`.
- Using strategy initial equity that differs from report initial capital.
- Inconsistent leverage semantics.
- Mixing price-return percent and margin/equity-return percent.
- Forgetting open-gap behavior through stops or entry triggers.
- Letting both long and short orders fill in one bar when pyramiding is disabled.
- Allowing same-bar re-entry after a close-all rule that should suppress it.
- Forgetting to cancel pending orders after close-all.
- Leaving pending orders or open positions unreported at the end.
- Returning non-finite PnL/returns.
- Returning trades with sparse metadata.
- Adding shell options without matching web/UI fields, or vice versa.

## Starter Template

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from candle_downloader.binance import interval_to_milliseconds
from candle_downloader.models import Candle

from .base import BacktestContext, BacktestStrategy, TradePerformance


@dataclass(frozen=True)
class YourStrategyConfig:
    symbol: str
    timeframe: str
    initial_equity: float = 100.0
    leverage: float = 1.0
    risk_pct: float = 1.0

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
        if self.risk_pct <= 0:
            raise ValueError("risk_pct must be positive")
        interval_to_milliseconds(timeframe)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timeframe", timeframe)


@dataclass
class _PositionState:
    direction: str
    entry_time: datetime
    entry_index: int
    entry_price: float
    qty: float
    risk_amount: float
    stop_price: float


class YourStrategy(BacktestStrategy):
    def __init__(self, config: YourStrategyConfig) -> None:
        self._config = config

    def name(self) -> str:
        return "YourStrategy"

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

        min_history = 50
        ignore_count = context.ignore_candles.get(cfg.symbol, {}).get(cfg.timeframe, 0)
        start_index = max(ignore_count, min_history)
        if len(candles) <= start_index:
            return [], {"note": "insufficient_data", "candles": len(candles)}

        trades: List[TradePerformance] = []
        stats: Dict[str, Any] = {
            "initial_equity": cfg.initial_equity,
            "final_equity": cfg.initial_equity,
            "signals_long": 0,
            "signals_short": 0,
            "entries_long": 0,
            "entries_short": 0,
            "forced_close_at_end": 0,
            "exit_reason_counts": {},
        }
        position: Optional[_PositionState] = None
        equity = cfg.initial_equity
        last_in_range: Candle | None = None
        last_in_range_index: int | None = None

        for idx in range(start_index, len(candles)):
            candle = candles[idx]
            if candle.close_time < context.config.start:
                continue
            if candle.open_time > context.config.end:
                break
            last_in_range = candle
            last_in_range_index = idx

            long_signal = False
            short_signal = False

            if position is not None:
                stop_hit = (
                    candle.low <= position.stop_price
                    if position.direction == "long"
                    else candle.high >= position.stop_price
                )
                if stop_hit:
                    exit_price = position.stop_price
                    pnl = self._close_trade(
                        position=position,
                        exit_price=exit_price,
                        exit_time=candle.close_time,
                        exit_index=idx,
                        reason="stop_loss",
                        trades=trades,
                        stats=stats,
                    )
                    equity += pnl
                    position = None
                    continue

            if position is None:
                if long_signal:
                    position = self._open_position(
                        direction="long",
                        candle=candle,
                        index=idx,
                        equity=equity,
                    )
                    stats["entries_long"] += 1
                elif short_signal:
                    position = self._open_position(
                        direction="short",
                        candle=candle,
                        index=idx,
                        equity=equity,
                    )
                    stats["entries_short"] += 1

        if position is not None and last_in_range is not None:
            pnl = self._close_trade(
                position=position,
                exit_price=last_in_range.close,
                exit_time=last_in_range.close_time,
                exit_index=last_in_range_index or position.entry_index,
                reason="forced_end_close",
                trades=trades,
                stats=stats,
            )
            equity += pnl
            stats["forced_close_at_end"] += 1

        stats["final_equity"] = equity
        stats["open_position_at_end"] = position.direction if position else None
        return trades, stats

    def _open_position(
        self, *, direction: str, candle: Candle, index: int, equity: float
    ) -> _PositionState:
        entry_price = candle.close
        stop_price = entry_price * (0.99 if direction == "long" else 1.01)
        risk_amount = equity * (self._config.risk_pct / 100.0)
        risk_per_unit = abs(entry_price - stop_price)
        qty = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
        return _PositionState(
            direction=direction,
            entry_time=candle.close_time,
            entry_index=index,
            entry_price=entry_price,
            qty=qty,
            risk_amount=risk_amount,
            stop_price=stop_price,
        )

    def _close_trade(
        self,
        *,
        position: _PositionState,
        exit_price: float,
        exit_time: datetime,
        exit_index: int,
        reason: str,
        trades: List[TradePerformance],
        stats: Dict[str, Any],
    ) -> float:
        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * position.qty
            return_pct = (exit_price - position.entry_price) / position.entry_price * 100.0
        else:
            pnl = (position.entry_price - exit_price) * position.qty
            return_pct = (position.entry_price - exit_price) / position.entry_price * 100.0

        r_multiple = pnl / position.risk_amount if position.risk_amount > 0 else 0.0
        trades.append(
            TradePerformance(
                entry_time=position.entry_time,
                exit_time=exit_time,
                pnl=pnl,
                return_pct=return_pct,
                notes=reason,
                metadata={
                    "direction": position.direction,
                    "entry_index": position.entry_index,
                    "entry_price": position.entry_price,
                    "exit_price": exit_price,
                    "qty": position.qty,
                    "risk_amount": position.risk_amount,
                    "r_multiple": r_multiple,
                    "holding_bars": max(exit_index - position.entry_index, 0),
                    "stop_at_exit": position.stop_price,
                    "reason": reason,
                },
            )
        )
        reason_counts = stats["exit_reason_counts"]
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        return pnl
```

## Final Review Checklist

Before trusting a strategy, verify:

- It compiles and imports.
- Config invalid values fail early.
- Missing/insufficient data returns cleanly.
- It does not trade before warmup is complete.
- It respects `start` and `end`.
- It does not use future data.
- Long and short PnL are symmetric and sign-correct.
- Stops, triggers, and trailing logic behave correctly on open gaps.
- Same-bar behavior is documented and tested.
- End-of-backtest open positions are closed exactly once or explicitly reported.
- Returned stats and trade metadata match the actual execution path.
- Shell and UI paths create the same effective strategy config.
