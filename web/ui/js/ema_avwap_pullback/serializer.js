function normalizeNumber(value) {
  if (typeof value !== "string") return value;

  const compact = value.trim().replace(/\s+/g, "");
  if (!compact) return compact;
  if (compact.includes(",") && compact.includes(".")) {
    return compact.replace(/,/g, "");
  }
  if (!compact.includes(",")) return compact;

  const commas = compact.match(/,/g) || [];
  if (commas.length === 1) {
    const [whole, fraction] = compact.split(",");
    if (fraction.length <= 2) return `${whole}.${fraction}`;
  }
  return compact.replace(/,/g, "");
}

function requiredText(value, fallback) {
  if (typeof value !== "string") return fallback;
  return value.trim() || fallback;
}

function numberValue(value, fallback) {
  const parsed = Number.parseFloat(normalizeNumber(value));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function integerValue(value, fallback) {
  const parsed = Number.parseInt(normalizeNumber(value), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function optionalInteger(value) {
  const parsed = Number.parseInt(normalizeNumber(value), 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function booleanValue(value, fallback) {
  if (value === "true") return true;
  if (value === "false") return false;
  return fallback;
}

function datetimeValue(value) {
  if (typeof value !== "string" || !value) return null;

  const date = new Date(`${value}Z`);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

function getAnalysis(data) {
  return {
    include_monte_carlo: booleanValue(
      data.get("analysis_include_monte_carlo"),
      false,
    ),
    monte_carlo_iterations: integerValue(
      data.get("monte_carlo_iterations"),
      1000,
    ),
    monte_carlo_seed: optionalInteger(data.get("monte_carlo_seed")),
    monte_carlo_block_size: integerValue(data.get("monte_carlo_block_size"), 5),
    monte_carlo_drawdown_threshold_pct: numberValue(
      data.get("monte_carlo_drawdown_threshold_pct"),
      30,
    ),
    monte_carlo_missed_fill_probability: numberValue(
      data.get("monte_carlo_missed_fill_probability"),
      0,
    ),
    monte_carlo_extra_spread_min_pct: numberValue(
      data.get("monte_carlo_extra_spread_min_pct"),
      0,
    ),
    monte_carlo_extra_spread_max_pct: numberValue(
      data.get("monte_carlo_extra_spread_max_pct"),
      0.001,
    ),
    include_out_of_sample: booleanValue(
      data.get("analysis_include_out_of_sample"),
      false,
    ),
    oos_training_fraction: numberValue(data.get("oos_training_fraction"), 0.6),
    oos_validation_fraction: numberValue(
      data.get("oos_validation_fraction"),
      0.2,
    ),
    include_walk_forward: booleanValue(
      data.get("analysis_include_walk_forward"),
      false,
    ),
    walk_forward_train_days: integerValue(
      data.get("walk_forward_train_days"),
      90,
    ),
    walk_forward_test_days: integerValue(data.get("walk_forward_test_days"), 30),
    walk_forward_step_days: optionalInteger(
      data.get("walk_forward_step_days"),
    ),
    walk_forward_anchored: booleanValue(data.get("walk_forward_anchored"), false),
    include_parameter_perturbation: booleanValue(
      data.get("analysis_include_parameter_perturbation"),
      false,
    ),
    parameter_perturbation_samples: integerValue(
      data.get("parameter_perturbation_samples"),
      25,
    ),
    parameter_perturbation_seed: optionalInteger(
      data.get("parameter_perturbation_seed"),
    ),
  };
}

function getStrategyParams(data) {
  return {
    symbol: requiredText(data.get("symbol"), "ETHUSDT.P"),
    timeframe: requiredText(data.get("timeframe"), "1h"),
    leverage: numberValue(data.get("leverage"), 10),
    max_entry_notional_usdt: numberValue(
      data.get("max_entry_notional_usdt"),
      15,
    ),
    max_position_size_pct: numberValue(data.get("max_position_size_pct"), 10),
    position_notional_pct: numberValue(data.get("position_notional_pct"), 1),
    minimum_balance_usdt: numberValue(data.get("minimum_balance_usdt"), 0),
    max_setup_age_bars: integerValue(data.get("max_setup_age_bars"), 3),
    max_entry_deviation_pct: numberValue(data.get("max_entry_deviation_pct"), 1),
    position_sizing_mode: requiredText(
      data.get("position_sizing_mode"),
      "risk_amount_per_price",
    ),
    entry_mode: requiredText(data.get("entry_mode"), "close"),
    exit_mode: requiredText(data.get("exit_mode"), "close"),
    exit_band: requiredText(data.get("exit_band"), "band_1"),
    ema_length: integerValue(data.get("ema_length"), 55),
    consecutive_count: integerValue(data.get("consecutive_count"), 4),
    ema_validation_mode: requiredText(data.get("ema_validation_mode"), "body"),
    setup_waiting_replacement_mode: requiredText(
      data.get("setup_waiting_replacement_mode"),
      "keep_waiting",
    ),
    avwap_multiplier_1: numberValue(data.get("avwap_multiplier_1"), 1),
    avwap_multiplier_2: numberValue(data.get("avwap_multiplier_2"), 2),
    avwap_multiplier_3: numberValue(data.get("avwap_multiplier_3"), 20),
    rigid_stop_loss_pct: numberValue(data.get("rigid_stop_loss_pct"), 3),
    trailing_activation_threshold_pct: numberValue(
      data.get("trailing_activation_threshold_pct"),
      0,
    ),
    trailing_gap_pct: numberValue(data.get("trailing_gap_pct"), 1),
    maker_fee_pct: numberValue(data.get("maker_fee_pct"), 0.0002),
    taker_fee_pct: numberValue(data.get("taker_fee_pct"), 0.0006),
    entry_slippage_pct: numberValue(data.get("entry_slippage_pct"), 0),
    exit_slippage_pct: numberValue(data.get("exit_slippage_pct"), 0),
    funding_mode: requiredText(data.get("funding_mode"), "historical"),
    use_gap_cross_detection: booleanValue(
      data.get("use_gap_cross_detection"),
      true,
    ),
    max_decision_log_entries: integerValue(
      data.get("max_decision_log_entries"),
      20000,
    ),
    risk_free_rate: numberValue(data.get("risk_free_rate"), 0),
  };
}

export function serializeBacktestForm(form) {
  const data = new FormData(form);
  const start = datetimeValue(data.get("start"));
  const end = datetimeValue(data.get("end"));
  if (!start || !end) return null;

  return {
    strategy: "ema_avwap_pullback",
    start,
    end,
    initial_capital: numberValue(data.get("initial_capital"), 10000),
    override_download: booleanValue(data.get("override_download"), false),
    warmup_days: integerValue(data.get("warmup_days"), 30),
    analysis: getAnalysis(data),
    params: getStrategyParams(data),
  };
}
