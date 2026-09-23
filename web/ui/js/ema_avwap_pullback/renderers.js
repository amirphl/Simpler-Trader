function getElement(id) {
  return document.getElementById(id);
}

export function formatNumber(value, digits = 4) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "n/a";
}

export function formatPercent(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(digits)}%`
    : "n/a";
}

export function renderMetricTiles(host, metrics) {
  if (!host) return;

  const fragment = document.createDocumentFragment();
  metrics.forEach(([label, value]) => {
    const tile = document.createElement("div");
    tile.className = "metric-tile";

    const labelElement = document.createElement("span");
    labelElement.className = "label";
    labelElement.textContent = label;

    const valueElement = document.createElement("span");
    valueElement.className = "value";
    valueElement.textContent = value ?? "n/a";

    tile.append(labelElement, valueElement);
    fragment.appendChild(tile);
  });

  host.replaceChildren(fragment);
}

function getStatistics(result) {
  return result?.report?.statistics || {};
}

export function renderQuickMetrics(result) {
  const stats = getStatistics(result);
  const performance = stats.performance || {};
  renderMetricTiles(getElement("quick-metrics"), [
    ["Trades", stats.total_trades ?? "n/a"],
    [
      "Win Rate",
      typeof stats.win_rate === "number"
        ? formatPercent(stats.win_rate * 100, 1)
        : "n/a",
    ],
    ["Net PnL", formatNumber(stats.net_profit, 2)],
    ["Profit Factor", formatNumber(stats.profit_factor, 2)],
    ["Max DD", formatPercent(stats.max_drawdown_pct)],
    ["CAGR", formatPercent(stats.cagr_pct)],
    ["Sortino", formatNumber(performance.sortino_ratio, 2)],
    ["Calmar", formatNumber(performance.calmar_ratio, 2)],
  ]);
}

export function renderPerformance(result) {
  const stats = getStatistics(result);
  const performance = stats.performance || {};

  renderMetricTiles(getElement("performance-metrics"), [
    ["Trades", performance.number_of_trades ?? stats.total_trades ?? "n/a"],
    ["Expectancy", formatNumber(performance.expectancy_per_trade, 2)],
    ["Avg Win", formatNumber(performance.average_win, 2)],
    ["Avg Loss", formatNumber(performance.average_loss, 2)],
    ["Profit Factor", formatNumber(performance.profit_factor, 2)],
    ["Sharpe", formatNumber(performance.sharpe_ratio, 2)],
    ["Sortino", formatNumber(performance.sortino_ratio, 2)],
    ["Calmar", formatNumber(performance.calmar_ratio, 2)],
    ["Max DD", formatPercent(performance.maximum_drawdown_pct)],
    ["Avg DD", formatPercent(performance.average_drawdown_pct)],
    ["Time Under Water", formatPercent(performance.time_under_water_pct)],
    ["Losing Streak", performance.longest_losing_streak ?? "n/a"],
    ["Skew", formatNumber(performance.return_skewness, 3)],
    ["Kurtosis", formatNumber(performance.return_excess_kurtosis, 3)],
    ["Exposure Adj.", formatPercent(performance.exposure_adjusted_return_pct)],
    ["Turnover", formatNumber(performance.turnover, 2)],
    ["Fee Drag", formatNumber(performance.fee_drag, 2)],
    ["Slippage Drag", formatNumber(performance.slippage_drag, 2)],
    ["Funding PnL", formatNumber(performance.funding_pnl, 2)],
    ["Deflated Sharpe", formatNumber(performance.deflated_sharpe_ratio, 3)],
  ]);

  const output = getElement("performance-details-output");
  if (output) {
    output.textContent = Object.keys(performance).length
      ? JSON.stringify(performance, null, 2)
      : "No detailed statistics yet.";
  }
}

export function renderRobustAnalysis(result) {
  const robust = result?.robust_analysis || {};
  const monteCarlo = robust.monte_carlo || {};
  const reshuffle = monteCarlo.trade_reshuffling || {};
  const block = monteCarlo.block_bootstrap || {};
  const walkForward = robust.walk_forward || {};
  const outOfSample = robust.out_of_sample || {};
  const perturbation = robust.parameter_perturbation || {};
  const finalStatistics = outOfSample.final_statistics || {};

  renderMetricTiles(getElement("robust-analysis-metrics"), [
    ["MC DD p95", formatPercent(block.max_drawdown_pct_p95)],
    ["MC DD p99", formatPercent(block.max_drawdown_pct_p99)],
    [
      "MC Loss Prob",
      typeof reshuffle.probability_of_loss === "number"
        ? formatPercent(reshuffle.probability_of_loss * 100)
        : "n/a",
    ],
    [
      "MC Threshold Prob",
      typeof reshuffle.probability_of_drawdown_ge_threshold === "number"
        ? formatPercent(reshuffle.probability_of_drawdown_ge_threshold * 100)
        : "n/a",
    ],
    [
      "WF Folds",
      Array.isArray(walkForward.folds) ? walkForward.folds.length : "n/a",
    ],
    [
      "WF Net",
      walkForward.combined_statistics
        ? formatPercent(walkForward.combined_statistics.net_profit_pct)
        : "n/a",
    ],
    ["Final OOS Net", formatPercent(finalStatistics.net_profit_pct)],
    ["Perturb p05", formatPercent(perturbation.score_p05)],
    ["Perturb p50", formatPercent(perturbation.score_p50)],
    ["Perturb p95", formatPercent(perturbation.score_p95)],
  ]);

  const output = getElement("robust-analysis-output");
  if (output) {
    output.textContent = Object.keys(robust).length
      ? JSON.stringify(robust, null, 2)
      : "No robust analysis requested yet.";
  }
}

export function renderConfigSnapshot(result) {
  const output = getElement("config-snapshot-output");
  if (!output) return;

  const report = result?.report || {};
  output.textContent = JSON.stringify(
    {
      strategy: result?.strategy || report.strategy || null,
      run_config: report.config || {},
      strategy_config: report.statistics?.config || {},
      execution_assumptions: report.statistics?.execution_assumptions || {},
    },
    null,
    2,
  );
}

export function createDecisionLogRenderer() {
  let entries = [];

  function render() {
    const output = getElement("decision-log-output");
    const filter = getElement("decision-log-filter");
    const limitInput = getElement("decision-log-limit");
    if (!output || !filter || !limitInput) return;

    const parsedLimit = Number.parseInt(limitInput.value, 10);
    const limit =
      Number.isFinite(parsedLimit) && parsedLimit > 0 ? parsedLimit : 250;
    const event = filter.value || "all";
    const filteredEntries = entries.filter(
      (entry) => event === "all" || entry?.event === event,
    );
    output.textContent = JSON.stringify(
      filteredEntries.slice(0, limit),
      null,
      2,
    );
  }

  return {
    setEntries(nextEntries) {
      entries = Array.isArray(nextEntries) ? nextEntries : [];
      render();
    },
    render,
    getEntries() {
      return entries;
    },
  };
}
