window.addEventListener("DOMContentLoaded", () => {
  const STORAGE_KEY = "ema_avwap_pullback_form_v1";
  const PRESETS = {
    baseline: {
      symbol: "ETHUSDT",
      timeframe: "1h",
      leverage: "10",
      position_notional_pct: "1",
      minimum_balance_usdt: "0",
      max_position_size_pct: "10",
      max_setup_age_bars: "3",
      max_entry_deviation_pct: "1",
      position_sizing_mode: "risk_amount_per_price",
      entry_mode: "close",
      exit_mode: "close",
      exit_band: "band_1",
      max_entry_notional_usdt: "15",
      ema_length: "55",
      consecutive_count: "4",
      ema_validation_mode: "body",
      setup_waiting_replacement_mode: "keep_waiting",
      avwap_multiplier_1: "1",
      avwap_multiplier_2: "2",
      avwap_multiplier_3: "3",
      rigid_stop_loss_pct: "3",
      trailing_activation_threshold_pct: "0",
      trailing_gap_pct: "1",
      maker_fee_pct: "0.0002",
      taker_fee_pct: "0.0006",
      entry_slippage_pct: "0",
      exit_slippage_pct: "0",
      use_gap_cross_detection: "true",
      max_decision_log_entries: "20000",
      risk_free_rate: "0",
      initial_capital: "10000",
      warmup_days: "30",
      override_download: "false",
      analysis_include_monte_carlo: "false",
      monte_carlo_iterations: "1000",
      monte_carlo_seed: "",
      monte_carlo_block_size: "5",
      monte_carlo_drawdown_threshold_pct: "30",
      monte_carlo_missed_fill_probability: "0",
      monte_carlo_extra_spread_min_pct: "0",
      monte_carlo_extra_spread_max_pct: "0.001",
      analysis_include_out_of_sample: "false",
      oos_training_fraction: "0.6",
      oos_validation_fraction: "0.2",
      analysis_include_walk_forward: "false",
      walk_forward_train_days: "90",
      walk_forward_test_days: "30",
      walk_forward_step_days: "",
      walk_forward_anchored: "false",
      analysis_include_parameter_perturbation: "false",
      parameter_perturbation_samples: "25",
      parameter_perturbation_seed: "",
    },
    trend_hunter: {
      ema_length: "89",
      consecutive_count: "5",
      ema_validation_mode: "wick",
      setup_waiting_replacement_mode: "keep_waiting",
      rigid_stop_loss_pct: "0.8",
      trailing_activation_threshold_pct: "0.2",
      trailing_gap_pct: "0.8",
      position_notional_pct: "0.75",
      avwap_multiplier_2: "2.2",
    },
    faster_retests: {
      timeframe: "15m",
      ema_length: "34",
      consecutive_count: "3",
      ema_validation_mode: "body",
      setup_waiting_replacement_mode: "replace_waiting",
      rigid_stop_loss_pct: "0.5",
      trailing_activation_threshold_pct: "0",
      trailing_gap_pct: "1.4",
      entry_slippage_pct: "0.0001",
      exit_slippage_pct: "0.0001",
      warmup_days: "14",
    },
    defensive_costs: {
      position_notional_pct: "0.5",
      setup_waiting_replacement_mode: "keep_waiting",
      rigid_stop_loss_pct: "0.4",
      maker_fee_pct: "0.0004",
      taker_fee_pct: "0.0008",
      entry_slippage_pct: "0.0002",
      exit_slippage_pct: "0.0002",
      use_gap_cross_detection: "false",
      trailing_activation_threshold_pct: "0.4",
    },
  };

  let latestDecisionLog = [];

  function isoLocalInputValue(date) {
    const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
    return local.toISOString().slice(0, 16);
  }

  function setDefaultDates(form) {
    const startInput = form.elements.namedItem("start");
    const endInput = form.elements.namedItem("end");
    if (
      !(startInput instanceof HTMLInputElement) ||
      !(endInput instanceof HTMLInputElement)
    ) {
      return;
    }
    if (startInput.value && endInput.value) {
      return;
    }
    const end = new Date();
    const start = new Date(end.getTime() - 90 * 24 * 60 * 60 * 1000);
    endInput.value = isoLocalInputValue(end);
    startInput.value = isoLocalInputValue(start);
  }

  function applyValues(form, values) {
    Object.entries(values || {}).forEach(([key, value]) => {
      const field = form.elements.namedItem(key);
      if (!field || typeof value !== "string") return;
      if ("value" in field) {
        field.value = value;
      }
    });
  }

  function captureValues(form) {
    const snapshot = {};
    Array.from(form.elements).forEach((element) => {
      if (!element.name) return;
      if (
        element instanceof HTMLInputElement ||
        element instanceof HTMLSelectElement
      ) {
        snapshot[element.name] = element.value;
      }
    });
    return snapshot;
  }

  function saveDraft(form) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(captureValues(form)));
  }

  function loadDraft(form) {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return;
      applyValues(form, parsed);
    } catch (error) {
      console.warn("Failed to restore EMA+AVWAP draft", error);
    }
  }

  function renderQuickMetrics(result) {
    const host = document.getElementById("quick-metrics");
    if (!host) return;
    const stats = result?.report?.statistics || {};
    const performance = stats.performance || {};
    const summary = [
      ["Trades", stats.total_trades],
      [
        "Win Rate",
        typeof stats.win_rate === "number"
          ? `${(stats.win_rate * 100).toFixed(1)}%`
          : "n/a",
      ],
      [
        "Net PnL",
        typeof stats.net_profit === "number"
          ? stats.net_profit.toFixed(2)
          : "n/a",
      ],
      [
        "Profit Factor",
        typeof stats.profit_factor === "number"
          ? stats.profit_factor.toFixed(2)
          : "n/a",
      ],
      [
        "Max DD",
        typeof stats.max_drawdown_pct === "number"
          ? `${stats.max_drawdown_pct.toFixed(2)}%`
          : "n/a",
      ],
      [
        "CAGR",
        typeof stats.cagr_pct === "number"
          ? `${stats.cagr_pct.toFixed(2)}%`
          : "n/a",
      ],
      [
        "Sortino",
        typeof performance.sortino_ratio === "number"
          ? performance.sortino_ratio.toFixed(2)
          : "n/a",
      ],
      [
        "Calmar",
        typeof performance.calmar_ratio === "number"
          ? performance.calmar_ratio.toFixed(2)
          : "n/a",
      ],
    ];
    host.innerHTML = "";
    summary.forEach(([label, value]) => {
      const tile = document.createElement("div");
      tile.className = "metric-tile";
      const labelEl = document.createElement("span");
      labelEl.className = "label";
      labelEl.textContent = label;
      const valueEl = document.createElement("span");
      valueEl.className = "value";
      valueEl.textContent = value;
      tile.appendChild(labelEl);
      tile.appendChild(valueEl);
      host.appendChild(tile);
    });
  }

  function renderConfigSnapshot(result) {
    const host = document.getElementById("config-snapshot-output");
    if (!host) return;
    const report = result?.report || {};
    const payload = {
      strategy: result?.strategy || report.strategy || null,
      run_config: report.config || {},
      strategy_config: report.statistics?.config || {},
      execution_assumptions: report.statistics?.execution_assumptions || {},
    };
    host.textContent = JSON.stringify(payload, null, 2);
  }

  function renderDecisionLog() {
    const output = document.getElementById("decision-log-output");
    const filterSelect = document.getElementById("decision-log-filter");
    const limitInput = document.getElementById("decision-log-limit");
    if (!output || !filterSelect || !limitInput) return;

    const selectedEvent = filterSelect.value || "all";
    const rawLimit = parseInt(limitInput.value, 10);
    const limit = Number.isFinite(rawLimit) && rawLimit > 0 ? rawLimit : 250;

    const filtered = latestDecisionLog.filter((entry) => {
      if (selectedEvent === "all") return true;
      return entry?.event === selectedEvent;
    });
    output.textContent = JSON.stringify(filtered.slice(0, limit), null, 2);
  }

  function formatFloat(value, digits = 4) {
    return typeof value === "number" && Number.isFinite(value)
      ? value.toFixed(digits)
      : "n/a";
  }

  function formatPct(value, digits = 2) {
    return typeof value === "number" && Number.isFinite(value)
      ? `${value.toFixed(digits)}%`
      : "n/a";
  }

  function clearAndRenderTiles(host, items) {
    if (!host) return;
    host.innerHTML = "";
    items.forEach(([label, value]) => {
      const tile = document.createElement("div");
      tile.className = "metric-tile";
      const labelEl = document.createElement("span");
      labelEl.className = "label";
      labelEl.textContent = label;
      const valueEl = document.createElement("span");
      valueEl.className = "value";
      valueEl.textContent = value;
      tile.appendChild(labelEl);
      tile.appendChild(valueEl);
      host.appendChild(tile);
    });
  }

  function renderDetailedPerformance(result) {
    const host = document.getElementById("performance-metrics");
    const output = document.getElementById("performance-details-output");
    const stats = result?.report?.statistics || {};
    const performance = stats.performance || {};

    clearAndRenderTiles(host, [
      ["Trades", performance.number_of_trades ?? stats.total_trades ?? "n/a"],
      ["Expectancy", formatFloat(performance.expectancy_per_trade, 2)],
      ["Avg Win", formatFloat(performance.average_win, 2)],
      ["Avg Loss", formatFloat(performance.average_loss, 2)],
      ["Profit Factor", formatFloat(performance.profit_factor, 2)],
      ["Sharpe", formatFloat(performance.sharpe_ratio, 2)],
      ["Sortino", formatFloat(performance.sortino_ratio, 2)],
      ["Calmar", formatFloat(performance.calmar_ratio, 2)],
      ["Max DD", formatPct(performance.maximum_drawdown_pct)],
      ["Avg DD", formatPct(performance.average_drawdown_pct)],
      ["Time Under Water", formatPct(performance.time_under_water_pct)],
      ["Losing Streak", performance.longest_losing_streak ?? "n/a"],
      ["Skew", formatFloat(performance.return_skewness, 3)],
      ["Kurtosis", formatFloat(performance.return_excess_kurtosis, 3)],
      ["Exposure Adj.", formatPct(performance.exposure_adjusted_return_pct)],
      ["Turnover", formatFloat(performance.turnover, 2)],
      ["Fee Drag", formatFloat(performance.fee_drag, 2)],
      ["Slippage Drag", formatFloat(performance.slippage_drag, 2)],
      ["Deflated Sharpe", formatFloat(performance.deflated_sharpe_ratio, 3)],
    ]);

    if (output) {
      output.textContent = Object.keys(performance).length
        ? JSON.stringify(performance, null, 2)
        : "No detailed statistics yet.";
    }
  }

  function renderRobustAnalysis(result) {
    const host = document.getElementById("robust-analysis-metrics");
    const output = document.getElementById("robust-analysis-output");
    const robust = result?.robust_analysis || {};
    const monteCarlo = robust.monte_carlo || {};
    const reshuffle = monteCarlo.trade_reshuffling || {};
    const block = monteCarlo.block_bootstrap || {};
    const walkForward = robust.walk_forward || {};
    const oos = robust.out_of_sample || {};
    const perturbation = robust.parameter_perturbation || {};
    const finalStats = oos.final_statistics || {};

    clearAndRenderTiles(host, [
      [
        "MC DD p95",
        typeof block.max_drawdown_pct_p95 === "number"
          ? formatPct(block.max_drawdown_pct_p95)
          : "n/a",
      ],
      [
        "MC DD p99",
        typeof block.max_drawdown_pct_p99 === "number"
          ? formatPct(block.max_drawdown_pct_p99)
          : "n/a",
      ],
      [
        "MC Loss Prob",
        typeof reshuffle.probability_of_loss === "number"
          ? formatPct(reshuffle.probability_of_loss * 100)
          : "n/a",
      ],
      [
        "MC Threshold Prob",
        typeof reshuffle.probability_of_drawdown_ge_threshold === "number"
          ? formatPct(reshuffle.probability_of_drawdown_ge_threshold * 100)
          : "n/a",
      ],
      [
        "WF Folds",
        Array.isArray(walkForward.folds) ? walkForward.folds.length : "n/a",
      ],
      [
        "WF Net",
        walkForward.combined_statistics
          ? formatPct(walkForward.combined_statistics.net_profit_pct)
          : "n/a",
      ],
      [
        "Final OOS Net",
        typeof finalStats.net_profit_pct === "number"
          ? formatPct(finalStats.net_profit_pct)
          : "n/a",
      ],
      [
        "Perturb p05",
        typeof perturbation.score_p05 === "number"
          ? formatPct(perturbation.score_p05)
          : "n/a",
      ],
      [
        "Perturb p50",
        typeof perturbation.score_p50 === "number"
          ? formatPct(perturbation.score_p50)
          : "n/a",
      ],
      [
        "Perturb p95",
        typeof perturbation.score_p95 === "number"
          ? formatPct(perturbation.score_p95)
          : "n/a",
      ],
    ]);

    if (output) {
      output.textContent = Object.keys(robust).length
        ? JSON.stringify(robust, null, 2)
        : "No robust analysis requested yet.";
    }
  }

  initBacktestPage({
    serializeForm(form) {
      const data = new FormData(form);
      const start = data.get("start");
      const end = data.get("end");
      const text = (value, fallback) => {
        if (typeof value !== "string") return fallback;
        const trimmed = value.trim();
        return trimmed ? trimmed : fallback;
      };
      const normalizeNumeric = (value) => {
        if (typeof value !== "string") {
          return value;
        }
        const trimmed = value.trim();
        if (!trimmed) {
          return trimmed;
        }
        const compact = trimmed.replace(/\s+/g, "");
        if (compact.includes(",") && compact.includes(".")) {
          return compact.replace(/,/g, "");
        }
        if (compact.includes(",") && !compact.includes(".")) {
          const commaCount = (compact.match(/,/g) || []).length;
          if (commaCount === 1) {
            const [left, right] = compact.split(",");
            if ((right || "").length <= 2) {
              return `${left}.${right}`;
            }
          }
          return compact.replace(/,/g, "");
        }
        return compact;
      };
      const num = (value, fallback) => {
        const parsed = parseFloat(normalizeNumeric(value));
        return Number.isFinite(parsed) ? parsed : fallback;
      };
      const int = (value, fallback) => {
        const parsed = parseInt(normalizeNumeric(value), 10);
        return Number.isFinite(parsed) ? parsed : fallback;
      };
      const optionalInt = (value) => {
        const parsed = parseInt(normalizeNumeric(value), 10);
        return Number.isFinite(parsed) ? parsed : null;
      };
      const optionalNum = (value) => {
        const parsed = parseFloat(normalizeNumeric(value));
        return Number.isFinite(parsed) ? parsed : null;
      };
      const bool = (value, fallback) => {
        if (value === "true") return true;
        if (value === "false") return false;
        return fallback;
      };

      if (!start || !end) {
        return null;
      }

      saveDraft(form);

      return {
        strategy: "ema_avwap_pullback",
        start: new Date(start).toISOString(),
        end: new Date(end).toISOString(),
        initial_capital: num(data.get("initial_capital"), 10000),
        override_download: bool(data.get("override_download"), false),
        warmup_days: int(data.get("warmup_days"), 30),
        analysis: {
          include_monte_carlo: bool(
            data.get("analysis_include_monte_carlo"),
            false,
          ),
          monte_carlo_iterations: int(data.get("monte_carlo_iterations"), 1000),
          monte_carlo_seed: optionalInt(data.get("monte_carlo_seed")),
          monte_carlo_block_size: int(data.get("monte_carlo_block_size"), 5),
          monte_carlo_drawdown_threshold_pct: num(
            data.get("monte_carlo_drawdown_threshold_pct"),
            30,
          ),
          monte_carlo_missed_fill_probability: num(
            data.get("monte_carlo_missed_fill_probability"),
            0,
          ),
          monte_carlo_extra_spread_min_pct: num(
            data.get("monte_carlo_extra_spread_min_pct"),
            0,
          ),
          monte_carlo_extra_spread_max_pct: num(
            data.get("monte_carlo_extra_spread_max_pct"),
            0.001,
          ),
          include_out_of_sample: bool(
            data.get("analysis_include_out_of_sample"),
            false,
          ),
          oos_training_fraction: num(data.get("oos_training_fraction"), 0.6),
          oos_validation_fraction: num(
            data.get("oos_validation_fraction"),
            0.2,
          ),
          include_walk_forward: bool(
            data.get("analysis_include_walk_forward"),
            false,
          ),
          walk_forward_train_days: int(data.get("walk_forward_train_days"), 90),
          walk_forward_test_days: int(data.get("walk_forward_test_days"), 30),
          walk_forward_step_days: optionalInt(
            data.get("walk_forward_step_days"),
          ),
          walk_forward_anchored: bool(data.get("walk_forward_anchored"), false),
          include_parameter_perturbation: bool(
            data.get("analysis_include_parameter_perturbation"),
            false,
          ),
          parameter_perturbation_samples: int(
            data.get("parameter_perturbation_samples"),
            25,
          ),
          parameter_perturbation_seed: optionalInt(
            data.get("parameter_perturbation_seed"),
          ),
        },
        params: {
          symbol: text(data.get("symbol"), "ETHUSDT"),
          timeframe: text(data.get("timeframe"), "1h"),
          leverage: num(data.get("leverage"), 10),
          max_entry_notional_usdt: num(data.get("max_entry_notional_usdt"), 15),
          max_position_size_pct: num(data.get("max_position_size_pct"), 10),
          position_notional_pct: num(data.get("position_notional_pct"), 1),
          minimum_balance_usdt: num(data.get("minimum_balance_usdt"), 0),
          max_setup_age_bars: int(data.get("max_setup_age_bars"), 3),
          max_entry_deviation_pct: num(data.get("max_entry_deviation_pct"), 1),
          position_sizing_mode: text(
            data.get("position_sizing_mode"),
            "risk_amount_per_price",
          ),
          entry_mode: text(data.get("entry_mode"), "close"),
          exit_mode: text(data.get("exit_mode"), "close"),
          exit_band: text(data.get("exit_band"), "band_1"),
          ema_length: int(data.get("ema_length"), 55),
          consecutive_count: int(data.get("consecutive_count"), 4),
          ema_validation_mode: text(data.get("ema_validation_mode"), "body"),
          setup_waiting_replacement_mode: text(
            data.get("setup_waiting_replacement_mode"),
            "keep_waiting",
          ),
          avwap_multiplier_1: num(data.get("avwap_multiplier_1"), 1),
          avwap_multiplier_2: num(data.get("avwap_multiplier_2"), 2),
          avwap_multiplier_3: num(data.get("avwap_multiplier_3"), 3),
          rigid_stop_loss_pct: num(data.get("rigid_stop_loss_pct"), 3),
          trailing_activation_threshold_pct: num(
            data.get("trailing_activation_threshold_pct"),
            0,
          ),
          trailing_gap_pct: num(data.get("trailing_gap_pct"), 1),
          maker_fee_pct: num(data.get("maker_fee_pct"), 0.0002),
          taker_fee_pct: num(data.get("taker_fee_pct"), 0.0006),
          entry_slippage_pct: num(data.get("entry_slippage_pct"), 0),
          exit_slippage_pct: num(data.get("exit_slippage_pct"), 0),
          use_gap_cross_detection: bool(
            data.get("use_gap_cross_detection"),
            true,
          ),
          max_decision_log_entries: int(
            data.get("max_decision_log_entries"),
            20000,
          ),
          risk_free_rate: num(data.get("risk_free_rate"), 0),
        },
      };
    },
    onReady({ form }) {
      const applyPresetButton = document.getElementById("apply-preset");
      const presetSelect = document.getElementById("preset-select");
      const saveDraftButton = document.getElementById("save-draft");
      const resetButton = document.getElementById("reset-form");
      const lastMonthButton = document.getElementById("fill-last-month");
      const decisionLogFilter = document.getElementById("decision-log-filter");
      const decisionLogLimit = document.getElementById("decision-log-limit");

      applyValues(form, PRESETS.baseline);
      setDefaultDates(form);
      loadDraft(form);

      if (applyPresetButton && presetSelect) {
        applyPresetButton.addEventListener("click", () => {
          const presetKey = presetSelect.value || "baseline";
          applyValues(form, PRESETS[presetKey] || PRESETS.baseline);
          setDefaultDates(form);
          saveDraft(form);
        });
      }

      if (saveDraftButton) {
        saveDraftButton.addEventListener("click", () => {
          saveDraft(form);
          saveDraftButton.textContent = "Draft Saved";
          setTimeout(() => {
            saveDraftButton.textContent = "Save Draft";
          }, 1200);
        });
      }

      if (resetButton) {
        resetButton.addEventListener("click", () => {
          localStorage.removeItem(STORAGE_KEY);
          applyValues(form, PRESETS.baseline);
          setDefaultDates(form);
        });
      }

      if (lastMonthButton) {
        lastMonthButton.addEventListener("click", () => {
          const end = new Date();
          const start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
          const startInput = form.elements.namedItem("start");
          const endInput = form.elements.namedItem("end");
          if (startInput instanceof HTMLInputElement) {
            startInput.value = isoLocalInputValue(start);
          }
          if (endInput instanceof HTMLInputElement) {
            endInput.value = isoLocalInputValue(end);
          }
          saveDraft(form);
        });
      }

      form.addEventListener("input", () => saveDraft(form));
      form.addEventListener("change", () => saveDraft(form));

      if (decisionLogFilter) {
        decisionLogFilter.addEventListener("change", renderDecisionLog);
      }
      if (decisionLogLimit) {
        decisionLogLimit.addEventListener("input", renderDecisionLog);
      }
    },
    onResult(result) {
      const stats = result?.report?.statistics || {};
      latestDecisionLog = Array.isArray(stats.decision_log)
        ? stats.decision_log
        : [];

      renderQuickMetrics(result);
      renderDetailedPerformance(result);
      renderRobustAnalysis(result);
      renderConfigSnapshot(result);
      renderDecisionLog();

      const statusStream = document.getElementById("status-stream");
      if (statusStream && latestDecisionLog.length) {
        const lastEvent = latestDecisionLog[latestDecisionLog.length - 1] || {};
        const eventLabel = lastEvent.event
          ? String(lastEvent.event).replaceAll("_", " ")
          : "n/a";
        statusStream.title = [
          `Last event: ${eventLabel}`,
          `Timestamp: ${lastEvent.timestamp || "n/a"}`,
          `Setup: ${lastEvent.setup_type || "n/a"}`,
          `VWAP: ${formatFloat(lastEvent.vwap_middle_line, 5)}`,
        ].join("\n");
      }
    },
  });
});
