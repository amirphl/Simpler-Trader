window.addEventListener("DOMContentLoaded", () => {
  const STORAGE_KEY = "ema_avwap_pullback_form_v1";
  const PRESETS = {
    baseline: {
      symbol: "ETHUSDT",
      timeframe: "1h",
      leverage: "1",
      equity_risk_pct: "1",
      position_sizing_mode: "risk_distance",
      ema_length: "55",
      consecutive_count: "4",
      ema_validation_mode: "body",
      setup_waiting_replacement_mode: "keep_waiting",
      avwap_multiplier_1: "1",
      avwap_multiplier_2: "2",
      avwap_multiplier_3: "3",
      rigid_stop_loss_pct: "0",
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
    },
    trend_hunter: {
      ema_length: "89",
      consecutive_count: "5",
      ema_validation_mode: "wick",
      setup_waiting_replacement_mode: "keep_waiting",
      position_sizing_mode: "risk_distance",
      rigid_stop_loss_pct: "0.8",
      trailing_activation_threshold_pct: "0.2",
      trailing_gap_pct: "0.8",
      equity_risk_pct: "0.75",
      avwap_multiplier_2: "2.2",
    },
    faster_retests: {
      timeframe: "15m",
      ema_length: "34",
      consecutive_count: "3",
      ema_validation_mode: "body",
      setup_waiting_replacement_mode: "replace_waiting",
      position_sizing_mode: "risk_amount_per_price",
      rigid_stop_loss_pct: "0.5",
      trailing_activation_threshold_pct: "0",
      trailing_gap_pct: "1.4",
      entry_slippage_pct: "0.0001",
      exit_slippage_pct: "0.0001",
      warmup_days: "14",
    },
    defensive_costs: {
      equity_risk_pct: "0.5",
      setup_waiting_replacement_mode: "keep_waiting",
      position_sizing_mode: "risk_distance",
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
        params: {
          symbol: text(data.get("symbol"), "ETHUSDT"),
          timeframe: text(data.get("timeframe"), "1h"),
          leverage: num(data.get("leverage"), 1),
          equity_risk_pct: num(data.get("equity_risk_pct"), 1),
          position_sizing_mode: text(
            data.get("position_sizing_mode"),
            "risk_distance",
          ),
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
          rigid_stop_loss_pct: num(data.get("rigid_stop_loss_pct"), 0),
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
