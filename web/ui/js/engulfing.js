window.addEventListener("DOMContentLoaded", () => {
  initBacktestPage({
    serializeForm(form) {
      const data = new FormData(form);
      const start = data.get("start");
      const end = data.get("end");
      const stopLossPct = data.get("stop_loss_pct");
      const tp = data.get("take_profit_pct");
      return {
        strategy: "engulfing",
        start: start ? new Date(start).toISOString() : null,
        end: end ? new Date(end).toISOString() : null,
        initial_capital: parseFloat(data.get("initial_capital")),
        override_download: false,
        params: {
          symbol: data.get("symbol"),
          timeframe: data.get("timeframe"),
          window_size: parseInt(data.get("window_size"), 10),
          leverage: parseFloat(data.get("leverage")),
          take_profit_pct: tp ? parseFloat(tp) : 0.02,
          doji_size: parseFloat(data.get("doji_size") || "0.05"),
          stop_loss_mode: data.get("stop_loss_mode"),
          stop_loss_pct: stopLossPct ? parseFloat(stopLossPct) : 0.005,
          exchange_fee_pct: parseFloat(data.get("exchange_fee_pct") || "0.0004"),
          skip_large_upper_wick: data.get("skip_large_upper_wick") === "true",
          skip_bollinger_cross: data.get("skip_bollinger_cross") === "true",
          bollinger_period: parseInt(data.get("bollinger_period") || "20", 10),
          bollinger_stddev: parseFloat(data.get("bollinger_stddev") || "2.0"),
          enable_volume_pressure_filter: data.get("enable_volume_pressure_filter") === "true",
          volume_window: parseInt(data.get("volume_window") || "20", 10),
          max_volume_pressure_score: parseFloat(data.get("max_volume_pressure_score") || "3.0"),
          enable_stochastic_filter: data.get("enable_stochastic_filter") === "true",
          stochastic_first_line: data.get("stochastic_first_line") || "k",
          stochastic_first_period: parseInt(data.get("stochastic_first_period") || "20", 10),
          stochastic_first_threshold: null,
          stochastic_second_line: data.get("stochastic_second_line") || "k",
          stochastic_second_period: parseInt(data.get("stochastic_second_period") || "100", 10),
          stochastic_second_threshold: null,
          stochastic_comparison: data.get("stochastic_comparison") || "gt",
          stochastic_d_smoothing: parseInt(data.get("stochastic_d_smoothing") || "3", 10),
        },
      };
    },
  });
});

