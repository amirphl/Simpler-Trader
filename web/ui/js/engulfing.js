window.addEventListener("DOMContentLoaded", () => {
  initBacktestPage({
    serializeForm(form) {
      const data = new FormData(form);
      const start = data.get("start");
      const end = data.get("end");
      const stopLossPct = data.get("stop_loss_pct");
      const tp = data.get("take_profit_pct");
      const num = (value, fallback) => {
        const parsed = parseFloat(value);
        return Number.isFinite(parsed) ? parsed : fallback;
      };
      const int = (value, fallback) => {
        const parsed = parseInt(value, 10);
        return Number.isFinite(parsed) ? parsed : fallback;
      };
      if (!start || !end) {
        return null;
      }
      return {
        strategy: "engulfing",
        start: start ? new Date(start).toISOString() : null,
        end: end ? new Date(end).toISOString() : null,
        initial_capital: num(data.get("initial_capital"), 10000),
        override_download: false,
        params: {
          symbol: data.get("symbol"),
          timeframe: data.get("timeframe"),
          window_size: int(data.get("window_size"), 5),
          leverage: num(data.get("leverage"), 1),
          take_profit_pct: tp ? num(tp, 0.02) : 0.02,
          doji_size: num(data.get("doji_size") || "0.05", 0.05),
          stop_loss_mode: data.get("stop_loss_mode"),
          stop_loss_pct: stopLossPct ? num(stopLossPct, 0.005) : 0.005,
          exchange_fee_pct: num(data.get("exchange_fee_pct") || "0.0004", 0.0004),
          skip_large_upper_wick: data.get("skip_large_upper_wick") === "true",
          skip_bollinger_cross: data.get("skip_bollinger_cross") === "true",
          bollinger_period: int(data.get("bollinger_period") || "20", 20),
          bollinger_stddev: num(data.get("bollinger_stddev") || "2.0", 2.0),
          enable_volume_pressure_filter: data.get("enable_volume_pressure_filter") === "true",
          volume_window: int(data.get("volume_window") || "20", 20),
          max_volume_pressure_score: num(data.get("max_volume_pressure_score") || "3.0", 3.0),
          enable_stochastic_filter: data.get("enable_stochastic_filter") === "true",
          stochastic_first_line: data.get("stochastic_first_line") || "k",
          stochastic_first_period: int(data.get("stochastic_first_period") || "20", 20),
          stochastic_first_threshold: null,
          stochastic_second_line: data.get("stochastic_second_line") || "k",
          stochastic_second_period: int(data.get("stochastic_second_period") || "100", 100),
          stochastic_second_threshold: null,
          stochastic_comparison: data.get("stochastic_comparison") || "gt",
          stochastic_d_smoothing: int(data.get("stochastic_d_smoothing") || "3", 3),
        },
      };
    },
  });
});
