function parseFloatList(value, fallback) {
  if (!value) return fallback;
  const parts = value.split(",").map((p) => p.trim()).filter(Boolean);
  const parsed = parts.map((p) => Number(p)).filter((n) => Number.isFinite(n));
  return parsed.length ? parsed : fallback;
}

window.addEventListener("DOMContentLoaded", () => {
  initBacktestPage({
    serializeForm(form) {
      const data = new FormData(form);
      const start = data.get("start");
      const end = data.get("end");
      const symbolsRaw = data.get("symbols") || "";
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

      const payload = {
        strategy: "stochastic_fsm",
        start: start ? new Date(start).toISOString() : null,
        end: end ? new Date(end).toISOString() : null,
        initial_capital: num(data.get("initial_capital") || "10000", 10000),
        override_download: false,
        params: {
          symbols: symbolsRaw
            .split(",")
            .map((s) => s.trim().toUpperCase())
            .filter(Boolean),
          base_timeframe: data.get("base_timeframe") || "1h",
          higher_timeframe: data.get("higher_timeframe") || "4h",
          higher_timeframe_2: data.get("higher_timeframe_2") || null,
          k_period: int(data.get("k_period") || "8", 8),
          k_slowing: int(data.get("k_slowing") || "1", 1),
          d_period: int(data.get("d_period") || "1", 1),
          use_d_line: data.get("use_d_line") === "true",
          oversold: num(data.get("oversold") || "20", 20),
          overbought: num(data.get("overbought") || "80", 80),
          initial_order_usdt: num(data.get("initial_order_usdt") || "100", 100),
          initial_leverage: num(data.get("initial_leverage") || "3", 3),
          martingale_multiplier: num(data.get("martingale_multiplier") || "1.1", 1.1),
          martingale_multipliers: parseFloatList(data.get("martingale_multipliers"), [1.5, 2.0, 2.5, 3.0]),
          martingale_leverages: parseFloatList(data.get("martingale_leverages"), [3, 3, 3, 3]),
          max_concurrent_positions: int(data.get("max_concurrent_positions") || "5", 5),
          take_profit_pct: num(data.get("take_profit_pct") || "0.02", 0.02),
          slippage_pct: num(data.get("slippage_pct") || "0.0002", 0.0002),
          maker_fee_pct: num(data.get("maker_fee_pct") || "0.0002", 0.0002),
          taker_fee_pct: num(data.get("taker_fee_pct") || "0.0006", 0.0006),
          funding_rate_per_day_pct: num(data.get("funding_rate_per_day_pct") || "0", 0),
          trailing_activation_pct: num(data.get("trailing_activation_pct") || "1.5", 1.5),
          trailing_gap_pct: num(data.get("trailing_gap_pct") || "1.0", 1.0),
          trailing_check_interval_seconds: num(
            data.get("trailing_check_interval_seconds") || "10",
            10
          ),
          max_position_days: num(data.get("max_position_days") || "30", 30),
          margin_mode: data.get("margin_mode") || "cross",
          aligned_high_stoch_mode: data.get("aligned_high_stoch_mode") || "v3",
          signal_offset: int(data.get("signal_offset") || "0", 0),
          enable_take_profit_check: data.get("enable_take_profit_check") === "true",
          enable_high_exit_cross: data.get("enable_high_exit_cross") === "true",
          use_midsold_filter: data.get("use_midsold_filter") === "true",
          enable_reversal_logic: data.get("enable_reversal_logic") === "true",
          enable_reversal_reentry: data.get("enable_reversal_reentry") === "true",
          trailing_use_first_entry_price: data.get("trailing_use_first_entry_price") === "true",
          trailing_use_close_for_stop_activation:
            data.get("trailing_use_close_for_stop_activation") === "true",
          take_profit_use_first_entry_price:
            data.get("take_profit_use_first_entry_price") === "true",
          enable_grid_martingales: data.get("enable_grid_martingales") === "true",
          grid_martingales_percent: num(data.get("grid_martingales_percent") || "3", 3),
          http_proxy: data.get("http_proxy") || null,
          https_proxy: data.get("https_proxy") || null,
          risk_free_rate: num(data.get("risk_free_rate") || "0", 0),
        },
      };

      if (!payload.params.symbols.length) {
        payload.params.symbols = ["BTCUSDT"];
      }
      return payload;
    },
  });
});
