window.addEventListener("DOMContentLoaded", () => {
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
        strategy: "strong_trend_stair",
        start: start ? new Date(start).toISOString() : null,
        end: end ? new Date(end).toISOString() : null,
        initial_capital: num(data.get("initial_capital"), 10000),
        override_download: false,
        params: {
          symbol: text(data.get("symbol"), "BTCUSDT"),
          timeframe: text(data.get("timeframe"), "1m"),
          leverage: num(data.get("leverage"), 100),
          position_size_pct: num(data.get("position_size_pct"), 2),
          starting_balance_usd: num(data.get("starting_balance_usd"), 10000),
          hard_stop_loss_pct: num(data.get("hard_stop_loss_pct"), 5),
          trail_start_pct: num(data.get("trail_start_pct"), 2),
          trail_offset_pct: num(data.get("trail_offset_pct"), 1),
          ema_fast_len: int(data.get("ema_fast_len"), 50),
          ema_mid_len: int(data.get("ema_mid_len"), 100),
          ema_slow_len: int(data.get("ema_slow_len"), 200),
          slope_lookback: int(data.get("slope_lookback"), 10),
          st_atr_len: int(data.get("st_atr_len"), 10),
          st_factor: num(data.get("st_factor"), 3),
          di_len: int(data.get("di_len"), 14),
          adx_smooth: int(data.get("adx_smooth"), 14),
          adx_min: num(data.get("adx_min"), 20),
          risk_free_rate: num(data.get("risk_free_rate"), 0),
        },
      };
    },
  });
});
