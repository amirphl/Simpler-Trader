window.addEventListener("DOMContentLoaded", () => {
  initBacktestPage({
    serializeForm(form) {
      const data = new FormData(form);
      const start = data.get("start");
      const end = data.get("end");
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
        strategy: "pinbar_magic_v2",
        start: start ? new Date(start).toISOString() : null,
        end: end ? new Date(end).toISOString() : null,
        initial_capital: num(data.get("initial_capital"), 10000),
        override_download: false,
        params: {
          symbol: data.get("symbol"),
          timeframe: data.get("timeframe"),
          leverage: num(data.get("leverage"), 1),
          equity_risk_pct: num(data.get("equity_risk_pct"), 3),
          atr_multiple: num(data.get("atr_multiple"), 0.5),
          trail_points: num(data.get("trail_points"), 1),
          trail_offset: num(data.get("trail_offset"), 1),
          slow_sma_period: int(data.get("slow_sma_period"), 50),
          medium_ema_period: int(data.get("medium_ema_period"), 18),
          fast_ema_period: int(data.get("fast_ema_period"), 6),
          atr_period: int(data.get("atr_period"), 14),
          entry_cancel_bars: int(data.get("entry_cancel_bars"), 3),
        },
      };
    },
  });
});
