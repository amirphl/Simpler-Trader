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
      if (!start || !end) {
        return null;
      }
      return {
        strategy: "pinbar",
        start: start ? new Date(start).toISOString() : null,
        end: end ? new Date(end).toISOString() : null,
        initial_capital: num(data.get("initial_capital"), 10000),
        override_download: false,
        params: {
          symbol: data.get("symbol"),
          timeframe: data.get("timeframe"),
          leverage: num(data.get("leverage"), 1),
          take_profit_pct: tp ? num(tp, 0.02) : 0.02,
          stop_loss_mode: data.get("stop_loss_mode"),
          stop_loss_pct: stopLossPct ? num(stopLossPct, 0.005) : 0.005,
          exchange_fee_pct: num(data.get("exchange_fee_pct") || "0.0004", 0.0004),
          min_shadow_body_ratio: num(data.get("min_shadow_body_ratio") || "0.5", 0.5),
          shadow_dominance_ratio: num(data.get("shadow_dominance_ratio") || "2.0", 2.0),
        },
      };
    },
  });
});
