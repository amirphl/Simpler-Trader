window.addEventListener("DOMContentLoaded", () => {
  initBacktestPage({
    serializeForm(form) {
      const data = new FormData(form);
      const start = data.get("start");
      const end = data.get("end");
      const stopLossPct = data.get("stop_loss_pct");
      const tp = data.get("take_profit_pct");
      return {
        strategy: "pinbar",
        start: start ? new Date(start).toISOString() : null,
        end: end ? new Date(end).toISOString() : null,
        initial_capital: parseFloat(data.get("initial_capital")),
        override_download: false,
        params: {
          symbol: data.get("symbol"),
          timeframe: data.get("timeframe"),
          leverage: parseFloat(data.get("leverage")),
          take_profit_pct: tp ? parseFloat(tp) : 0.02,
          stop_loss_mode: data.get("stop_loss_mode"),
          stop_loss_pct: stopLossPct ? parseFloat(stopLossPct) : 0.005,
          exchange_fee_pct: parseFloat(data.get("exchange_fee_pct") || "0.0004"),
          min_shadow_body_ratio: parseFloat(data.get("min_shadow_body_ratio") || "0.5"),
          shadow_dominance_ratio: parseFloat(data.get("shadow_dominance_ratio") || "2.0"),
        },
      };
    },
  });
});

