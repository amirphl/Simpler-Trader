window.addEventListener("DOMContentLoaded", () => {
  initBacktestPage({
    serializeForm(form) {
      const data = new FormData(form);
      const start = data.get("start");
      const end = data.get("end");
      return {
        strategy: "pinbar_magic_v2",
        start: start ? new Date(start).toISOString() : null,
        end: end ? new Date(end).toISOString() : null,
        initial_capital: parseFloat(data.get("initial_capital")),
        override_download: false,
        params: {
          symbol: data.get("symbol"),
          timeframe: data.get("timeframe"),
          leverage: parseFloat(data.get("leverage")),
          equity_risk_pct: parseFloat(data.get("equity_risk_pct")),
          atr_multiple: parseFloat(data.get("atr_multiple")),
          trail_points: parseFloat(data.get("trail_points")),
          trail_offset: parseFloat(data.get("trail_offset")),
          slow_sma_period: parseInt(data.get("slow_sma_period"), 10),
          medium_ema_period: parseInt(data.get("medium_ema_period"), 10),
          fast_ema_period: parseInt(data.get("fast_ema_period"), 10),
          atr_period: parseInt(data.get("atr_period"), 10),
          entry_cancel_bars: parseInt(data.get("entry_cancel_bars"), 10),
        },
      };
    },
  });
});

