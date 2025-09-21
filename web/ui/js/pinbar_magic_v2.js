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
      const bool = (value, fallback) => {
        if (value === "true") return true;
        if (value === "false") return false;
        return fallback;
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
          symbol: text(data.get("symbol"), "ETHUSDT"),
          timeframe: text(data.get("timeframe"), "1h"),
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
          trailing_tick_timeframe: text(
            data.get("trailing_tick_timeframe"),
            "15m",
          ),
          use_trailing_tick_emulation: bool(
            data.get("use_trailing_tick_emulation"),
            false,
          ),
          use_stop_fill_open_gap: bool(
            data.get("use_stop_fill_open_gap"),
            true,
          ),
          entry_activation_mode: text(
            data.get("entry_activation_mode"),
            "next_bar",
          ),
          enable_friday_close: bool(data.get("enable_friday_close"), true),
          friday_close_hour_utc: int(data.get("friday_close_hour_utc"), 16),
          enable_ema_cross_close: bool(
            data.get("enable_ema_cross_close"),
            true,
          ),
          risk_equity_include_unrealized: bool(
            data.get("risk_equity_include_unrealized"),
            true,
          ),
          risk_equity_mark_source: text(
            data.get("risk_equity_mark_source"),
            "close",
          ),
          risk_free_rate: num(data.get("risk_free_rate"), 0),
        },
      };
    },
  });
});
