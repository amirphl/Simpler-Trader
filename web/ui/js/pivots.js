(() => {
  const form = document.getElementById("pivot-form");
  const statusEl = document.getElementById("status");
  const list = document.getElementById("pivot-list");
  const chartContainer = document.getElementById("chart");

  function isoInput(dt) {
    const pad = (n) => String(n).padStart(2, "0");
    const yyyy = dt.getUTCFullYear();
    const mm = pad(dt.getUTCMonth() + 1);
    const dd = pad(dt.getUTCDate());
    const hh = pad(dt.getUTCHours());
    const mi = pad(dt.getUTCMinutes());
    return `${yyyy}-${mm}-${dd}T${hh}:${mi}`;
  }

  // seed defaults: last 7 days
  (() => {
    const end = new Date();
    const start = new Date(end.getTime() - 7 * 24 * 3600 * 1000);
    document.getElementById("start").value = isoInput(start);
    document.getElementById("end").value = isoInput(end);
  })();

  let chart;
  let candleSeries;
  let lastMarkers = [];

  const css = getComputedStyle(document.documentElement);
  const BEAR_ODD = css.getPropertyValue("--bear-odd").trim() || "#fef08a";
  const BEAR_EVEN = css.getPropertyValue("--bear-even").trim() || "#f59e0b";
  const BULL_ODD = css.getPropertyValue("--bull-odd").trim() || "#93c5fd";
  const BULL_EVEN = css.getPropertyValue("--bull-even").trim() || "#86efac";

  function ensureChart() {
    if (chart) return;
    chart = LightweightCharts.createChart(chartContainer, {
      layout: { background: { color: "#0b1220" }, textColor: "#e5e7eb" },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    candleSeries = chart.addCandlestickSeries({
      upColor: "#34d399",
      downColor: "#f87171",
      borderVisible: false,
      wickUpColor: "#34d399",
      wickDownColor: "#f87171",
    });
  }

  function render(candles, pivots) {
    ensureChart();
    const data = candles.map((c) => ({
      time: Math.floor(new Date(c.close_time).getTime() / 1000),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleSeries.setData(data);

    lastMarkers = [];
    const color = (p, idx) => {
      const isOdd = idx % 2 === 1;
      if (p.type === "bullish") return isOdd ? BULL_ODD : BULL_EVEN;
      return isOdd ? BEAR_ODD : BEAR_EVEN;
    };
    const position = (p) => (p.type === "bullish" ? "belowBar" : "aboveBar");

    const pushLabel = (time, label, pivot, pivotIndex, extra = {}) => {
      if (!time) return;
      lastMarkers.push({
        time,
        position: position(pivot),
        color: color(pivot, pivotIndex),
        text: label,
        shape: "circle",
        size: 1,
        textColor: "#0b1220",
        ...extra,
      });
    };

    pivots.forEach((p, idx) => {
      const pivotTime = Math.floor(new Date(p.time).getTime() / 1000);
      const refTime = Math.floor(new Date(p.reference_time).getTime() / 1000);
      const trgTime = Math.floor(new Date(p.trigger_time).getTime() / 1000);
      const invTime = p.invalidation_time ? Math.floor(new Date(p.invalidation_time).getTime() / 1000) : null;
      const nextBullTime = p.next_bullish_time ? Math.floor(new Date(p.next_bullish_time).getTime() / 1000) : null;
      const nextBearTime = p.next_bearish_time ? Math.floor(new Date(p.next_bearish_time).getTime() / 1000) : null;
      const prevBullTime = p.previous_bullish_time ? Math.floor(new Date(p.previous_bullish_time).getTime() / 1000) : null;
      const prevBearTime = p.previous_bearish_time ? Math.floor(new Date(p.previous_bearish_time).getTime() / 1000) : null;

      // Base labels
      pushLabel(refTime, "Ref", p, idx);
      pushLabel(pivotTime, "Pivot", p, idx, { shape: "square" });
      pushLabel(pivotTime, "High", p, idx, { shape: "square", size: p.haunted ? 1 : 2 });

      // Context labels
      pushLabel(nextBullTime || nextBearTime, "NBull", p, idx);
      pushLabel(prevBearTime || prevBullTime, "PBear", p, idx);
      pushLabel(invTime, "InvHigh", p, idx);
      pushLabel(trgTime, "Target", p, idx);
    });
    candleSeries.setMarkers(lastMarkers);

    list.innerHTML = "";
    pivots.forEach((p, idx) => {
      const option = document.createElement("option");
      const date = new Date(p.time).toISOString().replace("T", " ").slice(0, 16);
      const qText = p.invalidation_time ? ` · Q ${new Date(p.invalidation_time).toISOString().slice(11,16)}` : "";
      option.value = idx;
      option.textContent = `${p.type.toUpperCase()} R/P/Q @ ${date}${qText} · ${p.high.toFixed(4)}/${p.low.toFixed(4)}${p.haunted ? " · haunted" : ""}`;
      option.style.color = color(p, idx);
      list.appendChild(option);
    });

    if (lastMarkers.length) {
      chart.timeScale().setVisibleRange({
        from: lastMarkers[0].time,
        to: lastMarkers[lastMarkers.length - 1].time,
      });
    }
  }

  list?.addEventListener("change", () => {
    if (!chart || !candleSeries) return;
    const idx = Number(list.value);
    if (Number.isNaN(idx)) return;
    // jump to pivot P marker for selected pivot (3 markers per pivot, 4 with Q)
    const baseIndex = idx * 4; // R,P,T,(Q)
    const markerTime = lastMarkers[baseIndex + 1]?.time; // pivot marker
    if (markerTime != null) {
      chart.timeScale().setVisibleRange({
        from: markerTime - 50 * 60,
        to: markerTime + 50 * 60,
      });
    }
  });

  async function handleSubmit(event) {
    event.preventDefault();
    statusEl.textContent = "Requesting candles & pivots...";
    const formData = new FormData(form);
    const payload = {
      symbol: formData.get("symbol"),
      timeframe: formData.get("timeframe"),
      start: new Date(formData.get("start")).toISOString(),
      end: new Date(formData.get("end")).toISOString(),
      scan_length: Number(formData.get("scan_length") || 500),
    };
    try {
      const res = await fetch("/api/pivots", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const error = await res.json();
        statusEl.textContent = error.detail || "Request failed";
        return;
      }
      const json = await res.json();
      render(json.candles, json.pivots);
      statusEl.textContent = `Found ${json.pivots.length} pivots across ${json.candles.length} candles.`;
    } catch (err) {
      console.error(err);
      statusEl.textContent = "Unexpected error; check console.";
    }
  }

  form?.addEventListener("submit", handleSubmit);
})();
