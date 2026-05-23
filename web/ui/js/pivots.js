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

  // Seed a fixed UTC start date while keeping the end at the current time.
  (() => {
    const end = new Date();
    const start = new Date(Date.UTC(2026, 0, 1, 0, 0, 0));
    document.getElementById("start").value = isoInput(start);
    document.getElementById("end").value = isoInput(end);
  })();

  let chart;
  let candleSeries;
  let markerOverlay;
  let lastMarkers = [];
  let lastOverlayMarkers = [];
  let pivotMarkerTimes = [];
  let candleByTime = new Map();
  let candleSpacing = 3600;

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
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    candleSeries = chart.addCandlestickSeries({
      upColor: "#34d399",
      downColor: "#f87171",
      borderVisible: false,
      wickUpColor: "#34d399",
      wickDownColor: "#f87171",
    });
    markerOverlay = document.createElement("div");
    markerOverlay.className = "pivot-marker-overlay";
    chartContainer.appendChild(markerOverlay);
    chart.timeScale().subscribeVisibleTimeRangeChange(drawMarkerOverlay);
    window.addEventListener("resize", drawMarkerOverlay);
  }

  function drawMarkerOverlay() {
    if (!chart || !candleSeries || !markerOverlay) return;
    markerOverlay.innerHTML = "";

    lastOverlayMarkers.forEach((marker) => {
      const candle = candleByTime.get(marker.time);
      if (!candle) return;

      const x = chart.timeScale().timeToCoordinate(marker.time);
      const price = marker.position === "belowBar" ? candle.low : candle.high;
      const y = candleSeries.priceToCoordinate(price);
      if (x == null || y == null) return;

      const label = document.createElement("div");
      label.className = `pivot-marker-label ${marker.position === "belowBar" ? "below" : "above"}`;
      label.textContent = marker.text;
      label.style.color = marker.color;
      label.style.left = `${x}px`;
      label.style.top =
        marker.position === "belowBar" ? `${y + 18}px` : `${y - 18}px`;
      markerOverlay.appendChild(label);
    });
  }

  function render(candles, pivots) {
    ensureChart();
    const data = candles.map((c) => ({
      time: Math.floor(new Date(c.open_time).getTime() / 1000),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleByTime = new Map(data.map((c) => [c.time, c]));
    if (data.length >= 2) candleSpacing = data[1].time - data[0].time;
    candleSeries.setData(data);

    const rawMarkers = [];
    pivotMarkerTimes = [];
    const color = (p, idx) => {
      const isOdd = idx % 2 === 1;
      if (p.type === "bullish") return isOdd ? BULL_ODD : BULL_EVEN;
      return isOdd ? BEAR_ODD : BEAR_EVEN;
    };
    const position = (p) => (p.type === "bullish" ? "belowBar" : "aboveBar");

    const pushLabel = (time, label, pivot, pivotIndex, extra = {}) => {
      if (!time) return;
      rawMarkers.push({
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

      pushLabel(refTime, "Ref", p, idx);
      pushLabel(pivotTime, "Pivot", p, idx, { shape: "square" });
      pushLabel(trgTime, "Target", p, idx);
      pivotMarkerTimes[idx] = pivotTime;
    });

    const groupedMarkers = new Map();
    rawMarkers.forEach((marker) => {
      const key = `${marker.time}:${marker.position}`;
      const group = groupedMarkers.get(key);
      if (group) {
        group.text = `${group.text}\n${marker.text}`;
        group.size = Math.max(group.size || 1, marker.size || 1);
        if (marker.shape === "square") group.shape = "square";
      } else {
        groupedMarkers.set(key, { ...marker });
      }
    });
    lastOverlayMarkers = Array.from(groupedMarkers.values()).sort(
      (a, b) => a.time - b.time,
    );
    lastMarkers = lastOverlayMarkers.map((marker) => ({
      ...marker,
      text: "",
    }));
    candleSeries.setMarkers(lastMarkers);
    drawMarkerOverlay();

    list.innerHTML = "";
    pivots.forEach((p, idx) => {
      const option = document.createElement("option");
      const date = new Date(p.time)
        .toISOString()
        .replace("T", " ")
        .slice(0, 16);
      const targetText = ` · Target ${new Date(p.trigger_time).toISOString().slice(11, 16)}`;
      option.value = idx;
      const invText =
        p.invalidation_level != null
          ? ` · Inv:${p.invalidation_level.toFixed(4)}`
          : "";
      option.textContent = `${p.type.toUpperCase()} R/P/T @ ${date}${targetText} · H:${p.high.toFixed(4)} L:${p.low.toFixed(4)}${invText}${p.haunted ? " · haunted" : ""}`;
      option.style.color = color(p, idx);
      list.appendChild(option);
    });

    if (lastMarkers.length) {
      chart.timeScale().setVisibleRange({
        from: lastMarkers[0].time,
        to: lastMarkers[lastMarkers.length - 1].time,
      });
      drawMarkerOverlay();
    }
  }

  list?.addEventListener("change", () => {
    if (!chart || !candleSeries) return;
    const idx = Number(list.value);
    if (Number.isNaN(idx)) return;
    const markerTime = pivotMarkerTimes[idx];
    if (markerTime != null) {
      chart.timeScale().setVisibleRange({
        from: markerTime - 50 * candleSpacing,
        to: markerTime + 50 * candleSpacing,
      });
      drawMarkerOverlay();
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
      restart_on_invalidation: formData.get("restart_on_invalidation") === "on",
      min_swing_pct: Number(formData.get("min_swing_pct") || 0),
      use_structural_left_bound:
        formData.get("use_structural_left_bound") === "on",
      include_reference_candle:
        formData.get("include_reference_candle") === "on",
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
