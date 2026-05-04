(() => {
  const form = document.getElementById("bos-choch-form");
  const statusEl = document.getElementById("status");
  const list = document.getElementById("event-list");
  const chartContainer = document.getElementById("chart");
  const showLabelsInput = document.getElementById("show-marker-labels");

  const css = getComputedStyle(document.documentElement);
  const UP_COLOR = css.getPropertyValue("--line-up").trim() || "#22c55e";
  const DOWN_COLOR = css.getPropertyValue("--line-down").trim() || "#ef4444";

  let chart;
  let candleSeries;
  let overlay;
  let overlayCtx;
  let lastMarkers = [];

  function optionalText(value) {
    if (typeof value !== "string") return null;
    const trimmed = value.trim();
    return trimmed ? trimmed : null;
  }

  function isoInput(dt) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${dt.getUTCFullYear()}-${pad(dt.getUTCMonth() + 1)}-${pad(dt.getUTCDate())}T${pad(dt.getUTCHours())}:${pad(dt.getUTCMinutes())}`;
  }

  (() => {
    const end = new Date();
    const start = new Date(end.getTime() - 7 * 24 * 3600 * 1000);
    document.getElementById("start").value = isoInput(start);
    document.getElementById("end").value = isoInput(end);
  })();

  function syncSourceFields() {
    const source = form?.elements?.source?.value || "binance";
    const csvField = document.getElementById("csv-path-field");
    const csvInput = document.getElementById("csv_path");
    if (!csvField || !csvInput) return;
    const usingCsv = source === "csv";
    csvField.hidden = !usingCsv;
    csvInput.required = usingCsv;
  }

  function ensureOverlay() {
    if (overlay) return;
    overlay = document.createElement("canvas");
    overlay.style.position = "absolute";
    overlay.style.inset = "0";
    overlay.style.pointerEvents = "none";
    overlay.style.zIndex = "5";
    chartContainer.appendChild(overlay);
    overlayCtx = overlay.getContext("2d");
  }

  function resizeOverlay() {
    if (!overlay || !chartContainer) return;
    const pixelRatio = window.devicePixelRatio || 1;
    const width = chartContainer.clientWidth;
    const height = chartContainer.clientHeight;
    overlay.width = Math.floor(width * pixelRatio);
    overlay.height = Math.floor(height * pixelRatio);
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;
    overlayCtx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }

  function ensureChart() {
    if (chart) return;
    chart = LightweightCharts.createChart(chartContainer, {
      layout: { background: { color: "#10140f" }, textColor: "#edf2e8" },
      grid: {
        vertLines: { color: "rgba(237,242,232,0.05)" },
        horzLines: { color: "rgba(237,242,232,0.05)" },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });
    candleSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.16, bottom: 0.16 },
    });
    ensureOverlay();
    resizeOverlay();
    chart.timeScale().subscribeVisibleLogicalRangeChange(drawStructureMarkers);
    new ResizeObserver(() => {
      chart.applyOptions({
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
      });
      resizeOverlay();
      drawStructureMarkers();
    }).observe(chartContainer);
  }

  function markerColor(marker) {
    return marker.direction === "UPWARD" ? UP_COLOR : DOWN_COLOR;
  }

  function markerLineColor(marker) {
    if (marker.type === "BOS") {
      return markerColor(marker);
    }
    return marker.direction === "UPWARD" ? "#60a5fa" : "#f59e0b";
  }

  function markerLabelText(marker) {
    if (!showLabelsInput?.checked) return "";
    const side = marker.direction === "UPWARD" ? "UP" : "DOWN";
    return `${marker.type} ${marker.index} · ${side}`;
  }

  function markerY(marker, slot) {
    const top = candleSeries.priceToCoordinate(marker.high);
    const bottom = candleSeries.priceToCoordinate(marker.low);
    if (top == null || bottom == null) return null;
    const offset = 16 + slot * 14;
    return marker.direction === "UPWARD" ? top - offset : bottom + offset;
  }

  function markerSlot(marker, slots) {
    const key = `${marker.candle_index}:${marker.direction}`;
    const next = slots.get(key) || 0;
    slots.set(key, next + 1);
    return next;
  }

  function drawStructureMarkers() {
    if (!overlayCtx || !chart || !candleSeries) return;
    const width = chartContainer.clientWidth;
    const height = chartContainer.clientHeight;
    overlayCtx.clearRect(0, 0, width, height);

    const slots = new Map();
    const visible = lastMarkers
      .map((marker) => {
        const x = chart.timeScale().timeToCoordinate(marker.time);
        if (x == null) return null;
        const y = markerY(marker, markerSlot(marker, slots));
        const priceY = candleSeries.priceToCoordinate(marker.price);
        if (y == null) return null;
        return { marker, x, y, priceY };
      })
      .filter(Boolean);

    visible.forEach(({ marker, x, y, priceY }) => {
      const stemTop = Math.round(Math.min(y, priceY ?? y));
      const stemBottom = Math.round(Math.max(y, priceY ?? y));
      const label = markerLabelText(marker);
      const lineColor = markerLineColor(marker);
      const iconColor = markerColor(marker);
      const lineHalf = marker.type === "BOS" ? 34 : 28;

      overlayCtx.save();
      overlayCtx.strokeStyle = lineColor;
      overlayCtx.fillStyle = iconColor;
      overlayCtx.lineWidth = marker.type === "BOS" ? 2.4 : 2;
      overlayCtx.setLineDash(marker.type === "BOS" ? [] : [6, 4]);

      if (priceY != null) {
        overlayCtx.beginPath();
        overlayCtx.moveTo(Math.round(x), stemTop);
        overlayCtx.lineTo(Math.round(x), stemBottom);
        overlayCtx.stroke();
      }

      overlayCtx.beginPath();
      overlayCtx.moveTo(Math.round(x - lineHalf), Math.round(y));
      overlayCtx.lineTo(Math.round(x + lineHalf), Math.round(y));
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);

      if (marker.type === "BOS") {
        overlayCtx.fillRect(Math.round(x - 5), Math.round(y - 5), 10, 10);
        overlayCtx.strokeStyle = "#ecfdf5";
        overlayCtx.lineWidth = 1.2;
        overlayCtx.strokeRect(Math.round(x - 5) + 0.5, Math.round(y - 5) + 0.5, 9, 9);
      } else {
        overlayCtx.beginPath();
        overlayCtx.moveTo(Math.round(x), Math.round(y - 8));
        overlayCtx.lineTo(Math.round(x + 8), Math.round(y));
        overlayCtx.lineTo(Math.round(x), Math.round(y + 8));
        overlayCtx.lineTo(Math.round(x - 8), Math.round(y));
        overlayCtx.closePath();
        overlayCtx.fill();
        overlayCtx.strokeStyle = "#f8fafc";
        overlayCtx.lineWidth = 1.2;
        overlayCtx.stroke();
      }

      if (label) {
        overlayCtx.font = "700 11px ui-monospace, SFMono-Regular, monospace";
        const textWidth = Math.ceil(overlayCtx.measureText(label).width);
        const boxWidth = textWidth + 12;
        const boxHeight = 18;
        const boxX = Math.round(x - boxWidth / 2);
        const boxY = Math.round(y - boxHeight - 12);
        overlayCtx.fillStyle = "rgba(16, 20, 15, 0.94)";
        overlayCtx.fillRect(boxX, boxY, boxWidth, boxHeight);
        overlayCtx.strokeStyle = lineColor;
        overlayCtx.lineWidth = 1;
        overlayCtx.strokeRect(boxX + 0.5, boxY + 0.5, boxWidth - 1, boxHeight - 1);
        overlayCtx.fillStyle = lineColor;
        overlayCtx.textAlign = "center";
        overlayCtx.textBaseline = "middle";
        overlayCtx.fillText(label, Math.round(x), Math.round(boxY + boxHeight / 2 + 0.5));
      }
      overlayCtx.restore();
    });
  }

  function unixTime(value) {
    if (typeof value === "number") return value;
    return Math.floor(new Date(value).getTime() / 1000);
  }

  function render(candles, markers, directionState) {
    ensureChart();
    const data = candles.map((c) => ({
      time: unixTime(c.close_time),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleSeries.setData(data);

    lastMarkers = markers.map((marker) => ({
      ...marker,
      time: unixTime(marker.time),
    }));
    drawStructureMarkers();

    list.innerHTML = "";
    lastMarkers.forEach((marker, idx) => {
      const option = document.createElement("option");
      const date = new Date(marker.time * 1000).toISOString().replace("T", " ").slice(0, 16);
      option.value = idx;
      option.textContent = `${marker.label} · ${marker.direction} · candle ${marker.candle_index} · ${date}`;
      option.style.color = markerColor(marker);
      list.appendChild(option);
    });

    if (lastMarkers.length) {
      chart.timeScale().setVisibleRange({
        from: Math.max(data[0]?.time || lastMarkers[0].time, lastMarkers[0].time - 60 * 60 * 12),
        to: lastMarkers[lastMarkers.length - 1].time + 60 * 60 * 12,
      });
    } else {
      chart.timeScale().fitContent();
    }
    drawStructureMarkers();
    statusEl.textContent = `Found ${lastMarkers.length} events across ${candles.length} candles. Current direction: ${directionState.direction} since candle ${directionState.since_index}.`;
  }

  list?.addEventListener("change", () => {
    if (!chart) return;
    const marker = lastMarkers[Number(list.value)];
    if (!marker) return;
    chart.timeScale().setVisibleRange({
      from: marker.time - 60 * 60 * 12,
      to: marker.time + 60 * 60 * 12,
    });
    drawStructureMarkers();
  });

  async function handleSubmit(event) {
    event.preventDefault();
    statusEl.textContent = "Requesting candles & structure events...";
    const formData = new FormData(form);
    const payload = {
      symbol: formData.get("symbol"),
      timeframe: formData.get("timeframe"),
      start: new Date(formData.get("start")).toISOString(),
      end: new Date(formData.get("end")).toISOString(),
      direction_window: Number(formData.get("direction_window") || 3),
      hunt_mode: formData.get("hunt_mode"),
      include_bos_in_choch_range: formData.has("include_bos_in_choch_range"),
      include_hunt_candle_in_choch_range: formData.has("include_hunt_candle_in_choch_range"),
      source: formData.get("source"),
      csv_path: optionalText(formData.get("csv_path")),
      http_proxy: optionalText(formData.get("http_proxy")),
      https_proxy: optionalText(formData.get("https_proxy")),
    };

    try {
      const res = await fetch("/api/bos-choch", {
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
      render(json.candles, json.markers, json.direction_state);
    } catch (err) {
      console.error(err);
      statusEl.textContent = "Unexpected error; check console.";
    }
  }

  form?.elements?.source?.addEventListener("change", syncSourceFields);
  showLabelsInput?.addEventListener("change", drawStructureMarkers);
  syncSourceFields();
  form?.addEventListener("submit", handleSubmit);
})();
