(() => {
  const form = document.getElementById("liquidity-zone-form");
  const statusEl = document.getElementById("status");
  const zoneList = document.getElementById("zone-list");
  const segmentList = document.getElementById("segment-list");
  const chartContainer = document.getElementById("chart");
  const segmentCountEl = document.getElementById("segment-count");
  const zoneCountEl = document.getElementById("zone-count");
  const pivotCountEl = document.getElementById("pivot-count");
  const showZoneLabelsInput = document.getElementById("show-zone-labels");
  const showStructureLabelsInput = document.getElementById("show-structure-labels");
  const showPivotMarkersInput = document.getElementById("show-pivot-markers");

  const css = getComputedStyle(document.documentElement);
  const COLORS = {
    up: css.getPropertyValue("--up").trim() || "#31d17c",
    down: css.getPropertyValue("--down").trim() || "#ff6b57",
    level2Up: css.getPropertyValue("--level-2-up").trim() || "#4fb3ff",
    level2Down: css.getPropertyValue("--level-2-down").trim() || "#f5b51f",
    segmentUp: css.getPropertyValue("--segment-up").trim() || "rgba(49, 209, 124, 0.08)",
    segmentDown: css.getPropertyValue("--segment-down").trim() || "rgba(255, 107, 87, 0.08)",
  };

  let chart;
  let candleSeries;
  let overlay;
  let overlayCtx;
  let resizeObserver;
  let lastCandles = [];
  let lastPivots = [];
  let lastSegments = [];
  let lastZones = [];
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
    const start = new Date(end.getTime() - 14 * 24 * 3600 * 1000);
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
    overlay.style.zIndex = "6";
    chartContainer.appendChild(overlay);
    overlayCtx = overlay.getContext("2d");
  }

  function resizeOverlay() {
    if (!overlay) return;
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
      layout: { background: { color: "#09131a" }, textColor: "#e7f6ef" },
      grid: {
        vertLines: { color: "rgba(231,246,239,0.04)" },
        horzLines: { color: "rgba(231,246,239,0.04)" },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    candleSeries = chart.addCandlestickSeries({
      upColor: "#31d17c",
      downColor: "#ff6b57",
      borderVisible: false,
      wickUpColor: "#31d17c",
      wickDownColor: "#ff6b57",
    });
    candleSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.1, bottom: 0.1 },
    });
    ensureOverlay();
    resizeOverlay();
    chart.timeScale().subscribeVisibleLogicalRangeChange(drawOverlay);
    resizeObserver = new ResizeObserver(() => {
      chart.applyOptions({
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
      });
      resizeOverlay();
      drawOverlay();
    });
    resizeObserver.observe(chartContainer);
  }

  function unixTime(value) {
    if (typeof value === "number") return value;
    return Math.floor(new Date(value).getTime() / 1000);
  }

  function timeToX(time) {
    return chart.timeScale().timeToCoordinate(unixTime(time));
  }

  function priceToY(price) {
    return candleSeries.priceToCoordinate(price);
  }

  function zoneStyle(zone) {
    if (zone.level === 1 && zone.direction === "UPWARD") {
      return { stroke: COLORS.up, fill: "rgba(49, 209, 124, 0.14)", dash: [], badge: "L1 UP" };
    }
    if (zone.level === 1) {
      return { stroke: COLORS.down, fill: "rgba(255, 107, 87, 0.14)", dash: [], badge: "L1 DOWN" };
    }
    if (zone.direction === "UPWARD") {
      return { stroke: COLORS.level2Up, fill: "rgba(79, 179, 255, 0.12)", dash: [8, 5], badge: "L2 UP" };
    }
    return { stroke: COLORS.level2Down, fill: "rgba(245, 181, 31, 0.14)", dash: [8, 5], badge: "L2 DOWN" };
  }

  function pivotColor(pivot) {
    return pivot.type === "bullish" ? COLORS.up : COLORS.down;
  }

  function structureLineColor(marker) {
    if (marker.type === "BOS") {
      return marker.direction === "UPWARD" ? COLORS.up : COLORS.down;
    }
    return marker.direction === "UPWARD" ? COLORS.level2Up : COLORS.level2Down;
  }

  function structureIconColor(marker) {
    return structureLineColor(marker);
  }

  function drawSegmentBands(width) {
    const topBandHeight = 16;
    lastSegments.forEach((segment) => {
      const x0 = timeToX(segment.start_time);
      const x1 = timeToX(segment.end_time);
      if (x0 == null || x1 == null) return;
      overlayCtx.fillStyle = segment.direction === "UPWARD" ? COLORS.segmentUp : COLORS.segmentDown;
      overlayCtx.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0) || 2, topBandHeight);
    });
    overlayCtx.strokeStyle = "rgba(231,246,239,0.08)";
    overlayCtx.beginPath();
    overlayCtx.moveTo(0, topBandHeight + 0.5);
    overlayCtx.lineTo(width, topBandHeight + 0.5);
    overlayCtx.stroke();
  }

  function drawZones() {
    lastZones.forEach((zone) => {
      const x0 = timeToX(zone.start_time);
      const x1 = timeToX(zone.end_time);
      const y0 = priceToY(zone.price_high);
      const y1 = priceToY(zone.price_low);
      if (x0 == null || x1 == null || y0 == null || y1 == null) return;

      const left = Math.min(x0, x1);
      const top = Math.min(y0, y1);
      const width = Math.max(Math.abs(x1 - x0), 2);
      const height = Math.max(Math.abs(y1 - y0), 2);
      const style = zoneStyle(zone);

      overlayCtx.fillStyle = style.fill;
      overlayCtx.fillRect(left, top, width, height);
      overlayCtx.strokeStyle = style.stroke;
      overlayCtx.lineWidth = zone.level === 2 ? 2 : 1.5;
      overlayCtx.setLineDash(style.dash);
      overlayCtx.strokeRect(left + 0.5, top + 0.5, Math.max(width - 1, 1), Math.max(height - 1, 1));
      overlayCtx.setLineDash([]);

      if (showZoneLabelsInput?.checked) {
        overlayCtx.font = "500 11px 'IBM Plex Mono', monospace";
        const label = `${style.badge} · ${zone.id}${zone.is_hunted ? " *" : ""}`;
        const textWidth = Math.ceil(overlayCtx.measureText(label).width);
        const labelX = left + 6;
        const labelY = Math.max(top - 8, 28);
        overlayCtx.fillStyle = "rgba(9, 19, 26, 0.9)";
        overlayCtx.fillRect(labelX - 4, labelY - 11, textWidth + 8, 14);
        overlayCtx.strokeStyle = style.stroke;
        overlayCtx.strokeRect(labelX - 3.5, labelY - 10.5, textWidth + 7, 13);
        overlayCtx.fillStyle = style.stroke;
        overlayCtx.fillText(label, labelX, labelY);
      }
    });
  }

  function drawPivots() {
    if (!showPivotMarkersInput?.checked) return;
    lastPivots.forEach((pivot) => {
      const x = timeToX(pivot.time);
      const price = pivot.type === "bullish" ? pivot.low : pivot.high;
      const y = priceToY(price);
      if (x == null || y == null) return;

      overlayCtx.fillStyle = pivotColor(pivot);
      overlayCtx.beginPath();
      if (pivot.type === "bullish") {
        overlayCtx.moveTo(x, y + 6);
        overlayCtx.lineTo(x - 5, y - 4);
        overlayCtx.lineTo(x + 5, y - 4);
      } else {
        overlayCtx.moveTo(x, y - 6);
        overlayCtx.lineTo(x - 5, y + 4);
        overlayCtx.lineTo(x + 5, y + 4);
      }
      overlayCtx.closePath();
      overlayCtx.fill();

      if (pivot.haunted) {
        overlayCtx.strokeStyle = "#ffffff";
        overlayCtx.lineWidth = 1;
        overlayCtx.beginPath();
        overlayCtx.moveTo(x - 4, y - 4);
        overlayCtx.lineTo(x + 4, y + 4);
        overlayCtx.moveTo(x + 4, y - 4);
        overlayCtx.lineTo(x - 4, y + 4);
        overlayCtx.stroke();
      }
    });
  }

  function drawRepresentatives() {
    lastSegments.forEach((segment) => {
      if (!segment.representative_pivot_time) return;
      const x = timeToX(segment.representative_pivot_time);
      const candle = lastCandles[segment.representative_pivot_index];
      if (x == null || !candle) return;
      const y = priceToY(candle.close);
      if (y == null) return;

      overlayCtx.fillStyle = segment.direction === "UPWARD" ? COLORS.level2Up : COLORS.level2Down;
      overlayCtx.beginPath();
      overlayCtx.moveTo(x, y - 7);
      overlayCtx.lineTo(x + 7, y);
      overlayCtx.lineTo(x, y + 7);
      overlayCtx.lineTo(x - 7, y);
      overlayCtx.closePath();
      overlayCtx.fill();
    });
  }

  function structureMarkerY(marker, slot) {
    const candle = lastCandles[marker.candle_index];
    if (!candle) return null;
    const top = candleSeries.priceToCoordinate(candle.high);
    const bottom = candleSeries.priceToCoordinate(candle.low);
    if (top == null || bottom == null) return null;
    const offset = 16 + slot * 14;
    return marker.direction === "UPWARD" ? top - offset : bottom + offset;
  }

  function structureMarkerSlot(marker, slots) {
    const key = `${marker.candle_index}:${marker.direction}`;
    const next = slots.get(key) || 0;
    slots.set(key, next + 1);
    return next;
  }

  function drawStructureMarkers() {
    const slots = new Map();
    lastMarkers.forEach((marker) => {
      const x = timeToX(marker.time);
      const y = structureMarkerY(marker, structureMarkerSlot(marker, slots));
      const priceY = priceToY(marker.price);
      if (x == null || y == null) return;

      const lineColor = structureLineColor(marker);
      const iconColor = structureIconColor(marker);
      const lineHalf = marker.type === "BOS" ? 30 : 24;
      const label = showStructureLabelsInput?.checked
        ? `${marker.type} ${marker.index} · ${marker.direction === "UPWARD" ? "UP" : "DOWN"}`
        : "";

      overlayCtx.save();
      overlayCtx.strokeStyle = lineColor;
      overlayCtx.fillStyle = iconColor;
      overlayCtx.lineWidth = marker.type === "BOS" ? 2.2 : 1.9;
      overlayCtx.setLineDash(marker.type === "BOS" ? [] : [6, 4]);
      if (priceY != null) {
        overlayCtx.beginPath();
        overlayCtx.moveTo(Math.round(x), Math.round(Math.min(y, priceY)));
        overlayCtx.lineTo(Math.round(x), Math.round(Math.max(y, priceY)));
        overlayCtx.stroke();
      }
      overlayCtx.beginPath();
      overlayCtx.moveTo(Math.round(x - lineHalf), Math.round(y));
      overlayCtx.lineTo(Math.round(x + lineHalf), Math.round(y));
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);

      if (marker.type === "BOS") {
        overlayCtx.fillRect(Math.round(x - 4.5), Math.round(y - 4.5), 9, 9);
        overlayCtx.strokeStyle = "#eafff4";
        overlayCtx.lineWidth = 1;
        overlayCtx.strokeRect(Math.round(x - 4.5) + 0.5, Math.round(y - 4.5) + 0.5, 8, 8);
      } else {
        overlayCtx.beginPath();
        overlayCtx.moveTo(Math.round(x), Math.round(y - 7));
        overlayCtx.lineTo(Math.round(x + 7), Math.round(y));
        overlayCtx.lineTo(Math.round(x), Math.round(y + 7));
        overlayCtx.lineTo(Math.round(x - 7), Math.round(y));
        overlayCtx.closePath();
        overlayCtx.fill();
        overlayCtx.strokeStyle = "#f8fafc";
        overlayCtx.lineWidth = 1;
        overlayCtx.stroke();
      }

      if (label) {
        overlayCtx.font = "600 10px 'IBM Plex Mono', monospace";
        const textWidth = Math.ceil(overlayCtx.measureText(label).width);
        const boxWidth = textWidth + 12;
        const boxHeight = 17;
        const boxX = Math.round(x - boxWidth / 2);
        const boxY = Math.round(y - boxHeight - 11);
        overlayCtx.fillStyle = "rgba(8, 18, 25, 0.94)";
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

  function drawOverlay() {
    if (!overlayCtx || !chart || !candleSeries) return;
    const width = chartContainer.clientWidth;
    const height = chartContainer.clientHeight;
    overlayCtx.clearRect(0, 0, width, height);
    drawSegmentBands(width);
    drawZones();
    drawPivots();
    drawRepresentatives();
    drawStructureMarkers();
  }

  function render(candles, pivots, segments, zones, markers) {
    ensureChart();

    lastCandles = candles;
    lastPivots = pivots;
    lastSegments = segments;
    lastZones = zones;
    lastMarkers = markers.map((marker) => ({
      ...marker,
      time: unixTime(marker.time),
    }));

    candleSeries.setData(
      candles.map((c) => ({
        time: unixTime(c.close_time),
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
    );

    zoneList.innerHTML = "";
    zones.forEach((zone, idx) => {
      const pair = `${zone.left_pivot_type[0].toUpperCase()}${zone.right_pivot_type[0].toUpperCase()}`;
      const option = document.createElement("option");
      option.value = idx;
      option.textContent = `${zone.id} · L${zone.level} · ${zone.direction} · ${pair} · ${zone.is_hunted ? "hunted" : "active"} · ${zone.price_low.toFixed(4)}-${zone.price_high.toFixed(4)}`;
      option.style.color = zoneStyle(zone).stroke;
      zoneList.appendChild(option);
    });

    segmentList.innerHTML = "";
    segments.forEach((segment, idx) => {
      const option = document.createElement("option");
      option.value = idx;
      option.textContent = `SEG_${segment.index} · ${segment.direction} · pivots ${segment.pivot_count} · rep ${segment.representative_pivot_index ?? "-"}`;
      option.style.color = segment.direction === "UPWARD" ? COLORS.up : COLORS.down;
      segmentList.appendChild(option);
    });

    segmentCountEl.textContent = String(segments.length);
    zoneCountEl.textContent = String(zones.length);
    pivotCountEl.textContent = String(pivots.length);

    chart.timeScale().fitContent();
    drawOverlay();
  }

  function focusTimeWindow(centerTime, spanSeconds) {
    if (!chart) return;
    chart.timeScale().setVisibleRange({
      from: centerTime - spanSeconds,
      to: centerTime + spanSeconds,
    });
    drawOverlay();
  }

  zoneList?.addEventListener("change", () => {
    const zone = lastZones[Number(zoneList.value)];
    if (!zone) return;
    const center = Math.floor((unixTime(zone.start_time) + unixTime(zone.end_time)) / 2);
    const width = Math.max(unixTime(zone.end_time) - unixTime(zone.start_time), 6 * 3600);
    focusTimeWindow(center, Math.ceil(width * 0.8));
  });

  segmentList?.addEventListener("change", () => {
    const segment = lastSegments[Number(segmentList.value)];
    if (!segment) return;
    const center = Math.floor((unixTime(segment.start_time) + unixTime(segment.end_time)) / 2);
    const width = Math.max(unixTime(segment.end_time) - unixTime(segment.start_time), 8 * 3600);
    focusTimeWindow(center, Math.ceil(width * 0.65));
  });

  async function handleSubmit(event) {
    event.preventDefault();
    statusEl.textContent = "Requesting candles, pivots, segments, and zones...";

    const formData = new FormData(form);
    const maxPivotDistanceRaw = formData.get("maximum_pivot_distance");
    const payload = {
      symbol: formData.get("symbol"),
      timeframe: formData.get("timeframe"),
      start: new Date(formData.get("start")).toISOString(),
      end: new Date(formData.get("end")).toISOString(),
      scan_length: Number(formData.get("scan_length") || 500),
      direction_window: Number(formData.get("direction_window") || 3),
      hunt_mode: formData.get("hunt_mode"),
      up_pivot_filter: formData.get("up_pivot_filter"),
      down_pivot_filter: formData.get("down_pivot_filter"),
      include_hunted_pivots: formData.has("include_hunted_pivots"),
      representative_include_hunted: formData.has("representative_include_hunted"),
      maximum_pivot_distance: maxPivotDistanceRaw ? Number(maxPivotDistanceRaw) : null,
      minimum_overlap: Number(formData.get("minimum_overlap") || 0),
      minimum_overlap_ratio: Number(formData.get("minimum_overlap_ratio") || 0),
      allow_reuse: formData.has("allow_reuse"),
      relaxed_slope: formData.has("relaxed_slope"),
      slope_epsilon: Number(formData.get("slope_epsilon") || 0),
      epsilon: Number(formData.get("epsilon") || 0.000000001),
      include_bos_in_choch_range: formData.has("include_bos_in_choch_range"),
      include_hunt_candle_in_choch_range: formData.has("include_hunt_candle_in_choch_range"),
      source: formData.get("source"),
      csv_path: optionalText(formData.get("csv_path")),
      http_proxy: optionalText(formData.get("http_proxy")),
      https_proxy: optionalText(formData.get("https_proxy")),
    };

    try {
      const structurePayload = {
        symbol: payload.symbol,
        timeframe: payload.timeframe,
        start: payload.start,
        end: payload.end,
        direction_window: payload.direction_window,
        hunt_mode: payload.hunt_mode,
        include_bos_in_choch_range: payload.include_bos_in_choch_range,
        include_hunt_candle_in_choch_range: payload.include_hunt_candle_in_choch_range,
        source: payload.source,
        csv_path: payload.csv_path,
        http_proxy: payload.http_proxy,
        https_proxy: payload.https_proxy,
      };

      const [liquidityRes, structureRes] = await Promise.all([
        fetch("/api/liquidity-zones", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
        fetch("/api/bos-choch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(structurePayload),
        }),
      ]);

      if (!liquidityRes.ok) {
        const error = await liquidityRes.json();
        statusEl.textContent = error.detail || "Liquidity request failed";
        return;
      }

      const liquidityJson = await liquidityRes.json();
      let structureJson = { markers: [] };
      let structureWarning = "";
      if (structureRes.ok) {
        structureJson = await structureRes.json();
      } else {
        const error = await structureRes.json().catch(() => ({}));
        structureWarning = error.detail ? ` Structure overlay unavailable: ${error.detail}.` : " Structure overlay unavailable.";
      }

      render(
        liquidityJson.candles,
        liquidityJson.pivots,
        liquidityJson.segments,
        liquidityJson.zones,
        structureJson.markers || []
      );
      statusEl.textContent = `Loaded ${liquidityJson.candles.length} candles, ${liquidityJson.pivots.length} pivots, ${liquidityJson.segments.length} segments, ${liquidityJson.zones.length} liquidity zones, and ${(structureJson.markers || []).length} BOS/CHoCH markers.${structureWarning}`;
    } catch (err) {
      console.error(err);
      statusEl.textContent = "Unexpected error; check console.";
    }
  }

  form?.elements?.source?.addEventListener("change", syncSourceFields);
  showZoneLabelsInput?.addEventListener("change", drawOverlay);
  showStructureLabelsInput?.addEventListener("change", drawOverlay);
  showPivotMarkersInput?.addEventListener("change", drawOverlay);
  syncSourceFields();
  form?.addEventListener("submit", handleSubmit);
})();
