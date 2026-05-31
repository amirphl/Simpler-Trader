(() => {
  const form = document.getElementById("liquidity-zone-form");
  const statusEl = document.getElementById("status");
  const zoneTbody = document.getElementById("zone-tbody");
  const segmentTbody = document.getElementById("segment-tbody");
  const chartContainer = document.getElementById("chart");
  const segmentCountEl = document.getElementById("segment-count");
  const zoneCountEl = document.getElementById("zone-count");
  const pivotCountEl = document.getElementById("pivot-count");
  const zoneCountBadgeEl = document.getElementById("zone-count-badge");
  const segmentCountBadgeEl = document.getElementById("segment-count-badge");
  const showStructureLinesInput = document.getElementById(
    "show-structure-lines",
  );
  const showL1ZonesInput = document.getElementById("show-l1-zones");
  const showL2ZonesInput = document.getElementById("show-l2-zones");
  const showPivotMarkersInput = document.getElementById("show-pivot-markers");
  const pivotTypeFilterInput = document.getElementById("pivot-type-filter");
  const showRepresentativesInput = document.getElementById(
    "show-representatives",
  );
  const showZoneLabelsInput = document.getElementById("show-zone-labels");
  const showSegmentBandsInput = document.getElementById("show-segment-bands");
  const showCandleLabelsInput = document.getElementById("show-candle-labels");
  const candleLabelThresholdInput = document.getElementById(
    "candle-label-threshold",
  );

  const css = getComputedStyle(document.documentElement);
  const COLORS = {
    up: css.getPropertyValue("--up").trim() || "#31d17c",
    down: css.getPropertyValue("--down").trim() || "#ff6b57",
    level2Up: css.getPropertyValue("--level-2-up").trim() || "#4fb3ff",
    level2Down: css.getPropertyValue("--level-2-down").trim() || "#f5b51f",
    segmentUp:
      css.getPropertyValue("--segment-up").trim() || "rgba(49, 209, 124, 0.12)",
    segmentDown:
      css.getPropertyValue("--segment-down").trim() ||
      "rgba(255, 107, 87, 0.12)",
    text: css.getPropertyValue("--text").trim() || "#e7f6ef",
    panel: css.getPropertyValue("--panel-2").trim() || "rgba(8, 18, 25, 0.96)",
    borderSoft: "rgba(231, 246, 239, 0.08)",
    borderHard: "rgba(231, 246, 239, 0.18)",
    reversal: css.getPropertyValue("--reversal").trim() || "#e879f9",
    reversalBand: "rgba(232, 121, 249, 0.07)",
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
  let lastReversals = [];
  let pivotByIndex = new Map();
  let pivotByCandleIndex = new Map();

  // ── Utilities ──────────────────────────────────────────────────────────────

  function optionalText(value) {
    if (typeof value !== "string") return null;
    const trimmed = value.trim();
    return trimmed ? trimmed : null;
  }

  function parseDateInput(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date.toISOString();
  }

  function isoInput(dt) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${dt.getUTCFullYear()}-${pad(dt.getUTCMonth() + 1)}-${pad(dt.getUTCDate())}T${pad(dt.getUTCHours())}:${pad(dt.getUTCMinutes())}`;
  }

  function unixTime(value) {
    if (typeof value === "number") return value;
    return Math.floor(new Date(value).getTime() / 1000);
  }

  /** Format an ISO/datetime string → "YYYY-MM-DD HH:MM" (UTC) */
  function fmtTime(value) {
    const d = new Date(typeof value === "number" ? value * 1000 : value);
    if (isNaN(d)) return "—";
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  }

  function pivotTypeBadge(type) {
    return type === "bullish" ? "▲" : "▼";
  }

  function timeToX(time) {
    return chart.timeScale().timeToCoordinate(unixTime(time));
  }

  function priceToY(price) {
    return candleSeries.priceToCoordinate(price);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  // ── Init ───────────────────────────────────────────────────────────────────

  function setDefaultDates() {
    const end = new Date();
    const start = new Date(Date.UTC(2026, 0, 1, 0, 0, 0));
    document.getElementById("start").value = isoInput(start);
    document.getElementById("end").value = isoInput(end);
  }

  function syncSourceFields() {
    const source = form?.elements?.source?.value || "binance";
    const csvField = document.getElementById("csv-path-field");
    const csvInput = document.getElementById("csv_path");
    if (!csvField || !csvInput) return;
    const usingCsv = source === "csv";
    csvField.hidden = !usingCsv;
    csvInput.required = usingCsv;
  }

  // ── Chart / overlay ────────────────────────────────────────────────────────

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
      layout: { background: { color: "#09131a" }, textColor: COLORS.text },
      grid: {
        vertLines: { color: "rgba(231, 246, 239, 0.05)" },
        horzLines: { color: "rgba(231, 246, 239, 0.05)" },
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
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderVisible: false,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
    });
    candleSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.12, bottom: 0.12 },
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

  // ── Style helpers ──────────────────────────────────────────────────────────

  function zoneStyle(zone) {
    if (zone.level === 1 && zone.direction === "UPWARD") {
      return {
        stroke: COLORS.up,
        fill: zone.is_hunted
          ? "rgba(49, 209, 124, 0.09)"
          : "rgba(49, 209, 124, 0.16)",
        midline: "rgba(199, 255, 224, 0.68)",
        badge: "L1 UP",
        dash: [],
      };
    }
    if (zone.level === 1) {
      return {
        stroke: COLORS.down,
        fill: zone.is_hunted
          ? "rgba(255, 107, 87, 0.09)"
          : "rgba(255, 107, 87, 0.16)",
        midline: "rgba(255, 226, 220, 0.68)",
        badge: "L1 DOWN",
        dash: [],
      };
    }
    if (zone.direction === "UPWARD") {
      return {
        stroke: COLORS.level2Up,
        fill: zone.is_hunted
          ? "rgba(79, 179, 255, 0.08)"
          : "rgba(79, 179, 255, 0.14)",
        midline: "rgba(214, 238, 255, 0.72)",
        badge: "L2 UP",
        dash: [8, 4],
      };
    }
    return {
      stroke: COLORS.level2Down,
      fill: zone.is_hunted
        ? "rgba(245, 181, 31, 0.08)"
        : "rgba(245, 181, 31, 0.14)",
      midline: "rgba(255, 244, 201, 0.72)",
      badge: "L2 DOWN",
      dash: [8, 4],
    };
  }

  function structureLineColor(marker) {
    if (marker.type === "BOS") {
      return marker.direction === "UPWARD" ? COLORS.up : COLORS.down;
    }
    return marker.direction === "UPWARD" ? COLORS.level2Up : COLORS.level2Down;
  }

  function pivotColor(pivot) {
    return pivot.type === "bullish" ? COLORS.up : COLORS.down;
  }

  function pivotPrice(pivot) {
    return pivot.type === "bullish" ? pivot.low : pivot.high;
  }

  function representativePrice(segment) {
    const pivot = pivotByIndex.get(segment.representative_pivot_index);
    if (pivot) return pivotPrice(pivot);
    const candle = lastCandles[segment.representative_pivot_index];
    return candle ? candle.close : null;
  }

  // ── Canvas drawing helpers ─────────────────────────────────────────────────

  function drawTag(
    text,
    x,
    y,
    borderColor,
    fillColor = "rgba(8, 18, 25, 0.94)",
    align = "left",
  ) {
    overlayCtx.save();
    overlayCtx.font = "600 10px 'IBM Plex Mono', monospace";
    const textWidth = Math.ceil(overlayCtx.measureText(text).width);
    const boxWidth = textWidth + 12;
    const boxHeight = 18;
    const left = align === "right" ? x - boxWidth : x;
    overlayCtx.fillStyle = fillColor;
    overlayCtx.fillRect(left, y, boxWidth, boxHeight);
    overlayCtx.strokeStyle = borderColor;
    overlayCtx.lineWidth = 1;
    overlayCtx.strokeRect(left + 0.5, y + 0.5, boxWidth - 1, boxHeight - 1);
    overlayCtx.fillStyle = borderColor;
    overlayCtx.textAlign = "left";
    overlayCtx.textBaseline = "middle";
    overlayCtx.fillText(text, left + 6, y + boxHeight / 2 + 0.5);
    overlayCtx.restore();
  }

  function drawSegmentBands(width) {
    if (showSegmentBandsInput?.checked === false) return;
    const bandTop = 0;
    const bandHeight = 18;
    lastSegments.forEach((segment) => {
      const x0 = timeToX(segment.start_time);
      const x1 = timeToX(segment.end_time);
      if (x0 == null || x1 == null) return;
      const left = Math.min(x0, x1);
      const spanWidth = Math.max(Math.abs(x1 - x0), 2);
      overlayCtx.fillStyle =
        segment.direction === "UPWARD" ? COLORS.segmentUp : COLORS.segmentDown;
      overlayCtx.fillRect(left, bandTop, spanWidth, bandHeight);
      if (spanWidth > 84) {
        drawTag(
          `SEG_${segment.index}`,
          left + 3,
          2,
          segment.direction === "UPWARD" ? COLORS.up : COLORS.down,
          "rgba(8, 18, 25, 0.88)",
        );
      }
    });
    overlayCtx.strokeStyle = COLORS.borderHard;
    overlayCtx.beginPath();
    overlayCtx.moveTo(0, bandHeight + 0.5);
    overlayCtx.lineTo(width, bandHeight + 0.5);
    overlayCtx.stroke();
  }

  function drawPivotAnchor(x, y, color) {
    overlayCtx.save();
    overlayCtx.fillStyle = color;
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, 3.5, 0, Math.PI * 2);
    overlayCtx.fill();
    overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    overlayCtx.lineWidth = 1;
    overlayCtx.stroke();
    overlayCtx.restore();
  }

  function drawZones() {
    const showL1 = showL1ZonesInput?.checked !== false;
    const showL2 = showL2ZonesInput?.checked !== false;
    lastZones.forEach((zone) => {
      if (zone.level === 1 && !showL1) return;
      if (zone.level === 2 && !showL2) return;
      const x0 = timeToX(zone.start_time);
      const x1 = timeToX(zone.end_time);
      const yTop = priceToY(zone.price_high);
      const yBottom = priceToY(zone.price_low);
      if (x0 == null || x1 == null || yTop == null || yBottom == null) return;

      const style = zoneStyle(zone);
      const left = Math.min(x0, x1);
      const right = Math.max(x0, x1);
      const top = Math.min(yTop, yBottom);
      const bottom = Math.max(yTop, yBottom);
      const width = Math.max(right - left, 2);
      const height = Math.max(bottom - top, 2);
      const midY = (top + bottom) / 2;

      overlayCtx.save();
      overlayCtx.fillStyle = style.fill;
      overlayCtx.fillRect(left, top, width, height);

      overlayCtx.strokeStyle = style.stroke;
      overlayCtx.lineWidth = zone.level === 2 ? 2.4 : 2;
      overlayCtx.setLineDash(style.dash);
      overlayCtx.strokeRect(
        left + 0.5,
        top + 0.5,
        Math.max(width - 1, 1),
        Math.max(height - 1, 1),
      );
      overlayCtx.setLineDash([]);

      overlayCtx.strokeStyle = style.stroke;
      overlayCtx.lineWidth = zone.level === 2 ? 2.2 : 1.8;
      overlayCtx.beginPath();
      overlayCtx.moveTo(left, top);
      overlayCtx.lineTo(right, top);
      overlayCtx.moveTo(left, bottom);
      overlayCtx.lineTo(right, bottom);
      overlayCtx.stroke();

      overlayCtx.strokeStyle = style.midline;
      overlayCtx.lineWidth = 1.2;
      overlayCtx.setLineDash([5, 4]);
      overlayCtx.beginPath();
      overlayCtx.moveTo(left, midY);
      overlayCtx.lineTo(right, midY);
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);

      overlayCtx.strokeStyle = style.stroke;
      overlayCtx.lineWidth = 1;
      overlayCtx.beginPath();
      overlayCtx.moveTo(left, top);
      overlayCtx.lineTo(left, bottom);
      overlayCtx.moveTo(right, top);
      overlayCtx.lineTo(right, bottom);
      overlayCtx.stroke();

      const leftPivot = pivotByIndex.get(zone.left_pivot_index);
      const rightPivot = pivotByIndex.get(zone.right_pivot_index);
      if (leftPivot) {
        const y = priceToY(pivotPrice(leftPivot));
        if (y != null) drawPivotAnchor(left, y, style.stroke);
      }
      if (rightPivot) {
        const y = priceToY(pivotPrice(rightPivot));
        if (y != null) drawPivotAnchor(right, y, style.stroke);
      }

      if (zone.is_hunted) {
        overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.34)";
        overlayCtx.lineWidth = 1;
        overlayCtx.beginPath();
        overlayCtx.moveTo(left + 4, top + 4);
        overlayCtx.lineTo(right - 4, bottom - 4);
        overlayCtx.moveTo(left + 4, bottom - 4);
        overlayCtx.lineTo(right - 4, top + 4);
        overlayCtx.stroke();
      }

      overlayCtx.restore();

      if (showZoneLabelsInput?.checked) {
        const huntedText = zone.is_hunted ? " H" : "";
        const label = `${style.badge} ${zone.id}${huntedText}`;
        const labelY = top + 6;
        if (height > 26 && width > 120) {
          drawTag(
            label,
            left + 6,
            labelY,
            style.stroke,
            "rgba(8, 18, 25, 0.9)",
          );
        } else {
          drawTag(
            label,
            left + 6,
            Math.max(top - 22, 24),
            style.stroke,
            "rgba(8, 18, 25, 0.96)",
          );
        }
      }
    });
  }

  function drawPivots() {
    if (!showPivotMarkersInput?.checked) return;
    const pivotFilter = pivotTypeFilterInput?.value || "all";
    lastPivots.forEach((pivot) => {
      if (pivotFilter !== "all" && pivot.type !== pivotFilter) return;
      const x = timeToX(pivot.time);
      const y = priceToY(pivotPrice(pivot));
      if (x == null || y == null) return;

      overlayCtx.save();
      overlayCtx.fillStyle = "rgba(9, 19, 26, 0.92)";
      overlayCtx.beginPath();
      overlayCtx.arc(x, y, 8, 0, Math.PI * 2);
      overlayCtx.fill();

      overlayCtx.strokeStyle = pivotColor(pivot);
      overlayCtx.lineWidth = 1.6;
      overlayCtx.beginPath();
      overlayCtx.arc(x, y, 7.2, 0, Math.PI * 2);
      overlayCtx.stroke();

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
        overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.92)";
        overlayCtx.lineWidth = 1;
        overlayCtx.beginPath();
        overlayCtx.moveTo(x - 4.5, y - 4.5);
        overlayCtx.lineTo(x + 4.5, y + 4.5);
        overlayCtx.moveTo(x + 4.5, y - 4.5);
        overlayCtx.lineTo(x - 4.5, y + 4.5);
        overlayCtx.stroke();
      }
      overlayCtx.restore();
    });
  }

  function drawRepresentatives() {
    if (showRepresentativesInput && !showRepresentativesInput.checked) return;
    lastSegments.forEach((segment) => {
      if (
        segment.representative_pivot_index == null ||
        !segment.representative_pivot_time
      ) {
        return;
      }
      const x = timeToX(segment.representative_pivot_time);
      const price = representativePrice(segment);
      const y = price == null ? null : priceToY(price);
      if (x == null || y == null) return;

      const color =
        segment.direction === "UPWARD" ? COLORS.level2Up : COLORS.level2Down;
      overlayCtx.save();
      overlayCtx.fillStyle = "rgba(9, 19, 26, 0.94)";
      overlayCtx.beginPath();
      overlayCtx.moveTo(x, y - 10);
      overlayCtx.lineTo(x + 10, y);
      overlayCtx.lineTo(x, y + 10);
      overlayCtx.lineTo(x - 10, y);
      overlayCtx.closePath();
      overlayCtx.fill();

      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 2;
      overlayCtx.beginPath();
      overlayCtx.moveTo(x, y - 9);
      overlayCtx.lineTo(x + 9, y);
      overlayCtx.lineTo(x, y + 9);
      overlayCtx.lineTo(x - 9, y);
      overlayCtx.closePath();
      overlayCtx.stroke();

      overlayCtx.fillStyle = color;
      overlayCtx.font = "700 9px 'IBM Plex Mono', monospace";
      overlayCtx.textAlign = "center";
      overlayCtx.textBaseline = "middle";
      overlayCtx.fillText("R", x, y + 0.5);
      overlayCtx.restore();
    });
  }

  function drawStructureMarkers() {
    if (showStructureLinesInput?.checked === false) return;
    const width = chartContainer.clientWidth;
    lastMarkers.forEach((marker) => {
      const priceY = priceToY(marker.price);
      if (priceY == null) return;

      const rawX0 =
        marker.line_start_time != null
          ? chart.timeScale().timeToCoordinate(marker.line_start_time)
          : null;
      const rawX1 =
        marker.line_end_time != null
          ? chart.timeScale().timeToCoordinate(marker.line_end_time)
          : null;
      if (rawX0 == null && rawX1 == null) return;

      const x0 = Math.round(rawX0 != null ? rawX0 : 0);
      const x1 = Math.round(rawX1 != null ? rawX1 : width);
      const iconX = marker.type === "BOS" ? x1 : x0;
      const lineColor = structureLineColor(marker);

      overlayCtx.save();
      overlayCtx.strokeStyle = lineColor;
      overlayCtx.lineWidth = marker.type === "BOS" ? 2 : 1.7;
      overlayCtx.setLineDash(marker.type === "BOS" ? [] : [6, 4]);
      overlayCtx.beginPath();
      overlayCtx.moveTo(x0, priceY);
      overlayCtx.lineTo(x1, priceY);
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);

      overlayCtx.fillStyle = "rgba(9, 19, 26, 0.94)";
      if (marker.type === "BOS") {
        overlayCtx.fillRect(iconX - 6, priceY - 6, 12, 12);
        overlayCtx.strokeStyle = lineColor;
        overlayCtx.lineWidth = 1.6;
        overlayCtx.strokeRect(iconX - 5.5, priceY - 5.5, 11, 11);
      } else {
        overlayCtx.beginPath();
        overlayCtx.moveTo(iconX, priceY - 8);
        overlayCtx.lineTo(iconX + 8, priceY);
        overlayCtx.lineTo(iconX, priceY + 8);
        overlayCtx.lineTo(iconX - 8, priceY);
        overlayCtx.closePath();
        overlayCtx.fill();
        overlayCtx.strokeStyle = lineColor;
        overlayCtx.lineWidth = 1.6;
        overlayCtx.stroke();
      }
      overlayCtx.restore();
    });
  }

  function drawCandleLabels() {
    if (!showCandleLabelsInput?.checked) return;
    if (!lastCandles.length) return;
    const range = chart.timeScale().getVisibleLogicalRange();
    if (!range) return;
    const visibleCount = Math.abs(range.to - range.from);
    const threshold = parseInt(candleLabelThresholdInput?.value || "150", 10);
    if (visibleCount > (isNaN(threshold) ? 150 : Math.max(10, threshold)))
      return;

    const LABEL_BG = "rgba(8, 18, 25, 0.92)";
    const LABEL_NONE = "rgba(156,163,175,0.6)";
    const W = chartContainer.clientWidth;
    const H = chartContainer.clientHeight;

    overlayCtx.save();
    overlayCtx.beginPath();
    overlayCtx.rect(0, 0, W, H);
    overlayCtx.clip();

    lastCandles.forEach((c, idx) => {
      const x = chart.timeScale().timeToCoordinate(unixTime(c.open_time));
      if (x == null) return;
      const cx = Math.round(x);

      const pivot = pivotByCandleIndex.get(idx);
      const hasPivot = !!pivot;
      const topLine = String(idx);
      const bottomLine = hasPivot ? String(pivot.index) : "N/A";
      const color = hasPivot
        ? pivot.type === "bearish"
          ? "#fbbf24"
          : COLORS.up
        : LABEL_NONE;

      const highY = candleSeries.priceToCoordinate(c.high);
      if (highY == null) return;

      overlayCtx.save();
      overlayCtx.font = "600 9px 'IBM Plex Mono', monospace";
      const tw1 = overlayCtx.measureText(topLine).width;
      const tw2 = overlayCtx.measureText(bottomLine).width;
      const maxW = Math.max(tw1, tw2);
      const padX = 4;
      const lineH = 11;
      const boxW = maxW + padX * 2;
      const boxH = lineH * 2 + 4;
      const boxX = Math.round(cx - boxW / 2);
      const boxY = Math.round(highY) - boxH - 6;

      overlayCtx.fillStyle = LABEL_BG;
      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 0.8;
      overlayCtx.beginPath();
      if (overlayCtx.roundRect) {
        overlayCtx.roundRect(boxX, boxY, boxW, boxH, 3);
      } else {
        overlayCtx.rect(boxX, boxY, boxW, boxH);
      }
      overlayCtx.fill();
      overlayCtx.stroke();

      overlayCtx.fillStyle = color;
      overlayCtx.textAlign = "center";
      overlayCtx.textBaseline = "top";
      overlayCtx.fillText(topLine, cx, boxY + 2);
      overlayCtx.fillText(bottomLine, cx, boxY + lineH + 2);
      overlayCtx.restore();
    });

    overlayCtx.restore();
  }

  function drawReversalLines() {
    if (!lastReversals.length) return;
    const W = chartContainer.clientWidth;
    const H = chartContainer.clientHeight;

    lastReversals.forEach((rev) => {
      const rawX = chart.timeScale().timeToCoordinate(rev.time);
      if (rawX == null) return;
      const x = Math.round(clamp(rawX, 0, W));

      overlayCtx.save();
      overlayCtx.fillStyle = COLORS.reversalBand;
      overlayCtx.fillRect(x - 3, 0, 6, H);
      overlayCtx.strokeStyle = COLORS.reversal;
      overlayCtx.lineWidth = 1.5;
      overlayCtx.setLineDash([4, 6]);
      overlayCtx.globalAlpha = 0.6;
      overlayCtx.beginPath();
      overlayCtx.moveTo(x, 0);
      overlayCtx.lineTo(x, H);
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);
      overlayCtx.globalAlpha = 0.9;
      overlayCtx.fillStyle = COLORS.reversal;
      overlayCtx.font = "700 11px 'IBM Plex Mono', monospace";
      overlayCtx.textAlign = "center";
      overlayCtx.textBaseline = "top";
      // Place arrow below the 18px segment band at the top
      overlayCtx.fillText(rev.direction === "UPWARD" ? "▲" : "▼", x, 22);
      overlayCtx.restore();
    });
  }

  function drawOverlay() {
    if (!overlayCtx || !chart || !candleSeries) return;
    const width = chartContainer.clientWidth;
    const height = chartContainer.clientHeight;
    overlayCtx.clearRect(0, 0, width, height);
    drawSegmentBands(width);
    drawReversalLines();
    drawZones();
    drawStructureMarkers();
    drawPivots();
    drawRepresentatives();
    drawCandleLabels();
  }

  // ── Table rendering ────────────────────────────────────────────────────────

  /**
   * Render the zones into <tbody id="zone-tbody">.
   * Clicking a row scrolls the chart to that zone's time window.
   */
  function renderZoneTable(zones) {
    if (!zoneTbody) return;
    zoneTbody.innerHTML = "";
    zones.forEach((zone, idx) => {
      const style = zoneStyle(zone);
      const dirClass =
        zone.level === 1
          ? zone.direction === "UPWARD"
            ? "td-up"
            : "td-down"
          : zone.direction === "UPWARD"
            ? "td-l2up"
            : "td-l2down";

      const thickPct =
        zone.metadata?.thickness_pct != null
          ? zone.metadata.thickness_pct.toFixed(3) + "%"
          : "—";

      const row = document.createElement("tr");
      row.dataset.idx = String(idx);
      if (zone.is_hunted) row.classList.add("td-hunted");

      appendCell(row, zone.id, { color: style.stroke });
      appendCell(row, zone.direction === "UPWARD" ? "UP" : "DN", {
        className: dirClass,
      });
      appendCell(row, `L${zone.level}`, { className: "td-muted" });
      appendCell(
        row,
        `${pivotTypeBadge(zone.left_pivot_type)}/${pivotTypeBadge(zone.right_pivot_type)}`,
        { className: "td-muted" },
      );
      appendCell(row, zone.price_low.toFixed(4));
      appendCell(row, zone.price_high.toFixed(4));
      appendCell(row, thickPct, { className: "td-muted" });
      appendCell(row, zone.is_hunted ? "hunted" : "active", {
        color: zone.is_hunted ? "var(--down)" : "var(--up)",
      });

      row.addEventListener("click", () => {
        zoneTbody
          .querySelectorAll("tr.selected")
          .forEach((r) => r.classList.remove("selected"));
        row.classList.add("selected");
        const center = Math.floor(
          (unixTime(zone.start_time) + unixTime(zone.end_time)) / 2,
        );
        const span = Math.max(
          unixTime(zone.end_time) - unixTime(zone.start_time),
          6 * 3600,
        );
        focusTimeWindow(center, Math.ceil(span * 0.8));
      });

      zoneTbody.appendChild(row);
    });
  }

  /**
   * Render the direction segments into <tbody id="segment-tbody">.
   * Clicking a row scrolls the chart to that segment's time window.
   * Must be called after lastZones is populated (uses it for L1 zone count).
   */
  function renderSegmentTable(segments) {
    if (!segmentTbody) return;
    segmentTbody.innerHTML = "";
    segments.forEach((segment, idx) => {
      const dirClass = segment.direction === "UPWARD" ? "td-up" : "td-down";

      // Count L1 zones whose both pivots fall within this segment's index range
      const l1ZoneCount = lastZones.filter(
        (z) =>
          z.level === 1 &&
          z.left_pivot_index >= segment.start_index &&
          z.right_pivot_index <= segment.end_index,
      ).length;

      const row = document.createElement("tr");
      row.dataset.idx = String(idx);

      appendCell(row, segment.index, { className: "td-muted" });
      appendCell(row, segment.direction === "UPWARD" ? "UP" : "DN", {
        className: dirClass,
      });
      appendCell(row, fmtTime(segment.start_time), { className: "td-muted" });
      appendCell(row, fmtTime(segment.end_time), { className: "td-muted" });
      appendCell(row, segment.pivot_count);
      appendCell(row, l1ZoneCount);
      appendCell(row, segment.representative_pivot_index ?? "-", {
        className: "td-muted",
      });

      row.addEventListener("click", () => {
        segmentTbody
          .querySelectorAll("tr.selected")
          .forEach((r) => r.classList.remove("selected"));
        row.classList.add("selected");
        const center = Math.floor(
          (unixTime(segment.start_time) + unixTime(segment.end_time)) / 2,
        );
        const span = Math.max(
          unixTime(segment.end_time) - unixTime(segment.start_time),
          8 * 3600,
        );
        focusTimeWindow(center, Math.ceil(span * 0.65));
      });

      segmentTbody.appendChild(row);
    });
  }

  function appendCell(row, value, options = {}) {
    const cell = document.createElement("td");
    if (options.className) cell.className = options.className;
    if (options.color) cell.style.color = options.color;
    cell.textContent = String(value);
    row.appendChild(cell);
  }

  // ── Chart render ───────────────────────────────────────────────────────────

  function render(candles, pivots, segments, zones, markers, reversals = []) {
    ensureChart();

    lastCandles = candles;
    lastPivots = pivots;
    lastSegments = segments;
    lastZones = zones; // must be set before renderSegmentTable
    lastReversals = reversals.map((rev) => ({
      ...rev,
      time: unixTime(rev.time),
    }));
    lastMarkers = markers.map((marker) => ({
      ...marker,
      time: unixTime(marker.time),
      line_start_time: marker.line_start_time
        ? unixTime(marker.line_start_time)
        : null,
      line_end_time: marker.line_end_time
        ? unixTime(marker.line_end_time)
        : null,
    }));
    pivotByIndex = new Map(pivots.map((pivot) => [pivot.index, pivot]));

    const candleIndexByTime = new Map(
      candles.map((c, idx) => [unixTime(c.open_time), idx]),
    );
    pivotByCandleIndex = new Map();
    pivots.forEach((pivot) => {
      const ci = candleIndexByTime.get(unixTime(pivot.time));
      if (ci != null) pivotByCandleIndex.set(ci, pivot);
    });

    candleSeries.setData(
      candles.map((c) => ({
        time: unixTime(c.open_time),
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );

    // Render info tables (renderSegmentTable uses lastZones for L1 counts)
    renderZoneTable(zones);
    renderSegmentTable(segments);

    // Update all stat counters + badges
    const segCount = String(segments.length);
    const zCount = String(zones.length);
    if (segmentCountEl) segmentCountEl.textContent = segCount;
    if (zoneCountEl) zoneCountEl.textContent = zCount;
    if (pivotCountEl) pivotCountEl.textContent = String(pivots.length);
    if (segmentCountBadgeEl) segmentCountBadgeEl.textContent = segCount;
    if (zoneCountBadgeEl) zoneCountBadgeEl.textContent = zCount;

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

  // ── Form submission ────────────────────────────────────────────────────────

  async function handleSubmit(event) {
    event.preventDefault();
    const formData = new FormData(form);
    const start = parseDateInput(formData.get("start"));
    const end = parseDateInput(formData.get("end"));
    if (!start || !end) {
      statusEl.textContent = "Valid start and end dates are required.";
      return;
    }
    statusEl.textContent = "Requesting candles, pivots, segments, and zones...";

    const maxPivotDistanceRaw = formData.get("maximum_pivot_distance");

    const payload = {
      // ── Data source ──────────────────────────────────────────────
      symbol: formData.get("symbol"),
      timeframe: formData.get("timeframe"),
      start,
      end,
      source: formData.get("source"),
      csv_path: optionalText(formData.get("csv_path")),
      http_proxy: optionalText(formData.get("http_proxy")),
      https_proxy: optionalText(formData.get("https_proxy")),

      // ── BOS / CHoCH detection ────────────────────────────────────
      direction_window: Number(formData.get("direction_window") || 3),
      hunt_mode: formData.get("hunt_mode"),
      min_swing_pct: Number(formData.get("min_swing_pct") ?? 0),
      choch_display_mode: formData.get("choch_display_mode") || "all",
      include_pullback_in_bos_level: formData.has(
        "include_pullback_in_bos_level",
      ),
      include_hunt_candle_in_choch_range: formData.has(
        "include_hunt_candle_in_choch_range",
      ),

      // ── Pivot detection ──────────────────────────────────────────
      scan_length: Number(formData.get("scan_length") || 500),
      pivot_min_swing_pct: Number(formData.get("pivot_min_swing_pct") ?? 0),

      // ── Zone matching ────────────────────────────────────────────
      intersection_method: formData.get("intersection_method"),
      slope_attribute: formData.get("slope_attribute"),
      up_pivot_filter: formData.get("up_pivot_filter"),
      down_pivot_filter: formData.get("down_pivot_filter"),
      pivot_grouping: formData.get("pivot_grouping"),
      pair_scan_order: formData.get("pair_scan_order"),
      zone_hunt_mode: formData.get("zone_hunt_mode"),
      maximum_pivot_distance: maxPivotDistanceRaw
        ? Number(maxPivotDistanceRaw)
        : null,
      minimum_overlap: Number(formData.get("minimum_overlap") || 0),
      minimum_overlap_ratio: Number(formData.get("minimum_overlap_ratio") || 0),
      slope_epsilon: Number(formData.get("slope_epsilon") || 0),
      epsilon: Number(formData.get("epsilon") || 0.0000000001),
      include_hunted_pivots: formData.has("include_hunted_pivots"),
      allow_reuse: formData.has("allow_reuse"),
      relaxed_slope: formData.has("relaxed_slope"),

      // ── Representative pivots ────────────────────────────────────
      representative_mode: formData.get("representative_mode"),
      representative_include_hunted: formData.has(
        "representative_include_hunted",
      ),
      allow_representative_fallback: formData.has(
        "allow_representative_fallback",
      ),
    };

    try {
      const res = await fetch("/api/liquidity-zones", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const error = await res.json().catch(() => ({}));
        statusEl.textContent = error.detail || "Request failed";
        return;
      }

      const json = await res.json();
      render(
        json.candles,
        json.pivots,
        json.segments,
        json.zones,
        json.markers || [],
        json.direction_reversals || [],
      );
      statusEl.textContent =
        `Loaded ${json.candles.length} candles, ` +
        `${json.pivots.length} pivots, ` +
        `${json.segments.length} segments, ` +
        `${json.zones.length} liquidity zones, ` +
        `and ${(json.markers || []).length} BOS/CHoCH markers.`;
    } catch (err) {
      console.error(err);
      statusEl.textContent = "Unexpected error; check console.";
    }
  }

  // ── Bootstrap ──────────────────────────────────────────────────────────────

  setDefaultDates();
  syncSourceFields();

  form?.elements?.source?.addEventListener("change", syncSourceFields);
  showStructureLinesInput?.addEventListener("change", drawOverlay);
  showL1ZonesInput?.addEventListener("change", drawOverlay);
  showL2ZonesInput?.addEventListener("change", drawOverlay);
  showPivotMarkersInput?.addEventListener("change", drawOverlay);
  pivotTypeFilterInput?.addEventListener("change", drawOverlay);
  showRepresentativesInput?.addEventListener("change", drawOverlay);
  showZoneLabelsInput?.addEventListener("change", drawOverlay);
  showSegmentBandsInput?.addEventListener("change", drawOverlay);
  showCandleLabelsInput?.addEventListener("change", drawOverlay);
  candleLabelThresholdInput?.addEventListener("change", drawOverlay);
  form?.addEventListener("submit", handleSubmit);
})();
