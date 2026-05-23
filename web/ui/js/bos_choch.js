(() => {
  const form = document.getElementById("bos-choch-form");
  const statusEl = document.getElementById("status");
  const list = document.getElementById("event-list");
  const chartContainer = document.getElementById("chart");
  const showLabelsInput = document.getElementById("show-marker-labels");
  const showVerticalLabelsInput = document.getElementById(
    "show-vertical-labels",
  );
  const showZonesInput = document.getElementById("show-event-zones");
  const showPointsInput = document.getElementById("show-event-points");
  const showPivotsInput = document.getElementById("show-pivots");
  const eventCountEl = document.getElementById("event-count");
  const bosCountEl = document.getElementById("bos-count");
  const chochCountEl = document.getElementById("choch-count");
  const pivotCountEl = document.getElementById("pivot-count");
  const reversalCountEl = document.getElementById("reversal-count");
  const detailEl = document.getElementById("event-detail");
  const showReversalsInput = document.getElementById("show-reversals");
  const chochUpdatesListEl = document.getElementById("choch-updates-list");
  const chochUpdateDetailEl = document.getElementById("choch-update-detail");
  const detectionEventsListEl = document.getElementById(
    "detection-events-list",
  );
  const detectionEventDetailEl = document.getElementById(
    "detection-event-detail",
  );

  const css = getComputedStyle(document.documentElement);
  const COLORS = {
    up: css.getPropertyValue("--line-up").trim() || "#31d17c",
    down: css.getPropertyValue("--line-down").trim() || "#ff6b57",
    chochUp: css.getPropertyValue("--choch-up").trim() || "#4fb3ff",
    chochDown: css.getPropertyValue("--choch-down").trim() || "#f5b51f",
    zoneUp:
      css.getPropertyValue("--zone-up").trim() || "rgba(49, 209, 124, 0.15)",
    zoneDown:
      css.getPropertyValue("--zone-down").trim() || "rgba(255, 107, 87, 0.15)",
    zoneChochUp:
      css.getPropertyValue("--zone-choch-up").trim() ||
      "rgba(79, 179, 255, 0.14)",
    zoneChochDown:
      css.getPropertyValue("--zone-choch-down").trim() ||
      "rgba(245, 181, 31, 0.14)",
    pivotBull: css.getPropertyValue("--pivot-bull").trim() || "#b6f3cf",
    pivotBear: css.getPropertyValue("--pivot-bear").trim() || "#ffcc66",
    pivotZoneBull:
      css.getPropertyValue("--pivot-zone-bull").trim() ||
      "rgba(182, 243, 207, 0.11)",
    pivotZoneBear:
      css.getPropertyValue("--pivot-zone-bear").trim() ||
      "rgba(255, 204, 102, 0.11)",
    labelBg:
      css.getPropertyValue("--label-bg").trim() || "rgba(7, 11, 20, 0.94)",
    chartBg: css.getPropertyValue("--chart-bg").trim() || "#070b14",
    text: css.getPropertyValue("--text").trim() || "#f3f6fb",
    reversal: css.getPropertyValue("--reversal").trim() || "#e879f9",
    reversalBand: "rgba(232, 121, 249, 0.07)",
  };

  let chart;
  let candleSeries;
  let overlay;
  let overlayCtx;
  let lastMarkers = [];
  let lastPivots = [];
  let lastReversals = [];
  let lastChochUpdates = [];
  let lastDetectionEvents = [];

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
    const start = new Date(Date.UTC(2026, 0, 1, 0, 0, 0));
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
    if (overlayCtx) {
      overlayCtx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }
  }

  function ensureChart() {
    if (chart) return;
    chart = LightweightCharts.createChart(chartContainer, {
      layout: { background: { color: COLORS.chartBg }, textColor: COLORS.text },
      grid: {
        vertLines: { color: "rgba(243,246,251,0.06)" },
        horzLines: { color: "rgba(243,246,251,0.06)" },
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
      scaleMargins: { top: 0.12, bottom: 0.14 },
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
    return marker.direction === "UPWARD" ? COLORS.up : COLORS.down;
  }

  function markerLineColor(marker) {
    if (marker.type === "BOS") {
      return markerColor(marker);
    }
    return marker.direction === "UPWARD" ? COLORS.chochUp : COLORS.chochDown;
  }

  function markerZoneColor(marker) {
    if (marker.type === "BOS") {
      return marker.direction === "UPWARD" ? COLORS.zoneUp : COLORS.zoneDown;
    }
    return marker.direction === "UPWARD"
      ? COLORS.zoneChochUp
      : COLORS.zoneChochDown;
  }

  function pivotColor(pivot) {
    return pivot.type === "bullish" ? COLORS.pivotBull : COLORS.pivotBear;
  }

  function pivotZoneColor(pivot) {
    return pivot.type === "bullish"
      ? COLORS.pivotZoneBull
      : COLORS.pivotZoneBear;
  }

  function pivotPrice(pivot) {
    return pivot.type === "bullish" ? Number(pivot.low) : Number(pivot.high);
  }

  function markerLabelText(marker) {
    if (!showLabelsInput?.checked) return "";
    const side = marker.direction === "UPWARD" ? "UP" : "DOWN";
    return `${marker.type} ${marker.index} ${side} @ ${Number(marker.price).toPrecision(6)}`;
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function normalizeRect(x0, y0, x1, y1) {
    const left = Math.min(x0, x1);
    const top = Math.min(y0, y1);
    return {
      left,
      top,
      width: Math.max(Math.abs(x1 - x0), 1),
      height: Math.max(Math.abs(y1 - y0), 1),
    };
  }

  function roundedRectPath(x, y, width, height, radius) {
    if (typeof overlayCtx.roundRect === "function") {
      overlayCtx.roundRect(x, y, width, height, radius);
      return;
    }
    const r = Math.min(radius, width / 2, height / 2);
    overlayCtx.moveTo(x + r, y);
    overlayCtx.lineTo(x + width - r, y);
    overlayCtx.quadraticCurveTo(x + width, y, x + width, y + r);
    overlayCtx.lineTo(x + width, y + height - r);
    overlayCtx.quadraticCurveTo(
      x + width,
      y + height,
      x + width - r,
      y + height,
    );
    overlayCtx.lineTo(x + r, y + height);
    overlayCtx.quadraticCurveTo(x, y + height, x, y + height - r);
    overlayCtx.lineTo(x, y + r);
    overlayCtx.quadraticCurveTo(x, y, x + r, y);
  }

  function drawLabel({
    text,
    x,
    y,
    color,
    align = "right",
    vertical = "above",
  }) {
    if (!text) return;

    overlayCtx.font = "700 11px ui-monospace, SFMono-Regular, monospace";
    const paddingX = 7;
    const textWidth = Math.ceil(overlayCtx.measureText(text).width);

    if (showVerticalLabelsInput?.checked) {
      const boxW = 18;
      const boxH = Math.min(
        textWidth + paddingX * 2,
        chartContainer.clientHeight - 40,
      );
      const rawX = align === "right" ? x + 10 : x - boxW - 10;
      const boxX = clamp(rawX, 6, chartContainer.clientWidth - boxW - 6);
      const boxY = clamp(
        y - boxH / 2,
        22,
        chartContainer.clientHeight - boxH - 6,
      );

      overlayCtx.save();
      overlayCtx.fillStyle = COLORS.labelBg;
      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 1.2;
      overlayCtx.beginPath();
      roundedRectPath(boxX, boxY, boxW, boxH, 5);
      overlayCtx.fill();
      overlayCtx.stroke();
      overlayCtx.translate(boxX + boxW / 2, boxY + boxH / 2);
      overlayCtx.rotate(-Math.PI / 2);
      overlayCtx.fillStyle = color;
      overlayCtx.textAlign = "center";
      overlayCtx.textBaseline = "middle";
      overlayCtx.fillText(text, 0, 0, boxH - paddingX * 2);
      overlayCtx.restore();
    } else {
      const boxH = 20;
      const boxW = Math.min(
        textWidth + paddingX * 2,
        chartContainer.clientWidth - 12,
      );
      const rawX = align === "right" ? x + 10 : x - boxW - 10;
      const rawY = vertical === "above" ? y - boxH - 9 : y + 9;
      const boxX = clamp(rawX, 6, chartContainer.clientWidth - boxW - 6);
      const boxY = clamp(rawY, 22, chartContainer.clientHeight - boxH - 6);

      overlayCtx.save();
      overlayCtx.fillStyle = COLORS.labelBg;
      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 1.2;
      overlayCtx.beginPath();
      roundedRectPath(boxX, boxY, boxW, boxH, 5);
      overlayCtx.fill();
      overlayCtx.stroke();
      overlayCtx.fillStyle = color;
      overlayCtx.textAlign = "left";
      overlayCtx.textBaseline = "middle";
      overlayCtx.fillText(
        text,
        boxX + paddingX,
        boxY + boxH / 2,
        boxW - paddingX * 2,
      );
      overlayCtx.restore();
    }
  }

  function drawPoint(marker, x, y, color) {
    if (!showPointsInput?.checked) return;

    overlayCtx.save();
    overlayCtx.shadowColor = "rgba(0, 0, 0, 0.65)";
    overlayCtx.shadowBlur = 4;
    overlayCtx.fillStyle = color;
    overlayCtx.strokeStyle = "#ffffff";
    overlayCtx.lineWidth = 1.4;

    if (marker.type === "BOS") {
      overlayCtx.fillRect(x - 5, y - 5, 10, 10);
      overlayCtx.strokeRect(x - 4.5, y - 4.5, 9, 9);
    } else {
      overlayCtx.beginPath();
      overlayCtx.moveTo(x, y - 7);
      overlayCtx.lineTo(x + 7, y);
      overlayCtx.lineTo(x, y + 7);
      overlayCtx.lineTo(x - 7, y);
      overlayCtx.closePath();
      overlayCtx.fill();
      overlayCtx.stroke();
    }
    overlayCtx.restore();
  }

  function drawEventZone(marker, x0, x1) {
    if (!showZonesInput?.checked) return;
    const highY = candleSeries.priceToCoordinate(marker.high);
    const lowY = candleSeries.priceToCoordinate(marker.low);
    if (highY == null || lowY == null) return;

    const rect = normalizeRect(x0, highY, x1, lowY);
    const left = clamp(rect.left, 0, chartContainer.clientWidth);
    const right = clamp(rect.left + rect.width, 0, chartContainer.clientWidth);
    const top = clamp(rect.top, 0, chartContainer.clientHeight);
    const bottom = clamp(
      rect.top + rect.height,
      0,
      chartContainer.clientHeight,
    );
    if (right - left < 1 || bottom - top < 1) return;

    overlayCtx.save();
    overlayCtx.fillStyle = markerZoneColor(marker);
    overlayCtx.strokeStyle = markerLineColor(marker);
    overlayCtx.lineWidth = 1;
    overlayCtx.setLineDash(marker.type === "BOS" ? [] : [6, 5]);
    overlayCtx.fillRect(left, top, right - left, bottom - top);
    overlayCtx.strokeRect(
      left + 0.5,
      top + 0.5,
      Math.max(right - left - 1, 1),
      Math.max(bottom - top - 1, 1),
    );
    overlayCtx.restore();
  }

  function drawTriangle(x, y, size, pointsUp) {
    overlayCtx.beginPath();
    if (pointsUp) {
      overlayCtx.moveTo(x, y - size);
      overlayCtx.lineTo(x + size, y + size);
      overlayCtx.lineTo(x - size, y + size);
    } else {
      overlayCtx.moveTo(x, y + size);
      overlayCtx.lineTo(x + size, y - size);
      overlayCtx.lineTo(x - size, y - size);
    }
    overlayCtx.closePath();
  }

  function drawPivotZones() {
    if (!showPivotsInput?.checked || !showZonesInput?.checked) return;

    lastPivots.forEach((pivot) => {
      const highY = candleSeries.priceToCoordinate(pivot.high);
      const lowY = candleSeries.priceToCoordinate(pivot.low);
      const rawX0 = chart.timeScale().timeToCoordinate(pivot.reference_time);
      const rawX1 = chart.timeScale().timeToCoordinate(pivot.trigger_time);
      if (highY == null || lowY == null || rawX0 == null || rawX1 == null)
        return;

      const rect = normalizeRect(rawX0, highY, rawX1, lowY);
      const left = clamp(rect.left, 0, chartContainer.clientWidth);
      const right = clamp(
        rect.left + rect.width,
        0,
        chartContainer.clientWidth,
      );
      const top = clamp(rect.top, 0, chartContainer.clientHeight);
      const bottom = clamp(
        rect.top + rect.height,
        0,
        chartContainer.clientHeight,
      );
      if (right - left < 1 || bottom - top < 1) return;

      overlayCtx.save();
      overlayCtx.fillStyle = pivotZoneColor(pivot);
      overlayCtx.strokeStyle = pivotColor(pivot);
      overlayCtx.lineWidth = 1;
      overlayCtx.setLineDash([3, 5]);
      overlayCtx.fillRect(left, top, right - left, bottom - top);
      overlayCtx.strokeRect(
        left + 0.5,
        top + 0.5,
        Math.max(right - left - 1, 1),
        Math.max(bottom - top - 1, 1),
      );
      overlayCtx.restore();
    });
  }

  function drawPivotLinesAndPoints() {
    if (!showPivotsInput?.checked) return;

    lastPivots.forEach((pivot) => {
      const price = pivotPrice(pivot);
      const y = candleSeries.priceToCoordinate(price);
      const rawPivotX = chart.timeScale().timeToCoordinate(pivot.time);
      const rawRefX = chart.timeScale().timeToCoordinate(pivot.reference_time);
      const rawTriggerX = chart
        .timeScale()
        .timeToCoordinate(pivot.trigger_time);
      const rawEndX = chart
        .timeScale()
        .timeToCoordinate(pivot.invalidation_time || pivot.trigger_time);
      if (y == null || rawPivotX == null) return;

      const color = pivotColor(pivot);
      const x0 = clamp(
        Math.round(rawRefX ?? rawPivotX),
        0,
        chartContainer.clientWidth,
      );
      const x1 = clamp(
        Math.round(rawEndX ?? rawTriggerX ?? rawPivotX),
        0,
        chartContainer.clientWidth,
      );
      const pivotX = Math.round(rawPivotX);
      const py = Math.round(y);
      const isBullish = pivot.type === "bullish";

      overlayCtx.save();
      overlayCtx.strokeStyle = color;
      overlayCtx.fillStyle = color;
      overlayCtx.lineWidth = pivot.haunted ? 1.4 : 2.2;
      overlayCtx.setLineDash(pivot.haunted ? [2, 5] : [7, 4]);
      overlayCtx.beginPath();
      overlayCtx.moveTo(x0, py);
      overlayCtx.lineTo(x1, py);
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);

      if (showPointsInput?.checked) {
        overlayCtx.shadowColor = "rgba(0, 0, 0, 0.75)";
        overlayCtx.shadowBlur = 5;
        overlayCtx.lineWidth = 1.5;
        overlayCtx.strokeStyle = "#ffffff";
        drawTriangle(pivotX, py, 8, isBullish);
        overlayCtx.fill();
        overlayCtx.stroke();
        overlayCtx.shadowBlur = 0;
      }

      if (showLabelsInput?.checked) {
        const side = isBullish ? "BULL" : "BEAR";
        drawLabel({
          text: `P ${pivot.index} ${side} @ ${price.toPrecision(6)}${pivot.haunted ? " hunted" : ""}`,
          x: pivotX,
          y: py,
          color,
          align: pivotX < chartContainer.clientWidth * 0.62 ? "right" : "left",
          vertical: isBullish ? "below" : "above",
        });
      }

      overlayCtx.restore();
    });
  }

  function drawReversalLines() {
    if (!showReversalsInput?.checked || !lastReversals.length) return;
    const W = chartContainer.clientWidth;
    const H = chartContainer.clientHeight;

    lastReversals.forEach((rev) => {
      const rawX = chart.timeScale().timeToCoordinate(rev.time);
      if (rawX == null) return;
      const x = Math.round(clamp(rawX, 0, W));

      overlayCtx.save();
      // semi-transparent band behind everything
      overlayCtx.fillStyle = COLORS.reversalBand;
      overlayCtx.fillRect(x - 3, 0, 6, H);
      // dashed center line
      overlayCtx.strokeStyle = COLORS.reversal;
      overlayCtx.lineWidth = 1.5;
      overlayCtx.setLineDash([4, 6]);
      overlayCtx.globalAlpha = 0.6;
      overlayCtx.beginPath();
      overlayCtx.moveTo(x, 0);
      overlayCtx.lineTo(x, H);
      overlayCtx.stroke();
      // arrow label at top edge
      overlayCtx.setLineDash([]);
      overlayCtx.globalAlpha = 0.9;
      overlayCtx.fillStyle = COLORS.reversal;
      overlayCtx.font = "700 11px ui-monospace, SFMono-Regular, monospace";
      overlayCtx.textAlign = "center";
      overlayCtx.textBaseline = "top";
      overlayCtx.fillText(rev.direction === "UPWARD" ? "▲" : "▼", x, 5);
      overlayCtx.restore();
    });
  }

  function drawStructureMarkers() {
    if (!overlayCtx || !chart || !candleSeries) return;
    const W = chartContainer.clientWidth;
    const H = chartContainer.clientHeight;
    overlayCtx.clearRect(0, 0, W, H);

    overlayCtx.save();
    overlayCtx.beginPath();
    overlayCtx.rect(0, 0, W, H);
    overlayCtx.clip();

    drawReversalLines();
    drawPivotZones();

    lastMarkers.forEach((marker) => {
      const priceY = candleSeries.priceToCoordinate(marker.price);
      if (priceY == null) return;

      const rawX0 = chart.timeScale().timeToCoordinate(marker.line_start_time);
      const rawX1 = chart.timeScale().timeToCoordinate(marker.line_end_time);
      const rawEventX = chart.timeScale().timeToCoordinate(marker.time);
      if (rawX0 == null && rawX1 == null && rawEventX == null) return;

      const x0 = Math.round(rawX0 ?? rawEventX ?? 0);
      const x1 = Math.round(rawX1 ?? rawEventX ?? W);
      const eventX = Math.round(rawEventX ?? x1);
      const py = Math.round(priceY);
      const isBOS = marker.type === "BOS";
      const lineColor = markerLineColor(marker);
      const iconColor = markerColor(marker);
      const visibleX0 = clamp(x0, 0, W);
      const visibleX1 = clamp(x1, 0, W);

      overlayCtx.save();
      drawEventZone(marker, visibleX0, visibleX1);

      // horizontal price-level line
      overlayCtx.strokeStyle = lineColor;
      overlayCtx.lineWidth = isBOS ? 2.4 : 2;
      overlayCtx.setLineDash(isBOS ? [] : [8, 5]);
      overlayCtx.globalAlpha = 0.96;
      overlayCtx.beginPath();
      overlayCtx.moveTo(visibleX0, py);
      overlayCtx.lineTo(visibleX1, py);
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);
      overlayCtx.globalAlpha = 1;

      // Vertical anchor makes the point readable even when several horizontal
      // levels share a nearby price.
      overlayCtx.strokeStyle = lineColor;
      overlayCtx.lineWidth = 1;
      overlayCtx.globalAlpha = 0.7;
      overlayCtx.beginPath();
      overlayCtx.moveTo(eventX, clamp(py - 18, 0, H));
      overlayCtx.lineTo(eventX, clamp(py + 18, 0, H));
      overlayCtx.stroke();
      overlayCtx.globalAlpha = 1;

      drawPoint(marker, eventX, py, iconColor);

      const label = markerLabelText(marker);
      drawLabel({
        text: label,
        x: eventX,
        y: py,
        color: lineColor,
        align: eventX < W * 0.62 ? "right" : "left",
        vertical: marker.direction === "UPWARD" ? "above" : "below",
      });

      overlayCtx.restore();
    });

    drawPivotLinesAndPoints();
    overlayCtx.restore();
  }

  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function fmtTime(unix) {
    if (unix == null) return "—";
    return (
      new Date(unix * 1000).toISOString().replace("T", " ").slice(0, 19) +
      " UTC"
    );
  }

  function renderDetailTo(el, obj) {
    if (!el) return;
    const timeFields = new Set(["time", "line_start_time", "line_end_time"]);
    const rows = Object.entries(obj)
      .map(([key, val]) => {
        let display;
        if (timeFields.has(key) && typeof val === "number") {
          display = fmtTime(val);
        } else if (typeof val === "number") {
          display = Number.isInteger(val)
            ? String(val)
            : Number(val).toPrecision(8);
        } else {
          display = val != null ? String(val) : "—";
        }
        return `<tr><td class="dk">${escHtml(key)}</td><td>${escHtml(display)}</td></tr>`;
      })
      .join("");
    el.innerHTML = `<table class="detail-table"><tbody>${rows}</tbody></table>`;
  }

  function renderDetail(marker) {
    renderDetailTo(detailEl, marker);
  }

  function buildSubList({
    listEl,
    detailEl: subDetailEl,
    items,
    rowText,
    rowColor,
    getTime,
  }) {
    if (!listEl) return;
    listEl.innerHTML = "";
    if (subDetailEl) {
      subDetailEl.innerHTML =
        '<p class="detail-placeholder">Click an item to view its full spec</p>';
    }
    items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "event-row";
      if (rowColor) row.style.color = rowColor(item);
      row.textContent = rowText(item);
      row.addEventListener("click", () => {
        listEl
          .querySelectorAll(".event-row.selected")
          .forEach((el) => el.classList.remove("selected"));
        row.classList.add("selected");
        const t = getTime ? getTime(item) : null;
        if (chart && t != null) {
          chart.timeScale().setVisibleRange({
            from: t - 60 * 60 * 12,
            to: t + 60 * 60 * 12,
          });
          drawStructureMarkers();
        }
        renderDetailTo(subDetailEl, item);
      });
      listEl.appendChild(row);
    });
  }

  function unixTime(value) {
    if (typeof value === "number") return value;
    return Math.floor(new Date(value).getTime() / 1000);
  }

  function normalizePivot(pivot) {
    return {
      ...pivot,
      time: unixTime(pivot.time),
      reference_time: unixTime(pivot.reference_time),
      trigger_time: unixTime(pivot.trigger_time),
      invalidation_time: pivot.invalidation_time
        ? unixTime(pivot.invalidation_time)
        : null,
      high: Number(pivot.high),
      low: Number(pivot.low),
    };
  }

  function render(
    candles,
    markers,
    directionState,
    pivots = [],
    reversals = [],
    chochUpdates = [],
    detectionEvents = [],
  ) {
    ensureChart();
    const data = candles.map((c) => ({
      time: unixTime(c.open_time),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleSeries.setData(data);

    lastMarkers = markers.map((marker) => ({
      ...marker,
      time: unixTime(marker.time),
      line_start_time: unixTime(marker.line_start_time),
      line_end_time: unixTime(marker.line_end_time),
    }));
    lastPivots = pivots.map(normalizePivot);
    lastReversals = reversals.map((rev) => ({
      ...rev,
      time: unixTime(rev.time),
    }));
    lastChochUpdates = chochUpdates.map((u) => ({
      ...u,
      time: unixTime(u.time),
    }));
    lastDetectionEvents = detectionEvents.map((ev) => ({
      ...ev,
      time: unixTime(ev.time),
    }));
    drawStructureMarkers();

    list.innerHTML = "";
    if (detailEl) {
      detailEl.innerHTML =
        '<p class="detail-placeholder">Click an event to view its full spec</p>';
    }
    lastMarkers.forEach((marker, idx) => {
      const row = document.createElement("div");
      const date = new Date(marker.time * 1000)
        .toISOString()
        .replace("T", " ")
        .slice(0, 16);
      const px = marker.price.toPrecision(6);
      const typeLabel =
        marker.type === "BOS"
          ? `BOS #${marker.index}`
          : `CHoCH #${marker.index} [B#${marker.bos_index ?? "?"}]`;
      const candleRange = `${Number(marker.low).toPrecision(6)}-${Number(marker.high).toPrecision(6)}`;
      row.className = "event-row";
      row.dataset.idx = String(idx);
      row.style.color = markerLineColor(marker);
      row.textContent = `${typeLabel.padEnd(22)} · ${marker.direction.padEnd(8)} · ${px} · ${candleRange} · ${date}`;
      row.addEventListener("click", () => {
        list
          .querySelectorAll(".event-row.selected")
          .forEach((el) => el.classList.remove("selected"));
        row.classList.add("selected");
        if (chart) {
          chart.timeScale().setVisibleRange({
            from: marker.time - 60 * 60 * 12,
            to: marker.time + 60 * 60 * 12,
          });
          drawStructureMarkers();
        }
        renderDetail(marker);
      });
      list.appendChild(row);
    });

    const bosCount = lastMarkers.filter(
      (marker) => marker.type === "BOS",
    ).length;
    const chochCount = lastMarkers.length - bosCount;
    if (eventCountEl) eventCountEl.textContent = String(lastMarkers.length);
    if (bosCountEl) bosCountEl.textContent = String(bosCount);
    if (chochCountEl) chochCountEl.textContent = String(chochCount);
    if (pivotCountEl) pivotCountEl.textContent = String(lastPivots.length);
    if (reversalCountEl)
      reversalCountEl.textContent = String(lastReversals.length);

    if (lastMarkers.length) {
      chart.timeScale().setVisibleRange({
        from: Math.max(
          data[0]?.time || lastMarkers[0].time,
          lastMarkers[0].time - 60 * 60 * 18,
        ),
        to: Math.min(
          data[data.length - 1]?.time ||
            lastMarkers[lastMarkers.length - 1].time,
          lastMarkers[lastMarkers.length - 1].time + 60 * 60 * 18,
        ),
      });
    } else {
      chart.timeScale().fitContent();
    }
    drawStructureMarkers();
    buildSubList({
      listEl: chochUpdatesListEl,
      detailEl: chochUpdateDetailEl,
      items: lastChochUpdates,
      rowText: (u) => {
        const date = new Date(u.time * 1000)
          .toISOString()
          .replace("T", " ")
          .slice(0, 16);
        return `BOS #${String(u.bos_index).padEnd(4)} · ${u.reason.padEnd(16)} · ${Number(u.level).toPrecision(6)} · ${date}`;
      },
      rowColor: (u) =>
        u.direction === "UPWARD" ? COLORS.chochUp : COLORS.chochDown,
      getTime: (u) => u.time,
    });
    buildSubList({
      listEl: detectionEventsListEl,
      detailEl: detectionEventDetailEl,
      items: lastDetectionEvents,
      rowText: (ev) => {
        const date = new Date(ev.time * 1000)
          .toISOString()
          .replace("T", " ")
          .slice(0, 16);
        return `${ev.event.padEnd(22)} · ${ev.direction.padEnd(8)} · ${ev.details} · ${date}`;
      },
      rowColor: (ev) => {
        if (ev.event === "DIRECTION_REVERSED") return COLORS.reversal;
        if (ev.event === "BOS_CONFIRMED")
          return ev.direction === "UPWARD" ? COLORS.up : COLORS.down;
        return ev.direction === "UPWARD" ? COLORS.chochUp : COLORS.chochDown;
      },
      getTime: (ev) => ev.time,
    });
    statusEl.textContent = `Found ${lastMarkers.length} events and ${lastPivots.length} pivots across ${candles.length} candles. Current direction: ${directionState.direction} since candle ${directionState.since_index}.`;
  }

  async function handleSubmit(event) {
    event.preventDefault();
    statusEl.textContent = "Requesting candles & structure events...";
    const formData = new FormData(form);
    const payload = {
      symbol: formData.get("symbol"),
      timeframe: formData.get("timeframe"),
      start: new Date(formData.get("start")).toISOString(),
      end: new Date(formData.get("end")).toISOString(),
      direction_window: Number(formData.get("direction_window") ?? 3) || 3,
      scan_length: Number(formData.get("scan_length") ?? 500) || 500,
      hunt_mode: formData.get("hunt_mode"),
      choch_display_mode: formData.get("choch_display_mode") || "all",
      include_bos_in_choch_range: formData.has("include_bos_in_choch_range"),
      include_hunt_candle_in_choch_range: formData.has(
        "include_hunt_candle_in_choch_range",
      ),
      min_swing_pct: Number(formData.get("min_swing_pct") ?? 0),
      include_pullback_in_bos_level: formData.has(
        "include_pullback_in_bos_level",
      ),
      restart_on_invalidation: formData.has("restart_on_invalidation"),
      use_structural_left_bound: formData.has("use_structural_left_bound"),
      include_reference_candle: formData.has("include_reference_candle"),
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
      render(
        json.candles,
        json.markers,
        json.direction_state,
        json.pivots || [],
        json.direction_reversals || [],
        json.choch_updates || [],
        json.detection_events || [],
      );
    } catch (err) {
      console.error(err);
      statusEl.textContent = "Unexpected error; check console.";
    }
  }

  form?.elements?.source?.addEventListener("change", syncSourceFields);
  showLabelsInput?.addEventListener("change", drawStructureMarkers);
  showVerticalLabelsInput?.addEventListener("change", drawStructureMarkers);
  showZonesInput?.addEventListener("change", drawStructureMarkers);
  showPointsInput?.addEventListener("change", drawStructureMarkers);
  showPivotsInput?.addEventListener("change", drawStructureMarkers);
  showReversalsInput?.addEventListener("change", drawStructureMarkers);
  syncSourceFields();
  form?.addEventListener("submit", handleSubmit);
})();
