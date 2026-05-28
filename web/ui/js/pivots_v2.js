(() => {
  const form = document.getElementById("pivot-form");
  const statusEl = document.getElementById("status");
  const entryList = document.getElementById("entry-list");
  const chartContainer = document.getElementById("chart");

  function isoInput(dt) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${dt.getUTCFullYear()}-${pad(dt.getUTCMonth() + 1)}-${pad(dt.getUTCDate())}T${pad(dt.getUTCHours())}:${pad(dt.getUTCMinutes())}`;
  }

  function optionalText(value) {
    if (typeof value !== "string") return null;
    const trimmed = value.trim();
    return trimmed ? trimmed : null;
  }

  function parseDateInput(value) {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date.toISOString();
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

  const BEAR_COLOR = "#fbbf24";
  const BULL_COLOR = "#6ee7b7";
  const LABEL_BG = "rgba(11,18,32,0.90)";
  const LABEL_BORDER_BEAR = "#fbbf24";
  const LABEL_BORDER_BULL = "#6ee7b7";
  const LABEL_NONE = "rgba(156,163,175,0.6)";

  let chart;
  let candleSeries;
  let overlay;
  let overlayCtx;

  // Indexed by candle_index: { pivot_index, pivot_type, hunt_index, candle_time, pivot_time }
  let pivotMap = new Map();
  // Sorted candle data array (time in unix seconds)
  let candleData = [];

  function ensureOverlay() {
    if (overlay) return;
    overlay = document.createElement("canvas");
    overlay.style.position = "absolute";
    overlay.style.inset = "0";
    overlay.style.pointerEvents = "none";
    overlay.style.zIndex = "4";
    chartContainer.appendChild(overlay);
    overlayCtx = overlay.getContext("2d");
  }

  function resizeOverlay() {
    if (!overlay) return;
    const ratio = window.devicePixelRatio || 1;
    const w = chartContainer.clientWidth;
    const h = chartContainer.clientHeight;
    overlay.width = Math.floor(w * ratio);
    overlay.height = Math.floor(h * ratio);
    overlay.style.width = `${w}px`;
    overlay.style.height = `${h}px`;
    if (overlayCtx) overlayCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

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
    candleSeries
      .priceScale()
      .applyOptions({ scaleMargins: { top: 0.14, bottom: 0.06 } });
    ensureOverlay();
    resizeOverlay();
    chart.timeScale().subscribeVisibleLogicalRangeChange(drawOverlay);
    new ResizeObserver(() => {
      chart.applyOptions({
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
      });
      resizeOverlay();
      drawOverlay();
    }).observe(chartContainer);
  }

  function getLabelThreshold() {
    const el = document.getElementById("label_threshold");
    const v = parseInt(el?.value || "100", 10);
    return isNaN(v) ? 100 : Math.max(10, v);
  }

  function visibleCandleCount() {
    if (!chart || !candleData.length) return Infinity;
    const range = chart.timeScale().getVisibleLogicalRange();
    if (!range) return Infinity;
    return Math.abs(range.to - range.from);
  }

  function drawOverlay() {
    if (!overlayCtx || !chart || !candleSeries || !candleData.length) return;
    const W = chartContainer.clientWidth;
    const H = chartContainer.clientHeight;
    overlayCtx.clearRect(0, 0, W, H);

    const threshold = getLabelThreshold();
    const visible = visibleCandleCount();
    const showLabels = visible <= threshold;

    overlayCtx.save();
    overlayCtx.beginPath();
    overlayCtx.rect(0, 0, W, H);
    overlayCtx.clip();

    candleData.forEach((c, idx) => {
      const x = chart.timeScale().timeToCoordinate(c.time);
      if (x == null) return;
      const cx = Math.round(x);

      const entry = pivotMap.get(idx);
      const hasPivot = entry && entry.pivot_index != null;

      // Draw pivot triangle marker
      if (hasPivot) {
        const isBear = entry.pivot_type === "bearish";
        const pivotCandle = candleData[entry.pivot_index];
        if (pivotCandle) {
          const px = chart.timeScale().timeToCoordinate(pivotCandle.time);
          if (px != null) {
            const pivotX = Math.round(px);
            const price = isBear ? pivotCandle.high : pivotCandle.low;
            const py = candleSeries.priceToCoordinate(price);
            if (py != null) {
              const triY = Math.round(py);
              const color = isBear ? BEAR_COLOR : BULL_COLOR;
              overlayCtx.save();
              overlayCtx.fillStyle = color;
              overlayCtx.strokeStyle = "#ffffff";
              overlayCtx.lineWidth = 1.2;
              overlayCtx.shadowColor = "rgba(0,0,0,0.6)";
              overlayCtx.shadowBlur = 4;
              overlayCtx.beginPath();
              if (isBear) {
                overlayCtx.moveTo(pivotX, triY - 10);
                overlayCtx.lineTo(pivotX + 7, triY - 2);
                overlayCtx.lineTo(pivotX - 7, triY - 2);
              } else {
                overlayCtx.moveTo(pivotX, triY + 10);
                overlayCtx.lineTo(pivotX + 7, triY + 2);
                overlayCtx.lineTo(pivotX - 7, triY + 2);
              }
              overlayCtx.closePath();
              overlayCtx.fill();
              overlayCtx.stroke();
              overlayCtx.restore();
            }
          }
        }
      }

      // Draw per-candle label: index / pivot_index
      if (!showLabels) return;

      const pivotLabel = hasPivot ? String(entry.pivot_index) : "N/A";
      const topLine = String(idx);
      const bottomLine = pivotLabel;
      const labelText = `${topLine}\n${bottomLine}`;
      const color = hasPivot
        ? entry.pivot_type === "bearish"
          ? LABEL_BORDER_BEAR
          : LABEL_BORDER_BULL
        : LABEL_NONE;

      const highY = candleSeries.priceToCoordinate(c.high);
      if (highY == null) return;

      overlayCtx.save();
      overlayCtx.font = "600 9px ui-monospace, SFMono-Regular, monospace";
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
      overlayCtx.roundRect
        ? overlayCtx.roundRect(boxX, boxY, boxW, boxH, 3)
        : overlayCtx.rect(boxX, boxY, boxW, boxH);
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

  function render(candles, pivotEntries) {
    ensureChart();

    candleData = candles.map((c) => ({
      time: Math.floor(new Date(c.open_time).getTime() / 1000),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleSeries.setData(candleData);

    pivotMap = new Map();
    pivotEntries.forEach((e) => {
      pivotMap.set(e.candle_index, e);
    });

    // Build entry list (only candles that produced a pivot)
    entryList.innerHTML = "";
    const withPivot = pivotEntries.filter((e) => e.pivot_index != null);
    withPivot.forEach((e) => {
      const opt = document.createElement("option");
      const cTime = new Date(e.candle_time)
        .toISOString()
        .replace("T", " ")
        .slice(0, 16);
      const dir = e.pivot_type === "bearish" ? "BEAR" : "BULL";
      const color = e.pivot_type === "bearish" ? BEAR_COLOR : BULL_COLOR;
      opt.value = e.candle_index;
      opt.textContent = `candle=${String(e.candle_index).padStart(4)} pivot=${String(e.pivot_index).padStart(4)} hunt=${String(e.hunt_index ?? "?").padStart(4)} ${dir} @ ${cTime}`;
      opt.style.color = color;
      entryList.appendChild(opt);
    });

    if (candleData.length) {
      chart.timeScale().fitContent();
    }
    drawOverlay();
  }

  entryList?.addEventListener("change", () => {
    if (!chart) return;
    const candleIdx = Number(entryList.value);
    if (isNaN(candleIdx) || !candleData[candleIdx]) return;
    const t = candleData[candleIdx].time;
    const spacing =
      candleData.length >= 2 ? candleData[1].time - candleData[0].time : 3600;
    chart
      .timeScale()
      .setVisibleRange({ from: t - 60 * spacing, to: t + 60 * spacing });
    drawOverlay();
  });

  document
    .getElementById("label_threshold")
    ?.addEventListener("change", drawOverlay);

  async function handleSubmit(event) {
    event.preventDefault();
    const fd = new FormData(form);
    const start = parseDateInput(fd.get("start"));
    const end = parseDateInput(fd.get("end"));
    if (!start || !end) {
      statusEl.textContent = "Valid start and end dates are required.";
      return;
    }
    statusEl.textContent = "Requesting candles & pivot entries...";
    const payload = {
      symbol: fd.get("symbol"),
      timeframe: fd.get("timeframe"),
      start,
      end,
      source: fd.get("source"),
      csv_path: optionalText(fd.get("csv_path")),
      http_proxy: optionalText(fd.get("http_proxy")),
      https_proxy: optionalText(fd.get("https_proxy")),
      scan_length: Number(fd.get("scan_length") || 500),
      min_swing_pct: Number(fd.get("min_swing_pct") || 0),
    };
    try {
      const res = await fetch("/api/pivots/v2", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        statusEl.textContent = err.detail || "Request failed";
        return;
      }
      const json = await res.json();
      const withPivot = json.pivot_entries.filter((e) => e.pivot_index != null);
      render(json.candles, json.pivot_entries);
      statusEl.textContent = `${json.candles.length} candles · ${json.pivot_entries.length} scanned · ${withPivot.length} with pivot.`;
    } catch (err) {
      console.error(err);
      statusEl.textContent = "Unexpected error; check console.";
    }
  }

  syncSourceFields();
  form?.elements?.source?.addEventListener("change", syncSourceFields);
  form?.addEventListener("submit", handleSubmit);
})();
