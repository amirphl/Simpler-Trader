(function () {
  function computeDrawdownCurve(equity) {
    const result = [];
    let peak = -Infinity;
    for (const value of equity) {
      peak = Math.max(peak, value);
      if (peak <= 0) {
        result.push(0);
      } else {
        result.push(((value - peak) / peak) * 100);
      }
    }
    return result;
  }

  const chartInstances = [];
  let zoomRegistered = false;
  let latestTrades = null;
  function destroyCharts() {
    chartInstances.splice(0).forEach((c) => {
      try {
        c.destroy();
      } catch {
        /* ignore */
      }
    });
  }

  function ensureZoomPlugin() {
    if (!window.Chart) return false;
    if (zoomRegistered) return true;
    if (window.ChartZoom) {
      Chart.register(window.ChartZoom);
      zoomRegistered = true;
      return true;
    }
    return false;
  }

  function createChart(container, type, data, options) {
    const canvas = document.createElement("canvas");
    canvas.height = 220;
    container.appendChild(canvas);
    ensureZoomPlugin();
    const chart = new Chart(canvas.getContext("2d"), {
      type,
      data,
      options: Object.assign(
        {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            zoom: {
              limits: { x: { min: "original", max: "original" }, y: { min: "original", max: "original" } },
              pan: { enabled: true, mode: "x", modifierKey: "shift" },
              zoom: {
                wheel: { enabled: true },
                pinch: { enabled: true },
                drag: { enabled: false },
                mode: "x",
              },
            },
          },
          interaction: { mode: "nearest", intersect: false },
          scales: {
            x: { ticks: { color: "#475569" }, grid: { display: false } },
            y: { ticks: { color: "#475569" }, grid: { color: "#e2e8f0" } },
          },
        },
        options || {}
      ),
    });
    chartInstances.push(chart);
    return chart;
  }

  function initBacktestPage(options) {
    const cfg = options || {};
    const formSelector = cfg.formSelector || "#backtest-form";
    const serializeForm = cfg.serializeForm;

    if (typeof serializeForm !== "function") {
      console.error("serializeForm must be provided for initBacktestPage.");
      return;
    }

    const form = document.querySelector(formSelector);
    const historyForm = document.getElementById("history-form");
    const statusStream = document.getElementById("status-stream");
    const statsOutput = document.getElementById("stats-output");
    const tradesOutput = document.getElementById("trades-output");
    const currentJob = document.getElementById("current-job");
    const chartsOutput = document.getElementById("charts-output");
    const tradeJumpForm = document.getElementById("trade-jump-form");
    const tradeJumpInput = document.getElementById("trade-jump-index");

    if (!form || !statusStream || !statsOutput || !tradesOutput) {
      console.error("Required DOM elements are missing from this page.");
      return;
    }

    let activeSocket = null;

    async function handleSubmit(event) {
      event.preventDefault();
      const payload = serializeForm(form);
      if (!payload || !payload.start || !payload.end) {
        statusStream.textContent = "Start and end dates are required.";
        statusStream.dataset.state = "failed";
        return;
      }
      statsOutput.textContent = "Awaiting result...";
      tradesOutput.textContent = "Awaiting result...";
      statusStream.textContent = "Submitting job...";
      statusStream.dataset.state = "queued";
      try {
        const response = await fetch("/api/backtests", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          const error = await response.json();
          statusStream.textContent = error.detail || "Failed to submit job.";
          statusStream.dataset.state = "failed";
          return;
        }
        const job = await response.json();
        if (currentJob) {
          currentJob.textContent = `Tracking job ${job.job_id}`;
        }
        statusStream.textContent = `Job ${job.job_id}: ${job.status.toUpperCase()}`;
        openSocket(job.job_id);
      } catch (error) {
        console.error(error);
        statusStream.textContent = "Unexpected error. See console for details.";
        statusStream.dataset.state = "failed";
      }
    }

    async function handleHistory(event) {
      event.preventDefault();
      const formData = new FormData(historyForm);
      const rawJobId = formData.get("job_id");
      const jobId = typeof rawJobId === "string" ? rawJobId.trim() : "";
      if (!jobId) {
        statusStream.textContent = "Job id is required.";
        statusStream.dataset.state = "failed";
        return;
      }
      statusStream.textContent = `Fetching result for ${jobId}...`;
      statusStream.dataset.state = "running";
      try {
        const response = await fetch(`/api/backtests/${jobId}/result`);
        if (!response.ok) {
          const error = await response.json();
          statusStream.textContent = error.detail || "Result not available.";
          statusStream.dataset.state = "failed";
          return;
        }
        const payload = await response.json();
        statusStream.textContent = `Job ${jobId}: ${payload.status.toUpperCase()}`;
        statusStream.dataset.state = payload.status;
        renderResult(payload.result);
      } catch (error) {
        console.error(error);
        statusStream.textContent = "Failed to fetch result.";
        statusStream.dataset.state = "failed";
      }
    }

    function openSocket(jobId) {
      if (activeSocket) {
        activeSocket.close();
      }
      const protocol = location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${protocol}://${location.host}/ws/backtests/${jobId}`);
      socket.onmessage = (event) => {
        try {
          handleEvent(JSON.parse(event.data));
        } catch (error) {
          console.error("Failed to parse socket payload", error);
        }
      };
      socket.onclose = () => {
        if (statusStream.dataset.state !== "completed") {
          statusStream.textContent = "Connection closed.";
          statusStream.dataset.state = "idle";
        }
      };
      socket.onerror = () => {
        statusStream.textContent = "WebSocket error.";
        statusStream.dataset.state = "failed";
      };
      activeSocket = socket;
    }

    function handleEvent(event) {
      if (event.event === "status") {
        statusStream.textContent = `Job ${event.job_id}: ${event.status.toUpperCase()}`;
        statusStream.dataset.state = event.status;
      } else if (event.event === "result") {
        statusStream.textContent = `Job ${event.job_id}: ${event.status.toUpperCase()}`;
        statusStream.dataset.state = event.status;
        renderResult(event.result);
      } else if (event.event === "error") {
        statusStream.textContent = `Job ${event.job_id}: ${event.error}`;
        statusStream.dataset.state = "failed";
      }
    }

    function renderResult(result) {
      if (!result) {
        statsOutput.textContent = "No result payload.";
        tradesOutput.textContent = "No result payload.";
        return;
      }
      const stats = result.report?.statistics || {};
      const statsWithIndex = Object.assign({}, stats, {
        equity_curve_indexed: Array.isArray(stats.equity_curve)
          ? stats.equity_curve.map((v, i) => ({ index: i, equity: v }))
          : [],
      });
      statsOutput.textContent = JSON.stringify(statsWithIndex, null, 2);
     const trades = result.report?.trades || [];
      renderTradesList(trades);

      if (chartsOutput) {
        destroyCharts();
        chartsOutput.innerHTML = "";
        chartsOutput.style.display = "grid";
        chartsOutput.style.gridTemplateColumns = "repeat(auto-fit, minmax(280px, 1fr))";
        chartsOutput.style.gap = "1rem";

        if (!window.Chart) {
          chartsOutput.textContent = "Chart.js failed to load.";
          return;
        }

        const equity = Array.isArray(stats.equity_curve) ? stats.equity_curve : [];
        if (equity.length) {
          const dd = computeDrawdownCurve(equity);
          const equityContainer = document.createElement("div");
          const ddContainer = document.createElement("div");
          equityContainer.className = ddContainer.className = "chart-card";
          chartsOutput.appendChild(equityContainer);
          chartsOutput.appendChild(ddContainer);
          createChart(
            equityContainer,
            "line",
            {
              labels: equity.map((_, i) => i + 1),
              datasets: [
                {
                  label: "Equity",
                  data: equity,
                  borderColor: "#2563eb",
                  backgroundColor: "rgba(37,99,235,0.15)",
                  tension: 0.2,
                  fill: true,
                },
              ],
            },
            { plugins: { legend: { display: false }, title: { display: true, text: "Equity Curve" } } }
          );
          createChart(
            ddContainer,
            "line",
            {
              labels: dd.map((_, i) => i + 1),
              datasets: [
                {
                  label: "Drawdown %",
                  data: dd,
                  borderColor: "#e11d48",
                  backgroundColor: "rgba(225,29,72,0.15)",
                  tension: 0.2,
                  fill: true,
                },
              ],
            },
            { plugins: { legend: { display: false }, title: { display: true, text: "Drawdown (%)" } } }
          );
        }

        if (trades.length) {
          const returns = trades.map((t, idx) => ({
            label: `#${idx + 1}`,
            value: t.return_pct || 0,
          }));
          const returnsContainer = document.createElement("div");
          returnsContainer.className = "chart-card";
          chartsOutput.appendChild(returnsContainer);
          createChart(
            returnsContainer,
            "bar",
            {
              labels: returns.map((r) => r.label),
              datasets: [
                {
                  label: "Return %",
                  data: returns.map((r) => r.value),
                  backgroundColor: returns.map((r) => (r.value >= 0 ? "rgba(34,197,94,0.6)" : "rgba(239,68,68,0.6)")),
                  borderColor: returns.map((r) => (r.value >= 0 ? "#22c55e" : "#ef4444")),
                  borderWidth: 1,
                },
              ],
            },
            {
              plugins: { legend: { display: false }, title: { display: true, text: "Return per Trade (%)" } },
            }
          );

          const wins = trades.filter((t) => (t.return_pct || 0) >= 0).length;
          const losses = trades.length - wins;
          const pieContainer = document.createElement("div");
          pieContainer.className = "chart-card";
          chartsOutput.appendChild(pieContainer);
          createChart(
            pieContainer,
            "doughnut",
            {
              labels: ["Wins", "Losses"],
              datasets: [
                {
                  data: [wins, losses],
                  backgroundColor: ["#22c55e", "#ef4444"],
                  borderWidth: 0,
                },
              ],
            },
            { plugins: { title: { display: true, text: "Win / Loss Count" } } }
          );

          const addsSeries = trades
            .map((t, idx) => {
              const raw = t.metadata ? t.metadata.adds : null;
              const value = raw === null || raw === undefined ? NaN : Number(raw);
              return { label: `#${idx + 1}`, value };
            })
            .filter((item) => Number.isFinite(item.value));
          if (addsSeries.length) {
            const addsContainer = document.createElement("div");
            addsContainer.className = "chart-card";
            chartsOutput.appendChild(addsContainer);
            createChart(
              addsContainer,
              "bar",
              {
                labels: addsSeries.map((a) => a.label),
                datasets: [
                  {
                    label: "Adds",
                    data: addsSeries.map((a) => a.value),
                    backgroundColor: addsSeries.map((a) =>
                      a.value > 0 ? "rgba(249,115,22,0.6)" : "rgba(148,163,184,0.6)"
                    ),
                    borderColor: addsSeries.map((a) => (a.value > 0 ? "#f97316" : "#94a3b8")),
                    borderWidth: 1,
                  },
                ],
              },
              {
                plugins: { legend: { display: false }, title: { display: true, text: "Martingale Adds" } },
                scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
              }
            );
          }
        }

        const metricEntries = [
          ["Net Profit", stats.net_profit],
          ["Net Profit %", stats.net_profit_pct],
          ["Win Rate %", stats.win_rate],
          ["Profit Factor", stats.profit_factor],
          ["Sharpe Ratio", stats.sharpe_ratio],
          ["Max DD %", stats.max_drawdown_pct],
          ["CAGR %", stats.cagr_pct],
          ["Expectancy", stats.expectancy],
          ["Avg Return %", stats.average_return_pct],
        ].filter(([, v]) => typeof v === "number" && !Number.isNaN(v));

        if (metricEntries.length) {
          const metricsContainer = document.createElement("div");
          metricsContainer.className = "chart-card";
          chartsOutput.appendChild(metricsContainer);
          createChart(
            metricsContainer,
            "bar",
            {
              labels: metricEntries.map((m) => m[0]),
              datasets: [
                {
                  label: "Metrics",
                  data: metricEntries.map((m) => m[1]),
                  backgroundColor: "rgba(59,130,246,0.6)",
                  borderColor: "#2563eb",
                  borderWidth: 1,
                },
              ],
            },
            {
              indexAxis: "y",
              plugins: { legend: { display: false }, title: { display: true, text: "Performance Metrics" } },
            }
          );
        }
      }
    }

    function jumpToTrade(trades, index) {
      if (!Array.isArray(trades) || trades.length === 0) return;
      if (Number.isNaN(index) || index < 0 || index >= trades.length) {
        alert(`Trade index must be between 0 and ${trades.length - 1}.`);
        return;
      }
      const trade = trades[index];
      renderTradesList(trades, index);
      const target = document.querySelector(`[data-trade-idx="${index}"]`);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.classList.add("highlight");
        setTimeout(() => target.classList.remove("highlight"), 1200);
      }
    }

    if (tradeJumpForm && tradeJumpInput) {
      tradeJumpForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const idx = parseInt(tradeJumpInput.value, 10);
       const trades = (() => {
          return Array.isArray(latestTrades) ? latestTrades : null;
        })();
        if (trades) {
          jumpToTrade(trades, idx);
        }
      });
    }

    function renderTradesList(trades, expandedIndex = null) {
      latestTrades = trades;
      tradesOutput.innerHTML = "";
      if (!Array.isArray(trades) || trades.length === 0) {
        tradesOutput.textContent = "No trades recorded.";
        return;
      }
      trades.forEach((trade, i) => {
        const summary = document.createElement("div");
        summary.dataset.tradeIdx = i;
        summary.style.border = "1px solid #e2e8f0";
        summary.style.borderRadius = "6px";
        summary.style.padding = "0.5rem 0.75rem";
        summary.style.marginBottom = "0.5rem";
        summary.style.cursor = "pointer";
        summary.style.background = "#fff";
        const entryTime = trade.entry_time
          ? new Date(trade.entry_time)
          : null;
        const labelTime = entryTime
          ? `${entryTime.getFullYear()}-${String(entryTime.getMonth() + 1).padStart(2, "0")}-${String(
              entryTime.getDate()
            ).padStart(2, "0")} ${String(entryTime.getHours()).padStart(2, "0")}:${String(
              entryTime.getMinutes()
            ).padStart(2, "0")}`
          : "n/a";
        summary.textContent = `#${i} | ${labelTime} | PnL: ${trade.pnl?.toFixed ? trade.pnl.toFixed(2) : trade.pnl}`;

        const detail = document.createElement("pre");
        detail.style.display = expandedIndex === i ? "block" : "none";
        detail.style.background = "#0f172a";
        detail.style.color = "#f8fafc";
        detail.style.padding = "0.5rem";
        detail.style.borderRadius = "4px";
        detail.style.marginTop = "0.5rem";
        detail.textContent = JSON.stringify(trade, null, 2);

        summary.addEventListener("click", () => {
          const isVisible = detail.style.display === "block";
          detail.style.display = isVisible ? "none" : "block";
        });

        const wrapper = document.createElement("div");
        wrapper.appendChild(summary);
        wrapper.appendChild(detail);
        tradesOutput.appendChild(wrapper);
      });
    }

    form.addEventListener("submit", handleSubmit);
    if (historyForm) {
      historyForm.addEventListener("submit", handleHistory);
    }
  }

  window.initBacktestPage = initBacktestPage;
})();
