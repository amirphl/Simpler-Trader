const form = document.getElementById("backtest-form");
const historyForm = document.getElementById("history-form");
const statusStream = document.getElementById("status-stream");
const statsOutput = document.getElementById("stats-output");
const tradesOutput = document.getElementById("trades-output");
const currentJob = document.getElementById("current-job");
let activeSocket = null;

function serializeForm(formEl) {
  const data = new FormData(formEl);
  const start = data.get("start");
  const end = data.get("end");
  const stopLossPct = data.get("stop_loss_pct");
  const tp = data.get("take_profit_pct");
  return {
    strategy: data.get("strategy"),
    start: start ? new Date(start).toISOString() : null,
    end: end ? new Date(end).toISOString() : null,
    initial_capital: parseFloat(data.get("initial_capital")),
    override_download: false,
    params: {
      symbol: data.get("symbol"),
      timeframe: data.get("timeframe"),
      window_size: parseInt(data.get("window_size"), 10),
      leverage: parseFloat(data.get("leverage")),
      take_profit_pct: tp ? parseFloat(tp) : 0.02,
      stop_loss_mode: data.get("stop_loss_mode"),
      stop_loss_pct: stopLossPct ? parseFloat(stopLossPct) : 0.005,
      skip_large_upper_wick: data.get("skip_large_upper_wick") === "true",
      skip_bollinger_cross: data.get("skip_bollinger_cross") === "true",
      enable_volume_pressure_filter: data.get("enable_volume_pressure_filter") === "true",
      enable_stochastic_filter: data.get("enable_stochastic_filter") === "true",
    },
  };
}

function openSocket(jobId) {
  if (activeSocket) {
    activeSocket.close();
  }
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/backtests/${jobId}`);
  socket.onmessage = (event) => handleEvent(JSON.parse(event.data));
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
  const report = result.report || {};
  const stats = report.statistics || {};
  statsOutput.textContent = JSON.stringify(stats, null, 2);
  const trades = report.trades || [];
  tradesOutput.textContent = trades.length ? JSON.stringify(trades, null, 2) : "No trades recorded.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = serializeForm(form);
  statsOutput.textContent = "Awaiting result...";
  tradesOutput.textContent = "Awaiting result...";
  statusStream.textContent = "Submitting job...";
  statusStream.dataset.state = "queued";
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
  currentJob.textContent = `Tracking job ${job.job_id}`;
  statusStream.textContent = `Job ${job.job_id}: ${job.status.toUpperCase()}`;
  openSocket(job.job_id);
});

historyForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const jobId = new FormData(historyForm).get("job_id");
  if (!jobId) return;
  statusStream.textContent = `Fetching result for ${jobId}...`;
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
});

