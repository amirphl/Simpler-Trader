(function () {
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

    if (!form || !statusStream || !statsOutput || !tradesOutput) {
      console.error("Required DOM elements are missing from this page.");
      return;
    }

    let activeSocket = null;

    async function handleSubmit(event) {
      event.preventDefault();
      const payload = serializeForm(form);
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
      const jobId = formData.get("job_id");
      if (!jobId) return;
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
      const stats = result.report?.statistics || {};
      statsOutput.textContent = JSON.stringify(stats, null, 2);
      const trades = result.report?.trades || [];
      tradesOutput.textContent = trades.length ? JSON.stringify(trades, null, 2) : "No trades recorded.";
    }

    form.addEventListener("submit", handleSubmit);
    if (historyForm) {
      historyForm.addEventListener("submit", handleHistory);
    }
  }

  window.initBacktestPage = initBacktestPage;
})();

