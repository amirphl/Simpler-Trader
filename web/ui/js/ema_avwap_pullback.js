import { STORAGE_KEY } from "./ema_avwap_pullback/constants.js";
import { initializeFormControls } from "./ema_avwap_pullback/form-controls.js";
import { createDraftStore } from "./ema_avwap_pullback/form-state.js";
import {
  createDecisionLogRenderer,
  formatNumber,
  renderConfigSnapshot,
  renderPerformance,
  renderQuickMetrics,
  renderRobustAnalysis,
} from "./ema_avwap_pullback/renderers.js";
import { serializeBacktestForm } from "./ema_avwap_pullback/serializer.js";

function updateStatusTooltip(decisionLog) {
  const statusStream = document.getElementById("status-stream");
  const entries = decisionLog.getEntries();
  if (!statusStream) return;
  if (!entries.length) {
    statusStream.removeAttribute("title");
    return;
  }

  const lastEvent = entries[entries.length - 1] || {};
  const eventLabel = lastEvent.event
    ? String(lastEvent.event).replaceAll("_", " ")
    : "n/a";
  statusStream.title = [
    `Last event: ${eventLabel}`,
    `Timestamp: ${lastEvent.timestamp || "n/a"}`,
    `Setup: ${lastEvent.setup_type || "n/a"}`,
    `VWAP: ${formatNumber(lastEvent.vwap_middle_line, 5)}`,
  ].join("\n");
}

function renderResultDetails(result, decisionLog) {
  const statistics = result?.report?.statistics || {};
  decisionLog.setEntries(statistics.decision_log);
  renderQuickMetrics(result);
  renderPerformance(result);
  renderRobustAnalysis(result);
  renderConfigSnapshot(result);
  updateStatusTooltip(decisionLog);
}

function initializePage() {
  const decisionLog = createDecisionLogRenderer();
  const draftStore = createDraftStore(STORAGE_KEY);

  window.initBacktestPage({
    serializeForm(form) {
      draftStore.save(form);
      return serializeBacktestForm(form);
    },
    onReady({ form }) {
      initializeFormControls({ form, draftStore, decisionLog });
    },
    onResult(result) {
      renderResultDetails(result, decisionLog);
    },
  });
}

document.addEventListener("DOMContentLoaded", initializePage);
