import { PRESETS } from "./constants.js";
import {
  applyFormValues,
  setDefaultDates,
  setLastMonthDates,
} from "./form-state.js";

function byId(id) {
  return document.getElementById(id);
}

function showSavedState(button) {
  const originalLabel = button.textContent;
  button.textContent = "Draft Saved";
  window.setTimeout(() => {
    button.textContent = originalLabel;
  }, 1200);
}

export function initializeFormControls({ form, draftStore, decisionLog }) {
  const applyPresetButton = byId("apply-preset");
  const presetSelect = byId("preset-select");
  const saveDraftButton = byId("save-draft");
  const resetButton = byId("reset-form");
  const lastMonthButton = byId("fill-last-month");
  const decisionLogFilter = byId("decision-log-filter");
  const decisionLogLimit = byId("decision-log-limit");

  applyFormValues(form, PRESETS.baseline);
  setDefaultDates(form);
  draftStore.load(form);

  applyPresetButton?.addEventListener("click", () => {
    const preset = PRESETS[presetSelect?.value] || PRESETS.baseline;
    applyFormValues(form, preset);
    setDefaultDates(form);
    draftStore.save(form);
  });

  saveDraftButton?.addEventListener("click", () => {
    draftStore.save(form);
    showSavedState(saveDraftButton);
  });

  resetButton?.addEventListener("click", () => {
    draftStore.clear();
    form.reset();
    applyFormValues(form, PRESETS.baseline);
    setDefaultDates(form);
  });

  lastMonthButton?.addEventListener("click", () => {
    setLastMonthDates(form);
    draftStore.save(form);
  });

  form.addEventListener("input", () => draftStore.save(form));
  form.addEventListener("change", () => draftStore.save(form));
  decisionLogFilter?.addEventListener("change", decisionLog.render);
  decisionLogLimit?.addEventListener("input", decisionLog.render);
}
