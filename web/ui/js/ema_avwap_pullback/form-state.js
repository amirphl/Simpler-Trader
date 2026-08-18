function isFormField(element) {
  return (
    element instanceof HTMLInputElement || element instanceof HTMLSelectElement
  );
}

export function toUtcInputValue(date) {
  return date.toISOString().slice(0, 16);
}

export function setDefaultDates(form) {
  const startInput = form.elements.namedItem("start");
  const endInput = form.elements.namedItem("end");
  if (!isFormField(startInput) || !isFormField(endInput)) return;

  const end = new Date();
  const start = new Date(end.getTime() - 90 * 24 * 60 * 60 * 1000);
  if (!endInput.value) endInput.value = toUtcInputValue(end);
  if (!startInput.value) startInput.value = toUtcInputValue(start);
}

export function setLastMonthDates(form) {
  const end = new Date();
  const start = new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);
  applyFormValues(form, {
    start: toUtcInputValue(start),
    end: toUtcInputValue(end),
  });
}

export function applyFormValues(form, values) {
  Object.entries(values || {}).forEach(([name, value]) => {
    const field = form.elements.namedItem(name);
    if (isFormField(field) && typeof value === "string") {
      field.value = value;
    }
  });
}

export function captureFormValues(form) {
  return Array.from(form.elements).reduce((values, element) => {
    if (element.name && isFormField(element)) {
      values[element.name] = element.value;
    }
    return values;
  }, {});
}

export function createDraftStore(storageKey) {
  return {
    save(form) {
      localStorage.setItem(storageKey, JSON.stringify(captureFormValues(form)));
    },

    load(form) {
      try {
        const raw = localStorage.getItem(storageKey);
        if (!raw) return;

        const values = JSON.parse(raw);
        if (values && typeof values === "object") {
          applyFormValues(form, values);
        }
      } catch (error) {
        console.warn("Failed to restore EMA+AVWAP draft", error);
      }
    },

    clear() {
      localStorage.removeItem(storageKey);
    },
  };
}
