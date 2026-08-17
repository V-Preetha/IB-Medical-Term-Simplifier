"use strict";

const reportSelect = document.querySelector("#report-select");
const textInput = document.querySelector("#ner-text");
const runButton = document.querySelector("#run-button");

function showAlert(message, kind = "danger") {
  const region = document.querySelector("#alert-region");
  region.replaceChildren();
  const alert = document.createElement("div");
  alert.className = `alert alert-${kind}`;
  alert.textContent = message;
  region.append(alert);
}

async function loadHealth() {
  const [healthResponse, modelsResponse] = await Promise.all([
    fetch("/api/v1/ner/health"),
    fetch("/api/v1/ner/models"),
  ]);
  const health = await healthResponse.json();
  const models = await modelsResponse.json();
  const model = models.models?.[0];
  const region = document.querySelector("#model-health");
  region.replaceChildren();
  const values = [
    ["Health", health.status],
    ["Model", model?.model_name],
    ["Revision", model?.model_revision],
    ["Device", model?.device],
    ["Provider status", model?.status],
  ];
  values.forEach(([label, value]) => {
    const line = document.createElement("p");
    line.className = "mb-2";
    const strong = document.createElement("strong");
    strong.textContent = `${label}: `;
    line.append(strong, document.createTextNode(value ?? "Unavailable"));
    region.append(line);
  });
}

async function loadReports() {
  const response = await fetch("/api/v1/ocr/recent");
  if (!response.ok) return;
  const payload = await response.json();
  payload.requests.filter((item) => item.status === "completed").forEach((item) => {
    reportSelect.add(new Option(`${item.request_id} - ${item.updated_at}`, item.request_id));
  });
}

reportSelect.addEventListener("change", async () => {
  if (!reportSelect.value) return;
  const response = await fetch(`/api/v1/ocr/${encodeURIComponent(reportSelect.value)}`);
  const payload = await response.json();
  if (!response.ok) {
    showAlert(payload.error?.message || "The OCR result is unavailable.");
    return;
  }
  textInput.value = payload.normalized_text;
});

function renderEntities(text, entities) {
  const pane = document.querySelector("#highlighted-text");
  pane.replaceChildren();
  let cursor = 0;
  [...entities].sort((a, b) => a.start - b.start).forEach((entity) => {
    if (entity.start < cursor) return;
    pane.append(document.createTextNode(text.slice(cursor, entity.start)));
    const mark = document.createElement("mark");
    mark.className = "entity-highlight";
    mark.textContent = text.slice(entity.start, entity.end);
    mark.title = `${entity.label} (${entity.confidence.toFixed(3)})`;
    pane.append(mark);
    cursor = entity.end;
  });
  pane.append(document.createTextNode(text.slice(cursor)));

  const table = document.querySelector("#entity-table");
  table.replaceChildren();
  entities.forEach((entity) => {
    const row = table.insertRow();
    [entity.text, entity.label, entity.start, entity.end, entity.confidence.toFixed(4)]
      .forEach((value) => {
        const cell = row.insertCell();
        cell.textContent = value;
      });
  });
}

function renderMetrics(payload) {
  const grid = document.querySelector("#metrics-grid");
  grid.replaceChildren();
  const metrics = {
    confidence: payload.confidence,
    processing_time_ms: payload.processing_time_ms,
    entity_count: payload.entities.length,
    tokens_per_second: payload.inference_metadata.tokens_per_second,
    review_required: payload.review_required,
    cache_hit: payload.cache_hit,
  };
  Object.entries(metrics).forEach(([name, value]) => {
    const column = document.createElement("div");
    column.className = "col-sm-6 col-lg-4";
    const card = document.createElement("div");
    card.className = "metric-card h-100";
    const label = document.createElement("span");
    label.textContent = name.replaceAll("_", " ");
    const strong = document.createElement("strong");
    strong.textContent = value === null ? "Not available" : String(value);
    card.append(label, strong);
    column.append(card);
    grid.append(column);
  });
}

runButton.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) {
    showAlert("Provide normalized OCR text before running NER.", "warning");
    return;
  }
  runButton.disabled = true;
  try {
    const response = await fetch("/api/v1/ner", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "NER inference failed.");
    renderEntities(payload.text, payload.entities);
    renderMetrics(payload);
    document.querySelector("#json-response").textContent = JSON.stringify(payload, null, 2);
    showAlert(`NER completed with ${payload.entities.length} entities.`, "success");
  } catch (error) {
    showAlert(error.message);
  } finally {
    runButton.disabled = false;
  }
});

Promise.all([loadHealth(), loadReports()]).catch((error) => showAlert(error.message));
