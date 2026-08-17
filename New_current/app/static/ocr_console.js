const ownerId = document.querySelector('meta[name="ocr-owner-id"]').content;
const headers = { "X-Owner-ID": ownerId };
const form = document.querySelector("#ocr-form");
const fileInput = document.querySelector("#document-file");
const dropZone = document.querySelector("#drop-zone");
const runButton = document.querySelector("#run-button");
const refreshButton = document.querySelector("#refresh-button");
const pdfPreview = document.querySelector("#pdf-preview");
const imagePreview = document.querySelector("#image-preview");
const previewEmpty = document.querySelector("#preview-empty");
let previewUrl = null;

fileInput.addEventListener("change", () => renderPreview(fileInput.files[0]));
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}
dropZone.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (files.length === 1) {
    fileInput.files = files;
    renderPreview(files[0]);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return showAlert("Choose a document before running OCR.", "danger");
  const body = new FormData();
  body.append("file", file);
  setRunning(true);
  try {
    const result = await requestJson("/api/v1/ocr", { method: "POST", headers, body });
    renderResult(result);
    showAlert("OCR completed successfully.", "success");
  } catch (error) {
    showAlert(error.message, "danger");
    document.querySelector("#pipeline-status").textContent = "Failed";
    document.querySelector("#pipeline-status").className = "badge text-bg-danger";
  } finally {
    setRunning(false);
    await refreshEngineeringData();
  }
});

refreshButton.addEventListener("click", refreshEngineeringData);

function renderPreview(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  pdfPreview.classList.add("d-none");
  imagePreview.classList.add("d-none");
  if (!file) return;
  previewUrl = URL.createObjectURL(file);
  previewEmpty.classList.add("d-none");
  if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
    pdfPreview.src = previewUrl;
    pdfPreview.classList.remove("d-none");
  } else {
    imagePreview.src = previewUrl;
    imagePreview.classList.remove("d-none");
  }
}

function renderResult(result) {
  document.querySelector("#pipeline-status").textContent = result.status;
  document.querySelector("#pipeline-status").className = "badge text-bg-success";
  const progress = document.querySelector("#pipeline-progress");
  progress.style.width = "100%";
  progress.textContent = "100%";
  document.querySelector("#document-type").textContent = result.document_type;
  document.querySelector("#confidence").textContent =
    result.confidence == null ? "Unavailable" : result.confidence.toFixed(4);
  document.querySelector("#processing-time").textContent = `${result.processing_time_ms.toFixed(1)} ms`;
  document.querySelector("#page-count").textContent = result.page_count;
  document.querySelector("#cache-status").textContent = result.cache_hit ? "Hit" : "Miss";
  document.querySelector("#request-id").textContent = result.request_id;
  document.querySelector("#raw-output").textContent = result.raw_text;
  document.querySelector("#normalized-output").textContent = result.normalized_text;
  const warnings = document.querySelector("#warnings");
  warnings.textContent = result.warnings.join(" ");
  warnings.classList.toggle("d-none", result.warnings.length === 0);
}

async function refreshEngineeringData() {
  const [health, models, recent, logs] = await Promise.allSettled([
    requestJson("/api/v1/ocr/health"),
    requestJson("/api/v1/ocr/models"),
    requestJson("/api/v1/ocr/recent?limit=20", { headers }),
    requestJson("/api/v1/ocr/logs?limit=100"),
  ]);
  renderHealth(health);
  renderModels(models);
  renderRecent(recent);
  renderLogs(logs);
}

function renderHealth(result) {
  const panel = document.querySelector("#health-panel");
  if (result.status === "rejected") return (panel.textContent = result.reason.message);
  panel.replaceChildren(...result.value.providers.map((provider) => {
    const row = document.createElement("div");
    row.className = "d-flex justify-content-between border-bottom py-2";
    row.textContent = `${provider.provider_kind}: ${provider.provider_name}`;
    const badge = document.createElement("span");
    badge.className = provider.health_status === "ready" ? "badge text-bg-success" : "badge text-bg-danger";
    badge.textContent = provider.health_status;
    row.append(badge);
    return row;
  }));
}

function renderModels(result) {
  const table = document.querySelector("#provider-table");
  if (result.status === "rejected") return (table.textContent = result.reason.message);
  table.replaceChildren(...result.value.models.map((model) => {
    const row = document.createElement("tr");
    for (const value of [model.provider_kind, model.provider_name, model.provider_version, model.model_revision || "—"]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  }));
}

function renderRecent(result) {
  const table = document.querySelector("#recent-table");
  if (result.status === "rejected") return (table.textContent = result.reason.message);
  table.replaceChildren(...result.value.requests.map((request) => {
    const row = document.createElement("tr");
    for (const value of [request.request_id, request.status, request.pipeline_stage, `${request.progress}%`, request.updated_at]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  }));
}

function renderLogs(result) {
  document.querySelector("#logs-viewer").textContent = result.status === "rejected"
    ? result.reason.message
    : result.value.records.map((record) => JSON.stringify(record)).join("\n");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || `Request failed (${response.status}).`);
  return payload;
}

function setRunning(running) {
  runButton.disabled = running;
  runButton.textContent = running ? "Running OCR…" : "Run OCR";
  if (running) {
    document.querySelector("#pipeline-status").textContent = "Running";
    document.querySelector("#pipeline-status").className = "badge text-bg-primary";
  }
}

function showAlert(message, kind) {
  document.querySelector("#alert-region").innerHTML = `<div class="alert alert-${kind}"></div>`;
  document.querySelector("#alert-region .alert").textContent = message;
}

refreshEngineeringData();
