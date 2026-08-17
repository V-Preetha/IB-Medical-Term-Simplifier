(() => {
  "use strict";

  // ---------------------------------------------------------------------
  // This console calls only the existing production APIs. It never
  // reimplements OCR/NER/simplification/verification/translation logic and
  // never fabricates a metric the backend does not return.
  // ---------------------------------------------------------------------

  const byId = (id) => document.getElementById(id);
  const alertBox = byId("demo-alert");
  const NOT_EXPOSED = "NOT EXPOSED";

  const state = {
    ocr: null,
    ner: null,
    embeddings: null,
    simplification: null,
    verification: {}, // level -> response
    translation: null,
    apiLog: [],
    perf: [], // {stage, latencyMs, model, device}
    firstResultAt: null,
    runStartedAt: null,
    translationLanguages: {},
    runtime: null, // last /api/v1/runtime/metrics response
    runtimeByStage: {}, // stage name -> ModelRuntimeStatus
  };

  const PERF_ORDER = [
    "Upload / Read",
    "Decode / Render",
    "OCR inference (Qwen3-VL)",
    "OCR Post-processing",
    "NER (biomedical-ner-all)",
    "Embeddings (BioClinical ModernBERT)",
    "Simplification (Qwen3-0.6B)",
    "Verification · clinical",
    "Verification · general_public",
    "Verification · child_friendly",
    "Translation (IndicTrans2)",
  ];

  // ------------------------- generic helpers ----------------------------

  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
    })[character]);

  const showAlert = (message, style = "danger") => {
    alertBox.textContent = message;
    alertBox.className = `alert alert-${style}`;
  };

  const clearAlert = () => {
    alertBox.textContent = "";
    alertBox.className = "alert d-none";
  };

  const fmtMs = (value) => (value === null || value === undefined ? NOT_EXPOSED : `${Number(value).toFixed(2)} ms`);
  const fmtNum = (value, digits = 4) => (value === null || value === undefined ? NOT_EXPOSED : Number(value).toFixed(digits));
  const fmtBytes = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    const units = ["KiB", "MiB", "GiB"];
    let value = bytes / 1024;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }
    return `${value.toFixed(1)} ${units[unitIndex]}`;
  };
  const orDash = (value) => (value === null || value === undefined || value === "" ? "—" : value);
  const isHealthy = (body) => ["healthy", "ready"].includes(String(body?.status || "").toLowerCase());

  // Structured API call: records every request/response for the Raw API
  // Inspector and never leaks a stack trace, only the backend's safe error
  // envelope: {error: {code, message}, request_id}.
  async function callApi(stage, path, options = {}) {
    const entry = {
      stage,
      method: options.method || "GET",
      url: path,
      requestBody: options.__loggedBody ?? null,
      httpStatus: null,
      requestId: null,
      responseBody: null,
      ok: false,
      startedAt: performance.now(),
    };
    state.apiLog.push(entry);
    renderApiInspector();
    let response;
    try {
      response = await fetch(path, options);
    } catch (networkError) {
      entry.errorMessage = "Network request failed.";
      renderApiInspector();
      throw { stage, httpStatus: null, code: "network_error", message: networkError.message, requestId: null };
    }
    entry.httpStatus = response.status;
    entry.elapsedMs = performance.now() - entry.startedAt;
    let body = {};
    try {
      body = await response.json();
    } catch {
      body = {};
    }
    entry.responseBody = body;
    entry.requestId = body.request_id || response.headers.get("X-Request-ID") || null;
    entry.ok = response.ok;
    renderApiInspector();
    if (!response.ok) {
      throw {
        stage,
        httpStatus: response.status,
        code: body?.error?.code || "unknown_error",
        message: body?.error?.message || `Request failed (${response.status}).`,
        requestId: entry.requestId,
      };
    }
    return body;
  }

  function loggedJson(payload) {
    return {
      __loggedBody: payload,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    };
  }

  // ------------------------- status badges -------------------------------

  const STATUS_CLASS = {
    WAITING: "status-loading",
    RUNNING: "status-running",
    PASS: "status-ready",
    REVIEW: "status-pending",
    BLOCKED: "status-blocked",
    FAILED: "status-blocked",
    DEFERRED: "status-frozen",
  };

  function setOverview(stage, label) {
    const el = byId(`overview-${stage}`);
    if (!el) return;
    el.textContent = label;
    el.className = `status ${STATUS_CLASS[label] || "status-loading"}`;
  }

  function setStageState(elementId, label) {
    const el = byId(elementId);
    if (!el) return;
    el.textContent = label;
    el.className = `status ${STATUS_CLASS[label] || "status-loading"}`;
  }

  function setTracker(stage, label) {
    const item = document.querySelector(`#progress-tracker li[data-stage="${stage}"] .tracker-status`);
    if (!item) return;
    item.textContent = label;
    item.className = `status tracker-status ${STATUS_CLASS[label] || "status-loading"}`;
  }

  function setReadyBadge(elementId, ready) {
    const el = byId(elementId);
    if (!el) return;
    el.textContent = ready ? "Production Ready" : "Architecture Complete";
    el.className = ready ? "status status-ready" : "status status-architecture";
  }

  // ------------------------------ tabs ------------------------------------

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tab-target]");
    if (!button) return;
    const nav = button.closest(".nav-tabs");
    const content = nav.nextElementSibling;
    nav.querySelectorAll(".nav-link").forEach((link) => link.classList.toggle("active", link === button));
    content.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.toggle("show", pane.id === button.dataset.tabTarget));
  });

  // --------------------------- health / models -----------------------------

  async function loadOcrStatus() {
    const [health, models] = await Promise.allSettled([
      callApi("ocr-health", "/api/v1/ocr/health"),
      callApi("ocr-models", "/api/v1/ocr/models"),
    ]);
    const healthOk = health.status === "fulfilled" && isHealthy(health.value);
    setReadyBadge("ocr-state", healthOk);
    if (health.status === "fulfilled") byId("ocr-health").textContent = health.value.status;
    if (models.status === "fulfilled") {
      const model = (models.value.models || []).find((item) => item.provider_kind === "ocr") || {};
      state.ocrModel = model;
      byId("ocr-provider").textContent = orDash(model.provider_name);
      byId("ocr-repo").textContent = orDash(model.configuration?.model_name);
      byId("ocr-revision").textContent = orDash(model.model_revision || model.configuration?.model_revision);
      byId("ocr-device").textContent = orDash(model.configuration?.resolved_device || model.configuration?.device);
      byId("ocr-dtype").textContent = orDash(model.configuration?.dtype);
    }
    return healthOk;
  }

  async function loadNerStatus() {
    const [health, models] = await Promise.allSettled([
      callApi("ner-health", "/api/v1/ner/health"),
      callApi("ner-models", "/api/v1/ner/models"),
    ]);
    const healthOk = health.status === "fulfilled" && isHealthy(health.value);
    setReadyBadge("ner-state", healthOk);
    if (health.status === "fulfilled") byId("ner-health").textContent = health.value.status;
    if (models.status === "fulfilled") {
      const model = (models.value.models || [])[0] || {};
      state.nerModel = model;
      byId("ner-model").textContent = orDash(model.model_name);
      byId("ner-revision").textContent = orDash(model.model_revision);
      byId("ner-device").textContent = orDash(model.device);
    }
    return healthOk;
  }

  async function loadEmbeddingStatus() {
    const [health, models] = await Promise.allSettled([
      callApi("embeddings-health", "/api/v1/embeddings/health"),
      callApi("embeddings-models", "/api/v1/embeddings/models"),
    ]);
    const healthOk = health.status === "fulfilled" && isHealthy(health.value);
    byId("embedding-state").textContent = healthOk ? "Ready" : "Not Ready";
    byId("embedding-state").className = healthOk ? "status status-ready" : "status status-pending";
    setOverview("embeddings", healthOk ? "PASS" : "DEFERRED");
    if (models.status === "fulfilled") {
      const model = (models.value.models || [])[0] || {};
      state.embeddingModel = model;
      byId("embedding-model").textContent = orDash(model.model_name);
      byId("embedding-revision").textContent = orDash(model.model_revision);
      byId("embedding-health").textContent = orDash(model.status);
      byId("embedding-device").textContent = orDash(model.device);
      byId("embedding-dimension").textContent = orDash(model.dimensions);
      byId("embedding-pooling").textContent = orDash(model.pooling_method);
      byId("embedding-normalized").textContent = model.normalized === undefined ? NOT_EXPOSED : String(model.normalized);
    }
    byId("run-embeddings").disabled = !(healthOk && state.ocr);
    return healthOk;
  }

  async function loadSimplificationStatus() {
    const result = await callApi("simplification-health", "/api/v1/simplify/health").catch((error) => ({ __error: error }));
    const healthy = !result.__error && result.status === "HEALTHY";
    byId("simplification-health").textContent = result.__error ? `HTTP ${result.__error.httpStatus}` : result.status;
    if (!result.__error) {
      byId("simplification-model").textContent = orDash(result.model_name);
    }
    return healthy;
  }

  async function loadVerificationStatus() {
    const result = await callApi("verification-health", "/api/v1/verification/health").catch((error) => ({ __error: error }));
    const healthy = !result.__error && String(result.status).toLowerCase() === "ready";
    byId("verification-health").textContent = result.__error ? `HTTP ${result.__error.httpStatus}` : result.status;
    if (!result.__error) {
      byId("verification-model").textContent = orDash(result.model_name);
      byId("verification-revision").textContent = orDash(result.model_revision);
      byId("verification-device").textContent = orDash(result.device);
      byId("verification-license").textContent = orDash(result.configuration?.license_status);
    }
    return healthy;
  }

  async function loadTranslationStatus() {
    const result = await callApi("translation-health", "/api/v1/translations/health").catch((error) => ({ __error: error }));
    const healthy = !result.__error && String(result.status).toLowerCase() === "ready";
    byId("translation-health").textContent = result.__error ? `HTTP ${result.__error.httpStatus}` : result.status;
    if (!result.__error) {
      byId("translation-model").textContent = orDash(result.model_name);
      byId("translation-revision").textContent = orDash(result.model_revision);
      byId("translation-device").textContent = orDash(result.device);
      state.translationDevice = result.device;
      state.translationLanguages = result.supported_languages || {};
      populateLanguageSelect(byId("target-language"), state.translationLanguages);
      populateLanguageSelect(byId("full-target-language"), state.translationLanguages);
    }
    return healthy;
  }

  function populateLanguageSelect(select, languages) {
    if (!select || select.dataset.populated) return;
    const entries = Object.entries(languages);
    if (!entries.length) return;
    select.replaceChildren(...entries.map(([code, name]) => new Option(name, code)));
    select.dataset.populated = "true";
  }

  async function loadPendingStage(prefix, endpoint) {
    try {
      const result = await callApi(`${prefix}-health`, endpoint);
      const provider = (result.providers || [])[0] || {};
      byId(`${prefix}-health`).textContent = result.status || provider.status || NOT_EXPOSED;
      byId(`${prefix}-detail`).textContent = provider.detail || byId(`${prefix}-detail`).textContent;
    } catch (error) {
      byId(`${prefix}-health`).textContent = `HTTP ${error.httpStatus ?? "?"}`;
    }
  }

  async function loadInfrastructureStatus() {
    try {
      const result = await callApi("infrastructure-health", "/api/v1/infrastructure/health");
      const ready = isHealthy(result);
      setReadyBadge("infrastructure-state", ready);
      const mappings = { postgresql: "postgresql-state", redis: "redis-state", celery: "celery-state" };
      Object.entries(mappings).forEach(([component, elementId]) => {
        byId(elementId).textContent = result.components?.[component]?.status || "Not Configured";
      });
    } catch {
      // Infrastructure is optional for this console; leave defaults visible.
    }
  }

  async function refreshAllStatuses() {
    const [ocrOk, nerOk, simplificationOk, verificationOk, translationOk] = await Promise.all([
      loadOcrStatus().catch(() => false),
      loadNerStatus().catch(() => false),
      loadSimplificationStatus().catch(() => false),
      loadVerificationStatus().catch(() => false),
      loadTranslationStatus().catch(() => false),
    ]);
    await Promise.allSettled([
      loadEmbeddingStatus(),
      loadPendingStage("linking", "/api/v1/entity-linking/health"),
      loadPendingStage("relation", "/api/v1/relation-extraction/health"),
      loadInfrastructureStatus(),
    ]);
    setOverview("linking", "DEFERRED");
    setOverview("relation", "DEFERRED");
    if (!state.ocr) setOverview("ocr", ocrOk ? "WAITING" : "FAILED");
    if (!state.ner) setOverview("ner", nerOk ? "WAITING" : "FAILED");
    if (!state.simplification) setOverview("simplification", simplificationOk ? "WAITING" : "FAILED");
    if (!Object.keys(state.verification).length) setOverview("verification", verificationOk ? "WAITING" : "FAILED");
    if (!state.translation) setOverview("translation", translationOk ? "WAITING" : "FAILED");
    loadRuntimeMetrics();
  }

  // ---------------------------- document input -----------------------------

  const fileInput = byId("report-file");
  const dropZone = byId("drop-zone");
  const pdfPreview = byId("pdf-preview");
  const imagePreview = byId("image-preview");
  const previewEmpty = byId("preview-empty");
  let previewUrl = null;

  fileInput.addEventListener("change", () => describeDocument(fileInput.files[0]));
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
      describeDocument(files[0]);
    }
  });

  function describeDocument(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    pdfPreview.classList.add("d-none");
    imagePreview.classList.add("d-none");
    if (!file) {
      previewEmpty.classList.remove("d-none");
      return;
    }
    byId("doc-filename").textContent = file.name;
    byId("doc-mime").textContent = file.type || "application/octet-stream";
    byId("doc-size").textContent = fmtBytes(file.size);
    byId("doc-pages").textContent = "Pending OCR";
    byId("doc-hash").textContent = NOT_EXPOSED;
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

  // -------------------------------- OCR ------------------------------------

  async function runOcr(file) {
    setStageState("ocr-state", "RUNNING");
    setOverview("ocr", "RUNNING");
    setTracker("ocr", "RUNNING");
    const form = new FormData();
    form.append("file", file);
    const startedAt = performance.now();
    const body = await callApi("ocr", "/api/v1/ocr", { method: "POST", body: form, __loggedBody: { file: file.name } });
    recordFirstResult(startedAt);
    renderOcr(body);
    await loadOcrStatus();
    setStageState("ocr-state", "PASS");
    setOverview("ocr", "PASS");
    setTracker("ocr", "PASS");
    return body;
  }

  function renderOcr(body) {
    state.ocr = body;
    byId("ocr-health").textContent = "ready";
    byId("ocr-page-count").textContent = body.page_count;
    byId("doc-pages").textContent = body.page_count;
    byId("ocr-doc-type").textContent = body.document_type;
    byId("ocr-confidence").textContent = body.confidence === null ? NOT_EXPOSED : fmtNum(body.confidence);
    byId("ocr-confidence-method").textContent = orDash(body.confidence_method);
    byId("ocr-review").textContent = String(body.review_required);
    byId("ocr-cache").textContent = body.cache_hit ? "Hit" : "Miss";
    byId("ocr-latency").textContent = fmtMs(body.processing_time_ms);
    byId("ocr-request-id").textContent = body.request_id;
    byId("ocr-ocr-id").textContent = body.ocr_id;
    byId("ocr-report-id").textContent = body.report_id;
    byId("ocr-pipeline-version").textContent = body.pipeline_version;
    byId("ocr-normalized").value = body.normalized_text;
    byId("ocr-raw").value = body.raw_text;
    byId("ocr-json").textContent = JSON.stringify(body, null, 2);
    const uploadMs = body.metadata?.upload_time_ms ?? null;
    const decodeMs = body.metadata?.decode_time_ms ?? null;
    const ocrTotalMs = body.metadata?.ocr_processing_time_ms ?? null;
    const inferenceMs = decodeMs !== null && ocrTotalMs !== null
      ? Math.max(ocrTotalMs - decodeMs, 0)
      : ocrTotalMs;
    byId("ocr-stats").textContent = JSON.stringify({
      page_count: body.page_count,
      processing_time_ms: body.processing_time_ms,
      upload_time_ms: uploadMs ?? NOT_EXPOSED,
      decode_time_ms: decodeMs ?? NOT_EXPOSED,
      ocr_inference_time_ms: inferenceMs ?? NOT_EXPOSED,
      ocr_processing_time_ms: ocrTotalMs ?? NOT_EXPOSED,
      postprocessing_time_ms: body.metadata?.postprocessing_time_ms ?? NOT_EXPOSED,
      cache_hit: body.cache_hit,
      warnings: body.warnings,
    }, null, 2);
    const warnings = byId("ocr-warnings");
    warnings.textContent = body.warnings.join(" ");
    warnings.classList.toggle("d-none", body.warnings.length === 0);
    byId("run-ner").disabled = false;
    recordPerf("Upload / Read", uploadMs, "client -> server");
    recordPerf("Decode / Render", decodeMs, "cpu");
    recordPerf("OCR inference (Qwen3-VL)", inferenceMs, state.ocrModel?.configuration?.resolved_device);
    recordPerf("OCR Post-processing", body.metadata?.postprocessing_time_ms ?? null, "symspell");
    renderPerformancePanel();
    applyStageWarmth("ocr", "ocr-warm");
    loadRuntimeMetrics();
  }

  // -------------------------------- NER ------------------------------------

  async function runNer(text) {
    setStageState("ner-state", "RUNNING");
    setOverview("ner", "RUNNING");
    setTracker("ner", "RUNNING");
    const body = await callApi("ner", "/api/v1/ner", loggedJson({ text }));
    renderNer(body);
    await loadNerStatus();
    setStageState("ner-state", "PASS");
    setOverview("ner", "PASS");
    setTracker("ner", "PASS");
    return body;
  }

  function renderNer(body) {
    state.ner = body;
    byId("ner-latency").textContent = fmtMs(body.processing_time_ms);
    byId("ner-count").textContent = body.entities.length;
    byId("ner-confidence").textContent = body.confidence === null ? NOT_EXPOSED : fmtNum(body.confidence);
    const confidences = body.entities.map((entity) => entity.confidence);
    byId("ner-minmax").textContent = confidences.length
      ? `${fmtNum(Math.min(...confidences))} / ${fmtNum(Math.max(...confidences))} (computed from response entities)`
      : NOT_EXPOSED;
    byId("ner-tokens").textContent = orDash(body.inference_metadata.token_count);
    byId("ner-tps").textContent = body.inference_metadata.tokens_per_second === null ? NOT_EXPOSED : fmtNum(body.inference_metadata.tokens_per_second, 2);
    byId("ner-review").textContent = String(body.review_required);
    byId("ner-cache").textContent = body.cache_hit ? "Hit" : "Miss";
    byId("ner-json").textContent = JSON.stringify(body, null, 2);
    byId("ner-stats").textContent = JSON.stringify({
      entity_count: body.entities.length,
      by_label: countByLabel(body.entities),
      confidence_api: body.confidence,
      processing_time_ms: body.processing_time_ms,
      token_count: body.inference_metadata.token_count,
      tokens_per_second: body.inference_metadata.tokens_per_second,
    }, null, 2);
    const byLabel = countByLabel(body.entities);
    byId("ner-label-counts").innerHTML = Object.keys(byLabel).length
      ? Object.entries(byLabel).map(([label, count]) => `<span class="entity-chip">${escapeHtml(label)}: ${count}</span>`).join("")
      : '<span class="text-secondary">No entities detected.</span>';
    renderEntityTable(body.entities);
    renderHighlightedText(body.text, body.entities);
    byId("run-simplification").disabled = false;
    recordPerf("NER (biomedical-ner-all)", body.processing_time_ms, body.inference_metadata.device);
    renderPerformancePanel();
    applyStageWarmth("ner", "ner-warm");
    loadRuntimeMetrics();
  }

  function countByLabel(entities) {
    const counts = {};
    entities.forEach((entity) => { counts[entity.label] = (counts[entity.label] || 0) + 1; });
    return counts;
  }

  function renderEntityTable(entities) {
    const table = byId("entity-table");
    table.innerHTML = entities.length
      ? entities.map((entity) => `<tr><td>${escapeHtml(entity.text)}</td><td><span class="entity-badge entity-badge-${labelSlug(entity.label)}">${escapeHtml(entity.label)}</span></td><td>${entity.start}</td><td>${entity.end}</td><td>${fmtNum(entity.confidence)}</td></tr>`).join("")
      : '<tr><td colspan="5" class="text-secondary">No entities detected.</td></tr>';
  }

  function labelSlug(label) {
    return String(label).toLowerCase().replace(/[^a-z]+/g, "-");
  }

  function renderHighlightedText(text, entities) {
    const pane = byId("highlighted-text");
    pane.replaceChildren();
    let cursor = 0;
    [...entities].sort((a, b) => a.start - b.start).forEach((entity) => {
      if (entity.start < cursor) return;
      pane.append(document.createTextNode(text.slice(cursor, entity.start)));
      const mark = document.createElement("mark");
      mark.className = `entity-highlight entity-badge-${labelSlug(entity.label)}`;
      mark.textContent = text.slice(entity.start, entity.end);
      mark.title = `${entity.label} (${entity.confidence.toFixed(3)})`;
      pane.append(mark);
      cursor = entity.end;
    });
    pane.append(document.createTextNode(text.slice(cursor)));
  }

  // ---------------------------- Embeddings ----------------------------------

  async function runEmbeddings(text, inputId) {
    setOverview("embeddings", "RUNNING");
    try {
      const body = await callApi("embeddings", "/api/v1/embeddings", loggedJson({ inputs: [{ input_id: inputId, text }] }));
      renderEmbeddings(body);
      setOverview("embeddings", "PASS");
    } catch (error) {
      byId("embedding-preview").textContent = `Background embedding request failed: ${error.message}`;
      setOverview("embeddings", "FAILED");
    }
  }

  function renderEmbeddings(body) {
    state.embeddings = body;
    const item = (body.embeddings || [])[0];
    byId("embedding-latency").textContent = fmtMs(body.processing_time_ms);
    byId("embedding-throughput").textContent = body.tokens_per_second === null ? NOT_EXPOSED : `${fmtNum(body.tokens_per_second, 2)} tok/s`;
    byId("embedding-model").textContent = orDash(body.reproducibility?.model_name);
    byId("embedding-revision").textContent = orDash(body.reproducibility?.model_revision);
    byId("embedding-device").textContent = orDash(body.reproducibility?.device);
    byId("embedding-dimension").textContent = orDash(item?.dimensions ?? body.reproducibility?.dimensions);
    byId("embedding-pooling").textContent = orDash(body.reproducibility?.pooling_method);
    byId("embedding-normalized").textContent = String(body.reproducibility?.normalized);
    byId("embedding-norm").textContent = item ? fmtNum(item.vector_norm) : NOT_EXPOSED;
    byId("embedding-tokens").textContent = item ? item.token_count : NOT_EXPOSED;
    byId("embedding-preview").textContent = item
      ? JSON.stringify({ input_id: item.input_id, dimensions: item.dimensions, vector_norm: item.vector_norm, preview: item.vector.slice(0, 12) }, null, 2)
      : "No embedding returned.";
    byId("embedding-full-json").textContent = JSON.stringify(body, null, 2);
    recordPerf("Embeddings (BioClinical ModernBERT)", body.processing_time_ms, body.reproducibility?.device);
    renderPerformancePanel();
    applyStageWarmth("embeddings", "embedding-warm");
    loadRuntimeMetrics();
  }

  byId("embedding-full-json-toggle").addEventListener("click", () => {
    const pre = byId("embedding-full-json");
    const showing = !pre.classList.contains("d-none");
    pre.classList.toggle("d-none", showing);
    byId("embedding-full-json-toggle").textContent = showing ? "Show Full JSON" : "Hide Full JSON";
  });

  // ----------------------------- Simplification ------------------------------

  const LEVELS = [
    { key: "clinical", label: "Level 1 · Clinical" },
    { key: "general_public", label: "Level 2 · General Public" },
    { key: "child_friendly", label: "Level 3 · Child-Friendly" },
  ];

  async function runSimplification(text, entities) {
    setStageState("simplification-state", "RUNNING");
    setOverview("simplification", "RUNNING");
    setTracker("simplification", "RUNNING");
    const body = await callApi("simplification", "/api/v1/simplify", loggedJson({ text, entities, linked_concepts: [] }));
    renderSimplification(body);
    setStageState("simplification-state", "PASS");
    setOverview("simplification", "PASS");
    setTracker("simplification", "PASS");
    return body;
  }

  function renderSimplification(body) {
    state.simplification = body;
    byId("simplification-model").textContent = orDash(body.inference.model_name);
    byId("simplification-revision").textContent = orDash(body.inference.model_revision);
    byId("simplification-device").textContent = orDash(body.inference.device);
    byId("simplification-prompt-version").textContent = orDash(body.inference.prompt_version);
    byId("simplification-prompt-tokens").textContent = body.inference.prompt_tokens;
    byId("simplification-output-tokens").textContent = body.inference.output_tokens;
    byId("simplification-generation-time").textContent = fmtMs(body.inference.generation_time_ms);
    byId("simplification-total-time").textContent = fmtMs(body.processing_time_ms);
    byId("simplification-cache").textContent = body.cache_hit ? "Hit" : "Miss";

    const container = byId("simplification-levels");
    container.innerHTML = LEVELS.map(({ key, label }, index) => {
      const level = body[key];
      const terms = level.medical_terms_explained
        .map((item) => `<li><strong>${escapeHtml(item.term)}</strong>: ${escapeHtml(item.explanation)}</li>`)
        .join("") || '<li class="text-secondary">None returned.</li>';
      const findings = level.important_findings.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || '<li class="text-secondary">None returned.</li>';
      const questions = level.suggested_questions_for_doctor.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || '<li class="text-secondary">None returned.</li>';
      const warnings = level.warnings.length ? `<div class="alert alert-warning py-2 mb-3">${level.warnings.map(escapeHtml).join("<br>")}</div>` : "";
      return `<div class="tab-pane${index === 0 ? " show" : ""}" id="level-tab-${key}">
        <div class="metadata-grid mb-3">
          <div><span>Confidence (grounding)</span><strong>${level.confidence === null ? NOT_EXPOSED : fmtNum(level.confidence)}</strong></div>
          <div><span>Confidence method</span><strong>${escapeHtml(level.confidence_method)}</strong></div>
          <div><span>Review required</span><strong>${level.review_required}</strong></div>
          <div><span>Model revision</span><strong>${escapeHtml(level.model_revision)}</strong></div>
          <div><span>Prompt version</span><strong>${escapeHtml(level.prompt_version)}</strong></div>
        </div>
        ${warnings}
        <label class="form-label fw-semibold">Simplified report</label>
        <textarea class="form-control output-text mb-3" rows="5" readonly>${escapeHtml(level.simplified_report)}</textarea>
        <div class="row g-3">
          <div class="col-md-4"><strong>Important findings</strong><ul>${findings}</ul></div>
          <div class="col-md-4"><strong>Medical terms explained</strong><ul>${terms}</ul></div>
          <div class="col-md-4"><strong>Questions for doctor</strong><ul>${questions}</ul></div>
        </div>
      </div>`;
    }).join("") + `<div class="tab-pane" id="simplification-tab-json"><pre class="json-output">${escapeHtml(JSON.stringify(body, null, 2))}</pre></div>`;

    byId("run-verification").disabled = false;
    recordPerf("Simplification (Qwen3-0.6B)", body.inference.generation_time_ms, body.inference.device);
    renderPerformancePanel();
    applyStageWarmth("simplification", "simplification-warm");
    loadRuntimeMetrics();
  }

  // ----------------------------- Verification ---------------------------------

  const CHECK_LABELS = {
    numeric_or_unit: "Numeric / dosage / unit / percentage / date values",
    medication_frequency: "Medication frequency",
    negation: "Negation",
    laterality: "Laterality",
  };

  async function runVerificationForLevel(levelKey, premise, hypothesis) {
    const body = await callApi(`verification-${levelKey}`, "/api/v1/verification", loggedJson({ premise, hypothesis }));
    state.verification[levelKey] = body;
    return body;
  }

  async function runVerificationAll(premise, simplification) {
    setStageState("verification-state", "RUNNING");
    setOverview("verification", "RUNNING");
    setTracker("verification", "RUNNING");
    const results = {};
    for (const { key } of LEVELS) {
      try {
        results[key] = await runVerificationForLevel(key, premise, simplification[key].simplified_report);
      } catch (error) {
        results[key] = { __error: error };
      }
    }
    renderVerification(results);
    await loadVerificationStatus();
    const overall = overallVerificationState(results);
    setStageState("verification-state", overall);
    setOverview("verification", overall);
    setTracker("verification", overall);
    return results;
  }

  function overallVerificationState(results) {
    const verdicts = Object.values(results).map((item) => item.verification).filter(Boolean);
    if (!verdicts.length) return "FAILED";
    if (verdicts.some((v) => v === "BLOCKED")) return "BLOCKED";
    if (verdicts.some((v) => v === "REVIEW")) return "REVIEW";
    return "PASS";
  }

  function renderVerification(results) {
    const container = byId("verification-cards");
    container.innerHTML = LEVELS.map(({ key, label }) => {
      const result = results[key];
      if (!result || result.__error) {
        const error = result?.__error;
        return `<div class="col-lg-4"><div class="verification-card verification-failed">
          <h4 class="h6">${label}</h4>
          <span class="status status-blocked">FAILED</span>
          <p class="small mb-0 mt-2">${escapeHtml(error?.message || "Verification did not run.")}</p>
          <p class="small text-secondary mb-0">Request ID: ${escapeHtml(error?.requestId || NOT_EXPOSED)}</p>
        </div></div>`;
      }
      const verdictClass = { PASS: "verification-pass", REVIEW: "verification-review", BLOCKED: "verification-blocked" }[result.verification] || "";
      const mismatches = result.deterministic_mismatches.length
        ? result.deterministic_mismatches.map((item) => `<li><strong>${escapeHtml(CHECK_LABELS[item.check] || item.check)}</strong>: missing [${item.missing.map(escapeHtml).join(", ") || "none"}], added [${item.added.map(escapeHtml).join(", ") || "none"}]</li>`).join("")
        : '<li class="text-secondary">No deterministic mismatches.</li>';
      const probabilities = Object.entries(result.probabilities)
        .map(([label2, value]) => `<div><span>${escapeHtml(label2)}</span><strong>${fmtNum(value, 4)}</strong></div>`)
        .join("");
      return `<div class="col-lg-4"><div class="verification-card ${verdictClass}">
        <div class="d-flex justify-content-between align-items-start">
          <h4 class="h6">${label}</h4>
          <span class="status ${STATUS_CLASS[result.verification] || ""}">${result.verification}</span>
        </div>
        <p class="small mb-1"><strong>NLI label:</strong> ${escapeHtml(result.label)}</p>
        <div class="metadata-grid mb-2">${probabilities}</div>
        <p class="small mb-1"><strong>Review required:</strong> ${result.review_required}</p>
        <p class="small mb-1"><strong>Model revision:</strong> ${escapeHtml(result.model_revision)}</p>
        <p class="small mb-1"><strong>NLI latency:</strong> ${fmtMs(result.nli_latency_ms)} &middot; <strong>Total:</strong> ${fmtMs(result.processing_time_ms)}</p>
        <strong class="d-block mt-2">Deterministic mismatches</strong>
        <ul class="mb-0">${mismatches}</ul>
      </div></div>`;
    }).join("");

    LEVELS.forEach(({ key }) => {
      recordPerf(`Verification · ${key}`, results[key]?.processing_time_ms ?? null, "PubMedBERT-MNLI-MedNLI");
    });
    renderPerformancePanel();
    renderSafetyPanel(results);
    populateTranslateLevelOptions(results);
    applyStageWarmth("verification", "verification-warm");
    loadRuntimeMetrics();
  }

  function renderSafetyPanel(results) {
    const panel = byId("safety-panel");
    const rows = LEVELS.map(({ key, label }) => {
      const result = results[key];
      if (!result || result.__error) return `<tr><td>${label}</td><td colspan="4" class="text-secondary">Not verified.</td></tr>`;
      const has = (check) => result.deterministic_mismatches.some((item) => item.check === check);
      const cell = (bad) => (bad ? '<span class="status status-blocked">Mismatch</span>' : '<span class="status status-ready">OK</span>');
      return `<tr>
        <td>${label}</td>
        <td>${cell(has("numeric_or_unit"))}</td>
        <td>${cell(has("negation"))}</td>
        <td>${cell(has("laterality"))}</td>
        <td>${cell(has("medication_frequency"))}</td>
        <td><span class="status ${STATUS_CLASS[result.verification] || ""}">${result.verification}</span></td>
      </tr>`;
    }).join("");
    panel.innerHTML = `<div class="table-responsive"><table class="table table-sm align-middle">
      <thead><tr><th>Level</th><th>Numeric / dosage / unit / % / date</th><th>Negation</th><th>Laterality</th><th>Frequency</th><th>Overall</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
  }

  function populateTranslateLevelOptions(results) {
    const select = byId("translate-level");
    const passing = LEVELS.filter(({ key }) => results[key] && results[key].verification === "PASS");
    if (!passing.length) {
      select.replaceChildren(new Option("No level passed verification", ""));
      byId("run-translation").disabled = true;
      const blockedLabels = LEVELS
        .filter(({ key }) => results[key] && results[key].verification !== "PASS")
        .map(({ key, label }) => `${label}: ${results[key].verification}`);
      const notice = byId("translation-blocked-notice");
      notice.textContent = `Translation is not offered because no simplification level passed verification (${blockedLabels.join("; ")}).`;
      notice.classList.remove("d-none");
    } else {
      select.replaceChildren(...passing.map(({ key, label }) => new Option(label, key)));
      byId("run-translation").disabled = false;
      byId("translation-blocked-notice").classList.add("d-none");
    }
  }

  // ------------------------------- Translation --------------------------------

  async function runTranslation(levelKey, targetLanguage) {
    setStageState("translation-state", "RUNNING");
    setOverview("translation", "RUNNING");
    setTracker("translation", "RUNNING");
    const text = state.simplification[levelKey].simplified_report;
    const body = await callApi("translation", "/api/v1/translations", loggedJson({
      text, source_language: "eng_Latn", target_language: targetLanguage,
    }));
    renderTranslation(body);
    setStageState("translation-state", "PASS");
    setOverview("translation", "PASS");
    setTracker("translation", "PASS");
    return body;
  }

  function renderTranslation(body) {
    state.translation = body;
    byId("translation-model").textContent = orDash(body.model_name);
    byId("translation-revision").textContent = orDash(body.model_version);
    byId("translation-source").textContent = body.source_language;
    byId("translation-target").textContent = body.target_language;
    byId("translation-latency").textContent = fmtMs(body.processing_time_ms);
    byId("translated-output").value = body.translated_text;
    byId("translation-json").textContent = JSON.stringify(body, null, 2);
    const warnings = byId("translation-warnings");
    warnings.textContent = (body.warnings || []).join(" ");
    warnings.classList.toggle("d-none", !(body.warnings || []).length);
    recordPerf("Translation (IndicTrans2)", body.processing_time_ms, state.translationDevice || NOT_EXPOSED);
    renderPerformancePanel();
    applyStageWarmth("translation", "translation-warm");
    loadRuntimeMetrics();
  }

  // ---------------------------- performance panel -----------------------------

  function recordPerf(stage, latencyMs, model) {
    state.perf = state.perf.filter((item) => item.stage !== stage);
    state.perf.push({ stage, latencyMs, model });
  }

  function recordFirstResult(startedAt) {
    if (state.firstResultAt === null) state.firstResultAt = performance.now() - startedAt;
  }

  function renderPerformancePanel() {
    const ordered = [...state.perf].sort((a, b) => {
      const left = PERF_ORDER.indexOf(a.stage);
      const right = PERF_ORDER.indexOf(b.stage);
      return (left === -1 ? Number.MAX_SAFE_INTEGER : left) - (right === -1 ? Number.MAX_SAFE_INTEGER : right);
    });
    const known = ordered.filter((item) => item.latencyMs !== null && item.latencyMs !== undefined);
    const total = known.reduce((sum, item) => sum + Number(item.latencyMs), 0);
    const table = byId("performance-waterfall");
    table.innerHTML = ordered.length
      ? ordered.map((item) => {
          const pct = item.latencyMs === null || item.latencyMs === undefined || total === 0
            ? NOT_EXPOSED
            : `${((Number(item.latencyMs) / total) * 100).toFixed(1)}%`;
          return `<tr><td>${escapeHtml(item.stage)}</td><td>${fmtMs(item.latencyMs)}</td><td>${pct}</td><td>${escapeHtml(item.model || NOT_EXPOSED)}</td></tr>`;
        }).join("")
      : '<tr><td colspan="4" class="text-secondary">Run a stage to populate the waterfall.</td></tr>';
    byId("perf-total").textContent = known.length ? `${total.toFixed(2)} ms (sum of reported stage latencies)` : NOT_EXPOSED;
    byId("perf-first-result").textContent = state.firstResultAt === null
      ? NOT_EXPOSED
      : `${state.firstResultAt.toFixed(0)} ms (client-measured, upload through first OCR response)`;
    const warmSummary = Object.entries(state.runtimeByStage)
      .filter(([, item]) => item.loaded)
      .map(([stage, item]) => `${stage}: ${item.warm ? "warm" : "cold"}`);
    byId("perf-cold-warm").textContent = warmSummary.length ? warmSummary.join(", ") : NOT_EXPOSED;
  }

  // ------------------------------ runtime stats --------------------------------

  function applyStageWarmth(stage, elementId) {
    const item = state.runtimeByStage[stage];
    const element = byId(elementId);
    if (!element) return;
    element.textContent = item ? (item.warm ? "Warm (resident)" : "Cold (first load)") : NOT_EXPOSED;
  }

  async function loadRuntimeMetrics() {
    let body;
    try {
      body = await callApi("runtime-metrics", "/api/v1/runtime/metrics");
    } catch {
      byId("runtime-models-table").innerHTML =
        '<tr><td colspan="11" class="text-secondary">Runtime metrics endpoint unavailable.</td></tr>';
      return;
    }
    state.runtime = body;
    state.runtimeByStage = Object.fromEntries((body.models || []).map((item) => [item.stage, item]));

    const gpu = body.gpu || {};
    byId("runtime-gpu-name").textContent = orDash(gpu.device_name);
    byId("runtime-gpu-available").textContent = String(gpu.available);
    byId("runtime-gpu-device").textContent = gpu.available ? "cuda" : "cpu";
    byId("runtime-gpu-fallback").textContent = state.ocrModel?.configuration?.cpu_fallback === undefined
      ? NOT_EXPOSED
      : String(state.ocrModel.configuration.cpu_fallback);
    byId("runtime-gpu-allocated").textContent = gpu.allocated_mb == null ? NOT_EXPOSED : `${gpu.allocated_mb.toFixed(1)} MiB`;
    byId("runtime-gpu-reserved").textContent = gpu.reserved_mb == null ? NOT_EXPOSED : `${gpu.reserved_mb.toFixed(1)} MiB`;
    byId("runtime-gpu-peak").textContent = gpu.peak_allocated_mb == null ? NOT_EXPOSED : `${gpu.peak_allocated_mb.toFixed(1)} MiB`;
    byId("runtime-gpu-total").textContent = gpu.total_mb == null ? NOT_EXPOSED : `${gpu.total_mb.toFixed(1)} MiB`;
    byId("runtime-gpu-utilization").textContent = gpu.utilization_percent == null
      ? NOT_EXPOSED
      : `${gpu.utilization_percent.toFixed(1)}% (${gpu.utilization_source || "unknown source"})`;

    const cpu = body.cpu || {};
    byId("runtime-cpu-rss").textContent = cpu.process_rss_mb == null ? NOT_EXPOSED : `${cpu.process_rss_mb.toFixed(1)} MiB`;
    byId("runtime-cpu-percent").textContent = cpu.process_cpu_percent == null ? NOT_EXPOSED : `${cpu.process_cpu_percent.toFixed(1)}%`;

    byId("runtime-models-table").innerHTML = (body.models || []).length
      ? body.models.map((item) => `<tr>
          <td>${escapeHtml(item.stage)}</td>
          <td>${escapeHtml(orDash(item.provider_name))}</td>
          <td>${escapeHtml(orDash(item.model_name))}</td>
          <td>${escapeHtml(orDash(item.model_revision))}</td>
          <td>${NOT_EXPOSED}</td>
          <td>${escapeHtml(orDash(item.device))}</td>
          <td>${NOT_EXPOSED}</td>
          <td>${item.loaded ? (item.warm ? "Warm" : "Cold") : "Not loaded"}</td>
          <td>${item.request_count ?? NOT_EXPOSED}</td>
          <td>${item.load_timestamp ? new Date(item.load_timestamp).toLocaleTimeString() : NOT_EXPOSED}</td>
          <td>${fmtMs(item.load_duration_ms)}</td>
        </tr>`).join("")
      : '<tr><td colspan="11" class="text-secondary">No providers reported runtime status.</td></tr>';
    renderPerformancePanel();
  }

  // ------------------------------ API inspector --------------------------------

  function renderApiInspector() {
    const list = byId("api-inspector-list");
    if (!state.apiLog.length) {
      list.innerHTML = '<p class="text-secondary">No API calls executed yet.</p>';
      return;
    }
    list.innerHTML = state.apiLog.map((entry, index) => `
      <details class="api-entry mb-2">
        <summary>
          <span class="status ${entry.ok ? "status-ready" : entry.ok === false && entry.httpStatus ? "status-blocked" : "status-loading"}">${entry.httpStatus ?? "pending"}</span>
          ${escapeHtml(entry.method)} ${escapeHtml(entry.url)}
          <span class="text-secondary">&middot; ${escapeHtml(entry.stage)}</span>
        </summary>
        <div class="p-3">
          <p class="small mb-1"><strong>Request ID:</strong> ${escapeHtml(entry.requestId || NOT_EXPOSED)}</p>
          <p class="small mb-1"><strong>HTTP status:</strong> ${entry.httpStatus ?? "pending"}</p>
          <strong class="small d-block mt-2">Request</strong>
          <pre class="json-output compact">${escapeHtml(JSON.stringify(entry.requestBody ?? {}, null, 2))}</pre>
          <strong class="small d-block mt-2">Response</strong>
          <pre class="json-output compact">${escapeHtml(JSON.stringify(entry.responseBody ?? {}, null, 2))}</pre>
          <button class="btn btn-sm btn-outline-secondary mt-2" type="button" data-copy-index="${index}">Copy JSON</button>
        </div>
      </details>`).join("");
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-copy-index]");
    if (!button) return;
    const entry = state.apiLog[Number(button.dataset.copyIndex)];
    navigator.clipboard?.writeText(JSON.stringify(entry, null, 2)).then(() => {
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = "Copy JSON"; }, 1500);
    });
  });

  // -------------------------------- errors --------------------------------------

  function describeError(error) {
    if (error && typeof error === "object" && "stage" in error) {
      return `${error.stage}: ${error.message} (HTTP ${error.httpStatus ?? "?"}, code ${error.code ?? "?"}, request ${error.requestId ?? NOT_EXPOSED})`;
    }
    return error?.message || String(error);
  }

  // -------------------------------- mode toggle -----------------------------------

  byId("mode-step").addEventListener("click", () => setMode("step"));
  byId("mode-full").addEventListener("click", () => setMode("full"));

  function setMode(mode) {
    const stepMode = mode === "step";
    byId("mode-step").classList.toggle("active", stepMode);
    byId("mode-step").setAttribute("aria-pressed", String(stepMode));
    byId("mode-full").classList.toggle("active", !stepMode);
    byId("mode-full").setAttribute("aria-pressed", String(!stepMode));
    byId("full-pipeline-section").hidden = stepMode;
    byId("upload-section").hidden = !stepMode;
  }

  // -------------------------------- step-by-step wiring -----------------------------

  byId("pipeline-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    clearAlert();
    const file = fileInput.files[0];
    if (!file) return showAlert("Choose a document before running OCR.", "warning");
    byId("run-ocr").disabled = true;
    try {
      await runOcr(file);
      showAlert("OCR completed successfully.", "success");
    } catch (error) {
      setStageState("ocr-state", "FAILED");
      setOverview("ocr", "FAILED");
      showAlert(describeError(error));
    } finally {
      byId("run-ocr").disabled = false;
    }
  });

  byId("run-ner").addEventListener("click", async () => {
    if (!state.ocr) return;
    byId("run-ner").disabled = true;
    try {
      await runNer(state.ocr.normalized_text);
      runEmbeddings(state.ocr.normalized_text, state.ocr.request_id);
      showAlert("NER completed successfully.", "success");
    } catch (error) {
      setStageState("ner-state", "FAILED");
      setOverview("ner", "FAILED");
      showAlert(describeError(error));
    } finally {
      byId("run-ner").disabled = false;
    }
  });

  byId("run-embeddings").addEventListener("click", () => {
    if (!state.ocr) return;
    runEmbeddings(state.ocr.normalized_text, state.ocr.request_id);
  });

  byId("run-simplification").addEventListener("click", async () => {
    if (!state.ocr) return;
    byId("run-simplification").disabled = true;
    try {
      await runSimplification(state.ocr.normalized_text, state.ner?.entities || []);
      showAlert("Simplification completed successfully.", "success");
    } catch (error) {
      setStageState("simplification-state", "FAILED");
      setOverview("simplification", "FAILED");
      showAlert(describeError(error));
    } finally {
      byId("run-simplification").disabled = false;
    }
  });

  byId("run-verification").addEventListener("click", async () => {
    if (!state.ocr || !state.simplification) return;
    byId("run-verification").disabled = true;
    try {
      const results = await runVerificationAll(state.ocr.normalized_text, state.simplification);
      const overall = overallVerificationState(results);
      showAlert(`Verification completed: ${overall}.`, overall === "PASS" ? "success" : "warning");
    } catch (error) {
      setStageState("verification-state", "FAILED");
      setOverview("verification", "FAILED");
      showAlert(describeError(error));
    } finally {
      byId("run-verification").disabled = false;
    }
  });

  byId("run-translation").addEventListener("click", async () => {
    const levelKey = byId("translate-level").value;
    const targetLanguage = byId("target-language").value;
    if (!levelKey || !targetLanguage) return;
    byId("run-translation").disabled = true;
    try {
      await runTranslation(levelKey, targetLanguage);
      showAlert("Translation completed successfully.", "success");
    } catch (error) {
      setStageState("translation-state", "FAILED");
      setOverview("translation", "FAILED");
      showAlert(describeError(error));
    } finally {
      byId("run-translation").disabled = false;
    }
  });

  // -------------------------------- full pipeline mode ---------------------------

  byId("full-pipeline-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    clearAlert();
    const file = byId("full-report-file").files[0];
    const targetLanguage = byId("full-target-language").value;
    if (!file) return showAlert("Choose a document before running the pipeline.", "warning");
    const button = byId("run-full-pipeline");
    button.disabled = true;
    state.runStartedAt = performance.now();
    state.firstResultAt = null;
    ["ocr", "ner", "simplification", "verification", "translation"].forEach((stage) => setTracker(stage, "WAITING"));
    let completed = 0;
    const tick = () => {
      completed += 1;
      byId("full-stage-count").textContent = `${completed} / 5`;
    };
    try {
      describeDocument(file);
      const ocr = await runOcr(file);
      tick();
      runEmbeddings(ocr.normalized_text, ocr.request_id);

      const ner = await runNer(ocr.normalized_text);
      tick();

      const simplification = await runSimplification(ocr.normalized_text, ner.entities);
      tick();

      const verificationResults = await runVerificationAll(ocr.normalized_text, simplification);
      tick();

      const passing = LEVELS.find(({ key }) => verificationResults[key]?.verification === "PASS");
      if (passing) {
        await runTranslation(passing.key, targetLanguage);
        tick();
      } else {
        setTracker("translation", overallVerificationState(verificationResults));
        setOverview("translation", overallVerificationState(verificationResults));
        showAlert("Translation was skipped because no simplification level passed verification.", "warning");
      }
      showAlert("Full pipeline run completed.", "success");
    } catch (error) {
      showAlert(`Pipeline stopped: ${describeError(error)}`, "danger");
      if (error && error.stage) {
        const stageKey = error.stage.split("-")[0];
        setTracker(stageKey, "FAILED");
        setOverview(stageKey, "FAILED");
      }
    } finally {
      button.disabled = false;
      byId("full-total-elapsed").textContent = `${(performance.now() - state.runStartedAt).toFixed(0)} ms`;
      byId("full-first-result").textContent = state.firstResultAt === null ? NOT_EXPOSED : `${state.firstResultAt.toFixed(0)} ms`;
    }
  });

  byId("full-report-file").addEventListener("change", (event) => describeDocument(event.target.files[0]));

  // ------------------------------------ boot --------------------------------------

  setMode("step");
  refreshAllStatuses().catch((error) => showAlert(describeError(error), "warning"));
})();
