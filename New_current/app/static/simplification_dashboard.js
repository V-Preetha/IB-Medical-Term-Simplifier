(() => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const escape = (value) => String(value).replace(/[&<>"']/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"})[character]);
  const showError = (message) => { byId("alert").textContent = message; byId("alert").className = "alert alert-danger"; };
  const requestJson = async (url, options = {}) => {
    const response = await fetch(url, {headers: {"Content-Type": "application/json"}, ...options});
    const body = await response.json();
    if (!response.ok) throw new Error(body?.error?.message || `Request failed (${response.status})`);
    return body;
  };
  const refreshHealth = async () => {
    try { const response = await fetch("/api/v1/simplify/health"); byId("health-output").textContent = JSON.stringify(await response.json(), null, 2); }
    catch (error) { showError(error.message); }
  };
  const title = (name) => ({clinical: "Level 1 - Clinical", general_public: "Level 2 - General Public", child_friendly: "Level 3 - Child-Friendly"})[name];
  const renderLevel = (value) => `<article class="col-lg-4"><div class="card shadow-sm h-100"><div class="card-body"><h2 class="h5">${title(value.level)}</h2><p class="level-output">${escape(value.simplified_report)}</p><h3 class="h6">Important findings</h3><ul>${value.important_findings.map((item) => `<li>${escape(item)}</li>`).join("")}</ul><h3 class="h6">Medical terms</h3><ul>${value.medical_terms_explained.map((item) => `<li><strong>${escape(item.term)}:</strong> ${escape(item.explanation)}</li>`).join("")}</ul><h3 class="h6">Questions for the doctor</h3><ul>${value.suggested_questions_for_doctor.map((item) => `<li>${escape(item)}</li>`).join("")}</ul><p class="small text-secondary mb-0">Confidence: ${value.confidence ?? "not measurable"}</p></div></div></article>`;
  byId("refresh-health").addEventListener("click", refreshHealth);
  byId("run").addEventListener("click", async () => {
    const text = byId("report-text").value.trim();
    if (!text) return showError("Enter normalized OCR text.");
    byId("run").disabled = true;
    try {
      const body = await requestJson("/api/v1/simplify", {method: "POST", body: JSON.stringify({text, entities: [], linked_concepts: []})});
      const levels = [body.clinical, body.general_public, body.child_friendly];
      byId("levels").innerHTML = levels.map(renderLevel).join("");
      byId("latency").textContent = `${body.processing_time_ms.toFixed(2)} ms`;
      byId("model").textContent = body.inference.model_name;
      byId("revision").textContent = body.inference.model_revision;
      byId("prompt-version").textContent = body.inference.prompt_version;
      byId("raw-json").textContent = JSON.stringify(body, null, 2);
      byId("results").classList.remove("d-none");
      byId("alert").className = "alert d-none";
    } catch (error) { showError(error.message); }
    finally { byId("run").disabled = false; }
  });
  refreshHealth();
})();
