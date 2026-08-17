(() => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const escape = (value) => String(value).replace(/[&<>"']/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"})[character]);
  const showError = (message) => {
    byId("alert").textContent = message;
    byId("alert").className = "alert alert-danger";
  };
  const request = async (url, options = {}) => {
    const response = await fetch(url, {headers: {"Content-Type": "application/json"}, ...options});
    const body = await response.json();
    if (!response.ok) throw new Error(body?.error?.message || `Request failed (${response.status})`);
    return body;
  };
  const refreshHealth = async () => {
    try {
      const response = await fetch("/api/v1/embeddings/health");
      byId("health-output").textContent = JSON.stringify(await response.json(), null, 2);
    } catch (error) { showError(error.message); }
  };
  const render = (body) => {
    byId("response-output").textContent = JSON.stringify(body, null, 2);
    byId("batch-size").textContent = body.batch_size;
    byId("dimensions").textContent = body.reproducibility.dimensions ?? "—";
    byId("latency").textContent = `${body.processing_time_ms.toFixed(2)} ms`;
    byId("revision").textContent = body.reproducibility.model_revision;
    byId("vectors").innerHTML = body.embeddings.map((item) => `<tr><td>${escape(item.input_id)}</td><td>${item.dimensions}</td><td>${item.token_count}</td><td>${item.vector_norm.toFixed(6)}</td><td class="font-monospace">${escape(JSON.stringify(item.vector.slice(0, 6)))}…</td></tr>`).join("");
  };
  byId("refresh-health").addEventListener("click", refreshHealth);
  byId("generate").addEventListener("click", async () => {
    const lines = byId("batch-text").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    try {
      const body = await request("/api/v1/embeddings", {method: "POST", body: JSON.stringify({inputs: lines.map((text, index) => ({input_id: `segment-${index + 1}`, text}))})});
      render(body);
    } catch (error) { showError(error.message); }
  });
  refreshHealth();
})();
