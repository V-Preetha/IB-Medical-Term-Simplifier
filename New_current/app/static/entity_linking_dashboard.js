(() => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const alertBox = byId("alert");

  const showError = (message) => {
    alertBox.textContent = message;
    alertBox.className = "alert alert-danger";
  };
  const request = async (url, options = {}) => {
    const response = await fetch(url, {headers: {"Content-Type": "application/json"}, ...options});
    const body = await response.json();
    if (!response.ok) throw new Error(body?.error?.message || `Request failed (${response.status})`);
    return body;
  };
  const refreshHealth = async () => {
    try {
      const response = await fetch("/api/v1/entity-linking/health");
      byId("health-output").textContent = JSON.stringify(await response.json(), null, 2);
    } catch (error) { showError(error.message); }
  };
  const render = (body) => {
    byId("response-output").textContent = JSON.stringify(body, null, 2);
    const rows = body.links.map((link) => {
      const concept = link.normalized_concept;
      const cells = [link.original_entity.text, link.status, concept?.concept_id || "—", concept?.preferred_name || "—", concept?.semantic_types.join(", ") || "—", concept?.confidence?.toFixed(4) || "—"];
      return `<tr>${cells.map((value) => `<td>${String(value).replace(/[&<>"']/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"})[character])}</td>`).join("")}</tr>`;
    });
    byId("concepts").innerHTML = rows.join("");
  };
  const link = async (entities) => render(await request("/api/v1/entity-linking", {method: "POST", body: JSON.stringify({entities})}));
  byId("refresh-health").addEventListener("click", refreshHealth);
  byId("run-linking").addEventListener("click", async () => {
    try { await link(JSON.parse(byId("entity-json").value).entities); } catch (error) { showError(error.message); }
  });
  byId("run-pipeline").addEventListener("click", async () => {
    try {
      const ner = await request("/api/v1/ner", {method: "POST", body: JSON.stringify({text: byId("ocr-text").value})});
      byId("entity-json").value = JSON.stringify({entities: ner.entities}, null, 2);
      await link(ner.entities);
    } catch (error) { showError(error.message); }
  });
  refreshHealth();
})();
