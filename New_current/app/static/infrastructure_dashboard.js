"use strict";

const componentContainer = document.querySelector("#components");
const metricContainer = document.querySelector("#metrics");
const jobContainer = document.querySelector("#jobs");
const alertBox = document.querySelector("#alert");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[character]));
}

async function readJson(url) {
  const response = await fetch(url, {headers: {Accept: "application/json"}});
  const payload = await response.json();
  return {response, payload};
}

function renderHealth(health) {
  componentContainer.innerHTML = Object.entries(health.components).map(([name, item]) => `
    <div class="col-sm-6 col-xl-3"><article class="card health-card shadow-sm"><div class="card-body">
      <h2 class="h6 text-uppercase">${escapeHtml(name)}</h2>
      <p><span class="status-dot status-${escapeHtml(item.status)} me-2"></span><strong>${escapeHtml(item.status)}</strong></p>
      <small class="text-secondary">${escapeHtml(item.detail)}</small>
    </div></article></div>`).join("");
  const labels = {queue_length:"Queue length",pending_jobs:"Pending jobs",running_jobs:"Running jobs",completed_jobs:"Completed jobs",failed_jobs:"Failed jobs",cache_hits:"Cache hits",cache_misses:"Cache misses",cache_keys:"Cache keys",database_pool_size:"DB pool",database_checked_out_connections:"DB checked out",celery_workers:"Workers"};
  metricContainer.innerHTML = Object.entries(labels).map(([key, label]) => `<div class="col-6 col-md-3"><div class="text-secondary small">${label}</div><div class="metric-value">${escapeHtml(health.metrics[key] ?? "N/A")}</div></div>`).join("");
  document.querySelector("#migration").textContent = `Migration ${health.migration_current ?? "unknown"} / ${health.migration_head}`;
}

function renderJobs(items) {
  jobContainer.innerHTML = items.length ? items.map((job) => `<tr><td><code>${escapeHtml(job.job_id.slice(0, 8))}</code></td><td>${escapeHtml(job.stage)}</td><td>${escapeHtml(job.status)}</td><td>${escapeHtml(job.progress)}%</td><td>${escapeHtml(new Date(job.updated_at).toLocaleString())}</td></tr>`).join("") : '<tr><td colspan="5" class="text-secondary">No processing jobs.</td></tr>';
}

async function refresh() {
  alertBox.classList.add("d-none");
  try {
    const health = await readJson("/api/v1/infrastructure/health");
    renderHealth(health.payload);
    const jobs = await readJson("/api/v1/jobs?limit=25");
    if (jobs.response.ok) renderJobs(jobs.payload.items);
    else renderJobs([]);
  } catch (error) {
    alertBox.textContent = `Infrastructure API unavailable: ${error.message}`;
    alertBox.className = "alert alert-danger";
  }
}

document.querySelector("#refresh").addEventListener("click", refresh);
refresh();
