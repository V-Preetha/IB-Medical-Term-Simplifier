const byId = (id) => document.getElementById(id);

async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error?.message || "Request failed.");
  return body;
}

async function refreshHealth() {
  try {
    byId("health").textContent = JSON.stringify(
      await request("/api/v1/verification/health"), null, 2
    );
  } catch (error) {
    byId("health").textContent = error.message;
  }
}

byId("run").addEventListener("click", async () => {
  try {
    const body = await request("/api/v1/verification", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({premise: byId("premise").value, hypothesis: byId("hypothesis").value}),
    });
    byId("result").textContent = JSON.stringify(body, null, 2);
  } catch (error) {
    byId("result").textContent = error.message;
  }
});

refreshHealth();
