/* script.js — Solar Energy Dashboard */
const API = "http://127.0.0.1:5000/api";

// Chart colour palette
const COLORS = {
  green:  "rgba(46,204,113,",
  blue:   "rgba(52,152,219,",
  orange: "rgba(243,156,18,",
  red:    "rgba(231,76,60,",
  purple: "rgba(155,89,182,",
};
const solid = (c, a=1) => c + a + ")";
const fade  = (c, a=.25) => c + a + ")";

const chartInstances = {};

function mkChart(id, config) {
  if (chartInstances[id]) chartInstances[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  chartInstances[id] = new Chart(ctx, config);
}

const baseOpts = (title="") => ({
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    legend: { labels: { color: "#ecf0f1", font: { size: 11 } } },
    title:  { display: false },
  },
  scales: {
    x: { ticks: { color: "#95a5a6", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,.05)" } },
    y: { ticks: { color: "#95a5a6" }, grid: { color: "rgba(255,255,255,.05)" } },
  },
});

// ── Helpers ──────────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

async function apiFetch(endpoint) {
  const res = await fetch(API + endpoint);
  if (!res.ok) throw new Error(`${endpoint} → ${res.status}`);
  return res.json();
}

function showLoading(show) {
  document.getElementById("loading").classList.toggle("hidden", !show);
}

// ── Summary ───────────────────────────────────────────────────────────────────
async function loadSummary() {
  const d = await apiFetch("/summary");
  setText("c-total", d.total_energy_kwh.toLocaleString());
  setText("c-avg",   d.avg_hourly_generation);
  setText("c-inst",  d.num_installations);
  setText("c-low",   d.lowest_installation);
  setText("c-anom",  d.num_anomalies);
  setText("c-r2",    d.r2_score);
}

// ── Generation Trends ─────────────────────────────────────────────────────────
async function loadTrends() {
  const d = await apiFetch("/generation-trend");

  // Daily
  const daily = d.daily.filter((_, i) => i % 3 === 0); // thin out for readability
  mkChart("dailyChart", {
    type: "line",
    data: {
      labels: daily.map(r => r.date),
      datasets: [{ label: "Daily Generation (kWh)", data: daily.map(r => r.total_generation),
        borderColor: solid(COLORS.green), backgroundColor: fade(COLORS.green),
        fill: true, tension: .3, pointRadius: 0 }],
    },
    options: { ...baseOpts(), plugins: { ...baseOpts().plugins } },
  });

  // Monthly
  mkChart("monthlyChart", {
    type: "bar",
    data: {
      labels: d.monthly.map(r => r.month),
      datasets: [{ label: "Total (kWh)", data: d.monthly.map(r => r.total_generation),
        backgroundColor: solid(COLORS.orange, .8), borderRadius: 4 }],
    },
    options: baseOpts(),
  });

  // Hourly
  mkChart("hourlyChart", {
    type: "line",
    data: {
      labels: d.hourly.map(r => r.hour + ":00"),
      datasets: [{ label: "Avg Generation (kWh)", data: d.hourly.map(r => r.avg_generation),
        borderColor: solid(COLORS.blue), backgroundColor: fade(COLORS.blue),
        fill: true, tension: .4, pointRadius: 3 }],
    },
    options: baseOpts(),
  });

  // Seasonal
  const seasonOrder = ["Winter","Spring","Summer","Autumn"];
  const seasonal = seasonOrder.map(s => d.seasonal.find(r => r.Season === s) || { avg: 0 });
  mkChart("seasonalChart", {
    type: "bar",
    data: {
      labels: seasonOrder,
      datasets: [{ label: "Avg Generation (kWh)", data: seasonal.map(r => r.avg),
        backgroundColor: [solid(COLORS.blue,.7), solid(COLORS.green,.8),
                          solid(COLORS.orange,.8), solid(COLORS.purple,.7)],
        borderRadius: 4 }],
    },
    options: baseOpts(),
  });
}

// ── Installations ─────────────────────────────────────────────────────────────
async function loadInstallations() {
  const d = await apiFetch("/installation-performance");
  const sorted = [...d.installations].sort((a, b) => b.mean - a.mean);

  mkChart("instChart", {
    type: "bar",
    data: {
      labels: sorted.map(r => r.Installation_ID),
      datasets: [{ label: "Avg Generation (kWh)", data: sorted.map(r => r.mean),
        backgroundColor: sorted.map((r, i) =>
          r.Installation_ID === d.highest ? solid(COLORS.green,.85) :
          r.Installation_ID === d.lowest  ? solid(COLORS.red,.75) :
          solid(COLORS.blue,.7)),
        borderRadius: 4 }],
    },
    options: baseOpts(),
  });

  const tbody = document.querySelector("#instTable tbody");
  tbody.innerHTML = sorted.map(r => `
    <tr>
      <td>${r.Installation_ID}
        ${r.Installation_ID === d.highest ? '<span class="badge badge-low">Best</span>' : ""}
        ${r.Installation_ID === d.lowest  ? '<span class="badge badge-high">Lowest</span>' : ""}
      </td>
      <td>${r.mean}</td><td>${r.total.toLocaleString()}</td><td>${r.max}</td>
    </tr>`).join("");
}

// ── Weather Analysis ──────────────────────────────────────────────────────────
async function loadWeather() {
  const d = await apiFetch("/weather-analysis");
  const pts = d.scatter_data;

  const scatterCfg = (xKey, yKey, label, color) => ({
    type: "scatter",
    data: { datasets: [{ label, data: pts.map(p => ({ x: p[xKey], y: p[yKey] })),
      backgroundColor: fade(color, .4), pointRadius: 2 }] },
    options: { ...baseOpts(),
      scales: {
        x: { ...baseOpts().scales.x, title: { display: true, text: xKey, color: "#95a5a6" } },
        y: { ...baseOpts().scales.y, title: { display: true, text: "Generation (kWh)", color: "#95a5a6" } },
      }},
  });

  mkChart("sunScatter",  scatterCfg("Sunlight",    "Solar_Generation", "Sunlight vs Gen",     COLORS.orange));
  mkChart("tempScatter", scatterCfg("Temperature", "Solar_Generation", "Temperature vs Gen",  COLORS.red));
  mkChart("cloudScatter",scatterCfg("Cloud_Cover", "Solar_Generation", "Cloud Cover vs Gen",  COLORS.blue));

  // Panel condition bar
  const pd = await apiFetch("/panel-condition");
  mkChart("panelChart", {
    type: "bar",
    data: {
      labels: pd.map(r => r.Panel_Condition),
      datasets: [{ label: "Avg Generation (kWh)", data: pd.map(r => r.mean),
        backgroundColor: [solid(COLORS.green,.8), solid(COLORS.orange,.8), solid(COLORS.red,.75)],
        borderRadius: 4 }],
    },
    options: baseOpts(),
  });
}

// ── Anomalies ─────────────────────────────────────────────────────────────────
async function loadAnomalies() {
  const data = await apiFetch("/anomalies");
  const tbody = document.querySelector("#anomalyTable tbody");
  if (!data.length) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#95a5a6">No anomalies detected</td></tr>';
    return;
  }
  tbody.innerHTML = data.slice(0, 200).map(a => `
    <tr>
      <td>${a.timestamp.replace("T", " ").slice(0,16)}</td>
      <td>${a.installation_id}</td>
      <td>${a.actual_generation}</td>
      <td>${a.expected_generation}</td>
      <td style="color:#e74c3c">-${a.difference}</td>
      <td><span class="badge badge-${a.severity.toLowerCase()}">${a.severity}</span></td>
      <td>${a.reason}</td>
    </tr>`).join("");
}

// ── Model Metrics ─────────────────────────────────────────────────────────────
async function loadModel() {
  const m = await apiFetch("/model-metrics");
  setText("m-mae",  m.MAE);
  setText("m-mse",  m.MSE);
  setText("m-rmse", m.RMSE);
  setText("m-r2",   m.R2);

  // Actual vs Predicted
  const avp = await apiFetch("/actual-vs-predicted");
  const sample = avp.filter((_, i) => i % 5 === 0).slice(0, 100);
  mkChart("avpChart", {
    type: "line",
    data: {
      labels: sample.map(r => r.timestamp.slice(5, 16)),
      datasets: [
        { label: "Actual",    data: sample.map(r => r.actual),
          borderColor: solid(COLORS.green), pointRadius: 0, tension: .3 },
        { label: "Predicted", data: sample.map(r => r.predicted),
          borderColor: solid(COLORS.orange), pointRadius: 0, tension: .3, borderDash: [4,3] },
      ],
    },
    options: baseOpts(),
  });

  // Feature importance
  const fi = await apiFetch("/feature-importance");
  const fiEntries = Object.entries(fi).sort((a, b) => a[1] - b[1]);
  mkChart("fiChart", {
    type: "bar",
    data: {
      labels: fiEntries.map(e => e[0].replace("_enc", "").replace("_", " ")),
      datasets: [{ label: "Importance", data: fiEntries.map(e => e[1]),
        backgroundColor: solid(COLORS.green,.75), borderRadius: 4 }],
    },
    options: { ...baseOpts(), indexAxis: "y" },
  });
}

// ── Data Quality ──────────────────────────────────────────────────────────────
async function loadDataQuality() {
  const d = await apiFetch("/cleaning-summary");
  setText("dq-orig",  d.original_rows?.toLocaleString() ?? "—");
  setText("dq-miss",  d.missing_before ?? "—");
  setText("dq-dup",   d.duplicates_removed ?? "—");
  setText("dq-inv",   d.invalid_values_fixed ?? "—");
  setText("dq-clean", d.cleaned_rows?.toLocaleString() ?? "—");
  setText("dq-miss2", d.missing_after ?? "—");
}

// ── Insights & Action Plan ────────────────────────────────────────────────────
async function loadInsights() {
  const [insights, plan] = await Promise.all([
    apiFetch("/insights"),
    apiFetch("/action-plan"),
  ]);
  document.getElementById("insights-list").innerHTML =
    insights.map(i => `<li>${i}</li>`).join("");
  document.getElementById("action-list").innerHTML =
    plan.map(p => `<li>${p}</li>`).join("");
}

// ── Prediction ────────────────────────────────────────────────────────────────
async function submitPrediction(e) {
  e.preventDefault();
  const resultEl = document.getElementById("predict-result");
  const errorEl  = document.getElementById("predict-error");
  resultEl.classList.add("hidden");
  errorEl.classList.add("hidden");

  const payload = {
    Installation_ID: document.getElementById("p-inst").value,
    Sunlight:        parseFloat(document.getElementById("p-sun").value),
    Temperature:     parseFloat(document.getElementById("p-temp").value),
    Cloud_Cover:     parseFloat(document.getElementById("p-cloud").value),
    Panel_Condition: document.getElementById("p-panel").value,
    Hour:            parseInt(document.getElementById("p-hour").value),
    Day_of_Week:     parseInt(document.getElementById("p-dow").value),
    Month:           parseInt(document.getElementById("p-month").value),
    Season:          parseInt(document.getElementById("p-season").value),
  };

  try {
    const res = await fetch(API + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    setText("pred-value", data.predicted_generation_kwh);
    resultEl.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = "Error: " + err.message;
    errorEl.classList.remove("hidden");
  }
}

// ── Sidebar toggle ────────────────────────────────────────────────────────────
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

// Highlight active nav link on scroll
const sections = document.querySelectorAll(".section");
const navLinks  = document.querySelectorAll(".nav-link");
window.addEventListener("scroll", () => {
  let current = "";
  sections.forEach(s => { if (window.scrollY >= s.offsetTop - 100) current = s.id; });
  navLinks.forEach(l => {
    l.classList.toggle("active", l.getAttribute("href") === "#" + current);
  });
}, { passive: true });

// ── Load All ──────────────────────────────────────────────────────────────────
async function loadAll() {
  showLoading(true);
  try {
    await Promise.all([
      loadSummary(),
      loadTrends(),
      loadInstallations(),
      loadWeather(),
      loadAnomalies(),
      loadModel(),
      loadDataQuality(),
      loadInsights(),
    ]);
  } catch (err) {
    console.error("Dashboard load error:", err);
    alert("Could not load dashboard data.\nMake sure the Flask backend is running on port 5000.\n\n" + err.message);
  } finally {
    showLoading(false);
  }
}

document.addEventListener("DOMContentLoaded", loadAll);
