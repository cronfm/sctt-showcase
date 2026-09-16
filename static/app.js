"use strict";

const $ = (id) => document.getElementById(id);
const state = { colorMode: "rgb", backend: "plotly", request: null, busy: false };
const plotConfig = { responsive: true, displaylogo: false, scrollZoom: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d"], toImageButtonOptions: { format: "png", filename: "sctt-synthetic-scene", scale: 2 } };

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body)
  });
  const value = await response.json();
  if (!response.ok) {
    const detail = value.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || value));
  }
  return value;
}

function requestFromForm() {
  return { sample_id: $("sample").value, top_k: Number($("top-k").value),
    candidate_offset_m: ["x", "y", "z"].map((axis) => Number($("offset-" + axis).value)) };
}

function updateSnippet(request) {
  const payload = JSON.stringify(request, null, 2);
  $("request-code").textContent = `curl -X POST ${location.origin}/api/infer \\\n  -H 'Content-Type: application/json' \\\n  -d '${payload}'`;
}

function vector(position) {
  return Array.isArray(position) ? position.map((v) => Number(v).toFixed(3)).join("  /  ") : "Not provided";
}

function showResult(result) {
  const error = result.error_m;
  $("error-value").textContent = error == null ? "—" : Number(error).toFixed(3);
  $("latency-value").textContent = result.duration_ms == null ? "—" : `${Number(result.duration_ms).toFixed(1)} ms`;
  $("k-value").textContent = String(result.candidate_positions?.length || result.candidates?.length || state.request.top_k);
  $("prediction-position").textContent = vector(result.position || result.prediction?.position);
  $("truth-position").textContent = vector(result.ground_truth || result.truth?.position);
  const list = $("candidate-list");
  list.replaceChildren();
  const weights = result.candidate_weights || [];
  weights.forEach((axes, index) => {
    const average = axes.reduce((sum, item) => sum + item, 0) / axes.length;
    const row = document.createElement("div"); row.className = "candidate-row";
    const name = document.createElement("span"); name.textContent = result.candidate_ids?.[index] || `C${index + 1}`;
    const track = document.createElement("div"); track.className = "weight-track";
    const bar = document.createElement("div"); bar.className = "weight-bar"; bar.style.width = `${Math.max(0, Math.min(100, average * 100))}%`; track.append(bar);
    const number = document.createElement("span"); number.className = "weight-value"; number.textContent = average.toFixed(3);
    row.title = `XYZ weights: ${axes.map((v) => v.toFixed(3)).join(", ")}`;
    row.append(name, track, number); list.append(row);
  });
}

async function drawFigure() {
  $("view-error").hidden = true;
  const figure = await api("/api/figure", { ...state.request, color_mode: state.colorMode, backend: state.backend });
  // Preserve the user's orbit while changing colors, candidates or backend.
  const camera = $("plot").layout?.scene?.camera;
  if (camera && figure.layout?.scene) figure.layout.scene.camera = camera;
  figure.layout.height = Math.max(430, $("plot").clientHeight || 580);
  await Plotly.react("plot", figure.data, figure.layout, plotConfig);
}

function busy(value) {
  state.busy = value;
  ["run", "reset", "sample", "top-k", "offset-x", "offset-y", "offset-z", "backend", "rgb-mode", "feature-mode"].forEach((id) => { $(id).disabled = value; });
  $("run").textContent = value ? "Refining…" : "Run refinement →";
  $("inference-form").setAttribute("aria-busy", String(value));
}

async function run() {
  if (state.busy) return;
  busy(true); state.request = requestFromForm(); updateSnippet(state.request);
  $("status").textContent = "Running the toy PyTorch model…";
  try {
    const result = await api("/api/infer", state.request);
    showResult(result);
    await drawFigure();
    $("status").textContent = "Ready · displaying the latest query.";
  } catch (error) {
    $("status").textContent = "The query could not finish.";
    $("view-error").textContent = error.message;
    $("view-error").hidden = false;
  } finally { busy(false); }
}

async function changeView({ colorMode = state.colorMode, backend = state.backend } = {}) {
  if (state.busy || !state.request) return;
  const previous = { colorMode: state.colorMode, backend: state.backend };
  state.colorMode = colorMode; state.backend = backend;
  busy(true);
  try {
    await drawFigure();
    $("status").textContent = backend === "pytorch3d" ? "Ready · PyTorch3D tensors · Plotly view." : "Ready · portable Plotly scene.";
  } catch (error) {
    state.colorMode = previous.colorMode; state.backend = previous.backend;
    $("view-error").textContent = error.message;
    $("view-error").hidden = false;
  } finally {
    $("backend").value = state.backend;
    ["rgb", "feature"].forEach((mode) => { $(mode + "-mode").classList.toggle("active", state.colorMode === mode); $(mode + "-mode").setAttribute("aria-pressed", String(state.colorMode === mode)); });
    busy(false);
  }
}

$("inference-form").addEventListener("submit", (event) => { event.preventDefault(); run(); });
$("top-k").addEventListener("input", () => { $("top-k-value").textContent = $("top-k").value; });
$("rgb-mode").addEventListener("click", () => changeView({ colorMode: "rgb" }));
$("feature-mode").addEventListener("click", () => changeView({ colorMode: "feature" }));
$("backend").addEventListener("change", () => changeView({ backend: $("backend").value }));
$("reset").addEventListener("click", () => {
  $("sample").selectedIndex = 0; $("top-k").value = "4"; $("top-k-value").textContent = "4";
  ["x", "y", "z"].forEach((axis) => { $("offset-" + axis).value = "0"; }); run();
});
$("copy-request").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("request-code").textContent);
    $("copy-request").textContent = "Copied";
    setTimeout(() => { $("copy-request").textContent = "Copy"; }, 1800);
  } catch {
    $("status").textContent = "Select the request text to copy it; clipboard access is unavailable.";
  }
});

async function start() {
  try {
    const [scene, response, capabilities] = await Promise.all([api("/api/scene"), api("/api/samples"), api("/api/capabilities")]);
    const pytorch3dOption = $("backend").querySelector('option[value="pytorch3d"]');
    pytorch3dOption.disabled = !capabilities.pytorch3d;
    if (!capabilities.pytorch3d) {
      pytorch3dOption.textContent = "PyTorch3D · optional install needed";
      $("backend").title = "Install PyTorch3D to enable its tensor backend; see docs/pytorch3d.md.";
    }
    const samples = Array.isArray(response) ? response : (response.samples || scene.samples || []);
    if (!samples.length) throw new Error("No synthetic samples are available.");
    $("sample").replaceChildren(...samples.map((sample) => {
      const option = document.createElement("option"); option.value = sample.id || sample.sample_id;
      option.textContent = sample.label || sample.id || sample.sample_id; return option;
    }));
    $("point-count").textContent = scene.points.xyz.length.toLocaleString();
    $("camera-count").textContent = String(scene.trajectory?.length || 0);
    await run();
  } catch (error) {
    $("status").textContent = "Could not load the demo.";
    $("view-error").textContent = error.message;
    $("view-error").hidden = false;
    $("plot").replaceChildren();
  }
}
start();
