"use strict";

/**
 * This app assumes the embedded JSON bundle has:
 * - cat_cols: ["JobRole", ...]
 * - num_cols: ["Age", ...]
 * - ohe_categories: { JobRole: ["Sales Executive", ...], ... }
 * - feature_names: ["JobRole_Sales Executive", ..., "Age", ...]  (ORDER matters)
 * - scaler_min, scaler_scale: same length as feature_names (MinMaxScaler params)
 * - intercept: number
 * - coef: array same length as feature_names
 * - threshold: decision threshold (default 0.5)
 */

let BUNDLE = null;

const elFields   = document.getElementById("fields");
const elMeta     = document.getElementById("meta");
const elStatus   = document.getElementById("status");
const elProbText = document.getElementById("probText");
const elProbFill = document.getElementById("probFill");
const elDecision = document.getElementById("decision");
const elDebug    = document.getElementById("debug");

function sigmoid(z) {
  // Numerically stable-ish for typical LR ranges
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  } else {
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }
}

function loadBundleFromHtml() {
  const script = document.getElementById("bundle");
  if (!script) throw new Error("Embedded bundle <script id='bundle'> not found in index.html");
  const txt = script.textContent?.trim();
  if (!txt) throw new Error("Embedded bundle is empty");
  return JSON.parse(txt);
}

function validateBundle(b) {
  const required = ["cat_cols","num_cols","ohe_categories","feature_names","scaler_min","scaler_scale","intercept","coef"];
  for (const k of required) {
    if (!(k in b)) throw new Error(`Bundle missing key: ${k}`);
  }
  const n = b.feature_names.length;

  if (!Array.isArray(b.feature_names) || n === 0) {
    throw new Error("feature_names is empty. Export your bundle from Python and paste it into index.html.");
  }
  if (!Array.isArray(b.coef) || b.coef.length !== n) {
    throw new Error(`coef length mismatch. Expected ${n}, got ${b.coef?.length ?? "null"}`);
  }
  if (!Array.isArray(b.scaler_min) || b.scaler_min.length !== n) {
    throw new Error(`scaler_min length mismatch. Expected ${n}, got ${b.scaler_min?.length ?? "null"}`);
  }
  if (!Array.isArray(b.scaler_scale) || b.scaler_scale.length !== n) {
    throw new Error(`scaler_scale length mismatch. Expected ${n}, got ${b.scaler_scale?.length ?? "null"}`);
  }
}

function buildInputs() {
  elFields.innerHTML = "";

  // categorical dropdowns
  for (const col of BUNDLE.cat_cols) {
    const wrap = document.createElement("div");

    const lab = document.createElement("label");
    lab.textContent = col;
    wrap.appendChild(lab);

    const sel = document.createElement("select");
    sel.id = `inp_${col}`;

    const cats = BUNDLE.ohe_categories[col] || [];
    // Ensure we have at least one option
    if (cats.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(no categories in bundle)";
      sel.appendChild(opt);
    } else {
      for (const v of cats) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        sel.appendChild(opt);
      }
    }

    wrap.appendChild(sel);
    elFields.appendChild(wrap);
  }

  // numeric inputs
//   for (const col of BUNDLE.num_cols) {
//     const wrap = document.createElement("div");

//     const lab = document.createElement("label");
//     lab.textContent = col;
//     wrap.appendChild(lab);

//     const inp = document.createElement("input");
//     inp.type = "number";
//     inp.step = "1";
//     inp.value = "0";
//     inp.id = `inp_${col}`;
//     wrap.appendChild(inp);

//     elFields.appendChild(wrap);
//   }
}

/**
 * Recreate the exact training vector:
 * - one-hot columns named by sklearn: "Col_Value"
 * - numeric columns by their original name
 *
 * Vector order MUST match BUNDLE.feature_names
 */
function vectorizeInputs() {
  const n = BUNDLE.feature_names.length;
  const x = new Array(n).fill(0);

  // Map feature_name -> index (fast fill)
  const idx = new Map();
  for (let i = 0; i < n; i++) idx.set(BUNDLE.feature_names[i], i);

  // One-hot for categoricals
  for (const col of BUNDLE.cat_cols) {
    const el = document.getElementById(`inp_${col}`);
    const chosen = el ? el.value : "";
    if (!chosen) continue;

    // IMPORTANT: this must match get_feature_names_out() convention
    // For OneHotEncoder default: `${col}_${category}`
    const fname = `${col}_${chosen}`;
    const i = idx.get(fname);
    if (i !== undefined) x[i] = 1;
    // If undefined: category may contain special formatting differences; bundle should match exactly.
  }

  // Numeric
//   for (const col of BUNDLE.num_cols) {
//     const el = document.getElementById(`inp_${col}`);
//     const v = el ? Number(el.value) : 0;
//     const safe = Number.isFinite(v) ? v : 0;

//     const i = idx.get(col);
//     if (i !== undefined) x[i] = safe;
//   }

  return x;
}

/**
 * MinMaxScaler transform:
 * sklearn MinMaxScaler stores:
 *   X_scaled = X * scale_ + min_
 */
function minMaxScale(x) {
  const n = x.length;
  const xs = new Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = x[i] * BUNDLE.scaler_scale[i] + BUNDLE.scaler_min[i];
  }
  return xs;
}

function predict() {
  const raw = vectorizeInputs();
  const x = minMaxScale(raw);

  let score = BUNDLE.intercept;
  for (let i = 0; i < x.length; i++) {
    score += BUNDLE.coef[i] * x[i];
  }

  const p = sigmoid(score);
  return { p, score };
}

function setOutput(out) {
  const pct = Math.round(out.p * 1000) / 10; // 1 decimal %
  elProbText.textContent = `${pct}%`;
  elProbFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;

  const thr = (typeof BUNDLE.threshold === "number") ? BUNDLE.threshold : 0.5;
  elDecision.textContent = out.p >= thr
    ? `High risk (p ≥ ${thr}).`
    : `Lower risk (p < ${thr}).`;

  elDebug.textContent = JSON.stringify(
    { threshold: thr, ...out },
    null,
    2
  );
}

function resetUI() {
  buildInputs();
  elProbText.textContent = "—";
  elProbFill.style.width = "0%";
  elDecision.textContent = "—";
  elDebug.textContent = "—";
  elStatus.textContent = "";
}

function init() {
  try {
    BUNDLE = loadBundleFromHtml();
    // validateBundle(BUNDLE);

    elMeta.textContent =''
    //   `Loaded: ${BUNDLE.feature_names.length} features (${BUNDLE.cat_cols.length} categorical, ${BUNDLE.num_cols.length} numeric)`;

    buildInputs();

    document.getElementById("btnCalc").addEventListener("click", () => {
      try {
        const out = predict();
        setOutput(out);
        elStatus.textContent = "Calculated.";
        elStatus.classList.remove("error");
      } catch (e) {
        elStatus.textContent = `Error: ${e.message}`;
        elStatus.classList.add("error");
      }
    });

    document.getElementById("btnReset").addEventListener("click", resetUI);

  } catch (e) {
    elMeta.textContent = `Error: ${e.message}`;
    elMeta.classList.add("error");
    console.error(e);
  }
}

init();