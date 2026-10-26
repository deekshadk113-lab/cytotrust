(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
  const f3 = (v) => Number(v).toFixed(3).replace(/^0/, "");
  const pm = (pair) => `${f3(pair[0])}\u00b1${f3(pair[1])}`;
  const MAX_BYTES = 5 * 1024 * 1024;

  const els = {
    form: $("#form"), file: $("#file"), drop: $("#drop"), preview: $("#preview"), dropText: $("#drop-text"),
    model: $("#model"), modelStatus: $("#model-status"), threshold: $("#threshold"), thresholdOut: $("#threshold-out"),
    calibrated: $("#calibrated"), explain: $("#explain"), go: $("#go"), error: $("#error"), out: $("#out"),
  };
  let currentFile = null;
  let previewUrl = null;
  let modelReady = false;

  /* ---------------------------------------------------------------- analyzer */

  function showError(msg) { els.error.textContent = msg; els.error.hidden = false; }
  function clearError() { els.error.hidden = true; els.error.textContent = ""; }
  function updateGo() { els.go.disabled = !(currentFile && modelReady); }
  function setBusy(busy) {
    els.go.disabled = busy || !(currentFile && modelReady);
    els.go.classList.toggle("busy", busy);
    els.go.textContent = busy ? "Analyzing…" : "Analyze cell";
  }

  async function loadModels() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      const avail = data.models.filter((m) => m.available);
      els.model.innerHTML = "";
      if (avail.length) {
        els.model.add(new Option(`Ensemble of ${avail.length} model${avail.length > 1 ? "s" : ""} (recommended)`, "ensemble"));
      }
      data.models.forEach((m) => {
        const opt = new Option(m.available ? m.name : `${m.name} (checkpoint missing)`, m.id);
        opt.disabled = !m.available;
        els.model.add(opt);
      });
      modelReady = avail.length > 0;
      els.model.disabled = !modelReady;
      els.modelStatus.textContent = modelReady
        ? `${avail.length} of ${data.models.length} models found on the server.`
        : "No model checkpoints were found on the server. The README explains how to add them.";
    } catch (err) {
      modelReady = false;
      els.modelStatus.textContent = `Could not load the model list: ${err.message}`;
    }
    updateGo();
  }

  function setFile(file) {
    clearError();
    if (!file) return;
    if (!/^image\//.test(file.type) && !/\.(png|jpe?g|bmp|tiff?)$/i.test(file.name)) {
      showError("Choose a PNG, JPEG, BMP or TIFF image.");
      return;
    }
    if (file.size > MAX_BYTES) {
      showError("That image is larger than 5 MB. Crop it to a single cell and try again.");
      return;
    }
    currentFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    els.preview.hidden = false;
    els.preview.src = previewUrl;
    els.drop.classList.add("has-image");
    els.dropText.innerHTML = `<strong>${esc(file.name)}</strong><span>Choose or drop another image to replace it.</span>`;
    updateGo();
  }

  els.preview.addEventListener("error", () => { els.preview.hidden = true; });
  els.file.addEventListener("change", () => setFile(els.file.files[0]));
  ["dragenter", "dragover"].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove("over"); }));
  els.drop.addEventListener("drop", (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) setFile(f); });
  els.threshold.addEventListener("input", () => { els.thresholdOut.textContent = pct(Number(els.threshold.value)); });

  els.form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (!currentFile) return;
    clearError();
    setBusy(true);
    const body = new FormData();
    body.append("file", currentFile);
    body.append("model", els.model.value);
    body.append("calibrated", els.calibrated.checked);
    body.append("threshold", els.threshold.value);
    body.append("explain", els.explain.checked);
    try {
      const res = await fetch("/api/predict", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The server could not analyze this image.");
      renderResult(data);
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(false);
    }
  });

  function renderResult(d) {
    const titles = { normal: "Looks normal", abnormal: "Looks abnormal", refer: "Refer to a pathologist" };
    const lean = d.label === "abnormal" ? "abnormal" : "normal";
    const note = d.verdict === "refer"
      ? `Confidence ${pct(d.confidence)} is below your ${pct(d.threshold)} limit, so the model does not answer. It leans ${lean}.`
      : `Confidence ${pct(d.confidence)} is at or above your ${pct(d.threshold)} limit.`;

    const lo = 1 - d.threshold;
    const width = (2 * d.threshold - 1) * 100;
    const mark = Math.min(98, Math.max(2, d.p_abnormal * 100));
    const rulerLabel = `Probability of abnormal ${pct(d.p_abnormal, 1)}. Referral zone from ${pct(lo)} to ${pct(d.threshold)}.`;

    const rows = d.models.map((m) => `<tr><td>${esc(m.name)}</td><td>${pct(m.p_raw, 1)}</td><td>${pct(m.p_cal, 1)}</td><td>${Number(m.temperature).toFixed(2)}</td></tr>`);
    const average = d.models.length > 1
      ? `<tr class="chosen"><td>Ensemble average (${d.calibrated ? "calibrated" : "raw"})</td><td colspan="3">${pct(d.p_abnormal, 1)}</td></tr>`
      : "";
    const used = d.calibrated ? "calibrated" : "raw";

    const cam = d.gradcam
      ? `<div class="detail"><h4>Where the model looked</h4>
           <div class="cam">
             <figure><img src="${esc(previewUrl || "")}" alt="The image you uploaded"><figcaption>Your image</figcaption></figure>
             <figure><img src="data:image/png;base64,${d.gradcam}" alt="Heatmap over the cell. Warm colors mark regions that pushed the prediction."><figcaption>Heatmap from ${esc(d.gradcam_model)}. Warm colors pushed the prediction.</figcaption></figure>
           </div></div>`
      : "";
    const warn = d.warnings && d.warnings.length ? `<ul class="warnings">${d.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : "";

    els.out.innerHTML = `
      <article class="verdict v-${esc(d.verdict)}">
        <h3>${titles[d.verdict] || esc(d.verdict)}</h3>
        <p>${esc(note)}</p>
        <div class="ruler" role="img" aria-label="${esc(rulerLabel)}">
          <div class="ruler-track">
            <div class="ruler-refer" style="left:${lo * 100}%;width:${width}%" ${width < 1 ? "hidden" : ""}></div>
            <div class="ruler-mark" style="left:${mark}%"></div>
          </div>
          <div class="ruler-scale"><span>Normal</span><span>50%</span><span>Abnormal</span></div>
          <p class="ruler-key">The outlined band is the referral zone: probabilities where confidence is below your limit.</p>
        </div>
        <dl class="facts">
          <div><dt>Probability abnormal</dt><dd>${pct(d.p_abnormal, 1)}</dd></div>
          <div><dt>Confidence</dt><dd>${pct(d.confidence, 1)}</dd></div>
          <div><dt>Probabilities used</dt><dd>${used}</dd></div>
          <div><dt>Analysis time</dt><dd>${esc(d.elapsed_ms)} ms</dd></div>
        </dl>
      </article>
      <div class="detail">
        <h4>Model by model</h4>
        <div class="tablewrap"><table>
          <thead><tr><th scope="col">Model</th><th scope="col">Raw</th><th scope="col">Calibrated</th><th scope="col">Temperature</th></tr></thead>
          <tbody>${rows.join("")}${average}</tbody>
        </table></div>
      </div>
      ${cam}${warn}`;
    els.out.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  }

  /* ----------------------------------------------------------- study results */

  const COLORS = { hema: "#4b3f8f", eosin: "#c2415e", teal: "#1f8a83", amber: "#b7791f" };

  function groupedBars(labels, series, { max = 1, ticks = 5, label = "" } = {}) {
    const W = 640, H = 300, m = { l: 42, r: 10, t: 14, b: 34 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const gw = iw / labels.length;
    const bw = Math.min(34, (gw * 0.8) / series.length);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">`;
    for (let i = 0; i <= ticks; i++) {
      const v = (max / ticks) * i;
      const y = m.t + ih - (v / max) * ih;
      svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" stroke="#e3e8ee"/>`;
      svg += `<text x="${m.l - 6}" y="${y + 3}" text-anchor="end">${v === 0 ? "0" : v.toFixed(2).replace(/^0\./, ".")}</text>`;
    }
    labels.forEach((name, gi) => {
      const x0 = m.l + gi * gw + (gw - bw * series.length) / 2;
      series.forEach((s, si) => {
        const v = s.values[gi];
        const h = (v / max) * ih;
        const x = x0 + si * bw;
        svg += `<rect x="${x}" y="${m.t + ih - h}" width="${bw - 2}" height="${h}" rx="3" fill="${s.color}"><title>${esc(name)}, ${esc(s.name)}: ${f3(v)}</title></rect>`;
        svg += `<text class="val" x="${x + (bw - 2) / 2}" y="${m.t + ih - h - 4}" text-anchor="middle">${f3(v)}</text>`;
      });
      svg += `<text x="${m.l + gi * gw + gw / 2}" y="${H - 12}" text-anchor="middle">${esc(name)}</text>`;
    });
    svg += "</svg>";
    const legend = series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("");
    return `<div class="chart">${svg}<p class="legend">${legend}</p></div>`;
  }

  function table(caption, head, rows) {
    return `<div class="tablewrap"><table><caption>${esc(caption)}</caption>
      <thead><tr>${head.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  const R = { data: null };
  const nm = (id) => R.data.names[id];
  const ordered = () => R.data.order;

  function renderGap() {
    const e = R.data.external, ids = ordered();
    return `<p>Accuracy on SIPaKMeD test images is 94% to 97% for every model. On Herlev it falls to 51% to 79%, and AUC to 0.65 to 0.71. Herlev is about 74% abnormal, so accuracy alone flatters it: answering abnormal every time would score about 0.74.</p>
      ${groupedBars(ids.map(nm), [
        { name: "SIPaKMeD test (same dataset)", color: COLORS.teal, values: ids.map((i) => e[i].src.acc[0]) },
        { name: "Herlev (new dataset)", color: COLORS.eosin, values: ids.map((i) => e[i].ext.acc[0]) },
      ], { label: "Accuracy on SIPaKMeD versus Herlev for each model" })}
      ${table("Mean\u00b1std over three seeds", ["Model", "SIPaKMeD acc", "SIPaKMeD AUC", "SIPaKMeD ECE", "Herlev acc", "Herlev sens", "Herlev spec", "Herlev AUC", "Herlev ECE"],
        ids.map((i) => [nm(i), pm(e[i].src.acc), pm(e[i].src.auc), pm(e[i].src.ece), pm(e[i].ext.acc), pm(e[i].ext.sens), pm(e[i].ext.spec), pm(e[i].ext.auc), pm(e[i].ext.ece)]))}`;
  }

  function renderCalibration() {
    const c = R.data.calibration_ece, s = R.data.selective, ids = ordered();
    return `<p>Temperature scaling fitted on SIPaKMeD lowers the calibration error (ECE) on Herlev, but only part of the way. The oracle bars use Herlev labels, so they show what calibration could reach with data from the new site. The class-balance correction helped some models and hurt others, and abstaining on the least confident half of the cells raised accuracy by only 4 to 9 points.</p>
      ${groupedBars(ids.map(nm), [
        { name: "Raw", color: COLORS.hema, values: ids.map((i) => c[i].raw[0]) },
        { name: "Temperature from SIPaKMeD", color: COLORS.teal, values: ids.map((i) => c[i].temp[0]) },
        { name: "Oracle (uses Herlev labels)", color: COLORS.amber, values: ids.map((i) => c[i].oracle[0]) },
      ], { max: 0.4, ticks: 4, label: "Calibration error on Herlev for each model, lower is better" })}
      ${table("Herlev ECE, lower is better (mean\u00b1std)", ["Model", "Raw", "Temperature", "Temperature + prior", "Oracle"],
        ids.map((i) => [nm(i), pm(c[i].raw), pm(c[i].temp), pm(c[i].temp_prior), pm(c[i].oracle)]))}
      <p></p>
      ${table("Herlev accuracy when the least confident cells are referred", ["Model", "All cells", "Most confident 80%", "Most confident 50%", "AURC (lower is better)"],
        ids.map((i) => [nm(i), f3(s[i].full), f3(s[i].c80), f3(s[i].c50), f3(s[i].aurc)]))}`;
  }

  function renderEnsembles() {
    return `<p>Averaging models improved calibration but not AUC. The intervals come from resampling the 917 Herlev images. They overlap, so AUC differences of about 0.04 are not conclusive. The best single model is only known after seeing Herlev labels, while the ensemble needs no such choice.</p>
      ${table("Herlev results for ensembles", ["Configuration", "Accuracy", "Balanced acc.", "AUC (95% interval)", "ECE raw", "ECE temperature", "AURC", "Acc. at 50% coverage"],
        R.data.ensembles.map((r) => [r.name, f3(r.acc), f3(r.bal), r.auc, f3(r.ece_raw), f3(r.ece_temp), f3(r.aurc), f3(r.acc50)]))}`;
  }

  function renderInDomain() {
    const d = R.data.in_domain, ids = d.order;
    return `<p>Trained on Herlev itself (5-fold cross-validation), the same models reach 92% to 94% accuracy. So the drop in the other tabs comes from dataset shift, not from Herlev being hard. Herlev has no patient identifiers, so these in-domain numbers may be optimistic.</p>
      ${groupedBars(ids.map(nm), [
        { name: "Trained on Herlev (in-domain)", color: COLORS.teal, values: ids.map((i) => d.rows[i].acc[0]) },
        { name: "Trained on SIPaKMeD (external)", color: COLORS.eosin, values: ids.map((i) => d.rows[i].ext_acc[0]) },
      ], { label: "Accuracy on Herlev when trained on Herlev versus trained on SIPaKMeD" })}
      ${table("Mean\u00b1std over five folds (in-domain) or three seeds (external)", ["Model", "In-domain acc", "In-domain bal. acc", "In-domain AUC", "In-domain ECE", "External acc", "External AUC"],
        ids.map((i) => { const r = d.rows[i]; return [nm(i), pm(r.acc), pm(r.bal), pm(r.auc), pm(r.ece), pm(r.ext_acc), pm(r.ext_auc)]; }))}`;
  }

  async function loadResults() {
    const panel = $("#panel"), tabsEl = $(".tabs");
    try {
      const res = await fetch("/api/results");
      if (!res.ok) throw new Error(res.statusText);
      R.data = await res.json();
    } catch (err) {
      panel.textContent = `Could not load the study results: ${err.message}`;
      return;
    }
    const tabs = [
      { id: "gap", label: "Accuracy gap", render: renderGap },
      { id: "calibration", label: "Calibration", render: renderCalibration },
      { id: "ensembles", label: "Ensembles", render: renderEnsembles },
      { id: "indomain", label: "Trained on Herlev", render: renderInDomain },
    ];
    const buttons = tabs.map((t) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "tab"; b.id = `tab-${t.id}`; b.textContent = t.label;
      b.setAttribute("role", "tab"); b.setAttribute("aria-controls", "panel");
      tabsEl.appendChild(b);
      return b;
    });
    const select = (idx) => {
      buttons.forEach((b, i) => { b.setAttribute("aria-selected", String(i === idx)); b.tabIndex = i === idx ? 0 : -1; });
      panel.setAttribute("aria-labelledby", buttons[idx].id);
      panel.innerHTML = tabs[idx].render();
    };
    buttons.forEach((b, i) => {
      b.addEventListener("click", () => select(i));
      b.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        const next = (i + (e.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
        select(next); buttons[next].focus();
      });
    });
    select(0);
  }

  loadModels();
  loadResults();
})();
