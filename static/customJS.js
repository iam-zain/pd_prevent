const config = window.factorConfig;

// ── Globals to hold latest result for PDF/report ─────────────────────────────
let _lastResult   = null;   // server JSON
let _lastFormData = null;   // FormData used for /submit

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

const getRiskCategoryMessage = (riskCategory) =>
  riskCategory === "PROFILE_1"
    ? `<span style="color:#c0392b;font-size:1.4em;font-weight:700;">&#9888; High Risk (PD)</span>`
    : `<span style="color:#27ae60;font-size:1.4em;font-weight:700;">&#10003; Low Risk (Healthy)</span>`;

const getConfidenceBandColor = (band) => {
  if (band === "High Confidence")     return "#27ae60";
  if (band === "Moderate Confidence") return "#e67e22";
  return "#c0392b";
};

const miniBar = (prob, color, width) => {
  width = width || 160;
  const pct = Math.round(prob * 100);
  return '<div style="display:inline-flex;align-items:center;gap:6px;">'
    + '<span style="min-width:34px;font-size:12px;">' + pct + '%</span>'
    + '<div style="background:#e8e8e8;border-radius:4px;overflow:hidden;width:' + width + 'px;height:10px;">'
    + '<div style="background:' + color + ';width:' + pct + '%;height:100%;border-radius:4px;"></div>'
    + '</div></div>';
};

// ─────────────────────────────────────────────────────────────────────────────
// SUMMARY SIDEBAR
// ─────────────────────────────────────────────────────────────────────────────

const getSummaryItemHTML = function(item) {
  return '<li id="summary' + item.id + '" class="list-group-item d-flex justify-content-between lh-condensed">'
    + '<div><h6 class="my-0">' + item.label + '</h6>'
    + (item.description ? '<small class="text-muted">' + item.description + '</small>' : '')
    + '</div><span class="text-muted" id="summaryVal' + item.id + '">' + item.value + '</span></li>';
};

// ─────────────────────────────────────────────────────────────────────────────
// FEATURE SLIDER ROW
// ─────────────────────────────────────────────────────────────────────────────

const getItemHTML = function(item) {
  const step = item.step || 1;
  return '<div class="form-group mb-4" id="container' + item.id + '">'
    + '<label for="' + item.id + '">' + item.label
    + ' <span class="text-muted small">(min: ' + item.min + ', max: ' + item.max + ')</span>'
    + ' <span data-toggle="tooltip" title="Remove this factor" data-placement="top" class="remove-feature" onclick="removeFeature(' + item.id + ')">&#128683;</span>'
    + ' <span class="info-feature" onclick="showInfo(' + item.id + ')">?</span>'
    + '</label>'
    + '<div class="row">'
    + '<div class="col-9 d-flex">'
    + '<input id="' + item.id + '" min="' + item.min + '" max="' + item.max + '" type="range"'
    + ' step="' + step + '" value="' + item.value + '"'
    + ' class="form-control-range factors-input" oninput="handleRangeChange(this);" />'
    + '</div>'
    + '<div class="col-3 d-flex">'
    + '<input min="' + item.min + '" max="' + item.max + '" type="number"'
    + ' step="' + step + '" value="' + item.value + '" required'
    + ' id="' + item.id + '_current"'
    + ' class="form-control factors-input factors-input-box"'
    + ' onchange="handleRangeInput(this, ' + item.id + ');" />'
    + '</div></div></div>';
};

// ─────────────────────────────────────────────────────────────────────────────
// INFO MODAL
// ─────────────────────────────────────────────────────────────────────────────

const getInfoModalBody = function(item) {
  return '<p>' + item.info + '</p><img src="' + item.img + '" class="info-img"/>';
};

// ─────────────────────────────────────────────────────────────────────────────
// RESULT MODAL — SIMPLE VIEW (user-friendly)
// ─────────────────────────────────────────────────────────────────────────────

const getSimpleResultHTML = function(r, userName, nFeatures) {
  const bandColor   = getConfidenceBandColor(r.confidenceBand);
  const isPD        = r.userStatus === "PROFILE_1";
  const display_pct = isPD ? Math.round(r.finalProb * 100)
                            : Math.round((1 - r.finalProb) * 100);
  const pd_pct      = Math.round(r.finalProb * 100);
  const hl_pct      = 100 - pd_pct;
  const conf_pct    = r.confidenceScore;
  const name        = (userName && userName.trim()) ? userName.trim() : "the patient";
  const feats       = nFeatures || r.numProvided || "several";

  const sentence = isPD
    ? "Based on the analysis of " + feats + " features, the model predicts that "
      + name + " has a <b style=\"color:#c0392b;\">" + pd_pct + "% probability</b> "
      + "of having Parkinson&#39;s Disease (High Risk)."
    : "Based on the analysis of " + feats + " features, the model predicts that "
      + name + " has an <b style=\"color:#27ae60;\">" + hl_pct + "% probability</b> "
      + "of being Healthy, indicating Low Risk of Parkinson&#39;s Disease.";

  return ''
    + '<div style="text-align:center;padding:20px 10px 6px;">'
    + getRiskCategoryMessage(r.userStatus)
    + '<div style="margin:14px auto;width:180px;">'
    + '<div id="resultProgressBar" role="progressbar" style="--value:' + display_pct + '"></div>'
    + '</div>'
    + '<div style="font-size:13px;color:#666;margin-top:4px;">'
    + '<b style="color:#c0392b;">' + pd_pct + '% PD</b>'
    + '&nbsp;&nbsp;&middot;&nbsp;&nbsp;'
    + '<b style="color:#27ae60;">' + hl_pct + '% Healthy</b>'
    + '</div>'
    + '<div style="width:90%;margin:12px auto 0;font-size:13px;color:#555;line-height:1.6;background:#f8f9fa;border-radius:8px;padding:12px 16px;">'
    + sentence
    + '</div>'
    + '</div>'

    + '<hr style="width:90%;margin:12px auto;"/>'

    + '<div style="width:90%;margin:0 auto 18px;">'
    + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
    + '<span style="font-size:14px;font-weight:600;color:#333;">Assessment Confidence</span>'
    + '<span style="font-size:14px;font-weight:700;color:' + bandColor + ';">' + r.confidenceBand + '</span>'
    + '</div>'
    + '<div style="background:#e8e8e8;border-radius:8px;overflow:hidden;height:14px;">'
    + '<div style="background:' + bandColor + ';width:' + conf_pct + '%;height:100%;border-radius:8px;"></div>'
    + '</div>'
    + '<div style="display:flex;justify-content:space-between;font-size:11px;color:#aaa;margin-top:3px;">'
    + '<span>Low</span><span>High</span></div>'
    + '<div style="margin-top:6px;font-size:12px;color:#888;text-align:center;">'
    + 'See &quot;View Full Report&quot; for detailed model breakdown.'
    + '</div></div>'

    + '<div style="width:90%;margin:0 auto 10px;font-size:11.5px;color:#aaa;text-align:center;line-height:1.5;">'
    + 'This tool is for research &amp; screening purposes only and does <em>not</em> constitute a '
    + 'clinical diagnosis. Please consult a qualified neurologist for medical evaluation.'
    + '</div>';
};


// ─────────────────────────────────────────────────────────────────────────────
// RESULT MODAL — FULL REPORT VIEW
// ─────────────────────────────────────────────────────────────────────────────

const getFullReportHTML = function(r) {
  const bandColor = getConfidenceBandColor(r.confidenceBand);
  const isPD      = r.userStatus === "PROFILE_1";
  const pd_pct    = Math.round(r.finalProb * 100);
  const hl_pct    = 100 - pd_pct;

  let voteRows = "";
  Object.entries(r.subModelVotes || {}).forEach(function(entry) {
    const model = entry[0], vote = entry[1];
    const match = (vote === "PD") === isPD;
    const color  = vote === "PD" ? "#c0392b" : "#27ae60";
    const icon   = match ? "&#10003;" : "&#10007;";
    voteRows += '<tr>'
      + '<td style="padding:5px 10px;font-size:13px;">' + icon + ' ' + model + '</td>'
      + '<td style="padding:5px 10px;font-weight:700;color:' + color + ';font-size:13px;">' + vote + '</td>'
      + '</tr>';
  });

  const probRow = function(label, prob, color) {
    return '<tr>'
      + '<td style="padding:4px 10px;font-size:13px;">' + label + '</td>'
      + '<td style="padding:4px 10px;">' + miniBar(prob, color) + '</td>'
      + '</tr>';
  };

  // Feature count note — use numProvided (actual user input count)
  const nProvided   = r.numProvided || 0;
  const nTotal      = (r.usedFeatures||[]).length + (r.defaultedFeatures||[]).length;
  const nDefaulted  = Math.max(0, nTotal - nProvided);
  const usedNote    = nProvided > 0
    ? nProvided + ' feature(s) measured; ' + nDefaulted + ' used population-average defaults.'
    : '';

  const finalColor = isPD ? "#c0392b" : "#27ae60";

  return '<div style="padding:10px 5px;width:100%;">'
    + '<div style="text-align:center;border-bottom:2px solid #0b4f58;padding-bottom:12px;margin-bottom:16px;">'
    + '<h5 style="color:#0b4f58;font-weight:700;margin:0;">PD-PREVeNT — Full Assessment Report</h5>'
    + '<small style="color:#888;">Three-Model Ensemble &middot; Parkinson&#39;s Disease Risk Evaluation</small>'
    + '</div>'

    + '<div style="text-align:center;margin-bottom:16px;">'
    + getRiskCategoryMessage(r.userStatus)

    + '</div>'

    // Confidence
    + '<div style="background:#f8f9fa;border-radius:8px;padding:12px 16px;margin-bottom:14px;">'
    + '<div style="font-size:13px;font-weight:700;color:' + bandColor + ';margin-bottom:6px;">'
    + 'Confidence: ' + r.confidenceBand + ' (' + r.confidenceScore + '%)</div>'
    + '<table style="width:100%;font-size:12px;">'
    + '<tr><td style="padding:2px 6px;color:#666;">Probability Margin</td>'
    + '<td style="padding:2px 6px;">' + miniBar(r.probabilityMargin, bandColor, 100) + '</td>'
    + '<td style="padding:2px 6px;color:#aaa;font-size:11px;">Distance from 50/50 boundary</td></tr>'
    + '<tr><td style="padding:2px 6px;color:#666;">Model Consensus</td>'
    + '<td style="padding:2px 6px;">' + miniBar(r.modelConsensus, bandColor, 100) + '</td>'
    + '<td style="padding:2px 6px;color:#aaa;font-size:11px;">Fraction of sub-models agreeing</td></tr>'
    + '</table></div>'

    // Sub-model votes
    + '<div style="margin-bottom:14px;">'
    + '<div style="font-size:13px;font-weight:700;color:#333;margin-bottom:4px;">Sub-Model Votes</div>'
    + '<table style="width:100%;">' + voteRows + '</table></div>'

    // Probabilities — M3 sub-metrics hidden
    + '<div style="margin-bottom:14px;">'
    + '<div style="font-size:13px;font-weight:700;color:#333;margin-bottom:4px;">Per-Model Probabilities P(PD)</div>'
    + '<table style="width:100%;">'
    + probRow("M1  Feature-Stacker",  r.probM1,  "#3498db")
    + probRow("M2  XGB-Only",         r.probM2,  "#9b59b6")
    + probRow("M3  CB-Collab Filter", r.probM3,  "#1abc9c")
    + probRow("&#9733; Final Ensemble", r.finalProb, finalColor)
    + '</table>'
    + (usedNote ? '<div style="margin-top:6px;font-size:11px;color:#aaa;">' + usedNote + '</div>' : '')
    + '</div>'
    + '<hr/>'
    + '<div style="font-size:11.5px;color:#999;margin-bottom:10px;">'
    + '<b>Confidence</b> = 0.70 &times; Probability Margin + 0.30 &times; Model Consensus. '
    + 'M1 = LR+SVM+XGB feature stacker &middot; M2 = XGBoost global &middot; M3 = Collaborative Filter (cosine + Pearson + Euclidean, K=7).'
    + '</div>'
    + '<div style="font-size:11px;color:#bbb;text-align:center;">'
    + 'PD-PREVeNT is an academic research tool and does <em>not</em> constitute a clinical diagnosis. '
    + 'Please consult a qualified neurologist for medical evaluation.'
    + '</div></div>';
};


// ─────────────────────────────────────────────────────────────────────────────
// MODAL FOOTER BUTTONS
// ─────────────────────────────────────────────────────────────────────────────

const buildModalFooter = function(showingFull) {
  return '<button type="button" id="toggleReportBtn" class="btn btn-outline-secondary btn-sm" onclick="toggleFullReport()">'
    + (showingFull ? '&#8592; Back to Summary' : 'View Full Report')
    + '</button> '
    + '<button type="button" class="btn btn-outline-primary btn-sm" onclick="downloadPDF()">'
    + '&#11015; Download PDF</button> '
    + '<button type="button" class="btn btn-secondary btn-sm" data-dismiss="modal">Close</button>';
};

let _showingFullReport = false;

const toggleFullReport = function() {
  _showingFullReport = !_showingFullReport;
  document.getElementById("resultModalBody").innerHTML   = _showingFullReport ? getFullReportHTML(_lastResult) : getSimpleResultHTML(_lastResult, (document.getElementById("username")||{}).value||"", _lastResult.numProvided);
  document.getElementById("resultModalFooter").innerHTML = buildModalFooter(_showingFullReport);
};

// ─────────────────────────────────────────────────────────────────────────────
// PDF DOWNLOAD
// ─────────────────────────────────────────────────────────────────────────────

const downloadPDF = async function() {
  if (!_lastResult || !_lastFormData) return;

  const featuresForPdf = [];
  const n = parseInt(_lastFormData.get("num_features") || "0");
  for (let i = 1; i <= n; i++) {
    const name  = _lastFormData.get("Feature" + i)       || "";
    const score = _lastFormData.get("Score_Feature" + i) || "";
    const label = (config[name] && config[name].label) ? config[name].label : name;
    featuresForPdf.push({ name: name, label: label, score: score });
  }

  const payload = {
    patientName: (document.getElementById("username") || {}).value || "User",
    age:         _lastFormData.get("Age")    || "",
    gender:      _lastFormData.get("Gender") === "0" ? "Male" : "Female",
    features:    featuresForPdf,
    result:      _lastResult,
  };

  const btns = document.querySelectorAll("[onclick='downloadPDF()']");
  btns.forEach(function(b){ b.innerText = "Generating\u2026"; b.disabled = true; });

  try {
    const resp = await fetch("/generate_pdf", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error("PDF generation failed: " + resp.statusText);
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = "PD_PREVeNT_Report.pdf";
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Could not generate PDF: " + err.message);
  } finally {
    btns.forEach(function(b){ b.innerText = "\u2B07 Download PDF"; b.disabled = false; });
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// SWS ERROR HELPERS
// ─────────────────────────────────────────────────────────────────────────────

const showSWSError = function(opts) {
  document.querySelector("label.text-danger.small.font-weight-bold.error-helper").innerText = opts.message;
  document.querySelector(".selectpicker-wrp").classList.add("error-show");
};
const removeSWSError = function() {
  document.querySelector(".selectpicker-wrp").classList.remove("error-show");
};

// ─────────────────────────────────────────────────────────────────────────────
// ADD FACTOR
// ─────────────────────────────────────────────────────────────────────────────

const addNewFactor = function() {
  const selectedItemValue = $(".selected-item")[0].innerText.trim();
  if (!selectedItemValue || selectedItemValue === "Select a test or symptom") {
    showSWSError({ message: "Please select a value from the dropdown" });
    return;
  }
  removeSWSError();

  const key = Object.keys(config).find(function(k){ return config[k].label === selectedItemValue; });
  if (document.querySelectorAll("#container" + key).length) {
    showSWSError({ message: "This value already exists" });
    return;
  }

  $(getItemHTML(Object.assign({ id: key }, config[key]))).insertAfter(document.getElementById("swsTopEl"));
  $(getSummaryItemHTML(Object.assign({ id: key }, config[key]))).insertBefore(document.getElementById("listOfRisks"));
  updateCountOnFactorsChange();
  $('[data-toggle="tooltip"]').tooltip({ container: "body" });
};

// ─────────────────────────────────────────────────────────────────────────────
// RANGE SYNC
// ─────────────────────────────────────────────────────────────────────────────

const handleRangeChange = function(props) {
  document.querySelector("#" + props.id + "_current").value       = props.value;
  document.getElementById("summaryVal" + props.id).innerText = props.value;
};

const handleRangeInput = function(event, el) {
  if (event.stopPropagation) { event.stopPropagation(); event.preventDefault(); }
  else if (window.event)     { window.event.cancelBubble = true; }
  const min = config[el.id].min, max = config[el.id].max;
  if (Number(event.value) > Number(max)) { alert("Input out of range"); setTimeout(function(){ event.value = max; }, 10); }
  if (Number(event.value) < Number(min)) { alert("Input out of range"); setTimeout(function(){ event.value = min; }, 10); }
  document.getElementById(el.id).value = event.value;
  document.getElementById("summaryVal" + el.id).innerText = event.value;
};

// ─────────────────────────────────────────────────────────────────────────────
// DOCUMENT READY
// ─────────────────────────────────────────────────────────────────────────────

$(document).ready(function() {
  $('[data-toggle="tooltip"]').tooltip({ container: "body" });

  if (window.CSS && window.CSS.registerProperty) {
    window.CSS.registerProperty({ name: "--percentage", syntax: "<number>", inherits: true, initialValue: 0 });
  }

  const frag = new DocumentFragment();
  Object.values(config).forEach(function(item) {
    const el = document.createElement("div");
    el.classList.add("select-item");
    el.innerText = item.label;
    frag.append(el);
  });
  $("#selectOptionsContainer").append(frag);

  setTimeout(function() {
    registerSelectListeners();
    registerResetFormListener();
    setContainerHeightObserver();
  }, 10);
});

// ─────────────────────────────────────────────────────────────────────────────
// LISTENERS
// ─────────────────────────────────────────────────────────────────────────────

const registerResetFormListener = function() {
  document.getElementById("resetButton").addEventListener("click", resetForm);
};

const registerSelectListeners = function() {
  document.querySelector(".selected-item").addEventListener("click", function() {
    this.nextElementSibling.classList.toggle("hidden");
    this.nextElementSibling.nextElementSibling.classList.toggle("hidden");
  });
  document.querySelectorAll(".select-item").forEach(function(item) {
    item.addEventListener("click", function() {
      $(".selected-item")[0].textContent = this.textContent;
      this.parentNode.classList.add("hidden");
      this.parentNode.nextElementSibling.classList.add("hidden");
    });
  });
  document.querySelector(".select-search").addEventListener("keyup", function() {
    const filter = this.value.toUpperCase();
    Array.from(this.previousElementSibling.children).forEach(function(item) {
      item.style.display = (item.textContent || item.innerText).toUpperCase().indexOf(filter) > -1 ? "" : "none";
    });
  });
  document.addEventListener("click", function(event) {
    const box = document.querySelector(".select-box");
    if (!box.contains(event.target)) {
      document.querySelector(".select-items").classList.add("hidden");
      document.querySelector(".select-search").classList.add("hidden");
    }
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// FORM RESET
// ─────────────────────────────────────────────────────────────────────────────

const resetForm = function(event) {
  event.preventDefault();
  document.getElementById("pdpForm").reset();
  document.querySelectorAll(".remove-feature").forEach(function(el){ el.click(); });
  removeSWSError();
  $(".selected-item")[0].innerText = "Select a test or symptom ";
};

// ─────────────────────────────────────────────────────────────────────────────
// FORM VALIDATION & SUBMIT
// ─────────────────────────────────────────────────────────────────────────────

const validateAndGetFormData = function() {
  const genderEl        = document.querySelector('[name="gender"]:checked');
  const ageEl           = document.querySelector("#Age");
  const factorsInputArr = document.querySelectorAll("input.form-control-range.factors-input");
  if (!genderEl || !genderEl.value) throw new Error("Please select a gender");
  if (!ageEl    || !ageEl.value)    throw new Error("Please enter your age");
  if (factorsInputArr.length < 10)  throw new Error("Please select at least 10 features to continue");
  const formData = new FormData();
  formData.append("Gender", genderEl.value.toLowerCase() === "male" ? 0 : 1);
  formData.append("Age",    ageEl.value);
  formData.append("num_features", factorsInputArr.length);
  factorsInputArr.forEach(function(item, i) {
    formData.append("Feature" + (i+1),       item.id);
    formData.append("Score_Feature" + (i+1), item.value);
  });
  return formData;
};

const showLoadingOnSubmitCTA = function() {
  document.getElementById("fullPageLoader").classList.add("d-flex");
  const cta = document.getElementById("submitFormCTA");
  cta.innerText = "Predicting \u2014 please wait\u2026";
  cta.classList.add("disabled");
};

const resetSubmitButton = function() {
  document.getElementById("fullPageLoader").classList.remove("d-flex");
  const cta = document.getElementById("submitFormCTA");
  cta.innerText = "Calculate my risk profile";
  cta.classList.remove("disabled");
};

$("#pdpForm").bind("submit", async function(e) {
  e.preventDefault();
  try {
    const formData = validateAndGetFormData();
    _lastFormData  = formData;
    showLoadingOnSubmitCTA();

    const resp   = await fetch("/submit", { method: "POST", body: formData });
    const result = await resp.json();
    if (result.error) throw new Error(result.error);

    _lastResult        = result;
    _showingFullReport = false;

    document.getElementById("resultModalLabel").innerText  = "Prediction Results";
    const uName = (document.getElementById("username") || {}).value || "";
    document.getElementById("resultModalBody").innerHTML   = getSimpleResultHTML(result, uName, result.numProvided);
    document.getElementById("resultModalFooter").innerHTML = buildModalFooter(false);
    $("#resultModal").modal();
  } catch(err) {
    resetSubmitButton();
    alert(err.message || err);
  }
});

$("#resultModal").on("hide.bs.modal", resetSubmitButton);

// ─────────────────────────────────────────────────────────────────────────────
// MISC HELPERS
// ─────────────────────────────────────────────────────────────────────────────

const updateCountOnFactorsChange = function() {
  document.querySelector("span.badge.badge-secondary.badge-pill.factors-count").innerText =
    document.querySelectorAll("input.form-control-range.factors-input").length;
};

const stopOnEnter = function(e) {
  if (e && (e.which === 13 || e.keyCode === 13)) return false;
};
$("#pdpForm").keypress(stopOnEnter);

const removeFeature = function(element) {
  const c = document.getElementById("container" + element.id);
  const s = document.getElementById("summary"   + element.id);
  if (c) c.remove(); if (s) s.remove();
  updateCountOnFactorsChange();
  $(".tooltip").hide();
};

const showInfo = function(element) {
  const item = factorConfig[element.id];
  document.getElementById("resultModalLabel").innerText  = item.label;
  document.getElementById("resultModalBody").innerHTML   = getInfoModalBody(item);
  document.getElementById("resultModalFooter").innerHTML =
    '<button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>';
  $("#resultModal").modal();
};

const showDataModal = function(element) {
  document.getElementById("resultModalLabel").innerHTML  = element.innerText;
  document.getElementById("resultModalBody").innerHTML   = element.dataset["info"];
  document.getElementById("resultModalFooter").innerHTML =
    '<button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>';
  $("#resultModal").modal();
};

const setContainerHeightObserver = function() {
  const setMaxHeight = function() {
    const rect = document.querySelector(".selected-item").getBoundingClientRect();
    document.getElementById("selectOptionsContainer").style.maxHeight =
      Math.ceil(window.innerHeight - rect.bottom - 60) + "px";
  };
  window.addEventListener("resize", setMaxHeight);
  setMaxHeight();
};