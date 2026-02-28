"""
app.py — PD-PREVeNT Flask web server

Routes
------
GET  /              → index.html
GET  /box           → box.html
POST /submit        → prediction JSON
GET  /features      → list of clinical feature names
GET  /debug         → model health check (open in browser to verify setup)
POST /generate_pdf  → PDF download
"""

import os
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_DIR = "saved_models"

try:
    from joblib import load as joblib_load
    m1         = joblib_load(os.path.join(MODEL_DIR, "model1_feature_ensemble.joblib"))
    m1_stacker = joblib_load(os.path.join(MODEL_DIR, "m1_stacker.joblib"))
    m2         = joblib_load(os.path.join(MODEL_DIR, "model2_xgb_only.joblib"))
    collab     = joblib_load(os.path.join(MODEL_DIR, "model3_collab_filter.joblib"))
    meta       = joblib_load(os.path.join(MODEL_DIR, "meta_model.joblib"))

    # ── Fix: inject M2 feature columns saved separately during training ───────
    # Training saves list(X5.columns) → m2_feature_cols.joblib.
    # Without this, _get_m2_columns() falls back to reconstruction which
    # produces the wrong column count → XGBoost crashes → Flask returns
    # an HTML 500 page → frontend JSON.parse fails.
    m2_feature_cols_path = os.path.join(MODEL_DIR, "m2_feature_cols.joblib")
    if os.path.exists(m2_feature_cols_path):
        m2["feature_cols"] = joblib_load(m2_feature_cols_path)
        print(f"OK: m2_feature_cols injected ({len(m2['feature_cols'])} cols).", flush=True)
    else:
        print("Warning: m2_feature_cols.joblib not found — M2 column reconstruction will be used.", flush=True)

    # ── Load neutral imputation means (class-balanced, avoids PD bias) ────────
    neutral_m2_path = os.path.join(MODEL_DIR, "neutral_means_m2.joblib")
    neutral_m3_path = os.path.join(MODEL_DIR, "neutral_means_m3.joblib")
    neutral_means_m2 = joblib_load(neutral_m2_path) if os.path.exists(neutral_m2_path) else {}
    neutral_means_m3 = joblib_load(neutral_m3_path) if os.path.exists(neutral_m3_path) else {}
    if neutral_means_m2:
        m2["neutral_means"] = neutral_means_m2
        print(f"OK: neutral_means_m2 injected ({len(neutral_means_m2)} cols).", flush=True)
    if neutral_means_m3:
        collab["neutral_means"] = neutral_means_m3
        print(f"OK: neutral_means_m3 injected ({len(neutral_means_m3)} cols).", flush=True)

    MODELS_LOADED = True
    print("OK: All model files loaded.", flush=True)
except FileNotFoundError as e:
    print(f"Warning: Model not found: {e}", flush=True)
    MODELS_LOADED = False
    m1 = m1_stacker = m2 = collab = meta = None
    neutral_means_m2 = neutral_means_m3 = {}

from prediction import predict_patient   # noqa
from pdf_report import generate_report   # noqa

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/box")
def box():
    return render_template("box.html")


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG — open http://127.0.0.1:5000/debug in browser to verify setup
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/debug")
def debug():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503

    import numpy as np
    from prediction import _get_m2_columns

    m1_feats   = sorted(m1.keys())
    first_feat = m1_feats[0]
    bundle     = m1[first_feat]

    m2_cols       = _get_m2_columns(m2, m1, collab)
    collab_cols   = list(collab["imp_cols"])
    active_feats  = m1_stacker["active_features"]

    # Check if SVM is in bundles
    has_svm = "svm" in bundle

    # Sample bundle sub-model check
    sample_sub_models = list(bundle.keys())

    # M2 feature count vs actual XGB
    m2_xgb_n = m2["xgb"].n_features_in_

    info = {
        "status": "OK",
        "m1_total_features": len(m1_feats),
        "m1_first_feature_bundle_keys": sample_sub_models,
        "m1_has_svm": has_svm,
        "m1_active_features_count": len(active_feats),
        "m1_active_features": active_feats,
        "m1_all_features": m1_feats,
        "m2_xgb_expects_n_features": m2_xgb_n,
        "m2_reconstructed_columns_count": len(m2_cols),
        "m2_reconstructed_columns": m2_cols,
        "m2_column_count_match": len(m2_cols) == m2_xgb_n,
        "m3_collab_columns_count": len(collab_cols),
        "m3_collab_columns": collab_cols,
        "meta_coef": list(meta.coef_[0]),
        "warning_if_any": (
            "SVM missing from M1 bundle — predictions will be biased!"
            if not has_svm else
            ("M2 column count MISMATCH — M2 predictions unreliable!"
             if len(m2_cols) != m2_xgb_n else
             "None — all checks passed")
        )
    }
    return jsonify(info)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES LIST
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/features")
def features():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503
    clinical = [c for c in collab["imp_cols"] if c not in ("age", "sex")]
    return jsonify({"features": clinical})


# ─────────────────────────────────────────────────────────────────────────────
# SUBMIT — run prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/submit", methods=["POST"])
def submit():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded — run training first"}), 503

    try:
        age    = float(request.form.get("Age", ""))
        gender = int(request.form.get("Gender", ""))
        n_feat = int(request.form.get("num_features", "0"))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid Age / Gender / num_features"}), 400

    if n_feat < 10:
        return jsonify({"error": "Please select at least 10 features"}), 400

    input_dict = {"age": age, "sex": gender}
    for i in range(1, n_feat + 1):
        feat_name  = request.form.get(f"Feature{i}", "").strip()
        feat_value = request.form.get(f"Score_Feature{i}", "")
        if not feat_name:
            continue
        try:
            input_dict[feat_name] = float(feat_value)
        except (ValueError, TypeError):
            pass

    print(f"[submit] age={age}, gender={gender}, features={n_feat}", flush=True)
    print(f"[submit] input_dict keys: {list(input_dict.keys())}", flush=True)

    if len(input_dict) - 2 < 10:
        return jsonify({"error": "At least 10 valid feature values required"}), 400

    try:
        final_prob, risk_label, confidence, breakdown = predict_patient(
            input_dict, m1, m1_stacker, m2, collab, meta
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    user_status      = "PROFILE_1" if "High" in risk_label else "PROFILE_2"
    confidence_score = round(confidence["combined_score"] * 100)

    return jsonify({
        "userStatus":        user_status,
        "riskLabel":         risk_label,
        "confidenceScore":   confidence_score,
        "confidenceBand":    confidence["band"],
        "probabilityMargin": confidence["probability_margin"],
        "modelConsensus":    confidence["model_consensus"],
        "subModelVotes":     confidence["sub_model_votes"],
        "finalProb":         breakdown["final_prob"],
        "probM1":            breakdown["prob_m1"],
        "probM2":            breakdown["prob_m2"],
        "probM3":            breakdown["prob_m3"],
        "probM3Euc":         breakdown["prob_m3_euc"],
        "probM3Cos":         breakdown["prob_m3_cos"],
        "probM3Prs":         breakdown["prob_m3_prs"],
        "usedFeatures":      breakdown["used_features"],
        "defaultedFeatures": breakdown["defaulted_features"],
        "numProvided":       len(input_dict) - 2,   # actual clinical features user entered
    })


# ─────────────────────────────────────────────────────────────────────────────
# PDF DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400
    try:
        pdf_bytes = generate_report(data)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"PDF generation failed: {exc}"}), 500

    patient_name = (data.get("patientName") or "User").replace(" ", "_")
    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"PD_PREVeNT_{patient_name}_Report.pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
