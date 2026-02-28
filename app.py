import os
import logging
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file

# ── Logging (SAFE for PythonAnywhere) ─────────────────────────────
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────
MODEL_DIR = "saved_models"

try:
    from joblib import load as joblib_load

    m1         = joblib_load(os.path.join(MODEL_DIR, "model1_feature_ensemble.joblib"))
    m1_stacker = joblib_load(os.path.join(MODEL_DIR, "m1_stacker.joblib"))
    m2         = joblib_load(os.path.join(MODEL_DIR, "model2_xgb_only.joblib"))
    collab     = joblib_load(os.path.join(MODEL_DIR, "model3_collab_filter.joblib"))
    meta       = joblib_load(os.path.join(MODEL_DIR, "meta_model.joblib"))

    # Inject M2 columns
    m2_feature_cols_path = os.path.join(MODEL_DIR, "m2_feature_cols.joblib")
    if os.path.exists(m2_feature_cols_path):
        m2["feature_cols"] = joblib_load(m2_feature_cols_path)

    # Neutral means
    neutral_m2_path = os.path.join(MODEL_DIR, "neutral_means_m2.joblib")
    neutral_m3_path = os.path.join(MODEL_DIR, "neutral_means_m3.joblib")

    neutral_means_m2 = joblib_load(neutral_m2_path) if os.path.exists(neutral_m2_path) else {}
    neutral_means_m3 = joblib_load(neutral_m3_path) if os.path.exists(neutral_m3_path) else {}

    if neutral_means_m2:
        m2["neutral_means"] = neutral_means_m2

    if neutral_means_m3:
        collab["neutral_means"] = neutral_means_m3

    MODELS_LOADED = True

except Exception as e:
    logger.error("Model loading failed", exc_info=True)
    MODELS_LOADED = False
    m1 = m1_stacker = m2 = collab = meta = None

from prediction import predict_patient
from pdf_report import generate_report

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/box")
def box():
    return render_template("box.html")

# ─────────────────────────────────────────────────────────────────
@app.route("/features")
def features():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503

    clinical = [c for c in collab["imp_cols"] if c not in ("age", "sex")]
    return jsonify({"features": clinical})

# ─────────────────────────────────────────────────────────────────
@app.route("/submit", methods=["POST"])
def submit():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503

    try:
        age    = float(request.form.get("Age", ""))
        gender = int(request.form.get("Gender", ""))
        n_feat = int(request.form.get("num_features", "0"))
    except Exception:
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
        except:
            pass

    if len(input_dict) - 2 < 10:
        return jsonify({"error": "At least 10 valid feature values required"}), 400

    try:
        final_prob, risk_label, confidence, breakdown = predict_patient(
            input_dict, m1, m1_stacker, m2, collab, meta
        )
    except Exception as exc:
        logger.error("Prediction failed", exc_info=True)
        return jsonify({"error": "Prediction failed"}), 500

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
        "numProvided":       len(input_dict) - 2,
    })

# ─────────────────────────────────────────────────────────────────
@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    try:
        pdf_bytes = generate_report(data)
    except Exception:
        logger.error("PDF generation failed", exc_info=True)
        return jsonify({"error": "PDF generation failed"}), 500

    patient_name = (data.get("patientName") or "User").replace(" ", "_")

    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"PD_PREVeNT_{patient_name}_Report.pdf",
    )

# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
