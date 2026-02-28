import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CF_K = 5

def _sim_cosine(X_test, X_train):
    return cosine_similarity(X_test, X_train)

def _sim_pearson(X_test, X_train):
    def row_z(X):
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (X - mu) / sd
    return (row_z(X_test.copy().astype(float)) @
            row_z(X_train.copy().astype(float)).T) / X_test.shape[1]

def _sim_euclidean_rbf(X_test_sc, X_train_sc):
    from sklearn.metrics import pairwise_distances
    D     = pairwise_distances(X_test_sc, X_train_sc, metric="euclidean")
    sigma = np.median(D) + 1e-8
    return np.exp(-D / sigma)

def _distance_weighted_vote(sim, y_train, k):
    idx   = np.argsort(sim, axis=1)[:, -k:]
    probs = np.empty(len(sim))
    for i in range(len(sim)):
        nbrs = idx[i]
        w    = np.maximum(sim[i, nbrs], 0.0)
        wsum = w.sum()
        probs[i] = ((w * y_train[nbrs]).sum() / wsum
                    if wsum > 0 else y_train[nbrs].mean())
    return probs

def _predict_single_feature(bundle, Xf_num):
    Xf_s  = bundle["scaler"].transform(Xf_num)
    probs = []

    if "lr"  in bundle:
        probs.append(bundle["lr"].predict_proba(Xf_s)[:, 1])

    if "svm" in bundle:
        probs.append(bundle["svm"].predict_proba(Xf_s)[:, 1])

    if "xgb" in bundle:
        probs.append(bundle["xgb"].predict_proba(Xf_num.values)[:, 1])

    return float(np.mean(probs, axis=0)[0])

def _get_m2_columns(m2, m1, collab):

    if "feature_cols" in m2:
        return m2["feature_cols"]

    xgb = m2["xgb"]

    if hasattr(xgb, "feature_names_in_") and xgb.feature_names_in_ is not None:
        return list(xgb.feature_names_in_)

    clinical = list(m1.keys())
    cols = ["age", "sex"] + [f for f in clinical if f not in ("age", "sex")]
    return cols

def predict_patient(input_dict, m1, m1_stacker, m2, collab, meta):

    # ── M1 ─────────────────────────
    feat_to_prob = {}

    for feat, bundle in m1.items():
        if feat not in input_dict:
            continue

        Xf_raw = pd.DataFrame(
            [[input_dict[feat],
              input_dict.get("age", np.nan),
              input_dict.get("sex", np.nan)]],
            columns=[feat, "age", "sex"]
        ).apply(pd.to_numeric, errors="coerce")

        Xf_num = Xf_raw.copy()

        for col, med in bundle["medians"].items():
            Xf_num[col] = Xf_num[col].fillna(med)

        p = _predict_single_feature(bundle, Xf_num)
        feat_to_prob[feat] = np.array([p])

    active   = m1_stacker["active_features"]
    defaults = m1_stacker["default_probs"]
    stacker  = m1_stacker["model"]

    stacker_row, used, defaulted = [], [], []

    for feat in active:
        if feat in feat_to_prob:
            stacker_row.append(feat_to_prob[feat][0])
            used.append(feat)
        else:
            stacker_row.append(defaults[feat])
            defaulted.append(feat)

    prob1 = float(stacker.predict_proba([stacker_row])[0, 1])

    # ── M2 ─────────────────────────
    m2_cols = _get_m2_columns(m2, m1, collab)

    neutral_m2 = m2.get("neutral_means", {})
    m2_row = {col: neutral_m2.get(col, np.nan) for col in m2_cols}

    for k, v in input_dict.items():
        if k in m2_row:
            m2_row[k] = v

    df_m2 = pd.DataFrame([m2_row], columns=m2_cols)
    prob2 = float(m2["xgb"].predict_proba(df_m2.values)[0, 1])

    # ── M3 ─────────────────────────
    all_cols = collab["imp_cols"]

    patient_row = {col: np.nan for col in all_cols}
    for k, v in input_dict.items():
        if k in patient_row:
            patient_row[k] = v

    df_m3 = pd.DataFrame([patient_row], columns=all_cols)

    X_filled = df_m3.copy()

    fill_means = collab.get("neutral_means") or collab.get("col_means", {})

    for col, m_val in fill_means.items():
        if col in X_filled.columns:
            X_filled[col] = X_filled[col].fillna(m_val)

    X_imp    = X_filled.values
    X_scaled = collab["scaler"].transform(X_imp)
    y_tr     = collab["y_train"]

    sim_euc = _sim_euclidean_rbf(X_scaled, collab["X_scaled"])
    sim_cos = _sim_cosine(X_imp, collab["X_train"])
    sim_prs = _sim_pearson(X_imp, collab["X_train"])

    p_euc = float(_distance_weighted_vote(sim_euc, y_tr, CF_K)[0])
    p_cos = float(_distance_weighted_vote(sim_cos, y_tr, CF_K)[0])
    p_prs = float(_distance_weighted_vote(sim_prs, y_tr, CF_K)[0])

    prob3 = (p_euc + p_cos + p_prs) / 3.0

    # ── META ───────────────────────
    final_prob = float(meta.predict_proba([[prob1, prob2, prob3]])[0, 1])
    final_class = int(final_prob > 0.5)

    risk_label = "High Risk (PD)" if final_class == 1 else "Low Risk (Healthy)"

    # ── Confidence ─────────────────
    margin = abs(final_prob - 0.5) * 2.0
    sub_votes = [int(prob1 > 0.5), int(prob2 > 0.5), int(prob3 > 0.5)]
    consensus = sum(1 for v in sub_votes if v == final_class) / 3.0
    combined = 0.70 * margin + 0.30 * consensus

    if combined >= 0.80:
        conf_band = "High Confidence"
    elif combined >= 0.55:
        conf_band = "Moderate Confidence"
    elif combined >= 0.35:
        conf_band = "Low Confidence"
    else:
        conf_band = "Very Low Confidence — treat with caution"

    confidence = {
        "combined_score": round(combined, 4),
        "band": conf_band,
        "probability_margin": round(margin, 4),
        "model_consensus": round(consensus, 4),
        "sub_model_votes": {
            "M1 (Feature-Stacker)":  "PD" if prob1 > 0.5 else "Healthy",
            "M2 (XGB-Only)":         "PD" if prob2 > 0.5 else "Healthy",
            "M3 (CB-Collab-Filter)": "PD" if prob3 > 0.5 else "Healthy",
        },
    }

    breakdown = {
        "prob_m1": round(prob1, 4),
        "prob_m2": round(prob2, 4),
        "prob_m3": round(prob3, 4),
        "prob_m3_euc": round(p_euc, 4),
        "prob_m3_cos": round(p_cos, 4),
        "prob_m3_prs": round(p_prs, 4),
        "final_prob": round(final_prob, 4),
        "used_features": used,
        "defaulted_features": defaulted,
    }

    return final_prob, risk_label, confidence, breakdown