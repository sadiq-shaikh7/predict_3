import os
import json
import tempfile
from typing import List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# IMPORTANT: import the same module that defines URLFeatureizer
# so joblib can reconstruct the pipeline.
from url_features import URLFeatureizer  # noqa: F401  (import is required for joblib load)

st.set_page_config(page_title="Phishing URL Detector", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/phishing_xgb_pipeline.joblib")
META_PATH  = os.getenv("META_PATH",  "artifacts/metadata.json")

@st.cache_resource(show_spinner=False)
def _load_artifacts(model_path: str, meta_path: str):
    pipe = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return pipe, meta

def _get_classes(pipe):
    if hasattr(pipe, "classes_"):
        return list(pipe.classes_)
    if hasattr(pipe, "named_steps"):
        est = list(pipe.named_steps.values())[-1]
        if hasattr(est, "classes_"):
            return list(est.classes_)
    return None

def _resolve_positive_index(pipe, meta, probe_df: pd.DataFrame) -> int:
    # 1) metadata hint
    target_from_meta = (meta or {}).get("positive_class")
    classes_ = _get_classes(pipe)
    if classes_:
        if target_from_meta in classes_:
            return classes_.index(target_from_meta)
        for name in ("phishing", "malicious", "1", 1, True):
            if name in classes_:
                return classes_.index(name)
    # 2) quick probe
    probs = pipe.predict_proba(probe_df)
    if probs.shape[1] == 2:
        ph_mean = probs[: len(probe_df)//2].mean(axis=0)   # first half phishing-ish
        lg_mean = probs[len(probe_df)//2 :].mean(axis=0)   # second half legit-ish
        deltas = (ph_mean - lg_mean)
        return int(np.argmax(deltas))
    # 3) fallback
    return 1 if probs.shape[1] > 1 else 0

def _probe_df() -> pd.DataFrame:
    phishing_like = [
        "login-secure-update.com/verify",
        "paypal.com.account.security-check.ru/login",
        "http://192.168.1.10/verify?acc=123",
        "bit.ly/free-gift",
        "secure-google.com-login.verify-account.net",
        "bankofamerica.com.login-security-check.co/confirm",
    ]
    legit_like = [
        "https://www.google.com/",
        "https://www.bankofamerica.com/",
        "https://docs.python.org/3/",
        "https://www.wikipedia.org/",
        "https://www.amazon.com/",
        "https://www.microsoft.com/",
    ]
    return pd.DataFrame({"url": phishing_like + legit_like})

# ---------------- UI ----------------

st.sidebar.title("⚙️ Artifacts")
st.sidebar.caption("Load saved pipeline + metadata.")

if "attempted_load" not in st.session_state:
    st.session_state.attempted_load = True
    try:
        pipe, meta = _load_artifacts(MODEL_PATH, META_PATH)
        st.session_state.pipe = pipe
        st.session_state.meta = meta
        # resolve pos idx once
        st.session_state.pos_idx = _resolve_positive_index(pipe, meta, _probe_df())
        st.sidebar.success("Artifacts loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load artifacts: {e}")

st.title("🔍 Phishing URL Detector (Raw URL → Pipeline featureizer)")
st.caption("Paste one URL per line. The model pipeline computes features internally, then predicts phishing probability.")

if "pipe" not in st.session_state:
    st.error("Model not loaded. Ensure artifacts exist at the configured paths.")
else:
    threshold = st.slider("Decision threshold (probability for 'phishing')", 0.05, 0.9, 0.30, 0.01)
    show_probs = st.checkbox("Show probabilities", value=True)

    urls_text = st.text_area(
        "Enter one URL per line",
        placeholder="e.g.\nhttps://example.com/login\nbit.ly/free-gift\naccounts.google.com-security-check.com/verify",
        height=180
    )

    if st.button("Predict"):
        urls: List[str] = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
        if not urls:
            st.warning("Please paste at least one URL.")
        else:
            X_infer = pd.DataFrame({"url": urls})
            probs_all = st.session_state.pipe.predict_proba(X_infer)
            pos_col = st.session_state.pos_idx if probs_all.shape[1] > 1 else 0
            probs = probs_all[:, pos_col]
            preds = (probs >= threshold).astype(int)

            out = pd.DataFrame({
                "url": urls,
                "phishing_prob": probs,
                "prediction": np.where(preds == 1, "phishing", "legitimate")
            })

            st.success(f"Predicted {(out['prediction'] == 'phishing').sum()} phishing out of {len(out)} rows.")
            if show_probs:
                st.dataframe(out, use_container_width=True)

            st.download_button(
                "⬇️ Download predictions",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
