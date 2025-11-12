import os
import json
import re
import tempfile
from typing import Tuple, List, Dict
from urllib.parse import urlparse, parse_qs

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phishing URL Detector", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/phishing_xgb_pipeline.joblib")
META_PATH  = os.getenv("META_PATH",  "artifacts/metadata.json")

# =========================== Artifact Loading ===========================

@st.cache_resource(show_spinner=False)
def _load_artifacts_from_disk(model_path: str, meta_path: str):
    pipe = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    numeric_cols = list(map(str, meta.get("numeric_cols", [])))
    categorical_cols = list(map(str, meta.get("categorical_cols", [])))
    label_source = str(meta.get("label_column_used", "status/label"))
    used_smote = bool(meta.get("used_smote", False))
    return pipe, numeric_cols, categorical_cols, label_source, used_smote, meta

def _persist_uploaded_artifact(uploaded_file, suffix: str) -> str:
    tmpdir = st.session_state.setdefault("_artifacts_dir", tempfile.mkdtemp(prefix="artifacts_"))
    path = os.path.join(tmpdir, f"uploaded_{suffix}")
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def _artifacts_loaded() -> bool:
    return ("pipe" in st.session_state and
            "numeric_cols" in st.session_state and
            "categorical_cols" in st.session_state)

# =========================== URL Featureizer ===========================

SHORTENERS = {
    "bit.ly","goo.gl","t.co","ow.ly","tinyurl.com","is.gd","buff.ly","adf.ly","rb.gy","cutt.ly",
    "rebrand.ly","shorte.st","bl.ink","v.gd","t.ly","trib.al","lnkd.in"
}
SUSPICIOUS_TOKENS = [
    "login","verify","update","secure","account","bank","wallet","confirm","invoice","billing",
    "support","help","unlock","apple","google","microsoft","amazon","pay","paypal","meta",
    "facebook","instagram","webscr","signin","reset","limited","suspend","appeal"
]
SUSPICIOUS_TLDS = {
    "tk","ml","ga","cf","gq","top","xyz","club","work","click","country","gdn","kim","loan",
    "review","science","fit","men","party","date","stream"
}

def _safe_len(s: str) -> int:
    return len(s) if isinstance(s, str) else 0

def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text or ""))

def _is_ip(host: str) -> int:
    if not host: return 0
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host):
        parts = [int(p) for p in host.split(".")]
        return int(all(0 <= p <= 255 for p in parts))
    return int(bool(re.fullmatch(r"[0-9a-fA-F:]+", host) and ":" in host))

def _split_host(host: str) -> Dict[str, str]:
    h = (host or "").lower()
    if h.startswith("www."): h = h[4:]
    bits = h.split(".") if h else []
    tld = bits[-1] if len(bits) >= 1 else ""
    sld = bits[-2] if len(bits) >= 2 else ""
    sub = ".".join(bits[:-2]) if len(bits) > 2 else ""
    return {"sub": sub, "sld": sld, "tld": tld, "base": (sld + "." + tld) if sld and tld else h}

def _num_subdomains(host: str) -> int:
    if not host: return 0
    h = host.lower()
    h = h[4:] if h.startswith("www.") else h
    return max(0, len(h.split(".")) - 2)

def _has_tld_in(text: str, tld: str) -> int:
    if not text or not tld: return 0
    return int(("." + tld) in text.lower())

def _suspicious_token_count(url: str) -> int:
    low = (url or "").lower()
    return sum(t in low for t in SUSPICIOUS_TOKENS)

def _brand_hits(host: str, path: str) -> Dict[str, int]:
    brands = ["google","apple","microsoft","amazon","paypal","facebook","instagram","netflix","bank","meta"]
    return {
        "domain_in_brand": int(any(b in (host or "").lower() for b in brands)),
        "brand_in_subdomain": int(any(b in (host or "").lower().split(".")[:-2] for b in brands)) if host else 0,
        "brand_in_path": int(any(b in (path or "").lower() for b in brands)) if path else 0,
    }

def _path_ext(path: str) -> str:
    if not path: return ""
    last = path.split("/")[-1]
    if "." in last:
        return last.split(".")[-1].lower()
    return ""

def _word_stats(host: str, path: str, raw: str) -> Dict[str, float]:
    splitter = re.compile(r"[^a-zA-Z0-9]+")
    words_raw  = [w for w in splitter.split((raw or "").lower()) if w]
    words_host = [w for w in splitter.split((host or "").lower()) if w]
    words_path = [w for w in splitter.split((path or "").lower()) if w]

    def stats(ws: List[str]):
        if not ws:
            return 0, 0, 0.0
        lengths = [len(w) for w in ws]
        return min(lengths), max(lengths), float(np.mean(lengths))

    smin_raw, smax_raw, savg_raw = stats(words_raw)
    smin_host, smax_host, savg_host = stats(words_host)
    smin_path, smax_path, savg_path = stats(words_path)

    return {
        "length_words_raw": len(words_raw),
        "shortest_words_raw": smin_raw,
        "longest_words_raw": smax_raw,
        "avg_words_raw": savg_raw,
        "shortest_word_host": smin_host,
        "longest_word_host": smax_host,
        "avg_word_host": savg_host,
        "shortest_word_path": smin_path,
        "longest_word_path": smax_path,
        "avg_word_path": savg_path,
    }

def _extract_features_from_url(u: str) -> Dict[str, object]:
    url_text = (u or "").strip()
    parsed = urlparse(url_text if "://" in url_text else "http://" + url_text)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    frag  = parsed.fragment or ""
    scheme = (parsed.scheme or "").lower()
    qs = parse_qs(query)
    host_parts = _split_host(host)

    feats = {
        "length_url": _safe_len(url_text),
        "length_hostname": _safe_len(host),
        "ip": _is_ip(host),
        "nb_dots": url_text.count("."),
        "nb_hyphens": url_text.count("-"),
        "nb_at": url_text.count("@"),
        "nb_qm": url_text.count("?"),
        "nb_and": url_text.count("&"),
        "nb_or": url_text.count("|"),
        "nb_eq": url_text.count("="),
        "nb_underscore": url_text.count("_"),
        "nb_tilde": url_text.count("~"),
        "nb_percent": url_text.count("%"),
        "nb_slash": url_text.count("/"),
        "nb_star": url_text.count("*"),
        "nb_colon": url_text.count(":"),
        "nb_comma": url_text.count(","),
        "nb_semicolon": url_text.count(";"),
        "nb_dollar": url_text.count("$"),
        "nb_space": _count(r"\s", url_text),
        "nb_www": int("www" in host.lower()),
        "nb_com": int(".com" in url_text.lower()),
        "nb_dslash": url_text.count("//") - 1,
        "http_in_path": int("http" in path.lower()),
        "https_token": int("https" in url_text.lower() and scheme != "https"),
        "ratio_digits_url": (_count(r"\d", url_text) / max(1, len(url_text))),
        "ratio_digits_host": (_count(r"\d", host) / max(1, len(host))),
        "punycode": int("xn--" in host.lower()),
        "port": int(parsed.port is not None),
        "tld_in_path": _has_tld_in(path, host_parts["tld"]),
        "tld_in_subdomain": _has_tld_in(host_parts["sub"], host_parts["tld"]),
        "abnormal_subdomain": int(_num_subdomains(host) > 3),
        "nb_subdomains": _num_subdomains(host),
        "prefix_suffix": int("-" in host_parts["sld"]) if host_parts["sld"] else 0,
        "random_domain": int(bool(re.fullmatch(r"[a-z]{6,}\d{2,}|[a-z0-9]{12,}", host_parts["sld"] or ""))),
        "shortening_service": int((host or "").lower() in SHORTENERS),
        "path_extension": _path_ext(path),
        "nb_redirection": max(0, url_text.count("//") - 1),
        "nb_external_redirection": 0,
        "char_repeat": int(bool(re.search(r"(.)\1{3,}", url_text))),
        "suspicious_tld": int((host_parts["tld"] or "") in SUSPICIOUS_TLDS),
        "suspicious_token_count": _suspicious_token_count(url_text),
        "scheme": scheme,
        "top_level_domain": host_parts["tld"],
        "first_path_token": (path.split("/")[1].lower() if path.startswith("/") and len(path.split("/")) > 1 else ""),
        "num_params": len(qs),
        "frag_length": _safe_len(frag),
    }

    feats.update(_word_stats(host, path, url_text))
    feats.update(_brand_hits(host, path))

    # page/registry placeholders
    feats.update({
        "statistical_report": 0, "nb_hyperlinks": 0, "ratio_intHyperlinks": 0.0,
        "ratio_extHyperlinks": 0.0, "ratio_nullHyperlinks": 0.0, "nb_extCSS": 0,
        "ratio_intRedirection": 0.0, "ratio_extRedirection": 0.0, "ratio_intErrors": 0.0,
        "ratio_extErrors": 0.0, "login_form": 0, "external_favicon": 0, "links_in_tags": 0,
        "submit_email": 0, "ratio_intMedia": 0.0, "ratio_extMedia": 0.0, "sfh": 0,
        "iframe": 0, "popup_window": 0, "safe_anchor": 0, "on_mouseover": 0,
        "right_clic": 0, "empty_title": 0, "domain_in_title": 0, "domain_with_copyright": 0,
        "whois_registered_domain": 0, "domain_registration_length": 0.0, "domain_age": 0.0,
        "web_traffic": 0.0, "dns_record": 0, "google_index": 0, "page_rank": 0.0
    })

    feats.update({
        "url": url_text,
        "hostname": host,
        "domain": host_parts["base"]
    })
    return feats

def _featurize_urls(urls: List[str]) -> pd.DataFrame:
    return pd.DataFrame([_extract_features_from_url(u) for u in urls])

# =========================== Alignment (dtype-safe) ===========================

def _align_to_expected(df_feat: pd.DataFrame,
                       numeric_cols: List[str],
                       categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df_feat.replace([np.inf, -np.inf], np.nan)

    expected = list(numeric_cols) + list(categorical_cols)
    have = set(df.columns)
    missing = [c for c in expected if c not in have]

    # Add missing with safe defaults
    for m in missing:
        if m in numeric_cols:
            df[m] = 0
        else:
            df[m] = ""

    # Enforce numeric dtypes strictly
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Enforce categorical/string dtypes
    for c in categorical_cols:
        df[c] = df[c].astype("string").fillna("")

    # Reorder and final NA fill
    df = df[expected].fillna(0)
    return df, missing

# =========================== Class utilities ===========================

def _get_positive_index(pipe, meta) -> int:
    target_from_meta = (meta or {}).get("positive_class")
    classes_ = []
    if hasattr(pipe, "classes_"):
        classes_ = list(pipe.classes_)
    if not classes_ and hasattr(pipe, "named_steps"):
        est = list(pipe.named_steps.values())[-1]
        if hasattr(est, "classes_"):
            classes_ = list(est.classes_)
    if classes_:
        if target_from_meta and target_from_meta in classes_:
            return classes_.index(target_from_meta)
        for name in ("phishing", "1", 1, True):
            if name in classes_:
                return classes_.index(name)
        return int(np.argmax(classes_))
    return 1  # fallback

def _get_classes(pipe):
    if hasattr(pipe, "classes_"):
        return list(pipe.classes_)
    if hasattr(pipe, "named_steps"):
        est = list(pipe.named_steps.values())[-1]
        if hasattr(est, "classes_"):
            return list(est.classes_)
    return None

# =========================== Sidebar: artifacts ===========================

st.sidebar.title("⚙️ Artifacts")
st.sidebar.caption("Load saved pipeline + metadata from repo or upload them here.")

if "attempted_disk_load" not in st.session_state:
    st.session_state.attempted_disk_load = True
    try:
        pipe, numeric_cols, categorical_cols, label_source, used_smote, meta = _load_artifacts_from_disk(MODEL_PATH, META_PATH)
        st.session_state.pipe = pipe
        st.session_state.numeric_cols = numeric_cols
        st.session_state.categorical_cols = categorical_cols
        st.session_state.label_source = label_source
        st.session_state.used_smote = used_smote
        st.session_state.meta = meta
        st.session_state.pos_idx = _get_positive_index(pipe, meta)
        st.session_state.classes_ = _get_classes(pipe)
        st.sidebar.success(f"Loaded artifacts from {MODEL_PATH} / {META_PATH}")
    except Exception as e:
        st.sidebar.warning(f"Could not load from disk: {e}")

st.sidebar.markdown("---")
up_m = st.sidebar.file_uploader("Upload pipeline (.joblib)", type=["joblib", "pkl"], key="upl_model")
up_j = st.sidebar.file_uploader("Upload metadata (.json)", type=["json"], key="upl_meta")

if st.sidebar.button("Use uploaded artifacts"):
    if up_m and up_j:
        try:
            mpath = _persist_uploaded_artifact(up_m, "pipeline.joblib")
            jpath = _persist_uploaded_artifact(up_j, "metadata.json")
            pipe, numeric_cols, categorical_cols, label_source, used_smote, meta = _load_artifacts_from_disk(mpath, jpath)
            st.session_state.pipe = pipe
            st.session_state.numeric_cols = numeric_cols
            st.session_state.categorical_cols = categorical_cols
            st.session_state.label_source = label_source
            st.session_state.used_smote = used_smote
            st.session_state.meta = meta
            st.session_state.pos_idx = _get_positive_index(pipe, meta)
            st.session_state.classes_ = _get_classes(pipe)
            st.sidebar.success("Uploaded artifacts loaded.")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded artifacts: {e}")
    else:
        st.sidebar.error("Please upload both the pipeline and metadata files.")

# =========================== Main UI ===========================

st.title("🔍 Phishing URL Detector (Paste URLs)")
st.caption(
    "Paste one URL per line. The app computes URL features, aligns to the model’s expected schema (dtype-safe), "
    "and predicts phishing probability. Use the picker below if your model’s class order is inverted."
)

if _artifacts_loaded():
    with st.expander("Model information", expanded=False):
        st.write(f"**Used SMOTE (training):** `{st.session_state.used_smote}`")
        st.json(st.session_state.meta, expanded=False)

    threshold = st.slider("Decision threshold (probability for 'phishing')", 0.05, 0.9, 0.30, 0.01)
    show_probs = st.checkbox("Show probabilities", value=True)
    show_preview = st.checkbox("Show engineered features (first 10 rows)", value=False)

    # ---- Class mapping UI ----
    classes = st.session_state.get("classes_")
    pos_idx_ui = st.session_state.get("pos_idx", 1)
    if classes:
        st.caption(f"Model classes: {classes}")
        default_index = classes.index("phishing") if "phishing" in classes else min(len(classes)-1, 1)
        selected_label = st.selectbox("Treat this class as 'phishing'", options=classes, index=default_index)
        pos_idx_ui = classes.index(selected_label)
    else:
        st.info("Model classes unknown; using column index 1 for 'phishing'.")

    urls_text = st.text_area(
        "Enter one URL per line",
        placeholder="e.g.\nhttps://example.com/login\nbit.ly/free-gift\naccounts.google.com-security-check.com/verify",
        height=180
    )

    if st.button("Predict"):
        urls = [u.strip() for u in (urls_text or "").splitlines() if u.strip()]
        if not urls:
            st.warning("Please paste at least one URL.")
        else:
            feats = _featurize_urls(urls)
            if show_preview:
                st.subheader("Parsed URL features (preview)")
                st.dataframe(feats.head(10), use_container_width=True)

            X_aligned, missing_cols = _align_to_expected(
                feats.copy(), st.session_state.numeric_cols, st.session_state.categorical_cols
            )

            expected = len(st.session_state.numeric_cols) + len(st.session_state.categorical_cols)
            filled = len(missing_cols)
            pct = int(100 * (expected - filled) / max(1, expected))
            st.info(f"Aligned to expected columns. Filled missing: {missing_cols or '[]'}")
            if expected > 0 and (expected - filled) / expected < 0.6:
                st.warning(
                    f"Only {expected - filled}/{expected} expected features present ({pct}%). "
                    "Predictions may skew one-sided; consider moving full feature engineering into the saved pipeline."
                )

            probs_all = st.session_state.pipe.predict_proba(X_aligned)

            # Diagnostics: show both columns' stats if binary
            if probs_all.shape[1] == 2:
                st.write(
                    "Prob column 0 (mean/min/max):",
                    float(np.mean(probs_all[:, 0])),
                    float(np.min(probs_all[:, 0])),
                    float(np.max(probs_all[:, 0])),
                )
                st.write(
                    "Prob column 1 (mean/min/max):",
                    float(np.mean(probs_all[:, 1])),
                    float(np.min(probs_all[:, 1])),
                    float(np.max(probs_all[:, 1])),
                )

            # Use selected positive column (fallback to detected pos_idx)
            pos_col = pos_idx_ui if classes else st.session_state.pos_idx
            probs = probs_all[:, pos_col]
            preds = (probs >= threshold).astype(int)

            # Skew warning
            if probs_all.shape[1] == 2 and (np.mean(probs) > 0.9 or np.mean(probs) < 0.1):
                st.warning("Probability distribution is extremely skewed. If outputs look wrong, try selecting the other class as 'phishing' above.")

            out = pd.DataFrame({
                "url": urls,
                "phishing_prob": probs,
                "prediction": np.where(preds == 1, "phishing", "legitimate")
            })

            st.success(f"Predicted {(out['prediction'] == 'phishing').sum()} phishing out of {len(out)} rows.")
            if show_probs:
                st.subheader("Predictions (head)")
                st.dataframe(out.head(20), use_container_width=True)

            st.subheader("Counts")
            st.write(out["prediction"].value_counts())

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download predictions", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

else:
    st.error(
        "Artifacts not loaded. Either place the trained pipeline and metadata at "
        f"`{MODEL_PATH}` and `{META_PATH}` in your repo, or upload them from the sidebar."
    )
