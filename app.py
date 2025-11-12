import os
import json
import re
import tempfile
import types
import sys
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Define URLFeatureizer here, then register a fake module
# named "url_features" so joblib can resolve it on load.
# ------------------------------------------------------------

_RE_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
_RE_IPV4  = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}$")
_RE_IPV6  = re.compile(r"[0-9a-fA-F:]+:")

_SHORTENERS = {
    "bit.ly","goo.gl","t.co","ow.ly","tinyurl.com","is.gd","buff.ly","adf.ly","rb.gy","cutt.ly",
    "rebrand.ly","shorte.st","bl.ink","v.gd","t.ly","trib.al","lnkd.in"
}
_SUSPICIOUS_TOKENS = [
    "login","verify","update","secure","account","bank","wallet","confirm","invoice","billing",
    "support","help","unlock","apple","google","microsoft","amazon","pay","paypal","meta",
    "facebook","instagram","webscr","signin","reset","limited","suspend","appeal"
]
_SUSPICIOUS_TLDS = {
    "tk","ml","ga","cf","gq","top","xyz","club","work","click","country","gdn","kim","loan",
    "review","science","fit","men","party","date","stream"
}

def _safe_len(s: str) -> int:
    return len(s) if isinstance(s, str) else 0

def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text or ""))

def _is_ip(host: str) -> int:
    if not host: return 0
    if _RE_IPV4.fullmatch(host):
        parts = host.split(".")
        try:
            return int(all(0 <= int(p) <= 255 for p in parts))
        except ValueError:
            return 0
    return int(bool(_RE_IPV6.fullmatch(host)))

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
    return sum(t in low for t in _SUSPICIOUS_TOKENS)

def _brand_hits(host: str, path: str) -> Dict[str, int]:
    brands = ["google","apple","microsoft","amazon","paypal","facebook","instagram","netflix","bank","meta"]
    h = (host or "").lower()
    p = (path or "").lower()
    return {
        "domain_in_brand": int(any(b in h for b in brands)),
        "brand_in_subdomain": int(any(b in h.split(".")[:-2] for b in brands)) if host else 0,
        "brand_in_path": int(any(b in p for b in brands)) if path else 0,
    }

def _path_ext(path: str) -> str:
    if not path: return ""
    last = path.split("/")[-1]
    if "." in last:
        return last.split(".")[-1].lower()
    return ""

def _word_stats(host: str, path: str, raw: str) -> Dict[str, float]:
    words_raw  = [w for w in _RE_SPLIT.split((raw or "").lower()) if w]
    words_host = [w for w in _RE_SPLIT.split((host or "").lower()) if w]
    words_path = [w for w in _RE_SPLIT.split((path or "").lower()) if w]

    def stats(ws):
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

# --- the transformer class the pipeline was trained with ---
from sklearn.base import BaseEstimator, TransformerMixin

class URLFeatureizer(BaseEstimator, TransformerMixin):
    def __init__(self, url_col: str = "url"):
        self.url_col = url_col
        self._feature_names_ = []

    def fit(self, X: pd.DataFrame, y: Any = None):
        sample = pd.DataFrame({self.url_col: ["https://example.com/"]})
        self._feature_names_ = list(self._extract_row(sample.iloc[0][self.url_col]).keys())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        urls = X[self.url_col].astype(str).fillna("")
        rows = [self._extract_row(u) for u in urls]
        df = pd.DataFrame(rows)

        # enforce dtypes: numeric -> float; categoricals -> string
        for col in df.columns:
            if df[col].dtype == object and col not in ("scheme","top_level_domain","first_path_token","path_extension","url","hostname","domain"):
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().mean() >= 0.5:
                    df[col] = coerced.fillna(0.0)
                else:
                    df[col] = df[col].astype("string").fillna("")
            elif np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].astype(float)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names_)

    def _extract_row(self, url_text: str) -> Dict[str, Any]:
        url_text = (url_text or "").strip()
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
            "shortening_service": int((host or "").lower() in _SHORTENERS),
            "path_extension": _path_ext(path),
            "nb_redirection": max(0, url_text.count("//") - 1),
            "nb_external_redirection": 0,
            "char_repeat": int(bool(re.search(r"(.)\1{3,}", url_text))),
            "suspicious_tld": int((host_parts["tld"] or "") in _SUSPICIOUS_TLDS),
            "suspicious_token_count": _suspicious_token_count(url_text),
            "scheme": scheme,
            "top_level_domain": host_parts["tld"],
            "first_path_token": (path.split("/")[1].lower() if path.startswith("/") and len(path.split("/")) > 1 else ""),
            "num_params": len(qs),
            "frag_length": _safe_len(frag),
            "web_traffic": 0.0,
            "page_rank": 0.0,
        }

        feats.update(_word_stats(host, path, url_text))
        feats.update(_brand_hits(host, path))
        feats.update({"url": url_text, "hostname": host, "domain": host_parts["base"]})
        return feats

# Register fake module so joblib resolves "url_features.URLFeatureizer"
_mod = types.ModuleType("url_features")
_mod.URLFeatureizer = URLFeatureizer
sys.modules["url_features"] = _mod
# ------------------------------------------------------------

st.set_page_config(page_title="Phishing URL Detector", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/phishing_xgb_pipeline.joblib")
META_PATH  = os.getenv("META_PATH",  "artifacts/metadata.json")

@st.cache_resource(show_spinner=False)
def _load_artifacts(model_path: str, meta_path: str):
    pipe = joblib.load(model_path)   # will find url_features.URLFeatureizer via shim above
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

def _resolve_positive_index(pipe, meta, probe_df: pd.DataFrame) -> int:
    classes_ = _get_classes(pipe)
    target_from_meta = (meta or {}).get("positive_class")
    if classes_:
        if target_from_meta in classes_:
            return classes_.index(target_from_meta)
        for name in ("phishing", "malicious", "1", 1, True):
            if name in classes_:
                return classes_.index(name)
    probs = pipe.predict_proba(probe_df)
    if probs.shape[1] == 2:
        n = len(probe_df) // 2
        delta = probs[:n].mean(axis=0) - probs[n:].mean(axis=0)
        return int(np.argmax(delta))
    return 1 if probs.shape[1] > 1 else 0

# ---------------- UI ----------------

st.sidebar.title("⚙️ Artifacts")
st.sidebar.caption("Loads the unified pipeline + metadata from disk.")

if "attempted_load" not in st.session_state:
    st.session_state.attempted_load = True
    try:
        pipe, meta = _load_artifacts(MODEL_PATH, META_PATH)
        st.session_state.pipe = pipe
        st.session_state.meta = meta
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
