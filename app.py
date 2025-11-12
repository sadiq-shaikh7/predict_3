import os, json, re, types, sys, tempfile
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse, parse_qs

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# ================= URLFeatureizer and shim for joblib =================
_RE_SPLIT = re.compile(r"[^a-zA-Z0-9]+"); _RE_IPV4  = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}$"); _RE_IPV6  = re.compile(r"[0-9a-fA-F:]+:")
_SHORTENERS={"bit.ly","goo.gl","t.co","ow.ly","tinyurl.com","is.gd","buff.ly","adf.ly","rb.gy","cutt.ly","rebrand.ly","shorte.st","bl.ink","v.gd","t.ly","trib.al","lnkd.in"}
_SUSPICIOUS_TOKENS=["login","verify","update","secure","account","bank","wallet","confirm","invoice","billing","support","help","unlock","apple","google","microsoft","amazon","pay","paypal","meta","facebook","instagram","webscr","signin","reset","limited","suspend","appeal"]
_SUSPICIOUS_TLDS={"tk","ml","ga","cf","gq","top","xyz","club","work","click","country","gdn","kim","loan","review","science","fit","men","party","date","stream"}

def _safe_len(s:str)->int:return len(s) if isinstance(s,str) else 0
def _count(pat:str,t:str)->int:return len(re.findall(pat,t or ""))
def _is_ip(h:str)->int:
    if not h:return 0
    if _RE_IPV4.fullmatch(h):
        try:return int(all(0<=int(p)<=255 for p in h.split(".")))
        except: return 0
    return int(bool(_RE_IPV6.fullmatch(h)))
def _split_host(h:str)->Dict[str,str]:
    h=(h or "").lower(); h=h[4:] if h.startswith("www.") else h
    bits=h.split(".") if h else []; tld=bits[-1] if len(bits)>=1 else ""; sld=bits[-2] if len(bits)>=2 else ""; sub=".".join(bits[:-2]) if len(bits)>2 else ""
    return {"sub":sub,"sld":sld,"tld":tld,"base":(sld+"."+tld) if sld and tld else h}
def _num_subdomains(h:str)->int:
    if not h:return 0
    h=h.lower(); h=h[4:] if h.startswith("www.") else h
    return max(0,len(h.split("."))-2)
def _has_tld_in(text:str,tld:str)->int:
    if not text or not tld:return 0
    return int(("."+tld) in text.lower())
def _suspicious_token_count(u:str)->int:
    low=(u or "").lower(); return sum(t in low for t in _SUSPICIOUS_TOKENS)
def _brand_hits(host:str,path:str)->Dict[str,int]:
    brands=["google","apple","microsoft","amazon","paypal","facebook","instagram","netflix","bank","meta"]
    h=(host or "").lower(); p=(path or "").lower()
    return {"domain_in_brand":int(any(b in h for b in brands)),
            "brand_in_subdomain":int(any(b in h.split(".")[:-2] for b in brands)) if host else 0,
            "brand_in_path":int(any(b in p for b in brands)) if path else 0}
def _path_ext(path:str)->str:
    if not path:return ""
    last=path.split("/")[-1]; return last.split(".")[-1].lower() if "." in last else ""
def _word_stats(host:str,path:str,raw:str)->Dict[str,float]:
    wr=[w for w in _RE_SPLIT.split((raw or "").lower()) if w]
    wh=[w for w in _RE_SPLIT.split((host or "").lower()) if w]
    wp=[w for w in _RE_SPLIT.split((path or "").lower()) if w]
    def stats(ws): 
        if not ws:return 0,0,0.0
        L=[len(w) for w in ws]; return min(L),max(L),float(np.mean(L))
    sminr,smaxr,savgr=stats(wr); sminh,smaxh,savgh=stats(wh); sminp,smaxp,avgp=stats(wp)
    return {"length_words_raw":len(wr),"shortest_words_raw":sminr,"longest_words_raw":smaxr,"avg_words_raw":savgr,
            "shortest_word_host":sminh,"longest_word_host":smaxh,"avg_word_host":savgh,
            "shortest_word_path":sminp,"longest_word_path":smaxp,"avg_word_path":avgp}

class URLFeatureizer(BaseEstimator, TransformerMixin):
    def __init__(self,url_col:str="url"): self.url_col=url_col; self._feature_names_=[]
    def fit(self,X:pd.DataFrame,y=None):
        sample=pd.DataFrame({self.url_col:["https://example.com/"]})
        self._feature_names_=list(self._extract_row(sample.iloc[0][self.url_col]).keys()); return self
    def transform(self,X:pd.DataFrame)->pd.DataFrame:
        urls=X[self.url_col].astype(str).fillna(""); rows=[self._extract_row(u) for u in urls]; df=pd.DataFrame(rows)
        for c in df.columns:
            if df[c].dtype==object and c not in ("scheme","top_level_domain","first_path_token","path_extension","url","hostname","domain"):
                coer=pd.to_numeric(df[c],errors="coerce"); df[c]=coer.fillna(0.0) if coer.notna().mean()>=0.5 else df[c].astype("string").fillna("")
            elif np.issubdtype(df[c].dtype,np.number): df[c]=df[c].astype(float)
        return df
    def get_feature_names_out(self,input_features=None): return np.array(self._feature_names_)
    def _extract_row(self,url_text:str)->Dict[str,Any]:
        url_text=(url_text or "").strip()
        p=urlparse(url_text if "://" in url_text else "http://"+url_text)
        host=p.hostname or ""; path=p.path or ""; query=p.query or ""; frag=p.fragment or ""; scheme=(p.scheme or "").lower()
        qs=parse_qs(query); parts=_split_host(host)
        feats={"length_url":_safe_len(url_text),"length_hostname":_safe_len(host),"ip":_is_ip(host),
               "nb_dots":url_text.count("."),"nb_hyphens":url_text.count("-"),"nb_at":url_text.count("@"),"nb_qm":url_text.count("?"),
               "nb_and":url_text.count("&"),"nb_or":url_text.count("|"),"nb_eq":url_text.count("="),"nb_underscore":url_text.count("_"),
               "nb_tilde":url_text.count("~"),"nb_percent":url_text.count("%"),"nb_slash":url_text.count("/"),"nb_star":url_text.count("*"),
               "nb_colon":url_text.count(":"),"nb_comma":url_text.count(","),"nb_semicolon":url_text.count(";"),"nb_dollar":url_text.count("$"),
               "nb_space":_count(r"\s",url_text),"nb_www":int("www" in host.lower()),"nb_com":int(".com" in url_text.lower()),
               "nb_dslash":url_text.count("//")-1,"http_in_path":int("http" in path.lower()),
               "https_token":int("https" in url_text.lower() and scheme!="https"),
               "ratio_digits_url":(_count(r"\d",url_text)/max(1,len(url_text))),"ratio_digits_host":(_count(r"\d",host)/max(1,len(host))),
               "punycode":int("xn--" in host.lower()),"port":int(p.port is not None),
               "tld_in_path":_has_tld_in(path,parts["tld"]),"tld_in_subdomain":_has_tld_in(parts["sub"],parts["tld"]),
               "abnormal_subdomain":int(_num_subdomains(host)>3),"nb_subdomains":_num_subdomains(host),
               "prefix_suffix":int("-" in parts["sld"]) if parts["sld"] else 0,
               "random_domain":int(bool(re.fullmatch(r"[a-z]{6,}\d{2,}|[a-z0-9]{12,}",parts["sld"] or ""))),
               "shortening_service":int((host or "").lower() in _SHORTENERS),"path_extension":_path_ext(path),
               "nb_redirection":max(0,url_text.count("//")-1),"nb_external_redirection":0,
               "char_repeat":int(bool(re.search(r"(.)\1{3,}",url_text))),"suspicious_tld":int((parts["tld"] or "") in _SUSPICIOUS_TLDS),
               "suspicious_token_count":_suspicious_token_count(url_text),"scheme":scheme,"top_level_domain":parts["tld"],
               "first_path_token":(path.split("/")[1].lower() if path.startswith("/") and len(path.split("/"))>1 else ""),
               "num_params":len(qs),"frag_length":_safe_len(frag),
               "web_traffic":0.0,"page_rank":0.0}
        feats.update(_word_stats(host,path,url_text)); feats.update(_brand_hits(host,path))
        feats.update({"url":url_text,"hostname":host,"domain":parts["base"]}); return feats

# register shim so joblib resolves url_features.URLFeatureizer
_mod=types.ModuleType("url_features"); _mod.URLFeatureizer=URLFeatureizer; sys.modules["url_features"]=_mod
# ======================================================================

st.set_page_config(page_title="Phishing URL Detector", layout="wide")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/phishing_xgb_pipeline.joblib")
META_PATH  = os.getenv("META_PATH",  "artifacts/metadata.json")

@st.cache_resource(show_spinner=False)
def _load_artifacts(model_path: str, meta_path: str):
    pipe = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return pipe, meta

def _probe_df()->pd.DataFrame:
    phishing_like=["login-secure-update.com/verify","paypal.com.account.security-check.ru/login","http://192.168.1.10/verify?acc=123","bit.ly/free-gift","secure-google.com-login.verify-account.net","bankofamerica.com.login-security-check.co/confirm"]
    legit_like=["https://www.google.com/","https://www.bankofamerica.com/","https://docs.python.org/3/","https://www.wikipedia.org/","https://www.amazon.com/","https://www.microsoft.com/"]
    return pd.DataFrame({"url":phishing_like+legit_like})

# ---------- Legacy support ----------
def _featurize_urls(urls: List[str]) -> pd.DataFrame:
    rows=[]
    for u in urls:
        url_text=(u or "").strip()
        p=urlparse(url_text if "://" in url_text else "http://"+url_text)
        host=p.hostname or ""; path=p.path or ""; query=p.query or ""; frag=p.fragment or ""; scheme=(p.scheme or "").lower(); qs=parse_qs(query); parts=_split_host(host)
        feats={"length_url":_safe_len(url_text),"length_hostname":_safe_len(host),"ip":_is_ip(host),
               "nb_dots":url_text.count("."),"nb_hyphens":url_text.count("-"),"nb_at":url_text.count("@"),"nb_qm":url_text.count("?"),
               "nb_and":url_text.count("&"),"nb_or":url_text.count("|"),"nb_eq":url_text.count("="),"nb_underscore":url_text.count("_"),
               "nb_tilde":url_text.count("~"),"nb_percent":url_text.count("%"),"nb_slash":url_text.count("/"),"nb_star":url_text.count("*"),
               "nb_colon":url_text.count(":"),"nb_comma":url_text.count(","),"nb_semicolon":url_text.count(";"),"nb_dollar":url_text.count("$"),
               "nb_space":_count(r"\s",url_text),"nb_www":int("www" in host.lower()),"nb_com":int(".com" in url_text.lower()),
               "nb_dslash":url_text.count("//")-1,"http_in_path":int("http" in path.lower()),
               "https_token":int("https" in url_text.lower() and scheme!="https"),
               "ratio_digits_url":(_count(r"\d",url_text)/max(1,len(url_text))),"ratio_digits_host":(_count(r"\d",host)/max(1,len(host))),
               "punycode":int("xn--" in host.lower()),"port":int(p.port is not None),
               "tld_in_path":_has_tld_in(path,parts["tld"]),"tld_in_subdomain":_has_tld_in(parts["sub"],parts["tld"]),
               "abnormal_subdomain":int(_num_subdomains(host)>3),"nb_subdomains":_num_subdomains(host),
               "prefix_suffix":int("-" in parts["sld"]) if parts["sld"] else 0,
               "random_domain":int(bool(re.fullmatch(r"[a-z]{6,}\d{2,}|[a-z0-9]{12,}",parts["sld"] or ""))),
               "shortening_service":int((host or "").lower() in _SHORTENERS),"path_extension":_path_ext(path),
               "nb_redirection":max(0,url_text.count("//")-1),"nb_external_redirection":0,
               "char_repeat":int(bool(re.search(r"(.)\1{3,}",url_text))),"suspicious_tld":int((parts["tld"] or "") in _SUSPICIOUS_TLDS),
               "suspicious_token_count":_suspicious_token_count(url_text),"scheme":scheme,"top_level_domain":parts["tld"],
               "first_path_token":(path.split("/")[1].lower() if path.startswith("/") and len(path.split("/"))>1 else ""),
               "num_params":len(qs),"frag_length":_safe_len(frag),
               "web_traffic":0.0,"page_rank":0.0,
               "url":url_text,"hostname":host,"domain":parts["base"]}
        feats.update(_word_stats(host,path,url_text)); feats.update(_brand_hits(host,path))
        rows.append(feats)
    return pd.DataFrame(rows)

def _align_legacy(df_feat: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    df = df_feat.replace([np.inf, -np.inf], np.nan)
    expected = list(map(str, numeric_cols)) + list(map(str, categorical_cols))
    have = set(df.columns)
    missing = [c for c in expected if c not in have]
    for m in missing:
        if m in numeric_cols: df[m] = 0
        else: df[m] = ""
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in categorical_cols:
        df[c] = df[c].astype("string").fillna("")
    df = df[expected].fillna(0)
    return df, missing

def _predict_proba_hybrid(pipe, meta, urls: List[str]) -> Tuple[np.ndarray, Dict[str,Any]]:
    info = {}
    X_raw = pd.DataFrame({"url": urls})
    try:
        probs = pipe.predict_proba(X_raw)
        info["mode"] = "unified_pipeline(url)"
        return probs, info
    except Exception:
        numeric_cols = list(map(str, (meta or {}).get("numeric_cols", [])))
        categorical_cols = list(map(str, (meta or {}).get("categorical_cols", [])))
        if not (numeric_cols or categorical_cols):
            raise
        feats = _featurize_urls(urls)
        X_aligned, missing_cols = _align_legacy(feats, numeric_cols, categorical_cols)
        info["mode"] = "legacy(aligned_features)"
        info["missing_filled"] = missing_cols
        probs = pipe.predict_proba(X_aligned)
        return probs, info

# --------------- Auto-select positive column robustly ---------------
def _choose_positive_col(probs_probe: np.ndarray, probs_batch: np.ndarray, threshold: float) -> int:
    """
    Pick the column that 1) separates the mixed probe (phishing-like > legit-like),
    and 2) does not collapse the batch into a single class at the given threshold
       (use entropy/variance to prefer the more informative column).
    """
    ncols = probs_probe.shape[1]
    if ncols == 1: return 0
    n = probs_probe.shape[0] // 2
    deltas = probs_probe[:n].mean(axis=0) - probs_probe[n:].mean(axis=0)  # higher is 'phishing' col
    # initial guess
    cand = int(np.argmax(deltas))
    # if batch all one class with this column, try the other
    preds = (probs_batch[:, cand] >= threshold).astype(int)
    if preds.mean() in (0.0, 1.0):
        other = 1 - cand if ncols == 2 else int(np.argsort(deltas)[-2])
        preds_other = (probs_batch[:, other] >= threshold).astype(int)
        # pick the one with higher label entropy (less collapsed)
        def entropy(p): 
            if p in (0.0,1.0): return 0.0
            return - (p*np.log2(p) + (1-p)*np.log2(1-p))
        return other if entropy(preds_other.mean()) > entropy(preds.mean()) else cand
    return cand

# ---------------- Sidebar: load or upload ----------------
st.sidebar.title("⚙️ Artifacts")
st.sidebar.caption("Load unified pipeline + metadata from disk OR upload below.")

if "pipe" not in st.session_state:
    try:
        pipe, meta = _load_artifacts(MODEL_PATH, META_PATH)
        st.session_state.pipe = pipe
        st.session_state.meta = meta
        st.sidebar.success(f"Loaded from {MODEL_PATH} / {META_PATH}")
    except Exception as e:
        st.sidebar.error(f"Disk load failed: {e}")

st.sidebar.markdown("---")
up_m = st.sidebar.file_uploader("Upload pipeline (.joblib/.pkl)", type=["joblib","pkl"], key="upl_model")
up_j = st.sidebar.file_uploader("Upload metadata (.json)", type=["json"], key="upl_meta")
if st.sidebar.button("Use uploaded artifacts"):
    if up_m and up_j:
        try:
            tmpdir = st.session_state.setdefault("_artifacts_dir", tempfile.mkdtemp(prefix="artifacts_"))
            mpath = os.path.join(tmpdir, "uploaded_pipeline.joblib")
            jpath = os.path.join(tmpdir, "uploaded_metadata.json")
            with open(mpath,"wb") as f: f.write(up_m.getbuffer())
            with open(jpath,"wb") as f: f.write(up_j.getbuffer())
            pipe, meta = _load_artifacts(mpath, jpath)
            st.session_state.pipe = pipe
            st.session_state.meta = meta
            st.sidebar.success("Uploaded artifacts loaded.")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded artifacts: {e}")
    else:
        st.sidebar.warning("Please upload BOTH the pipeline and metadata files.")

# ---------------- Main UI ----------------
st.title("🔍 Phishing URL Detector")
st.caption("Paste one URL per line. Works with unified pipelines (raw URL) and legacy models (engineered columns).")

if "pipe" not in st.session_state:
    st.error("Model not loaded. Add artifacts to your repo OR upload them via the sidebar.")
else:
    threshold = st.slider("Decision threshold (probability for 'phishing')", 0.10, 0.90, 0.80, 0.01)
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
            # 1) Probe on fixed mix to detect column
            probe_urls = _probe_df()["url"].tolist()
            probs_probe, _ = _predict_proba_hybrid(st.session_state.pipe, st.session_state.meta, probe_urls)

            # 2) Get batch probs (try unified, else legacy)
            probs_all, info = _predict_proba_hybrid(st.session_state.pipe, st.session_state.meta, urls)
            mode = info.get("mode","?")

            # 3) Choose best positive column automatically
            pos_col = _choose_positive_col(probs_probe, probs_all, threshold)
            probs = probs_all[:, pos_col]
            preds = (probs >= threshold).astype(int)

            # Diagnostics + fallback warnings
            p_mean, p_min, p_max = float(np.mean(probs)), float(np.min(probs)), float(np.max(probs))
            st.caption(f"Mode: {mode} | Using prob column: {pos_col} | mean/min/max = {p_mean:.3f}/{p_min:.3f}/{p_max:.3f}")

            if mode.startswith("legacy") and info.get("missing_filled"):
                exp = len(info.get("missing_filled"))  # count we filled (informational)
                st.info(f"Legacy model detected. Filled {exp} missing engineered columns. Large gaps can skew predictions; consider retraining a unified pipeline.")

            if preds.mean() in (0.0, 1.0):
                st.warning("All predictions fell into a single class at this threshold. If this keeps happening, it indicates feature mismatch or a highly imbalanced/over-regularized model.")

            out = pd.DataFrame({"url": urls,
                                "phishing_prob": probs,
                                "prediction": np.where(preds==1,"phishing","legitimate")})

            st.success(f"Predicted {(out['prediction']=='phishing').sum()} phishing out of {len(out)} rows.")
            if show_probs:
                st.dataframe(out, use_container_width=True)

            st.download_button("⬇️ Download predictions",
                               data=out.to_csv(index=False).encode("utf-8"),
                               file_name="predictions.csv", mime="text/csv")
