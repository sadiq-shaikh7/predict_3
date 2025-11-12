from __future__ import annotations
import re
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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

_RE_SPLIT = re.compile(r"[^a-zA-Z0-9]+")
_RE_IPV4 = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}$")
_RE_IPV6 = re.compile(r"[0-9a-fA-F:]+:")


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


class URLFeatureizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer:
    Input: DataFrame with a 'url' column (string)
    Output: DataFrame with engineered numeric/categorical features.
    """

    def __init__(self, url_col: str = "url"):
        self.url_col = url_col
        self._feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None):
        # we compute feature names on a tiny sample for stability
        sample = pd.DataFrame({self.url_col: ["https://example.com/"]})
        self._feature_names_ = list(self._extract_row(sample.iloc[0][self.url_col]).keys())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        urls = X[self.url_col].astype(str).fillna("")
        rows = [self._extract_row(u) for u in urls]
        df = pd.DataFrame(rows)

        # enforce dtypes: numeric -> float, cat -> string (conservative rule)
        for col in df.columns:
            if df[col].dtype == object and col not in ("scheme","top_level_domain","first_path_token","path_extension","url","hostname","domain"):
                # try numeric
                coerced = pd.to_numeric(df[col], errors="coerce")
                # if at least half are numbers, take numeric path
                if coerced.notna().mean() >= 0.5:
                    df[col] = coerced.fillna(0.0)
                else:
                    df[col] = df[col].astype("string").fillna("")
            elif np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].astype(float)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self._feature_names_)

    # -------- internal: per-row extraction --------
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
            # placeholders for page/registry features if you add online lookups later
            "web_traffic": 0.0,
            "page_rank": 0.0,
        }

        feats.update(_word_stats(host, path, url_text))
        feats.update(_brand_hits(host, path))

        # keep raw strings for possible encoders
        feats.update({
            "url": url_text,
            "hostname": host,
            "domain": host_parts["base"]
        })
        return feats
