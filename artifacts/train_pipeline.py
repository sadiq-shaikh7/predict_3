from __future__ import annotations
import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from url_features import URLFeatureizer

# ---------------- Config ----------------
DATA_PATH = "data/urls_labeled.csv"   # columns: url, label  (label: 0=legitimate, 1=phishing)
LABEL_COL = "label"
URL_COL = "url"

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "phishing_xgb_pipeline.joblib")
META_PATH  = os.path.join(ARTIFACT_DIR, "metadata.json")
POSITIVE_CLASS = 1  # 1 == phishing

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --------------- Load data --------------
df = pd.read_csv(DATA_PATH)
df = df[[URL_COL, LABEL_COL]].dropna()
X = df[[URL_COL]].copy()
y = df[LABEL_COL].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# ---------- Build full pipeline ----------
featureizer = URLFeatureizer(url_col=URL_COL)

# Use selectors by dtype AFTER featureizer.
# The featureizer returns a DataFrame with mixed dtypes; we standardize numeric and one-hot encode categoricals.
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False, min_frequency=5), selector(dtype_include=object)),
        # string dtype in pandas 2 is "string[python]"; include that too:
        ("cat_str", OneHotEncoder(handle_unknown="ignore", sparse=False, min_frequency=5), selector(dtype_include="string"))
    ],
    remainder="drop",
    n_jobs=None
)

clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=4,
    random_state=42
)

pipe = Pipeline(steps=[
    ("featureizer", featureizer),
    ("preprocess", preprocess),
    ("classifier", clf),
])

# --------------- Train -------------------
pipe.fit(X_train, y_train)

# --------------- Eval --------------------
y_pred = pipe.predict(X_val)
y_proba = pipe.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred, digits=4))

# --------------- Save --------------------
joblib.dump(pipe, MODEL_PATH)

# Save minimal metadata — no need to carry column lists anymore,
# but we keep a couple of useful hints:
meta = {
    "label_column_used": LABEL_COL,
    "url_column_used": URL_COL,
    "positive_class": POSITIVE_CLASS,  # tells app which proba column to use
    "pipeline": "URLFeatureizer -> ColumnTransformer(Scale, OHE) -> XGBClassifier"
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved model to {MODEL_PATH}")
print(f"Saved metadata to {META_PATH}")
