"""
preprocessing.py
----------------
Loads the raw PS2_Dataset.csv, cleans it and encodes every column into a
purely numeric form that can be fed into ML / DL models.

All the fitted encoders (LabelEncoders for each categorical column + the
StandardScaler for the numeric columns) are saved to disk so that the
Streamlit app can apply the EXACT same transformation to a new student's
answers before asking the trained model for a prediction.
"""

import pandas as pd
import numpy as np
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "PS2_Dataset.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Columns that are already numeric in the raw file
NUMERIC_COLS = [
    "Logical quotient rating",
    "hackathons",
    "coding skills rating",
    "public speaking points",
]

# Columns that are categorical (text) and need Label/Ordinal encoding
CATEGORICAL_COLS = [
    "self-learning capability?",
    "Extra-courses did",
    "certifications",
    "workshops",
    "reading and writing skills",
    "memory capability score",
    "Interested subjects",
    "interested career area",
    "Type of company want to settle in?",
    "Taken inputs from seniors or elders",
    "Interested Type of Books",
    "Management or Technical",
    "hard/smart worker",
    "worked in teams ever?",
    "Introvert",
]

TARGET_COL = "Suggested Job Role"

# Ordinal columns have a natural order -> map explicitly instead of
# arbitrary LabelEncoder ordering, this keeps "excellent" > "medium" > "poor"
ORDINAL_MAPS = {
    "reading and writing skills": {"poor": 0, "medium": 1, "excellent": 2},
    "memory capability score": {"poor": 0, "medium": 1, "excellent": 2},
}

# Simple yes/no columns -> map explicitly to 0/1 (keeps things interpretable)
BINARY_MAPS = {
    "self-learning capability?": {"no": 0, "yes": 1},
    "Extra-courses did": {"no": 0, "yes": 1},
    "Taken inputs from seniors or elders": {"no": 0, "yes": 1},
    "worked in teams ever?": {"no": 0, "yes": 1},
    "Introvert": {"no": 0, "yes": 1},
}

# The remaining nominal (no natural order) categorical columns get a
# LabelEncoder each, saved so the app can reuse the identical mapping.
NOMINAL_COLS = [c for c in CATEGORICAL_COLS
                if c not in ORDINAL_MAPS and c not in BINARY_MAPS]


def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: strip whitespace, drop exact duplicates, fix dtypes."""
    df = df.copy()
    df = df.drop_duplicates()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna().reset_index(drop=True)
    return df


def fit_encoders(df: pd.DataFrame):
    """Fit a LabelEncoder for every nominal column + the target column."""
    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    for col in NOMINAL_COLS:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

    target_le = LabelEncoder()
    target_le.fit(df[TARGET_COL])
    encoders[TARGET_COL] = target_le
    return encoders


def encode_features(df: pd.DataFrame, encoders: dict, fit_scaler=False, scaler=None):
    """Turn the cleaned, raw dataframe into a fully numeric feature matrix X
    and the encoded target vector y (y is None if TARGET_COL isn't present,
    e.g. for a single new student coming from the Streamlit form)."""
    from sklearn.preprocessing import StandardScaler

    df = df.copy()

    # 1. binary yes/no columns
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping)

    # 2. ordinal columns
    for col, mapping in ORDINAL_MAPS.items():
        df[col] = df[col].map(mapping)

    # 3. nominal columns -> use the already-fitted encoders
    for col in NOMINAL_COLS:
        le = encoders[col]
        df[col] = df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    feature_cols = NUMERIC_COLS + list(BINARY_MAPS.keys()) + \
        list(ORDINAL_MAPS.keys()) + NOMINAL_COLS

    X = df[feature_cols].astype(float)

    # 4. scale the numeric features (helps DL model & SVM/LogReg converge)
    if fit_scaler:
        scaler = StandardScaler()
        X[NUMERIC_COLS] = scaler.fit_transform(X[NUMERIC_COLS])
    else:
        X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    y = None
    if TARGET_COL in df.columns:
        y = encoders[TARGET_COL].transform(df[TARGET_COL])

    return X, y, scaler, feature_cols


def build_dataset():
    """One call that returns train-ready X, y, encoders, scaler, feature_cols."""
    raw = load_raw_data()
    clean = clean_data(raw)
    encoders = fit_encoders(clean)
    X, y, scaler, feature_cols = encode_features(clean, encoders, fit_scaler=True)
    return X, y, encoders, scaler, feature_cols, clean


def save_artifacts(encoders, scaler, feature_cols):
    joblib.dump(encoders, os.path.join(ARTIFACT_DIR, "encoders.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(ARTIFACT_DIR, "feature_cols.pkl"))


def load_artifacts():
    encoders = joblib.load(os.path.join(ARTIFACT_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_cols.pkl"))
    return encoders, scaler, feature_cols


if __name__ == "__main__":
    X, y, encoders, scaler, feature_cols, clean = build_dataset()
    save_artifacts(encoders, scaler, feature_cols)
    print("Feature matrix shape:", X.shape)
    print("Target classes:", list(encoders[TARGET_COL].classes_))
    print("Artifacts saved to", ARTIFACT_DIR)
