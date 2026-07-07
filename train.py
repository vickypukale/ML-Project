"""
train.py
--------
Trains several ML models and a Deep Learning (Keras/TensorFlow) model on
the preprocessed Career Path dataset, evaluates all of them with the same
metrics, and persists the best-performing model + all preprocessing
artifacts to models/ so the Streamlit app can load them straight away.

Run:
    python train.py
"""

import os
import json
import warnings
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from preprocessing import build_dataset, save_artifacts, TARGET_COL

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def evaluate(name, y_true, y_pred, results):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    results[name] = {
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(prec), 4),
        "recall_weighted": round(float(rec), 4),
        "f1_weighted": round(float(f1), 4),
    }
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"Accuracy: {acc:.4f}  |  F1(weighted): {f1:.4f}  |  "
          f"Precision: {prec:.4f}  |  Recall: {rec:.4f}")
    return results


def build_dl_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading & preprocessing data ...")
    X, y, encoders, scaler, feature_cols, clean_df = build_dataset()
    save_artifacts(encoders, scaler, feature_cols)

    num_classes = len(encoders[TARGET_COL].classes_)
    print(f"Dataset ready: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{num_classes} target classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    trained_models = {}

    # ---------------------------------------------------------------
    # 1. Logistic Regression (simple linear baseline)
    # ---------------------------------------------------------------
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    results = evaluate("Logistic Regression", y_test, lr.predict(X_test), results)
    trained_models["Logistic Regression"] = lr

    # ---------------------------------------------------------------
    # 2. Random Forest (classic ensemble ML model)
    # ---------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results = evaluate("Random Forest", y_test, rf.predict(X_test), results)
    trained_models["Random Forest"] = rf

    # ---------------------------------------------------------------
    # 3. Support Vector Machine
    # ---------------------------------------------------------------
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)
    results = evaluate("SVM (RBF)", y_test, svm.predict(X_test), results)
    trained_models["SVM (RBF)"] = svm

    # ---------------------------------------------------------------
    # 4. XGBoost (gradient boosted ensemble)
    # ---------------------------------------------------------------
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", num_class=num_classes,
        random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    results = evaluate("XGBoost", y_test, xgb.predict(X_test), results)
    trained_models["XGBoost"] = xgb

    # ---------------------------------------------------------------
    # 5. Deep Learning model (Keras / TensorFlow MLP)
    # ---------------------------------------------------------------
    dl_model = build_dl_model(X_train.shape[1], num_classes)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                  restore_best_weights=True)
    history = dl_model.fit(
        X_train.values, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=0,
    )
    dl_pred_probs = dl_model.predict(X_test.values, verbose=0)
    dl_pred = np.argmax(dl_pred_probs, axis=1)
    results = evaluate("Deep Learning (Keras MLP)", y_test, dl_pred, results)
    trained_models["Deep Learning (Keras MLP)"] = dl_model

    # ---------------------------------------------------------------
    # Pick the best model by weighted F1 score
    # ---------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    print(f"\n{'#'*60}\nBEST MODEL: {best_name}  "
          f"(F1 weighted = {results[best_name]['f1_weighted']})\n{'#'*60}")

    best_model = trained_models[best_name]

    # Save every model's metrics for the record
    with open(os.path.join(OUT_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Detailed classification report for the winning model
    if best_name == "Deep Learning (Keras MLP)":
        final_pred = dl_pred
    else:
        final_pred = best_model.predict(X_test)

    report = classification_report(
        y_test, final_pred, target_names=encoders[TARGET_COL].classes_,
        zero_division=0
    )
    print("\nClassification report (best model):\n", report)
    with open(os.path.join(OUT_DIR, "classification_report_best_model.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n\n{report}")

    cm = confusion_matrix(y_test, final_pred)
    np.savetxt(os.path.join(OUT_DIR, "confusion_matrix_best_model.csv"),
               cm, fmt="%d", delimiter=",")

    # ---------------------------------------------------------------
    # Persist the winning model + metadata
    # ---------------------------------------------------------------
    if best_name == "Deep Learning (Keras MLP)":
        dl_model.save(os.path.join(MODEL_DIR, "best_model.keras"))
    else:
        joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

    with open(os.path.join(MODEL_DIR, "best_model_meta.json"), "w") as f:
        json.dump({
            "best_model_name": best_name,
            "is_keras": best_name == "Deep Learning (Keras MLP)",
            "metrics": results[best_name],
            "all_results": results,
            "feature_cols": feature_cols,
        }, f, indent=2)

    print(f"\nAll trained model artifacts saved under: {MODEL_DIR}")
    print(f"Metrics / reports saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
