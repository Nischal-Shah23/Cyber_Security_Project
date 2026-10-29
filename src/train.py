"""Train and evaluate a reproducible hybrid tabular + character n-gram model."""
from __future__ import annotations
import argparse, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.base import clone
from src.dataset import load_dataset
from src.features import extract_features

SEED = 42

def build_pipeline(model_name="random_forest"):
    numeric = list(extract_features("GET / HTTP/1.1\nHost: example.test").keys())
    pre = ColumnTransformer([
        ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                       max_features=50000, sublinear_tf=True), "request"),
        ("numeric", "passthrough", numeric),
    ], remainder="drop", sparse_threshold=0.3)
    if model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                     class_weight="balanced_subsample", random_state=SEED,
                                     n_jobs=-1)
    elif model_name == "extra_trees":
        clf = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1)
    elif model_name == "logistic_regression":
        # Sparse TF-IDF and numeric features are jointly handled; numeric features are
        # not scaled here to keep the transformer compact. Model is a comparison baseline.
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("features", pre), ("classifier", clf)])

def metrics_for(y_true, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_attack": float(precision_score(y_true, pred, zero_division=0)),
        "recall_attack": float(recall_score(y_true, pred, zero_division=0)),
        "f1_attack": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) == 2 else None,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix_labels_0_normal_1_attack": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw", help="Labeled CSV or directory of labeled files")
    ap.add_argument("--out", default="models")
    ap.add_argument("--model", choices=["random_forest","extra_trees","logistic_regression"], default="random_forest")
    ap.add_argument("--test-size", type=float, default=0.20)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    # Remove exact duplicates before splitting to reduce train/test contamination.
    df = df.drop_duplicates(subset=["request", "label"]).reset_index(drop=True)
    X = pd.DataFrame({"request": df["request"]})
    feats = pd.DataFrame([extract_features(r) for r in df["request"]])
    X = pd.concat([X, feats], axis=1)
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y
    )
    model = build_pipeline(args.model)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    infer_ms_per_request = (time.perf_counter() - start) * 1000 / max(len(X_test), 1)
    results = metrics_for(y_test, pred, proba)
    results.update({
        "model": args.model, "seed": SEED, "dataset_rows_after_deduplication": int(len(df)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "train_seconds": round(train_seconds, 4),
        "inference_ms_per_request_batch_average": round(infer_ms_per_request, 4),
        "label_mapping": {"0":"normal", "1":"attack"},
        "note": "Measured locally from this run. Do not substitute README-reported metrics.",
        "classification_report": classification_report(y_test, pred, target_names=["normal","attack"], output_dict=True, zero_division=0),
    })
    joblib.dump(model, out / "shadowguard_research_model.joblib")
    (out / "training_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, pred, labels=[0,1]),
                                  display_labels=["Normal","Attack"])
    disp.plot(values_format="d")
    plt.title("Held-out test confusion matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close()
    print(json.dumps({k:v for k,v in results.items() if k != "classification_report"}, indent=2))
    print(f"Saved model and report under: {out.resolve()}")

if __name__ == "__main__":
    main()
