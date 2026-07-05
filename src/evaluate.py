"""
src/evaluate.py - Model Evaluation Module

Loads the current trained models and the processed feature dataset,
re-runs the identical train/test split used in train.py, then computes
a comprehensive set of ML metrics.

Returns pure Python dicts and DataFrames -- no Streamlit dependency here.
All rendering lives in app.py (tab_analytics).
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLASSIFIER_PATH,
    HOME_GOALS_PATH,
    AWAY_GOALS_PATH,
    FEATURE_COLUMNS_PATH,
    PROCESSED_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
)


def evaluate_model_performance() -> dict:
    """
    Load the current trained models and processed feature dataset,
    reproduce the train/test split, and return a comprehensive metrics dict.

    Returns a dict with keys:
        classifier_metrics  - accuracy, log_loss, brier_score, report_df
        confusion_matrix    - matrix (np.ndarray 3x3), labels list
        home_goals_metrics  - mae, rmse
        away_goals_metrics  - mae, rmse
        feature_importance  - pd.DataFrame (Feature, Importance) sorted desc
        dataset_info        - train_size, test_size, total_samples
    """
    # 1. Load models
    for p in [CLASSIFIER_PATH, HOME_GOALS_PATH, AWAY_GOALS_PATH, FEATURE_COLUMNS_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model file missing: {p}. Run training first.")

    clf      = joblib.load(CLASSIFIER_PATH)
    home_reg = joblib.load(HOME_GOALS_PATH)
    away_reg = joblib.load(AWAY_GOALS_PATH)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_cols = json.load(f)

    # 2. Load processed features
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Processed features not found at {PROCESSED_PATH}.")

    df = pd.read_csv(PROCESSED_PATH)
    feature_cols = [c for c in feature_cols if c in df.columns]
    X            = df[feature_cols]
    y_outcome    = df["outcome_encoded"]
    y_home_goals = df["home_score"]
    y_away_goals = df["away_score"]

    # 3. Reproduce exact train/test split from train.py
    X_train, X_test, y_out_train, y_out_test, _, _ = train_test_split(
        X, y_outcome, df["date"], test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    _, _, y_hg_train, y_hg_test = train_test_split(
        X, y_home_goals, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    _, _, y_ag_train, y_ag_test = train_test_split(
        X, y_away_goals, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # 4. Classifier evaluation
    y_pred_xgb  = clf.predict(X_test)        # predicts {0,1,2}
    y_pred_out  = y_pred_xgb - 1             # back to {-1,0,1}
    y_proba_xgb = clf.predict_proba(X_test)  # shape (N,3)

    accuracy = float(accuracy_score(y_out_test, y_pred_out))

    y_out_test_xgb = (y_out_test + 1).astype(int)
    ll = float(log_loss(y_out_test_xgb, y_proba_xgb, labels=[0, 1, 2]))

    # Multi-class Brier: average of OvR per-class Brier scores
    brier_scores = []
    for cls_idx in range(3):
        y_bin = (y_out_test_xgb == cls_idx).astype(int)
        brier_scores.append(float(brier_score_loss(y_bin, y_proba_xgb[:, cls_idx])))
    brier_avg = float(np.mean(brier_scores))

    report_dict = classification_report(
        y_out_test, y_pred_out,
        labels=[-1, 0, 1],
        target_names=["Away Win", "Draw", "Home Win"],
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for cls_name in ["Away Win", "Draw", "Home Win"]:
        r = report_dict.get(cls_name, {})
        report_rows.append({
            "Class":     cls_name,
            "Precision": round(r.get("precision", 0.0), 3),
            "Recall":    round(r.get("recall",    0.0), 3),
            "F1-Score":  round(r.get("f1-score",  0.0), 3),
            "Support":   int(r.get("support",     0)),
        })
    report_df = pd.DataFrame(report_rows)

    cm = confusion_matrix(y_out_test, y_pred_out, labels=[-1, 0, 1])

    # 5. Regression evaluation
    hg_mae  = float(mean_absolute_error(y_hg_test, home_reg.predict(X_test)))
    hg_rmse = float(root_mean_squared_error(y_hg_test, home_reg.predict(X_test)))
    ag_mae  = float(mean_absolute_error(y_ag_test, away_reg.predict(X_test)))
    ag_rmse = float(root_mean_squared_error(y_ag_test, away_reg.predict(X_test)))

    # 6. Feature importance
    fi_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_df["Importance"] = fi_df["Importance"].round(4)

    return {
        "classifier_metrics": {
            "accuracy":    accuracy,
            "log_loss":    ll,
            "brier_score": brier_avg,
            "report_df":   report_df,
        },
        "confusion_matrix": {
            "matrix": cm,
            "labels": ["Away Win", "Draw", "Home Win"],
        },
        "home_goals_metrics": {"mae": hg_mae, "rmse": hg_rmse},
        "away_goals_metrics": {"mae": ag_mae, "rmse": ag_rmse},
        "feature_importance": fi_df,
        "dataset_info": {
            "train_size":    len(X_train),
            "test_size":     len(X_test),
            "total_samples": len(df),
        },
    }
