"""
src/train.py — Model Training

Load features → split → train 3 models → save as .pkl with feature column order.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CustomXGBClassifier,
    PROCESSED_PATH,
    CLASSIFIER_PATH,
    HOME_GOALS_PATH,
    AWAY_GOALS_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS,
    LEARNING_RATE,
    MAX_DEPTH,
    TOURNAMENT_WEIGHT_MULTIPLIER,
)


def run_training():
    """
    Main training pipeline: Load features → split → train 3 models → save.
    """
    # ──────────────────────────────────────
    # 1. Feature Matrix Setup
    # ──────────────────────────────────────
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_PATH)
    print(f"  Dataset shape: {df.shape}")

    # Use feature column order from config (or load from saved JSON if exists)
    feature_cols = FEATURE_COLUMNS

    X = df[feature_cols]
    y_outcome = df["outcome_encoded"]       # target for classifier
    y_home_goals = df["home_score"]          # target for home goals regressor
    y_away_goals = df["away_score"]          # target for away goals regressor

    # Verify no NaN values
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"NaN values found in feature columns: {nan_cols}")

    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Samples: {len(X)}")

    # ──────────────────────────────────────
    # 2. Train/Test Split
    # ──────────────────────────────────────
    X_train, X_test, y_out_train, y_out_test, date_train, _ = train_test_split(
        X, y_outcome, df["date"], test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    _, _, y_hg_train, y_hg_test = train_test_split(
        X, y_home_goals, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    _, _, y_ag_train, y_ag_test = train_test_split(
        X, y_away_goals, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    print(f"  Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    # ──────────────────────────────────────
    # 2.5 Time-Decay Sample Weights
    # ──────────────────────────────────────
    date_train_dt = pd.to_datetime(date_train)
    max_date = date_train_dt.max()
    days_ago = (max_date - date_train_dt).dt.days
    # k=0.00019 ensures 10 years ~ 0.5 weight, 33 years ~ 0.1 weight
    sample_weights = np.exp(-0.00019 * days_ago).values

    # ──────────────────────────────────────
    # 3. Model 1 — Outcome Classifier
    # ──────────────────────────────────────
    print("\n━━━ Training Outcome Classifier ━━━")
    clf = CustomXGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
    )
    # XGBoost requires classes to be 0, 1, 2. We shift our [-1, 0, 1] encoding.
    y_out_train_xgb = y_out_train + 1
    y_out_test_xgb = y_out_test + 1
    
    clf.fit(X_train, y_out_train_xgb, sample_weight=sample_weights)
    
    # Enable the fake classes for predict.py
    clf._fake_classes = True

    y_pred_out = clf.predict(X_test) - 1 # Shift back for reporting
    acc = accuracy_score(y_out_test, y_pred_out)
    
    report_dict = classification_report(y_out_test, y_pred_out, output_dict=True)
    print(classification_report(y_out_test, y_pred_out, target_names=["Away Win", "Draw", "Home Win"]))
    print(f"Classifier Accuracy: {acc:.3f}")
    
    # Diagnostic Warning for Draw Collapse
    draw_f1 = report_dict.get('0', {}).get('f1-score', 0.0)
    if draw_f1 < 0.25:
        print(f"⚠️ WARNING: The F1-score for 'Draw' has dropped below 0.25 (Current: {draw_f1:.3f}).")
        print("The model is losing the ability to differentiate tight, low-scoring draws from wins.")

    if acc < 0.44:
        print("⚠️  WARNING: Model accuracy below 0.44 threshold — check features")
    else:
        print(f"✅ Accuracy {acc:.3f} meets threshold (> 0.44)")

    # Ensure models directory exists
    os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
    joblib.dump(clf, CLASSIFIER_PATH)
    print(f"  Saved classifier to {CLASSIFIER_PATH}")

    # ──────────────────────────────────────
    # 4. Model 2 — Home Goals Regressor
    # ──────────────────────────────────────
    print("\n━━━ Training Home Goals Regressor ━━━")
    home_reg = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    home_reg.fit(X_train, y_hg_train, sample_weight=sample_weights)

    y_pred_hg = home_reg.predict(X_test)
    hg_mae = mean_absolute_error(y_hg_test, y_pred_hg)
    hg_rmse = root_mean_squared_error(y_hg_test, y_pred_hg)
    print(f"  Home Goals MAE: {hg_mae:.3f}")
    print(f"  Home Goals RMSE: {hg_rmse:.3f}")

    joblib.dump(home_reg, HOME_GOALS_PATH)
    print(f"  Saved home goals regressor to {HOME_GOALS_PATH}")

    # ──────────────────────────────────────
    # 5. Model 3 — Away Goals Regressor
    # ──────────────────────────────────────
    print("\n━━━ Training Away Goals Regressor ━━━")
    away_reg = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    away_reg.fit(X_train, y_ag_train, sample_weight=sample_weights)

    y_pred_ag = away_reg.predict(X_test)
    ag_mae = mean_absolute_error(y_ag_test, y_pred_ag)
    ag_rmse = root_mean_squared_error(y_ag_test, y_pred_ag)
    print(f"  Away Goals MAE: {ag_mae:.3f}")
    print(f"  Away Goals RMSE: {ag_rmse:.3f}")

    joblib.dump(away_reg, AWAY_GOALS_PATH)
    print(f"  Saved away goals regressor to {AWAY_GOALS_PATH}")

    # ──────────────────────────────────────
    # 6. Feature Column Persistence
    # ──────────────────────────────────────
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(list(X.columns), f)
    print(f"\n  Saved feature columns to {FEATURE_COLUMNS_PATH}")

    print("\n━━━ Saving Metrics ━━━")
    # Feature Importance for Classifier
    importance = clf.feature_importances_
    fi_dict = {col: float(imp) for col, imp in zip(list(X.columns), importance)}
    
    metrics = {
        "accuracy": float(acc),
        "home_goals_mae": float(hg_mae),
        "home_goals_rmse": float(hg_rmse),
        "away_goals_mae": float(ag_mae),
        "away_goals_rmse": float(ag_rmse),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_importance": fi_dict
    }
    
    metrics_path = os.path.join(os.path.dirname(FEATURE_COLUMNS_PATH), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Saved model metrics to {metrics_path}")

    print("\n━━━ Summary ━━━")
    print("All models and feature columns saved successfully.")
    print(f"  Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    print(f"  Classifier Accuracy: {acc:.3f}")
    print(f"  Home Goals MAE: {hg_mae:.3f} | RMSE: {hg_rmse:.3f}")
    print(f"  Away Goals MAE: {ag_mae:.3f} | RMSE: {ag_rmse:.3f}")


if __name__ == "__main__":
    run_training()
