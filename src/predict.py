"""
src/predict.py — Inference Engine

Given two team names, compute features using historical data and return a
structured prediction dict. This is the only module that loads models at runtime.
"""

import json
import os
import sys
from collections import defaultdict
import math

import joblib
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLASSIFIER_PATH,
    HOME_GOALS_PATH,
    AWAY_GOALS_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_COLUMNS,
    LOOKBACK_MATCHES,
    HOST_NATIONS,
)
from src.normalize import normalize_team_name


def load_models():
    """
    Load all 3 models and feature column order.
    
    Returns:
        Tuple of (classifier, home_regressor, away_regressor, feature_columns)
        or raises FileNotFoundError if any file is missing.
    """
    for path, name in [
        (CLASSIFIER_PATH, "outcome_classifier.pkl"),
        (HOME_GOALS_PATH, "home_goals_regressor.pkl"),
        (AWAY_GOALS_PATH, "away_goals_regressor.pkl"),
        (FEATURE_COLUMNS_PATH, "feature_columns.json"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {name}. Run: python src/train.py"
            )

    classifier = joblib.load(CLASSIFIER_PATH)
    home_reg = joblib.load(HOME_GOALS_PATH)
    away_reg = joblib.load(AWAY_GOALS_PATH)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_cols = json.load(f)

    return classifier, home_reg, away_reg, feature_cols


def build_feature_vector(home_team, away_team, historical_df, feature_columns):
    """
    Build a feature vector for a prediction, using the SAME logic as ETL.
    
    Args:
        home_team: Normalized home team name.
        away_team: Normalized away team name.
        historical_df: Full historical DataFrame (includes new results).
        feature_columns: Ordered list of feature column names.
    
    Returns:
        Tuple of (numpy array shape (1, 14), warning_message or None)
    """
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    warning_msg = None

    # Drop rows with NaN scores and filter to relevant teams for performance
    df = historical_df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Compute global averages for fallback
    global_avg_goals = df[["home_score", "away_score"]].values.mean()

    # Pre-filter to only rows involving either team (massive speedup)
    relevant = df[
        (df["home_team"] == home_team) | (df["away_team"] == home_team) |
        (df["home_team"] == away_team) | (df["away_team"] == away_team)
    ]

    # Build team history from historical data
    home_matches = []
    away_matches = []
    h2h_matches = []

    for _, row in relevant.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        h_score = row["home_score"]
        a_score = row["away_score"]

        # Home team history
        if h == home_team:
            result = "win" if h_score > a_score else ("draw" if h_score == a_score else "loss")
            pts = 3 if result == "win" else (1 if result == "draw" else 0)
            home_matches.append({
                "goals_scored": h_score,
                "goals_conceded": a_score,
                "result": result,
                "pts": pts,
            })
        elif a == home_team:
            result = "win" if a_score > h_score else ("draw" if a_score == h_score else "loss")
            pts = 3 if result == "win" else (1 if result == "draw" else 0)
            home_matches.append({
                "goals_scored": a_score,
                "goals_conceded": h_score,
                "result": result,
                "pts": pts,
            })

        # Away team history
        if h == away_team:
            result = "win" if h_score > a_score else ("draw" if h_score == a_score else "loss")
            pts = 3 if result == "win" else (1 if result == "draw" else 0)
            away_matches.append({
                "goals_scored": h_score,
                "goals_conceded": a_score,
                "result": result,
                "pts": pts,
            })
        elif a == away_team:
            result = "win" if a_score > h_score else ("draw" if a_score == h_score else "loss")
            pts = 3 if result == "win" else (1 if result == "draw" else 0)
            away_matches.append({
                "goals_scored": a_score,
                "goals_conceded": h_score,
                "result": result,
                "pts": pts,
            })

        # H2H history
        if (h == home_team and a == away_team) or (h == away_team and a == home_team):
            h2h_matches.append({
                "home": h,
                "away": a,
                "h_score": h_score,
                "a_score": a_score,
            })

    features = {}

    # Check data sufficiency
    if len(home_matches) == 0:
        warning_msg = f"No historical data for {home_team}. Predictions are estimates only."
    elif len(home_matches) < 3:
        warning_msg = f"Limited data for {home_team} ({len(home_matches)} matches). Predictions may be unreliable."

    if len(away_matches) == 0:
        msg = f"No historical data for {away_team}. Predictions are estimates only."
        warning_msg = f"{warning_msg} {msg}" if warning_msg else msg
    elif len(away_matches) < 3:
        msg = f"Limited data for {away_team} ({len(away_matches)} matches). Predictions may be unreliable."
        warning_msg = f"{warning_msg} {msg}" if warning_msg else msg

    # Home team features
    if len(home_matches) >= 3:
        recent = home_matches[-LOOKBACK_MATCHES:]
        features["home_avg_goals_scored"] = np.mean([m["goals_scored"] for m in recent])
        features["home_avg_goals_conceded"] = np.mean([m["goals_conceded"] for m in recent])
        features["home_win_rate"] = np.mean([1.0 if m["result"] == "win" else 0.0 for m in recent])
        features["home_draw_rate"] = np.mean([1.0 if m["result"] == "draw" else 0.0 for m in recent])
        features["home_goal_diff_avg"] = np.mean([m["goals_scored"] - m["goals_conceded"] for m in recent])
    else:
        features["home_avg_goals_scored"] = global_avg_goals
        features["home_avg_goals_conceded"] = global_avg_goals
        features["home_win_rate"] = 0.33
        features["home_draw_rate"] = 0.33
        features["home_goal_diff_avg"] = 0.0

    if len(home_matches) > 0:
        features["home_last_match_pts"] = home_matches[-1]["pts"]
        recent_3 = home_matches[-3:]
        features["home_form_last_3"] = np.mean([m["pts"] for m in recent_3]) * 3
        
        total_wins = sum(1 for m in home_matches if m["result"] == "win")
        total_draws = sum(1 for m in home_matches if m["result"] == "draw")
        total_points = sum(m["pts"] for m in home_matches)
        features["home_elo_score"] = (total_wins * 3 + total_draws) / len(home_matches)
        features["home_points_per_match"] = total_points / len(home_matches)
        features["home_goal_efficiency"] = sum(m["goals_scored"] for m in recent_3) / len(recent_3)
    else:
        features["home_last_match_pts"] = 1.0
        features["home_form_last_3"] = 3.0
        features["home_elo_score"] = 1.0
        features["home_points_per_match"] = 1.0
        features["home_goal_efficiency"] = global_avg_goals

    # Away team features
    if len(away_matches) >= 3:
        recent = away_matches[-LOOKBACK_MATCHES:]
        features["away_avg_goals_scored"] = np.mean([m["goals_scored"] for m in recent])
        features["away_avg_goals_conceded"] = np.mean([m["goals_conceded"] for m in recent])
        features["away_win_rate"] = np.mean([1.0 if m["result"] == "win" else 0.0 for m in recent])
        features["away_draw_rate"] = np.mean([1.0 if m["result"] == "draw" else 0.0 for m in recent])
        features["away_goal_diff_avg"] = np.mean([m["goals_scored"] - m["goals_conceded"] for m in recent])
    else:
        features["away_avg_goals_scored"] = global_avg_goals
        features["away_avg_goals_conceded"] = global_avg_goals
        features["away_win_rate"] = 0.33
        features["away_draw_rate"] = 0.33
        features["away_goal_diff_avg"] = 0.0

    if len(away_matches) > 0:
        features["away_last_match_pts"] = away_matches[-1]["pts"]
        recent_3 = away_matches[-3:]
        features["away_form_last_3"] = np.mean([m["pts"] for m in recent_3]) * 3
        
        total_wins = sum(1 for m in away_matches if m["result"] == "win")
        total_draws = sum(1 for m in away_matches if m["result"] == "draw")
        total_points = sum(m["pts"] for m in away_matches)
        features["away_elo_score"] = (total_wins * 3 + total_draws) / len(away_matches)
        features["away_points_per_match"] = total_points / len(away_matches)
        features["away_goal_efficiency"] = sum(m["goals_scored"] for m in recent_3) / len(recent_3)
    else:
        features["away_last_match_pts"] = 1.0
        features["away_form_last_3"] = 3.0
        features["away_elo_score"] = 1.0
        features["away_points_per_match"] = 1.0
        features["away_goal_efficiency"] = global_avg_goals
        
    features["strength_diff"] = features["home_elo_score"] - features["away_elo_score"]

    # Head-to-head features
    if len(h2h_matches) > 0:
        home_wins_h2h = sum(
            1 for m in h2h_matches
            if (m["home"] == home_team and m["h_score"] > m["a_score"])
            or (m["away"] == home_team and m["a_score"] > m["h_score"])
        )
        features["h2h_home_win_rate"] = home_wins_h2h / len(h2h_matches)
        features["h2h_total_matches"] = len(h2h_matches)
    else:
        features["h2h_home_win_rate"] = 0.33
        features["h2h_total_matches"] = 0

    # Context features (WC 2026 matches are neutral unless host nation)
    features["is_world_cup"] = 1
    if home_team in HOST_NATIONS or away_team in HOST_NATIONS:
        features["is_neutral_venue"] = 0
    else:
        features["is_neutral_venue"] = 1

    # --- NEW FEATURES (Patches 3.0, 3.1, 3.2) ---
    
    # 1. Adjusted Goal Efficiency
    mean_elo = historical_df["home_elo_score"].mean() if "home_elo_score" in historical_df.columns else 1.0
    features["home_adj_goal_efficiency"] = features["home_goal_efficiency"] * (features["away_elo_score"] / mean_elo)
    features["away_adj_goal_efficiency"] = features["away_goal_efficiency"] * (features["home_elo_score"] / mean_elo)

    # 2. Defensive Stats (Window=5)
    if len(home_matches) > 0:
        recent_home_def = home_matches[-5:]
        features["home_goals_conceded_per_match"] = np.mean([m["goals_conceded"] for m in recent_home_def])
        features["home_clean_sheet_ratio"] = np.mean([1 if m["goals_conceded"] == 0 else 0 for m in recent_home_def])
    else:
        features["home_goals_conceded_per_match"] = features.get("home_avg_goals_conceded", global_avg_goals)
        features["home_clean_sheet_ratio"] = 0.2

    if len(away_matches) > 0:
        recent_away_def = away_matches[-5:]
        features["away_goals_conceded_per_match"] = np.mean([m["goals_conceded"] for m in recent_away_def])
        features["away_clean_sheet_ratio"] = np.mean([1 if m["goals_conceded"] == 0 else 0 for m in recent_away_def])
    else:
        features["away_goals_conceded_per_match"] = features.get("away_avg_goals_conceded", global_avg_goals)
        features["away_clean_sheet_ratio"] = 0.2

    # 3. Draw Signals
    features['elo_gap'] = abs(features['home_elo_score'] - features['away_elo_score'])
    max_gap = historical_df["elo_gap"].max() if "elo_gap" in historical_df.columns else 1.0
    if max_gap == 0 or pd.isna(max_gap): max_gap = 1.0
    
    features['competitiveness'] = 1 - (features['elo_gap'] / max_gap)
    features['defensive_parity'] = 1 - abs(features['home_clean_sheet_ratio'] - features['away_clean_sheet_ratio'])
    features['draw_likelihood'] = (features['competitiveness'] + features['defensive_parity']) / 2

    # Assemble in the exact column order
    vector = np.array([[features[col] for col in feature_columns]])
    return vector, warning_msg

def get_ai_reasoning(home_team, away_team, X_fwd, feature_cols):
    """
    Generate an AI reasoning string based on feature vector values.
    """
    try:
        sd_idx = feature_cols.index("strength_diff")
        sd = X_fwd[0][sd_idx]
        
        if abs(sd) > 2.0:
            return "Prediction heavily influenced by significant power gap."
        elif abs(sd) < 1.0:
            return "Prediction based on recent goal-scoring efficiency and tight competitive balance."
        else:
            return "Prediction heavily influenced by historical metrics and neutral venue adjustment."
    except ValueError:
        return "Prediction heavily influenced by historical metrics and neutral venue adjustment."


def predict_match(home_team, away_team, historical_df, models=None):
    """
    Predict match outcome and scoreline.
    
    Args:
        home_team: Team name (will be normalized).
        away_team: Team name (will be normalized).
        historical_df: Full historical DataFrame.
        models: Optional pre-loaded models tuple. If None, loads from disk.
    
    Returns:
        Dict with prediction results.
    """
    if models is None:
        classifier, home_reg, away_reg, feature_cols = load_models()
    else:
        classifier, home_reg, away_reg, feature_cols = models

    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)

    # Build forward feature vector
    X_fwd, warning_msg = build_feature_vector(
        home_team, away_team, historical_df, feature_cols
    )
    # Build reverse feature vector
    X_rev, _ = build_feature_vector(
        away_team, home_team, historical_df, feature_cols
    )

    # Outcome probabilities (Forward)
    proba_fwd = classifier.predict_proba(X_fwd)[0]
    # Outcome probabilities (Reverse)
    proba_rev = classifier.predict_proba(X_rev)[0]
    
    classes = classifier.classes_
    fwd_dict = {cls: p for cls, p in zip(classes, proba_fwd)}
    rev_dict = {cls: p for cls, p in zip(classes, proba_rev)}

    # Map probabilities to outcome labels
    # Average them. Reverse's Away Win (-1) is Forward's Home Win (1), etc.
    home_win_pct = ((fwd_dict.get(1, 0) + rev_dict.get(-1, 0)) / 2) * 100
    draw_pct = ((fwd_dict.get(0, 0) + rev_dict.get(0, 0)) / 2) * 100
    away_win_pct = ((fwd_dict.get(-1, 0) + rev_dict.get(1, 0)) / 2) * 100

    # Exact scoreline
    raw_home_fwd = home_reg.predict(X_fwd)[0]
    raw_away_fwd = away_reg.predict(X_fwd)[0]
    
    raw_home_rev = away_reg.predict(X_rev)[0] # Reversed prediction for Home
    raw_away_rev = home_reg.predict(X_rev)[0] # Reversed prediction for Away

    raw_home = max(0.0, (raw_home_fwd + raw_home_rev) / 2)
    raw_away = max(0.0, (raw_away_fwd + raw_away_rev) / 2)

    # Context-Aware Rounding
    if home_win_pct > draw_pct and home_win_pct > away_win_pct:
        home_goals = int(round(raw_home))
        away_goals = int(round(raw_away))
        if home_goals <= away_goals:
            home_goals = math.ceil(raw_home)
            away_goals = math.floor(raw_away)
        if home_goals <= away_goals:
            home_goals = away_goals + 1
        if home_goals == 0 and away_goals == 0:
            home_goals = 1
            
    elif away_win_pct > draw_pct and away_win_pct > home_win_pct:
        home_goals = int(round(raw_home))
        away_goals = int(round(raw_away))
        if away_goals <= home_goals:
            home_goals = math.floor(raw_home)
            away_goals = math.ceil(raw_away)
        if away_goals <= home_goals:
            away_goals = home_goals + 1
        if home_goals == 0 and away_goals == 0:
            away_goals = 1
            
    else:
        # Draw is most likely outcome
        avg_goals = int(round((raw_home + raw_away) / 2))
        home_goals = avg_goals
        away_goals = avg_goals

    reasoning = get_ai_reasoning(home_team, away_team, X_fwd, feature_cols)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_win_pct": round(home_win_pct, 1),
        "draw_pct": round(draw_pct, 1),
        "away_win_pct": round(away_win_pct, 1),
        "warning": warning_msg,
        "reasoning": reasoning,
    }
