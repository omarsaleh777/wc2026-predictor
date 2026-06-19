"""
config.py — Global Configuration
Single source of truth for every constant used across the project.
All other modules import from here. No hardcoded paths or magic numbers anywhere else.
"""

import os
import numpy as np
from xgboost import XGBClassifier

class CustomXGBClassifier(XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fake_classes = False
        
    @property
    def classes_(self):
        if getattr(self, '_fake_classes', False):
            return np.array([-1, 0, 1])
        # In XGBoost 1.6+, classes_ is normally derived from n_classes_
        if hasattr(self, 'n_classes_'):
            return np.arange(self.n_classes_)
        return np.array([0, 1, 2])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECT ROOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAW_RESULTS_PATH     = os.path.join(PROJECT_ROOT, "data", "raw", "international_results.csv")
RAW_FIXTURES_PATH    = os.path.join(PROJECT_ROOT, "data", "raw", "wc2026_fixtures.csv")
PROCESSED_PATH       = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
NEW_RESULTS_PATH     = os.path.join(PROJECT_ROOT, "data", "updated", "new_results.csv")
STANDINGS_PATH       = os.path.join(PROJECT_ROOT, "data", "standings", "group_standings.json")

CLASSIFIER_PATH      = os.path.join(PROJECT_ROOT, "models", "outcome_classifier.pkl")
HOME_GOALS_PATH      = os.path.join(PROJECT_ROOT, "models", "home_goals_regressor.pkl")
AWAY_GOALS_PATH      = os.path.join(PROJECT_ROOT, "models", "away_goals_regressor.pkl")
FEATURE_COLUMNS_PATH = os.path.join(PROJECT_ROOT, "models", "feature_columns.json")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE ENGINEERING PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOOKBACK_MATCHES = 10        # Rolling window: last N matches per team
MIN_MATCH_YEAR   = 1993      # Filter historical data before this year
                              # (pre-1993 data lacks consistency + many nations didn't exist)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL HYPERPARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
N_ESTIMATORS   = 1000
LEARNING_RATE  = 0.05
MAX_DEPTH      = 5
TOURNAMENT_WEIGHT_MULTIPLIER = 5.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WORLD CUP 2026 STRUCTURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WC2026_GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
HOST_NATIONS = ["USA", "Canada", "Mexico"]
# 12 groups × 4 teams = 48 teams total (WC 2026 expanded format)
# 3 teams advance per group (top 2 + 8 best 3rd-place finishers)

# Official WC 2026 Group Assignments (after normalization)
# Source: FIFA Final Draw, December 5, 2025
WC2026_GROUP_TEAMS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Reverse lookup: team → group
TEAM_TO_GROUP = {}
for _group, _teams in WC2026_GROUP_TEAMS.items():
    for _team in _teams:
        TEAM_TO_GROUP[_team] = _group

STAGE_ORDER = [
    "Group Stage",
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Third Place Play-off",
    "Final",
]

KNOCKOUT_STAGES = [
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Third Place Play-off",
    "Final",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTCOME ENCODING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTCOME_ENCODING = {
    "Home Win": 1,
    "Draw": 0,
    "Away Win": -1,
}
OUTCOME_LABELS = ["Away Win", "Draw", "Home Win"]  # Order matches class [-1, 0, 1]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE COLUMNS (ordered list used in training and inference)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURE_COLUMNS = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "home_win_rate",
    "away_win_rate",
    "home_draw_rate",
    "away_draw_rate",
    "h2h_home_win_rate",
    "h2h_total_matches",
    "home_goal_diff_avg",
    "away_goal_diff_avg",
    "is_neutral_venue",
    "is_world_cup",
    "home_last_match_pts",
    "away_last_match_pts",
    "home_form_last_3",
    "away_form_last_3",
    "home_elo_score",
    "away_elo_score",
    "strength_diff",
    "home_goal_efficiency",
    "away_goal_efficiency",
    "home_points_per_match",
    "away_points_per_match",
    "home_adj_goal_efficiency",
    "away_adj_goal_efficiency",
    "home_goals_conceded_per_match",
    "away_goals_conceded_per_match",
    "home_clean_sheet_ratio",
    "away_clean_sheet_ratio",
    "elo_gap",
    "competitiveness",
    "defensive_parity",
    "draw_likelihood",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIXTURES COLUMN MAPPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The wc2026_fixtures.csv uses "round" instead of "stage".
# This mapping is applied when loading fixtures.
FIXTURES_COLUMN_MAP = {
    "round": "stage",
}

# Map the "stage" values from the fixtures CSV to our canonical stage names
FIXTURES_STAGE_MAP = {
    "Group stage": "Group Stage",
    "Round of 32": "Round of 32",
    "Round of 16": "Round of 16",
    "Quarter-finals": "Quarter-final",
    "Quarter-final": "Quarter-final",
    "Semi-finals": "Semi-final",
    "Semi-final": "Semi-final",
    "Third-place match": "Third Place Play-off",
    "Third Place Play-off": "Third Place Play-off",
    "Final": "Final",
}
