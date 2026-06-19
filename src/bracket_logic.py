"""
src/bracket_logic.py — Bracket & Standings Engine

Track the live state of the WC 2026 tournament. This module is the brain behind
dynamic knockout resolution. It calculates group standings from submitted results,
determines which teams advance, and resolves placeholder strings in the fixture
list so the UI can enable/disable match predictions correctly.
"""

import json
import os
import re

import pandas as pd

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    WC2026_GROUPS,
    WC2026_GROUP_TEAMS,
    STANDINGS_PATH,
)
from src.normalize import normalize_team_name


def initialize_standings(fixtures_df: pd.DataFrame) -> None:
    """
    Create initial group_standings.json with all group-stage teams
    initialized to zeroed stats.
    
    Only call this if STANDINGS_PATH does not exist.
    Never reset standings that already contain data.
    
    Args:
        fixtures_df: DataFrame of WC 2026 fixtures (already normalized).
    """
    if os.path.exists(STANDINGS_PATH):
        print("Standings file already exists. Skipping initialization.")
        return

    standings = {}
    for group in WC2026_GROUPS:
        standings[group] = {}
        for team in WC2026_GROUP_TEAMS.get(group, []):
            standings[group][team] = {
                "played": 0, "won": 0, "drawn": 0, "lost": 0,
                "gf": 0, "ga": 0, "gd": 0, "pts": 0,
            }

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(STANDINGS_PATH), exist_ok=True)
    with open(STANDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(standings, f, indent=2, ensure_ascii=False)

    total_groups = len(standings)
    total_teams = sum(len(g) for g in standings.values())
    print(f"Standings initialized for {total_groups} groups ({total_teams} teams).")


def load_standings() -> dict:
    """Load group standings from JSON file."""
    if not os.path.exists(STANDINGS_PATH):
        return {}
    with open(STANDINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_standings(standings: dict) -> None:
    """Save group standings to JSON file."""
    os.makedirs(os.path.dirname(STANDINGS_PATH), exist_ok=True)
    with open(STANDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(standings, f, indent=2, ensure_ascii=False)


def update_standings(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    group: str,
) -> None:
    """
    Update group table for one completed group-stage result.
    Only call for Group Stage matches. Skip for knockout matches.
    
    Args:
        home_team: Normalized home team name.
        away_team: Normalized away team name.
        home_score: Goals scored by home team.
        away_score: Goals scored by away team.
        group: Group letter ("A" through "L").
    """
    standings = load_standings()

    if group not in standings:
        standings[group] = {}

    # Initialize team entry if missing
    for team in [home_team, away_team]:
        if team not in standings[group]:
            standings[group][team] = {
                "played": 0, "won": 0, "drawn": 0, "lost": 0,
                "gf": 0, "ga": 0, "gd": 0, "pts": 0,
            }

    home_entry = standings[group][home_team]
    away_entry = standings[group][away_team]

    home_entry["played"] += 1
    away_entry["played"] += 1
    home_entry["gf"] += home_score
    away_entry["gf"] += away_score
    home_entry["ga"] += away_score
    away_entry["ga"] += home_score
    home_entry["gd"] = home_entry["gf"] - home_entry["ga"]
    away_entry["gd"] = away_entry["gf"] - away_entry["ga"]

    if home_score > away_score:
        home_entry["won"] += 1
        home_entry["pts"] += 3
        away_entry["lost"] += 1
    elif home_score < away_score:
        away_entry["won"] += 1
        away_entry["pts"] += 3
        home_entry["lost"] += 1
    else:
        home_entry["drawn"] += 1
        home_entry["pts"] += 1
        away_entry["drawn"] += 1
        away_entry["pts"] += 1

    save_standings(standings)


def sort_group(group_dict: dict) -> list:
    """
    Sort a group table by FIFA tiebreaker rules.
    
    Sort by (descending priority):
      1. pts  (total points)
      2. gd   (goal difference)
      3. gf   (goals scored)
      4. team name alphabetically (final tiebreaker for display stability)
    
    Args:
        group_dict: Dict of {team_name: stats_dict} for one group.
    
    Returns:
        Sorted list of (team_name, stats_dict) tuples.
    """
    return sorted(
        group_dict.items(),
        key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], x[0]),
        reverse=True,
    )


def is_group_complete(
    group: str,
    fixtures_df: pd.DataFrame,
    completed_results: pd.DataFrame,
) -> bool:
    """
    Check if all group-stage matches for a group are done.
    
    For a 4-team group: 6 matches (each team plays the other 3 once).
    
    Args:
        group: Group letter.
        fixtures_df: Full fixtures DataFrame.
        completed_results: DataFrame of submitted results.
    
    Returns:
        True if all group-stage matches for this group are completed.
    """
    group_matches_total = len(
        fixtures_df[
            (fixtures_df["group"] == group) &
            (fixtures_df["stage"] == "Group Stage")
        ]
    )

    if completed_results.empty:
        return group_matches_total == 0

    completed_in_group = len(
        completed_results[completed_results["group"] == group]
    )

    return completed_in_group >= group_matches_total


def build_placeholder_map(
    fixtures_df: pd.DataFrame,
    completed_results: pd.DataFrame,
) -> dict:
    """
    Return a mapping of placeholder string → resolved team name (or None if unresolved).
    This map is consumed by app.py to decide which fixtures are predictable.
    
    Args:
        fixtures_df: Full fixtures DataFrame.
        completed_results: DataFrame of submitted results.
    
    Returns:
        Dict mapping placeholder strings to resolved team names or None.
    """
    placeholder_map = {}
    standings = load_standings()

    all_groups_complete = True

    for group in WC2026_GROUPS:
        if group not in standings:
            placeholder_map[f"Winner Group {group}"] = None
            placeholder_map[f"Runner-up Group {group}"] = None
            placeholder_map[f"3rd Place Group {group}"] = None
            all_groups_complete = False
            continue

        if is_group_complete(group, fixtures_df, completed_results):
            sorted_group = sort_group(standings[group])
            if len(sorted_group) >= 2:
                winner = sorted_group[0][0]
                runner_up = sorted_group[1][0]
                placeholder_map[f"Winner Group {group}"] = winner
                placeholder_map[f"Runner-up Group {group}"] = runner_up
            else:
                placeholder_map[f"Winner Group {group}"] = None
                placeholder_map[f"Runner-up Group {group}"] = None

            # 3rd place: defer to separate resolution after all groups complete
            if len(sorted_group) >= 3:
                placeholder_map[f"3rd Place Group {group}"] = sorted_group[2][0]
            else:
                placeholder_map[f"3rd Place Group {group}"] = None
        else:
            placeholder_map[f"Winner Group {group}"] = None
            placeholder_map[f"Runner-up Group {group}"] = None
            placeholder_map[f"3rd Place Group {group}"] = None
            all_groups_complete = False

    # Best 3rd-place logic (WC 2026 specific):
    # Only resolve after ALL 12 groups are complete.
    # Collect all 12 third-place teams and rank them by pts → gd → gf.
    # Top 8 advance to Round of 32.
    if all_groups_complete and len(standings) == 12:
        third_place_teams = []
        for group in WC2026_GROUPS:
            sorted_group = sort_group(standings[group])
            if len(sorted_group) >= 3:
                team_name = sorted_group[2][0]
                team_stats = sorted_group[2][1]
                third_place_teams.append((team_name, team_stats, group))

        # Rank third-place teams by pts → gd → gf
        ranked_thirds = sorted(
            third_place_teams,
            key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
            reverse=True,
        )
        advancing_thirds = ranked_thirds[:8]

        # TODO: Map advancing thirds to their Round of 32 slots per official
        # FIFA 2026 bracket chart. The exact pairing depends on which specific
        # groups the advancing 3rd-place teams come from.
        # For now, mark them as resolved in the placeholder map.
        # The fixture CSV may already contain these matchups as specific
        # placeholder strings — use those directly if available.
        #
        # FIFA 2026 BRACKET PAIRING TABLE:
        # (To be populated once official bracket structure is confirmed)
        # THIRD_PLACE_BRACKET = {
        #     frozenset(["A","B","C","D"]): {"slot_1": "A", "slot_2": "B", ...},
        #     ...
        # }

        for team_name, _, group in advancing_thirds:
            placeholder_map[f"3rd Place Group {group}"] = team_name

    return placeholder_map


def resolve_fixtures(
    fixtures_df: pd.DataFrame,
    placeholder_map: dict,
) -> pd.DataFrame:
    """
    Return a copy of fixtures_df where resolved placeholders are replaced
    with actual team names. Unresolved placeholders remain as strings.
    
    Args:
        fixtures_df: Full fixtures DataFrame.
        placeholder_map: Dict of placeholder → resolved team name or None.
    
    Returns:
        Copy of fixtures_df with resolved placeholders replaced.
    """
    resolved_df = fixtures_df.copy()

    for placeholder, team in placeholder_map.items():
        if team is not None:
            resolved_df["home_team"] = resolved_df["home_team"].replace(
                placeholder, team
            )
            resolved_df["away_team"] = resolved_df["away_team"].replace(
                placeholder, team
            )

    return resolved_df


def is_match_predictable(
    home_team: str,
    away_team: str,
    placeholder_map: dict,
) -> bool:
    """
    Used by app.py to determine if a fixture is selectable for prediction.
    A match is predictable if neither team name is a remaining placeholder.
    
    Args:
        home_team: Team name string.
        away_team: Team name string.
        placeholder_map: Dict of placeholder → resolved team name or None.
    
    Returns:
        True if the match can be predicted (both teams are real/resolved).
    """
    unresolved_placeholders = {k for k, v in placeholder_map.items() if v is None}

    if home_team in unresolved_placeholders or away_team in unresolved_placeholders:
        return False

    placeholder_keywords = ["Winner", "Runner-up", "TBD", "3rd", "Loser"]
    for kw in placeholder_keywords:
        if kw in str(home_team) or kw in str(away_team):
            return False

    return True


def rebuild_all_standings(fixtures_df: pd.DataFrame, completed_results_df: pd.DataFrame) -> None:
    """
    Total standings rebuild. Wipes group_standings.json to all-zero stats,
    then replays every row in completed_results_df to regenerate correct standings.
    Called after ANY edit or delete operation so ghost points can never accumulate.
    """
    # STEP 1 — RE-INITIALIZE STANDINGS TO ZERO:
    standings = {}

    for group in WC2026_GROUPS:
        standings[group] = {}
        for team in WC2026_GROUP_TEAMS.get(group, []):
            standings[group][team] = {
                "played": 0,
                "won":    0,
                "drawn":  0,
                "lost":   0,
                "gf":     0,
                "ga":     0,
                "gd":     0,
                "pts":    0
            }

    # STEP 2 — REPLAY ALL COMPLETED RESULTS:
    if not completed_results_df.empty:
        group_results = completed_results_df[completed_results_df["stage"] == "Group Stage"]

        for row in group_results.itertuples():
            home = normalize_team_name(row.home_team)
            away = normalize_team_name(row.away_team)
            hs   = int(row.home_score)
            as_  = int(row.away_score)
            grp  = str(row.group)

            if grp not in standings: continue
            if home not in standings[grp]: continue
            if away not in standings[grp]: continue

            standings[grp][home]["played"] += 1
            standings[grp][away]["played"] += 1
            standings[grp][home]["gf"]     += hs
            standings[grp][away]["gf"]     += as_
            standings[grp][home]["ga"]     += as_
            standings[grp][away]["ga"]     += hs
            standings[grp][home]["gd"]      = standings[grp][home]["gf"] - standings[grp][home]["ga"]
            standings[grp][away]["gd"]      = standings[grp][away]["gf"] - standings[grp][away]["ga"]

            if hs > as_:
                standings[grp][home]["won"]  += 1
                standings[grp][home]["pts"]  += 3
                standings[grp][away]["lost"] += 1
            elif hs < as_:
                standings[grp][away]["won"]  += 1
                standings[grp][away]["pts"]  += 3
                standings[grp][home]["lost"] += 1
            else:
                standings[grp][home]["drawn"] += 1
                standings[grp][home]["pts"]   += 1
                standings[grp][away]["drawn"] += 1
                standings[grp][away]["pts"]   += 1

    # SAVE standings to STANDINGS_PATH as JSON
    with open(STANDINGS_PATH, "w") as f:
        json.dump(standings, f, indent=4)
    print(f"Standings rebuilt from {len(group_results) if not completed_results_df.empty else 0} group-stage result(s).")

