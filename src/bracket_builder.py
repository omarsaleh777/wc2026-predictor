"""
bracket_builder.py — ML-Powered Knockout Stage Generator

This module bridges the World Cup 2026 group stage standings with the XGBoost 
prediction engine. It dynamically parses the 32 qualified teams, maps the 8 best 
3rd-place teams into the 48-team bracket using a constraint satisfaction algorithm, 
and invokes the ML model to generate the exact UI-ready `bracket_data` payload.
"""

from src.predict import predict_match

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OFFICIAL 48-TEAM SEEDING CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Base Matchups: Mapping match_id -> (Home Seed, Away Seed)
# Format: "1A" means Winner Group A, "2B" means Runner-up Group B, "3C/D/E" is a 3rd-place pool

FLAG_CODES = {
  "Argentina":"ar","Australia":"au","Austria":"at","Albania":"al",
  "Bahrain":"bh","Belgium":"be","Bolivia":"bo","Brazil":"br",
  "Cameroon":"cm","Canada":"ca","Chile":"cl","China PR":"cn",
  "Colombia":"co","Costa Rica":"cr","Croatia":"hr","Czech Republic":"cz",
  "Czechia":"cz","Denmark":"dk","DR Congo":"cd",
  "Ecuador":"ec","Egypt":"eg","England":"gb-eng",
  "Finland":"fi","France":"fr",
  "Georgia":"ge","Germany":"de","Ghana":"gh",
  "Honduras":"hn","Hungary":"hu",
  "Iceland":"is","Indonesia":"id","Iran":"ir","Iraq":"iq",
  "Israel":"il","Italy":"it","Ivory Coast":"ci",
  "Jamaica":"jm","Japan":"jp","Jordan":"jo",
  "Kenya":"ke",
  "Mali":"ml","Mexico":"mx","Montenegro":"me","Morocco":"ma",
  "Netherlands":"nl","New Zealand":"nz","Nigeria":"ng","North Macedonia":"mk","Norway":"no",
  "Oman":"om",
  "Panama":"pa","Paraguay":"py","Peru":"pe","Philippines":"ph","Poland":"pl","Portugal":"pt",
  "Qatar":"qa",
  "Republic of Ireland":"ie","Romania":"ro","Russia":"ru",
  "Saudi Arabia":"sa","Scotland":"gb-sct","Senegal":"sn","Serbia":"rs",
  "Slovakia":"sk","Slovenia":"si","South Africa":"za","South Korea":"kr",
  "Spain":"es","Sweden":"se","Switzerland":"ch",
  "Thailand":"th","Trinidad and Tobago":"tt","Tunisia":"tn","Turkey":"tr",
  "Uganda":"ug","Ukraine":"ua","United Arab Emirates":"ae",
  "United States":"us","Uruguay":"uy","Uzbekistan":"uz",
  "Venezuela":"ve",
  "Wales":"gb-wls"
}

BASE_MATCHUPS = {
    # Left Side
    "r32_1":  ("1A", "3C/D/E"),
    "r32_2":  ("2B", "2E"),
    "r32_3":  ("1G", "3H/I/J"),
    "r32_4":  ("2H", "2K"),
    "r32_5":  ("1E", "3A/B/C/D/F"),
    "r32_6":  ("2F", "2I"),
    "r32_7":  ("1K", "3D/E/I/J/L"),
    "r32_8":  ("2L", "2A"),
    # Right Side
    "r32_9":  ("1B", "3E/F/G"),
    "r32_10": ("2C", "2F"),
    "r32_11": ("1H", "3J/K/L"),
    "r32_12": ("2I", "2L"),
    "r32_13": ("1D", "3B/E/F/G/I"),
    "r32_14": ("2G", "2J"),
    "r32_15": ("1J", "3A/B/C/D/F/H"),
    "r32_16": ("2D", "2G")
}

# The 8 specific slots that require a 3rd-place team, mapped to allowed groups
THIRD_PLACE_SLOTS = [
    {"match_id": "r32_1",  "allowed": ["C", "D", "E"]},
    {"match_id": "r32_3",  "allowed": ["H", "I", "J"]},
    {"match_id": "r32_5",  "allowed": ["A", "B", "C", "D", "F"]},
    {"match_id": "r32_7",  "allowed": ["D", "E", "I", "J", "L"]},
    {"match_id": "r32_9",  "allowed": ["E", "F", "G"]},
    {"match_id": "r32_11", "allowed": ["J", "K", "L"]},
    {"match_id": "r32_13", "allowed": ["B", "E", "F", "G", "I"]},
    {"match_id": "r32_15", "allowed": ["A", "B", "C", "D", "F", "H"]},
]

def map_third_place_teams(third_place_teams):
    """
    Backtracking Constraint Solver.
    Maps 8 specific 3rd-place teams perfectly into the 8 available slots 
    without causing group conflicts, avoiding the need for a 495-permutation hardcoded table.
    """
    def solve(slot_idx, current_assignment, used_indices):
        if slot_idx == len(THIRD_PLACE_SLOTS):
            return current_assignment
            
        slot = THIRD_PLACE_SLOTS[slot_idx]
        for t_idx, team in enumerate(third_place_teams):
            if t_idx not in used_indices:
                if team["group"] in slot["allowed"]:
                    # Attempt this path
                    used_indices.add(t_idx)
                    current_assignment[slot["match_id"]] = team
                    
                    result = solve(slot_idx + 1, current_assignment, used_indices)
                    if result is not None:
                        return result
                        
                    # Backtrack
                    used_indices.remove(t_idx)
                    del current_assignment[slot["match_id"]]
                    
        return None
        
    return solve(0, {}, set())


def resolve_qualified_teams(group_standings_df):
    """
    Parses a single DataFrame containing all group standings.
    Expected columns include: 'Group', 'Team', 'Pts', 'GD', 'GF'
    Returns a list of resolved R32 matchups as dictionaries.
    """
    group_winners = {}
    runners_up = {}
    third_places = []
    
    for group_name, group_df in group_standings_df.groupby('Group'):
        sorted_df = group_df.sort_values(by=['Pts', 'GD', 'GF'], ascending=[False, False, False])
        teams = sorted_df.to_dict('records')
        
        if len(teams) >= 1:
            group_winners[group_name] = teams[0]
        if len(teams) >= 2:
            runners_up[group_name] = teams[1]
        if len(teams) >= 3:
            third_places.append(teams[2])
            
    best_thirds = sorted(third_places, key=lambda x: (x.get("Pts", 0), x.get("GD", 0), x.get("GF", 0)), reverse=True)[:8]
    
    for t in best_thirds:
        t["group"] = t["Group"]
        
    third_place_map = map_third_place_teams(best_thirds)
    if third_place_map is None:
        third_place_map = {slot["match_id"]: team for slot, team in zip(THIRD_PLACE_SLOTS, best_thirds)}
        
    resolved_matchups = []
    for i in range(16):
        m_id = f"r32_{i+1}"
        home_seed_code, away_seed_code = BASE_MATCHUPS[m_id]
        
        if home_seed_code.startswith("1"):
            grp = home_seed_code[1]
            home_team = group_winners.get(grp, {"Team": f"{home_seed_code} TBD"})
        elif home_seed_code.startswith("2"):
            grp = home_seed_code[1]
            home_team = runners_up.get(grp, {"Team": f"{home_seed_code} TBD"})
        else:
            home_team = third_place_map.get(m_id, {"Team": "3rd Place TBD"})
            
        if away_seed_code.startswith("1"):
            grp = away_seed_code[1]
            away_team = group_winners.get(grp, {"Team": f"{away_seed_code} TBD"})
        elif away_seed_code.startswith("2"):
            grp = away_seed_code[1]
            away_team = runners_up.get(grp, {"Team": f"{away_seed_code} TBD"})
        else:
            away_team = third_place_map.get(m_id, {"Team": "3rd Place TBD"})
            
        resolved_matchups.append({
            "match_id": m_id,
            "home": home_team["Team"],
            "away": away_team["Team"]
        })
        
    return resolved_matchups


def generate_bracket_data(resolved_matchups, historical_df, models=None):
    """
    Takes the FINAL list of resolved (and potentially manually edited) R32 matchups,
    calls ML predictions on them, and builds the UI-ready bracket_data dict.
    """
    def empty_matches(prefix, count):
        return [{
            "match_id": f"{prefix}_{i+1}", 
            "status": "tbd", 
            "winner": None,
            "home": {"name": "TBD", "code": "xx"}, 
            "away": {"name": "TBD", "code": "xx"},
            "actual_home_goals": "-",
            "actual_away_goals": "-"
        } for i in range(count)]
        
    bracket_data = {
        "r32": empty_matches("r32", 16),
        "r16": empty_matches("r16", 8),
        "qf":  empty_matches("qf", 4),
        "sf":  empty_matches("sf", 2),
        "final": empty_matches("final", 1)
    }
    
    # Fill R32 with Teams and ML Predictions
    for match, resolved in zip(bracket_data["r32"], resolved_matchups):
        h_name = resolved["home"]
        a_name = resolved["away"]
        
        match["home"] = {"name": h_name, "code": FLAG_CODES.get(h_name, "xx")}
        match["away"] = {"name": a_name, "code": FLAG_CODES.get(a_name, "xx")}
        match["status"] = "upcoming"
        
        # Invoke ML Model
        if "TBD" not in h_name and "TBD" not in a_name:
            pred = predict_match(h_name, a_name, historical_df, models)
            
            w_prob = pred.get("home_win_pct", 50.0)
            l_prob = pred.get("away_win_pct", 50.0)
            total = w_prob + l_prob
            if total > 0:
                match["win_prob"] = float((w_prob / total) * 100.0)
                match["loss_prob"] = float((l_prob / total) * 100.0)
            else:
                match["win_prob"] = 50.0
                match["loss_prob"] = 50.0
            match["draw_prob"] = 0.0
            match["reasoning"] = pred.get("reasoning", "Awaiting AI Analysis")
        else:
            match["win_prob"] = 50.0
            match["draw_prob"] = 0.0
            match["loss_prob"] = 50.0
            match["reasoning"] = "Awaiting match resolution"

    return bracket_data
