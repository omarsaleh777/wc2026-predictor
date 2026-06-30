"""
bracket_component.py — 32-Team Knockout Bracket Dashboard
Renders a World Cup 2026 elimination bracket inside Streamlit
using st.components.v1.html with a fully self-contained HTML/CSS/JS payload.
"""

import json
import math
import streamlit as st
import streamlit.components.v1 as components
from src.predict import predict_match

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIFA 2026 SEEDING & PROGRESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEEDING_MAP = [
    # Left side (matches 1-8)
    ("1A", "3C/D/E"), ("2B", "2E"), ("1G", "3H/I/J"), ("2H", "2K"),
    ("1E", "3A/B/C/D/F"), ("2F", "2I"), ("1K", "3D/E/I/J/L"), ("2L", "2A"),
    # Right side (matches 9-16)
    ("1B", "3E/F/G"), ("2C", "2F"), ("1H", "3J/K/L"), ("2I", "2L"),
    ("1D", "3B/E/F/G/I"), ("2G", "2J"), ("1J", "3A/B/C/D/F/H"), ("2D", "2G")
]

def build_initial_bracket():
    """Builds a completely blank placeholder bracket (fallback)."""
    data = {"r32": [], "r16": [], "qf": [], "sf": [], "final": []}
    
    for rnd, count in [("r16", 8), ("qf", 4), ("sf", 2), ("final", 1)]:
        for i in range(count):
            data[rnd].append({
                "match_id": f"{rnd}_{i+1}", "status": "tbd", "winner": None,
                "home": {"name": "TBD", "code": "xx"}, "away": {"name": "TBD", "code": "xx"}
            })

    for i in range(16):
        home_seed, away_seed = SEEDING_MAP[i]
        data["r32"].append({
            "match_id": f"r32_{i+1}",
            "home": {"name": home_seed, "code": "xx"},
            "away": {"name": away_seed, "code": "xx"},
            "status": "upcoming"  
        })
        
    return data

def get_next_round(current_round):
    round_order = ["r32", "r16", "qf", "sf", "final"]
    if current_round not in round_order:
        return None
    idx = round_order.index(current_round)
    if idx + 1 < len(round_order):
        return round_order[idx + 1]
    return None

def process_match_result(bracket_data, match_id, home_goals, away_goals, is_penalty, home_pen, away_pen, historical_df=None, models=None):
    # 1. Parse current match ID (e.g. "r32_1" -> "r32", 1)
    parts = match_id.split("_")
    current_round = parts[0]
    match_num = int(parts[1])
    
    # Locate current match in data
    target_match = None
    for m in bracket_data.get(current_round, []):
        if m["match_id"] == match_id:
            target_match = m
            break
            
    if not target_match:
        return bracket_data
        
    # 2. Set Status and Determine Winner
    target_match["actual_home_goals"] = home_goals
    target_match["actual_away_goals"] = away_goals
    target_match["is_penalty"] = is_penalty
    if is_penalty:
        target_match["home_penalties"] = home_pen
        target_match["away_penalties"] = away_pen
    target_match["status"] = "completed"
    
    winner = None
    if home_goals > away_goals:
        winner = "home"
    elif away_goals > home_goals:
        winner = "away"
    elif is_penalty:
        if home_pen > away_pen:
            winner = "home"
        elif away_pen > home_pen:
            winner = "away"
            
    target_match["winner"] = winner
    
    if not winner:
        return bracket_data
        
    winning_team = target_match["home"] if winner == "home" else target_match["away"]
    
    # 3. Math Cascading for Progression
    next_round = get_next_round(current_round)
    if not next_round:
        return bracket_data 
        
    next_match_num = math.ceil(match_num / 2)
    next_match_id = f"{next_round}_{next_match_num}"
    slot = "home" if match_num % 2 != 0 else "away"
    
    # 4. Inject Winner into next match
    next_match = None
    for m in bracket_data.get(next_round, []):
        if m["match_id"] == next_match_id:
            next_match = m
            break
            
    if next_match:
        next_match[slot] = {"name": winning_team["name"], "code": winning_team.get("code", "xx")}
        
        # 5. Trigger ML Model when matchup is locked
        if next_match["home"]["name"] != "TBD" and next_match["away"]["name"] != "TBD":
            next_match["status"] = "upcoming"
            if historical_df is not None:
                pred = predict_match(next_match["home"]["name"], next_match["away"]["name"], historical_df, models)
                
                w_prob = pred.get("home_win_pct", 50.0)
                l_prob = pred.get("away_win_pct", 50.0)
                total = w_prob + l_prob
                
                if total > 0:
                    next_match["win_prob"] = float((w_prob / total) * 100.0)
                    next_match["loss_prob"] = float((l_prob / total) * 100.0)
                else:
                    next_match["win_prob"] = 50.0
                    next_match["loss_prob"] = 50.0
                next_match["reasoning"] = pred.get("reasoning", "Awaiting AI Analysis")
            else:
                next_match["win_prob"] = 50.0
                next_match["loss_prob"] = 50.0
                next_match["reasoning"] = "Awaiting match resolution"
            next_match["draw_prob"] = 0.0

    return bracket_data

def render_data_entry_ui(bracket_data, historical_df=None, models=None):
    """Renders the Streamlit Data Entry UI with Live ML Prediction pre-filling."""
    with st.expander("📝 Log Match Result", expanded=True):
        upcoming_matches = []
        for rnd, matches in bracket_data.items():
            for m in matches:
                if m.get("status") == "upcoming":
                    upcoming_matches.append((rnd, m))
                    
        if not upcoming_matches:
            st.info("No upcoming matches available.")
            return

        match_options = {
            f"{m['match_id']}": f"{m['home']['name']} vs {m['away']['name']} ({rnd.upper()})"
            for rnd, m in upcoming_matches
        }
        
        selected_match_id = st.selectbox("Select Match:", options=list(match_options.keys()), format_func=lambda x: match_options[x])
        selected_rnd, selected_match = next((r, m) for r, m in upcoming_matches if m["match_id"] == selected_match_id)
        
        home_name = selected_match['home']['name']
        away_name = selected_match['away']['name']
        
        pred_home_goals = 0
        pred_away_goals = 0
        
        if historical_df is not None and home_name != "TBD" and away_name != "TBD":
            pred = predict_match(home_name, away_name, historical_df, models)
            w_prob = pred.get("home_win_pct", 50.0)
            l_prob = pred.get("away_win_pct", 50.0)
            total = w_prob + l_prob
            if total > 0:
                win_prob = float((w_prob / total) * 100.0)
                loss_prob = float((l_prob / total) * 100.0)
            else:
                win_prob = 50.0
                loss_prob = 50.0
                
            prob_str = f"{win_prob:.1f}% chance for {home_name} to advance" if win_prob >= 50 else f"{loss_prob:.1f}% chance for {away_name} to advance"
            st.info(f"🤖 **Antigravity ML Prediction:** {prob_str}")
            
            # Simple heuristic to pre-fill goals
            if win_prob > 60:
                pred_home_goals = 2
                pred_away_goals = 0
            elif loss_prob > 60:
                pred_home_goals = 0
                pred_away_goals = 2
            elif win_prob > 50:
                pred_home_goals = 2
                pred_away_goals = 1
            else:
                pred_home_goals = 1
                pred_away_goals = 2
        else:
            st.info("🤖 **Antigravity ML Prediction:** Awaiting match resolution")
        
        col1, col2 = st.columns(2)
        with col1:
            actual_home = st.number_input(f"{home_name} Goals", min_value=0, value=pred_home_goals, step=1)
        with col2:
            actual_away = st.number_input(f"{away_name} Goals", min_value=0, value=pred_away_goals, step=1)
            
        went_to_pens = st.checkbox("Went to Penalties?")
        pens_home, pens_away = 0, 0
        if went_to_pens:
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                pens_home = st.number_input(f"{home_name} Penalties", min_value=0, value=0, step=1)
            with pcol2:
                pens_away = st.number_input(f"{away_name} Penalties", min_value=0, value=0, step=1)
        
        if st.button("Submit Result", type="primary"):
            updated_data = process_match_result(bracket_data, selected_match_id, actual_home, actual_away, went_to_pens, pens_home, pens_away, historical_df, models)
            
            st.session_state.bracket_data = updated_data
            with open("bracket_state.json", "w") as f:
                json.dump(updated_data, f)
            st.rerun()

def render_bracket(bracket_data: dict):
    """
    Render an interactive 32-team knockout bracket as a Streamlit HTML component.
    """
    bracket_json = json.dumps(bracket_data)
    html_string = _BRACKET_TEMPLATE.replace("'%%BRACKET_JSON%%'", bracket_json)
    components.html(html_string, height=900, scrolling=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-contained HTML template — CSS + JS + DOM-relative SVG connector logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BRACKET_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>

/* ─────────────────────── Reset ─────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

html,body{
  background:#0a0a0f;
  font-family:'Rajdhani',sans-serif;
  color:rgba(255,255,255,0.92);
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
  overflow:hidden;
  width:100%;
  height:100%;
}

/* ─────────────────── Bracket Layout ─────────────────── */
#bracket-root{
  overflow:hidden;
  width:100%;
  height:900px;
  cursor:grab;
  position:relative;
  padding:0;
}

.bracket-inner{
  display:flex;
  flex-direction:row;
  align-items:stretch;
  justify-content:center;
  position:absolute;
  height:900px;
  transform-origin:0 0;
  will-change:transform;
}

.bracket-half {
  display:flex;
  flex-direction:row;
  flex:1;
  justify-content:flex-end;
}

.bracket-half.right-half {
  justify-content:flex-start;
}

.bracket-center {
  display:flex;
  flex-direction:column;
  justify-content:space-around;
  margin:0 20px;
}

/* Make SVG absolute to the scaled .bracket-inner so lines match perfectly */
.bracket-svg{
  position:absolute;
  top:0; left:0;
  width:100%; height:100%;
  pointer-events:none;
  z-index:0;
}

.round-col{
  display:flex;
  flex-direction:column;
  height:100%;
  padding:0 20px;
  z-index:10;
  position:relative;
}

.round-header{
  text-align:center;
  margin-bottom:15px;
  margin-top:15px;
  font-weight:700;
  font-size:16px;
  letter-spacing:1.5px;
  color:rgba(255,255,255,0.4);
  text-transform:uppercase;
  border-bottom:1px solid rgba(255,255,255,0.1);
  padding-bottom:5px;
  width:220px;
}

.matches-wrap{
  flex:1;
  display:flex;
  flex-direction:column;
  justify-content:space-around;
  min-height:0;
}

/* ─────────────────── Match Card ─────────────────── */
.match-card{
  width:220px;
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:8px;
  backdrop-filter:blur(8px);
  position:relative;
  transition:all 0.25s cubic-bezier(0.2,0.8,0.2,1);
  box-shadow:0 4px 15px rgba(0,0,0,0.2);
  animation:cardSlideIn 0.5s ease-out backwards;
}

@keyframes cardSlideIn{
  from{opacity:0;transform:translateX(-10px)}
  to{opacity:1;transform:translateX(0)}
}

.match-card:hover{
  transform:translateY(-2px) scale(1.04);
  box-shadow:0 8px 30px rgba(255,75,75,0.18);
  z-index:20;
}

.match-card.has-winner{
  border-color:rgba(0,180,255,0.22);
  box-shadow:0 0 14px rgba(0,180,255,0.1),
             inset 0 0 8px rgba(0,180,255,0.03);
}

/* ─────────────── Visual States ─────────────── */
.match-card.tbd{
  opacity:0.3;
  border-style:dashed;
}
.match-card.tbd .team-name{
  color:rgba(255,255,255,0.4);
  font-style:italic;
}
.match-card.tbd .prob-badge,
.match-card.tbd .score,
.match-card.tbd .flag{
  display:none;
}
.penalty-score{
  font-size:10px;
  color:#ffc107;
  margin-left:4px;
  font-weight:700;
}

/* ─────────────────── Team Row ─────────────────── */
.team-row{
  display:flex;
  align-items:center;
  padding:5px 8px;
  gap:5px;
  transition:opacity 0.3s ease;
}

.team-row:first-child{
  border-bottom:1px solid rgba(255,255,255,0.05);
  border-radius:8px 8px 0 0;
}

.team-row.winner{
  background:rgba(0,180,255,0.15);
  font-weight:700;
  text-shadow:0 0 8px rgba(0,180,255,0.4);
}
.team-row.winner:first-child { border-radius: 8px 8px 0 0; }
.team-row.winner:last-child { border-radius: 0 0 8px 8px; }

.team-row.eliminated{
  opacity:0.4;
  filter:grayscale(100%);
}

.flag{
  width:20px; height:15px;
  border-radius:2px;
  object-fit:cover;
  background:#222;
}

.team-name{
  flex:1;
  font-size:15px;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
  letter-spacing:0.3px;
}

.score{
  font-weight:700;
  font-size:16px;
  color:rgba(255,255,255,0.9);
  min-width:18px;
  text-align:right;
}

/* ─────────────────── Badges ─────────────────── */
.prob-badge{
  font-size:11px;
  font-weight:700;
  padding:2px 5px;
  border-radius:4px;
  margin-left:5px;
}

.prob-badge.fav{
  background:rgba(255,75,75,0.15);
  color:#ff4b4b;
  border:1px solid rgba(255,75,75,0.3);
}
.prob-badge.und{
  background:rgba(255,255,255,0.05);
  color:rgba(255,255,255,0.5);
}

/* ─────────────────── Tooltip ─────────────────── */
.tooltip-box{
  position:absolute;
  top:-10px; left:105%;
  width:160px;
  background:rgba(15,15,22,0.95);
  border:1px solid rgba(255,255,255,0.15);
  padding:10px;
  border-radius:6px;
  opacity:0;
  visibility:hidden;
  transform:translateX(-10px);
  transition:all 0.2s;
  pointer-events:none;
  z-index:999;
  box-shadow:0 10px 25px rgba(0,0,0,0.5);
  backdrop-filter:blur(5px);
}

.match-card:hover .tooltip-box{
  opacity:1;
  visibility:visible;
  transform:translateX(0);
}

.tip-row{
  display:flex;
  justify-content:space-between;
  font-size:12px;
  margin-bottom:4px;
}
.tip-row:last-child{ margin-bottom:0; }
.tip-label{ color:rgba(255,255,255,0.6); text-transform:uppercase; font-weight:600; font-size:10px; letter-spacing:0.5px;}
.tip-val{ font-weight:700; color:#fff; }

</style>
</head>
<body>

<div id="bracket-root">
  <div class="bracket-inner" id="bracket-inner">
    
    <!-- SVG injected INSIDE bracket-inner so it scales natively with the DOM -->
    <svg class="bracket-svg" id="svg-canvas"></svg>

    <!-- LEFT HALF -->
    <div class="bracket-half left-half">
      <div class="round-col" id="col-r32-left">
        <div class="round-header">ROUND OF 32</div>
        <div class="matches-wrap" id="r32-left-wrap"></div>
      </div>
      <div class="round-col" id="col-r16-left">
        <div class="round-header">ROUND OF 16</div>
        <div class="matches-wrap" id="r16-left-wrap"></div>
      </div>
      <div class="round-col" id="col-qf-left">
        <div class="round-header">QUARTER-FINALS</div>
        <div class="matches-wrap" id="qf-left-wrap"></div>
      </div>
      <div class="round-col" id="col-sf-left">
        <div class="round-header">SEMI-FINALS</div>
        <div class="matches-wrap" id="sf-left-wrap"></div>
      </div>
    </div>

    <!-- CENTER (FINAL) -->
    <div class="bracket-center">
      <div class="round-col" id="col-final">
        <div class="round-header" style="color:#ff4b4b;border-bottom-color:#ff4b4b;">FINAL</div>
        <div class="matches-wrap" id="final-wrap"></div>
      </div>
    </div>

    <!-- RIGHT HALF -->
    <div class="bracket-half right-half">
      <div class="round-col" id="col-sf-right">
        <div class="round-header">SEMI-FINALS</div>
        <div class="matches-wrap" id="sf-right-wrap"></div>
      </div>
      <div class="round-col" id="col-qf-right">
        <div class="round-header">QUARTER-FINALS</div>
        <div class="matches-wrap" id="qf-right-wrap"></div>
      </div>
      <div class="round-col" id="col-r16-right">
        <div class="round-header">ROUND OF 16</div>
        <div class="matches-wrap" id="r16-right-wrap"></div>
      </div>
      <div class="round-col" id="col-r32-right">
        <div class="round-header">ROUND OF 32</div>
        <div class="matches-wrap" id="r32-right-wrap"></div>
      </div>
    </div>

  </div>
</div>

<script>
/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Data Injection
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
var DATA = '%%BRACKET_JSON%%';

/* Complete flag code mapping for all 48 WC 2026 qualified nations */
var FLAG_CODES = {
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
};

/* ━━━━━━━━━━━━━━━━━ Flag Helper ━━━━━━━━━━━━━━━━━ */
function flagImg(team) {
  if (!team || !team.name || team.name === 'TBD' || team.name.includes('TBD')) return '';
  var code = team.code;
  if (!code || code === 'xx') {
    code = FLAG_CODES[team.name] || 'xx';
  }
  if (code === 'xx') return '';
  return '<img class="flag team-flag" src="https://flagcdn.com/w40/' + code + '.png" alt="' + code + '">';
}

function matchCard(m, delay) {
  var isTbd = (m.status === 'tbd');
  var isCompleted = (m.status === 'completed');
  var isUpcoming = (m.status === 'upcoming');
  
  var hName = m.home ? m.home.name : 'TBD';
  var aName = m.away ? m.away.name : 'TBD';
  
  var hw = (m.winner === 'home');
  var aw = (m.winner === 'away');
  var hasW = hw || aw;

  /* Render probabilities */
  var hProb = m.win_prob || 50.0;
  var aProb = m.loss_prob || 50.0;

  var hScoreStr = '';
  var aScoreStr = '';

  if (isCompleted) {
    hScoreStr = m.actual_home_goals !== undefined ? m.actual_home_goals : '-';
    aScoreStr = m.actual_away_goals !== undefined ? m.actual_away_goals : '-';
  }
  
  if (isCompleted && m.is_penalty) {
    if (hw) hScoreStr += '<span class="penalty-score">(' + (m.home_penalties||0) + ')</span>';
    if (aw) aScoreStr += '<span class="penalty-score">(' + (m.away_penalties||0) + ')</span>';
  }

  var s = '<div class="match-card' + (isTbd ? ' tbd' : '') + (hasW ? ' has-winner' : '') + '" style="animation-delay:' + delay + 's" data-id="' + (m.match_id || '') + '">';

  /* Home row */
  s += '<div class="team-row' + (isCompleted && hw ? ' winner' : (isCompleted && aw ? ' eliminated' : '')) + '">';
  s += flagImg(m.home);
  s += '<span class="team-name">' + hName + '</span>';
  s += '<span class="score">' + hScoreStr + '</span>';
  
  if (isCompleted && hw) {
      s += '<span class="prob-badge fav">' + hProb.toFixed(1) + '%</span>';
  } else if (isUpcoming) {
      s += '<span class="prob-badge ' + (hProb >= 50 ? 'fav' : 'und') + '">' + hProb.toFixed(1) + '%</span>';
  }
  s += '</div>';

  /* Away row */
  s += '<div class="team-row' + (isCompleted && aw ? ' winner' : (isCompleted && hw ? ' eliminated' : '')) + '">';
  s += flagImg(m.away);
  s += '<span class="team-name">' + aName + '</span>';
  s += '<span class="score">' + aScoreStr + '</span>';
  
  if (isCompleted && aw) {
      s += '<span class="prob-badge fav">' + aProb.toFixed(1) + '%</span>';
  } else if (isUpcoming) {
      s += '<span class="prob-badge ' + (aProb >= 50 ? 'fav' : 'und') + '">' + aProb.toFixed(1) + '%</span>';
  }
  s += '</div>';

  /* Tooltip */
  if (isUpcoming && m.reasoning) {
      s += '<div class="tooltip-box" style="width:220px; font-weight:normal; line-height:1.4;">';
      s += '<div style="color:#ff4b4b;font-weight:700;margin-bottom:6px;font-size:11px;">🤖 AI REASONING</div>';
      s += '<div style="font-size:11px;color:#ccc;">' + m.reasoning + '</div>';
      s += '</div>';
  } else {
      s += '<div class="tooltip-box">';
      s += '<div class="tip-row"><span class="tip-label">Favored Win Prob</span><span class="tip-val">' + Math.max(hProb, aProb).toFixed(1) + '%</span></div>';
      if (isCompleted) {
        s += '<div class="tip-row" style="margin-top:4px;color:#aaa;border-top:1px solid rgba(255,255,255,0.1);padding-top:4px;text-align:center;">Result Final</div>';
      }
      s += '</div>';
  }

  s += '</div>';
  return s;
}

/* ━━━━━━━━━━━━━━━━━ Pan & Zoom ━━━━━━━━━━━━━━━━━ */
var scale = 1, panning = false, pointX = 0, pointY = 0, startX = 0, startY = 0;
var bracketRoot = document.getElementById("bracket-root");
var bracketInner = document.getElementById("bracket-inner");

function setTransform() {
  bracketInner.style.transform = "translate(" + pointX + "px, " + pointY + "px) scale(" + scale + ")";
  // No need to redraw connections during pan/zoom because SVG is now scaled natively inside .bracket-inner
}

bracketRoot.onmousedown = function (e) {
  e.preventDefault();
  startX = e.clientX - pointX;
  startY = e.clientY - pointY;
  panning = true;
  bracketRoot.style.cursor = "grabbing";
};

bracketRoot.onmouseup = function (e) {
  panning = false;
  bracketRoot.style.cursor = "grab";
};

bracketRoot.onmouseleave = function (e) {
  panning = false;
  bracketRoot.style.cursor = "grab";
};

bracketRoot.onmousemove = function (e) {
  e.preventDefault();
  if (!panning) return;
  pointX = (e.clientX - startX);
  pointY = (e.clientY - startY);
  setTransform();
};

bracketRoot.onwheel = function (e) {
  e.preventDefault();
  var xs = (e.clientX - pointX) / scale;
  var ys = (e.clientY - pointY) / scale;
  var delta = (e.wheelDelta ? e.wheelDelta : -e.deltaY);
  
  if (delta > 0) scale *= 1.1;
  else scale /= 1.1;

  if (scale < 0.3) scale = 0.3;
  if (scale > 2) scale = 2;

  pointX = e.clientX - xs * scale;
  pointY = e.clientY - ys * scale;
  setTransform();
};

/* ━━━━━━━━━━━━━━━━━ Rendering ━━━━━━━━━━━━━━━━━ */
function initBracket() {
  var d = DATA;
  if (!d || !d.r32) return;
  
  // Left Side (Indices 0-7)
  document.getElementById('r32-left-wrap').innerHTML = renderHalfRound(d.r32.slice(0,8), 0.1);
  document.getElementById('r16-left-wrap').innerHTML = renderHalfRound(d.r16.slice(0,4), 0.2);
  document.getElementById('qf-left-wrap').innerHTML  = renderHalfRound(d.qf.slice(0,2),  0.3);
  document.getElementById('sf-left-wrap').innerHTML  = renderHalfRound(d.sf.slice(0,1),  0.4);

  // Right Side (Indices 8-15)
  document.getElementById('r32-right-wrap').innerHTML = renderHalfRound(d.r32.slice(8,16), 0.1);
  document.getElementById('r16-right-wrap').innerHTML = renderHalfRound(d.r16.slice(4,8),  0.2);
  document.getElementById('qf-right-wrap').innerHTML  = renderHalfRound(d.qf.slice(2,4),   0.3);
  document.getElementById('sf-right-wrap').innerHTML  = renderHalfRound(d.sf.slice(1,2),   0.4);

  // Center (Final)
  var fHtml = '';
  if (d.final && d.final.length > 0) {
    fHtml = matchCard(d.final[0], 0.6);
  }
  document.getElementById('final-wrap').innerHTML = fHtml;

  // Initial centering
  setTimeout(function() {
    var bw = bracketInner.scrollWidth;
    var bh = bracketInner.scrollHeight;
    var vw = bracketRoot.clientWidth;
    var vh = bracketRoot.clientHeight;
    
    scale = Math.min((vw - 40) / bw, (vh - 40) / bh);
    if(scale > 1) scale = 1;
    
    pointX = (vw - (bw * scale)) / 2;
    pointY = (vh - (bh * scale)) / 2;
    setTransform();
    
    drawConnections();
  }, 100);
}

function renderHalfRound(matches, delayBase) {
  var html = '';
  for (var i=0; i<matches.length; i++) {
    html += matchCard(matches[i], delayBase + (i*0.05));
  }
  return html;
}

/* ━━━━━━━━━━━━━━━━━ SVG Connectors ━━━━━━━━━━━━━━━━━ */
function getRelativePos(el) {
  var x = 0, y = 0, w = el.offsetWidth, h = el.offsetHeight;
  var inner = document.getElementById('bracket-inner');
  while (el && el !== inner) {
    x += el.offsetLeft;
    y += el.offsetTop;
    el = el.offsetParent;
  }
  return { left: x, top: y, width: w, height: h, right: x + w, bottom: y + h };
}

function drawConnections() {
  var svg = document.getElementById('svg-canvas');
  svg.innerHTML = '';
  
  var d = DATA;
  if (!d || !d.r32) return;
  
  connectHalf(d.r32.slice(0,8), d.r16.slice(0,4), svg, 'right');
  connectHalf(d.r16.slice(0,4), d.qf.slice(0,2),  svg, 'right');
  connectHalf(d.qf.slice(0,2),  d.sf.slice(0,1),  svg, 'right');
  if(d.final && d.final.length>0) {
    connectSingle(d.sf[0], d.final[0], svg, 'right');
  }

  connectHalf(d.r32.slice(8,16), d.r16.slice(4,8), svg, 'left');
  connectHalf(d.r16.slice(4,8),  d.qf.slice(2,4),  svg, 'left');
  connectHalf(d.qf.slice(2,4),   d.sf.slice(1,2),  svg, 'left');
  if(d.final && d.final.length>0 && d.sf.length>1) {
    connectSingle(d.sf[1], d.final[0], svg, 'left');
  }
}

function connectHalf(sourceRound, targetRound, svg, direction) {
  if(!sourceRound || !targetRound) return;
  for (var i=0; i<targetRound.length; i++) {
    var tMatch = targetRound[i];
    var s1 = sourceRound[i*2];
    var s2 = sourceRound[i*2 + 1];
    if(s1 && tMatch) drawLine(s1.match_id, tMatch.match_id, svg, direction, 'top');
    if(s2 && tMatch) drawLine(s2.match_id, tMatch.match_id, svg, direction, 'bottom');
  }
}

function connectSingle(sourceMatch, targetMatch, svg, direction) {
  if(!sourceMatch || !targetMatch) return;
  var targetSlot = direction === 'right' ? 'top' : 'bottom';
  drawLine(sourceMatch.match_id, targetMatch.match_id, svg, direction, targetSlot);
}

function drawLine(id1, id2, svg, direction, targetSlot) {
  var el1 = document.querySelector('[data-id="'+id1+'"]');
  var el2 = document.querySelector('[data-id="'+id2+'"]');
  if(!el1 || !el2) return;

  var r1 = getRelativePos(el1);
  var r2 = getRelativePos(el2);

  var x1, y1, x2, y2;
  
  y1 = r1.top + (r1.height / 2);
  
  if (targetSlot === 'top') {
    y2 = r2.top + (r2.height / 4);
  } else {
    y2 = r2.top + (r2.height * 3 / 4);
  }

  if (direction === 'right') {
    x1 = r1.right;
    x2 = r2.left;
  } else {
    x1 = r1.left;
    x2 = r2.right;
  }

  var midX = (x1 + x2) / 2;
  var color = 'rgba(255,255,255,0.15)';
  
  if (el1.classList.contains('has-winner')) {
    color = 'rgba(0,180,255,0.3)';
  }

  var path = '<path d="M '+x1+' '+y1+' C '+midX+' '+y1+', '+midX+' '+y2+', '+x2+' '+y2+'" fill="none" stroke="'+color+'" stroke-width="2" />';
  svg.innerHTML += path;
}

window.addEventListener('resize', drawConnections);
initBracket();
</script>
</body>
</html>
"""
