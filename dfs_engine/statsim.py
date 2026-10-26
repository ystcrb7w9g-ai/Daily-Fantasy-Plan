"""
Stat-level game simulation: each trial plays every game out as box scores
(dropbacks, sacks, targets, catches, yards, carries, TDs, turnovers, field
goals), then scores them the DraftKings way, bonuses included. Players move
together because they share one game, the way SaberSim's play-by-play sims
work, instead of through correlation settings:

- a QB's passing yards are exactly his receivers' receiving yards (plus
  receivers outside the pool), and every passing TD is one receiver's TD;
- targets and carries are split by each player's share of his team's
  volume, with week-to-week noise in the shares (Dirichlet), so teammates
  compete for the same looks;
- a shared game environment (pace, scoring) moves both teams; a team that
  trails passes more and runs less;
- kickers score the drives that stall (fewer when the team scores TDs),
  and a DST's points come from the opponent's sacks, interceptions,
  fumbles and points.

Base rates are 2021-25 regular-season averages from nflverse weekly stats
(33.2 pass attempts, 2.4 sacks, 26.9 carries per team-game; catch rates and
yards per catch by position; TD rates rising with the Vegas team total).
Usage comes from the pool: SaberSim's stat projections (catches, carries,
receiving / rushing TDs per game; `dk-pool` keeps them) when present, then
targets per game (`tgt_pg`) and the QB's rushing share (`rush_share`),
otherwise each player's projection split by position. The output is raw DK
points; `simulate.simulate_player_scores` then fits each player's mean and
spread to his `proj` / `ceiling` (or maps him onto provided percentiles),
so the stat sim decides who booms together and the inputs decide by how
much.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 2021-25 regular-season per team-game averages (nflverse stats_player_week).
DROPBACKS, CARRIES = 35.6, 26.9            # pass attempts + sacks; carries
SACK_RATE = 0.068                           # sacks per dropback
TARGETS_PER_ATT = 0.957
INT_RATE = 0.0229                           # per attempt
FUMBLES_LOST = 0.51                         # per team-game
CATCH = {"WR": 0.631, "TE": 0.708, "RB": 0.779, "QB": 0.6, "other": 0.66}
YDS_PER_CATCH = {"WR": 12.56, "TE": 10.3, "RB": 7.39, "QB": 8.0, "other": 11.0}
CATCH_CV = {"WR": 0.90, "TE": 0.82, "RB": 1.07, "QB": 1.0, "other": 0.9}
YDS_PER_CARRY = {"RB": 4.30, "QB": 4.39, "WR": 5.78, "TE": 3.0, "other": 4.3}
CARRY_CV = {"RB": 1.47, "QB": 1.30, "WR": 1.30, "TE": 1.0, "other": 1.4}
REC_TD_PER_TGT = {"WR": 0.0481, "TE": 0.0518, "RB": 0.0302, "QB": 0.03, "other": 0.045}
RUSH_TD_PER_CARRY = {"RB": 0.0308, "QB": 0.0458, "WR": 0.0443, "TE": 0.05, "other": 0.03}
DK_PER_TARGET = {"WR": 1.71, "TE": 1.75, "RB": 1.54}  # DK points per target incl. TDs
DK_PER_CARRY = {"RB": 0.616, "QB": 0.715}
PASS_TDS, RUSH_TDS, AVG_TOTAL = 1.46, 0.91, 22.5
PASS_TD_ELASTICITY, RUSH_TD_ELASTICITY = 1.4, 1.1   # TDs vs implied total (2021-25 bins)
FG_ATTEMPTS, FG_MAKE, XP_PER_TD = 1.97, 0.851, 0.902
FG_POINTS = (3, 4, 5)                       # DK: 0-39 / 40-49 / 50+ yards
FG_DIST = (0.56, 0.263, 0.177)
DEF_TD_PER_TURNOVER, ST_TDS, SAFETIES = 0.052, 0.034, 0.024

# Knobs fit to real 2021-25 dependence (scratch random search + local refinement, Oct 2026) with the
# same targets as the factor model's TD_MODEL fit: all same-team / opponent role-pair correlations,
# co-boom / co-bust lifts, QB boom -> receiver boom, team-total spread and the opponent link. Fit loss
# 0.97 (factor TD_MODEL 1.15); 3 seeds x 12k trials: same-team pair RMS 0.034 (factor 0.031), opponent
# 0.028 (0.036); opposing team totals 0.18 (factor 0.25, real 0.19); QB ~ opposing QB 0.16 (0.20,
# real 0.20); QB boom -> receiver boom 84% (83%, real 86%); team-total spread 0.31 (0.30, real 0.33).
# A stronger shared game shock (game_score_sd 0.235) matches QB ~ opposing QB at loss 1.03, but it
# ranked both 2026 Showdowns' real entries worse in `backtest` (AUC 0.647 / 0.553 vs 0.697 / 0.609).
STAT_KNOBS = {
    "game_score_sd": 0.203,    # shared scoring environment (both offenses)
    "team_eff_sd": 0.231,      # one offense's efficiency (yards, TDs)
    "pace_sd": 0.045,          # shared plays
    "script": 0.149,           # pass-share change per unit of efficiency deficit vs the opponent
    "dropback_noise": 0.067,   # pass-share noise per team-game
    "target_conc": 94.0,       # Dirichlet concentration of target shares, beyond binomial noise
    "carry_conc": 11.5,        # ... of carry shares
    "yards_load": 0.207,       # yards per catch / carry vs efficiency
    "pass_td_load": 1.891,     # passing TD rate vs efficiency
    "rush_td_load": 0.302,
    "script_rush_td": 0.303,   # leading teams punch in more rushing TDs
    "turnover_pts": 0.15,      # log scoring rate per extra turnover by the opponent (short fields)
    "fg_eff_load": 0.0,        # field goal attempts vs offensive efficiency
    "fg_td_tradeoff": 0.25,    # fewer field goals when the drives end in TDs
}
# Off by default: "rush_eff_corr" (1 = rushing shares the passing efficiency shock; below 1 part of it is
# its own, so a big passing day no longer lifts the run game with it). "td_mix_sd" (0) tilts each team-game's TD rates toward passing or rushing (pass x exp(s*u),
# rush x exp(-s*u)); "share_tilt" moves its pass share with the same u. Real games: a QB and his RB1 both
# finish in their top 10% at 0.74x the independent rate; the stat sim gives ~2x without this.


def _pos_key(p: str, table: dict) -> str:
    return p if p in table else "other"


def _usage(team_df: pd.DataFrame) -> dict:
    """Expected targets and carries (and own TD rates) per player of one team: SaberSim stat
    projections, else tgt_pg / rush_share, else each projection split by position."""
    pos = team_df["position"].to_numpy()
    proj = team_df["proj"].to_numpy(dtype=float).clip(min=0.0)
    tgt_col = pd.to_numeric(team_df["tgt_pg"], errors="coerce").to_numpy() if "tgt_pg" in team_df else np.full(len(pos), np.nan)
    rush_col = pd.to_numeric(team_df["rush_share"], errors="coerce").to_numpy() if "rush_share" in team_df else np.full(len(pos), np.nan)
    qbs = np.flatnonzero(pos == "QB")
    passer = qbs[np.argmax(proj[qbs])] if len(qbs) else None
    # QBs projected near the starter (a questionable starter, a QB battle) each start in a share of
    # the trials, in proportion to projection; deep backups never do.
    starters = [q for q in qbs if proj[q] >= 0.4 * proj[passer]] if passer is not None else []
    targets = np.zeros(len(pos))
    carries = np.zeros(len(pos))
    stat = lambda c: (pd.to_numeric(team_df[c], errors="coerce").to_numpy(dtype=float)  # noqa: E731
                      if c in team_df else np.full(len(pos), np.nan))
    p_rec, p_rush = stat("proj_rec"), stat("proj_rush_att")
    p_rec_td, p_rush_td = stat("proj_rec_td"), stat("proj_rush_td")
    for i, p in enumerate(pos):
        if p in ("WR", "TE"):
            targets[i] = tgt_col[i] if np.isfinite(tgt_col[i]) and tgt_col[i] > 0 else proj[i] * 0.97 / DK_PER_TARGET[p]
            carries[i] = proj[i] * 0.03 / 0.9 if p == "WR" else 0.0
        elif p == "RB":
            rec_pts = tgt_col[i] * DK_PER_TARGET["RB"] if np.isfinite(tgt_col[i]) and tgt_col[i] > 0 else 0.35 * proj[i]
            rec_pts = min(rec_pts, 0.8 * proj[i])
            targets[i] = rec_pts / DK_PER_TARGET["RB"]
            carries[i] = max(proj[i] - rec_pts, 0.0) / DK_PER_CARRY["RB"]
        elif p == "QB" and i == passer:
            rs = rush_col[i] if np.isfinite(rush_col[i]) else 0.15
            carries[i] = rs * proj[i] / DK_PER_CARRY["QB"]
    # Stat projections (SaberSim's per-game catches / carries / TDs) beat shares guessed from DK points.
    has = np.isfinite(p_rec) | np.isfinite(p_rush)
    rec_td_rate = np.full(len(pos), np.nan)
    rush_td_rate = np.full(len(pos), np.nan)
    for i, p in enumerate(pos):
        if not has[i] or p not in ("WR", "TE", "RB", "QB") or (p == "QB" and i != passer):
            continue
        if p != "QB":
            targets[i] = max(np.nan_to_num(p_rec[i]), 0.0) / CATCH[p]
            if np.isfinite(p_rec_td[i]) and targets[i] > 0:
                rec_td_rate[i] = p_rec_td[i] / targets[i]
        carries[i] = max(np.nan_to_num(p_rush[i]), 0.0)
        if np.isfinite(p_rush_td[i]) and carries[i] > 0:
            rush_td_rate[i] = p_rush_td[i] / carries[i]
    start_p = proj[starters] / proj[starters].sum() if starters else np.zeros(0)
    return {"targets": targets, "carries": carries, "passer": passer, "starters": starters, "start_p": start_p,
            "rec_td_rate": rec_td_rate, "rush_td_rate": rush_td_rate}


def _gamma_sum(rng, k, mean, cv, scale_mult):
    """Total yards over k catches/carries, each with this mean and CV (a gamma with matching moments)."""
    k = np.asarray(k, dtype=float)
    shape = np.maximum(k / cv ** 2, 1e-9)
    out = rng.gamma(shape, mean * cv ** 2 * scale_mult)
    return np.where(k > 0, out, 0.0)


def _dirichlet_multinomial(rng, n_total, expected, conc):
    """Per-trial counts: shares ~ Dirichlet(conc * expected share), counts ~ Multinomial(n_total)."""
    share = expected / expected.sum()
    alpha = np.maximum(conc * share, 1e-3)
    shares = rng.gamma(alpha, size=(len(n_total), len(alpha)))
    shares /= shares.sum(axis=1, keepdims=True)
    return rng.multinomial(n_total.astype(np.int64), shares)


def _dst_points(sacks, ints, fum, tds, safeties, allowed):
    tiers = np.select([allowed <= 0, allowed <= 6, allowed <= 13, allowed <= 20, allowed <= 27, allowed <= 34],
                      [10, 7, 4, 1, 0, -1], -4)
    return sacks + 2 * ints + 2 * fum + 6 * tds + 2 * safeties + tiers


def stat_game_scores(df: pd.DataFrame, n_trials: int, rng: np.random.Generator,
                     knobs: dict | None = None) -> np.ndarray:
    """Raw DK points (n_trials, len(df)) from simulated box scores."""
    k = {**STAT_KNOBS, **(knobs or {})}
    n = len(df)
    out = np.zeros((n_trials, n))
    pos_all = df["position"].to_numpy()
    teams = list(dict.fromkeys(df["team"]))
    opp_of = df.groupby("team")["opp"].first().to_dict()
    tt = pd.to_numeric(df.get("team_total", pd.Series(np.nan, index=df.index)), errors="coerce")
    implied = tt.groupby(df["team"]).first().to_dict()
    # every team in a game (opponents outside the pool still get simulated, as generic offenses)
    all_teams = list(dict.fromkeys(teams + [o for o in opp_of.values() if isinstance(o, str)]))
    imp = {t: float(implied.get(t)) if pd.notna(implied.get(t, np.nan)) else AVG_TOTAL for t in all_teams}
    for t in all_teams:  # an opponent without its own rows: its total = game total - ours
        if t not in implied and any(opp_of.get(x) == t for x in teams):
            mate = next(x for x in teams if opp_of.get(x) == t)
            gt = pd.to_numeric(df.loc[df["team"] == mate, "game_total"], errors="coerce").iloc[0] if "game_total" in df else np.nan
            imp[t] = float(gt - imp[mate]) if pd.notna(gt) else AVG_TOTAL
    games, game_of = [], {}
    for t in all_teams:
        o = opp_of.get(t)
        key = tuple(sorted([t, o])) if isinstance(o, str) else (t,)
        if key not in game_of.values():
            games.append(key)
        game_of[t] = key
    g_index = {g: i for i, g in enumerate(dict.fromkeys(game_of.values()))}
    n_games = len(g_index)
    g_score = rng.normal(0.0, k["game_score_sd"], (n_trials, n_games))
    g_pace = rng.normal(0.0, k["pace_sd"], (n_trials, n_games))
    eff = {t: rng.normal(0.0, k["team_eff_sd"], n_trials) for t in all_teams}
    # rushing efficiency: the same shock as passing unless rush_eff_corr < 1 (then partly its own)
    rc = float(k.get("rush_eff_corr", 1.0))
    eff_rush = eff if rc >= 1.0 else {
        t: rc * eff[t] + np.sqrt(1.0 - rc ** 2) * rng.normal(0.0, k["team_eff_sd"], n_trials) for t in all_teams}

    # A pool can project a team for more volume than real teams get (deep synthetic pools, generous
    # projections): scale that team's dropbacks / carries up so each player's box scores can reach
    # his projection.
    team_rows = {t: np.flatnonzero(df["team"].to_numpy() == t) for t in teams}
    usage = {t: _usage(df.iloc[team_rows[t]]) for t in teams}
    team_tgts = DROPBACKS * (1 - SACK_RATE) * TARGETS_PER_ATT
    vol = {t: (max(1.0, usage[t]["targets"].sum() / (0.92 * team_tgts)), max(1.0, usage[t]["carries"].sum() / (0.95 * CARRIES)))
           for t in teams}
    # 1. volume and turnovers for every offense
    box = {}
    for t in all_teams:
        gi = g_index[game_of[t]]
        o = opp_of.get(t)
        e = g_score[:, gi] + eff[t]                         # this offense's scoring shock
        e_r = g_score[:, gi] + eff_rush[t]                  # ... on the ground (= e unless rush_eff_corr < 1)
        deficit = (eff[o] - eff[t]) if isinstance(o, str) and o in eff else np.zeros(n_trials)
        exp_margin = imp[t] - (imp[o] if isinstance(o, str) and o in imp else imp[t])
        mix_u = None
        if k.get("td_mix_sd", 0.0) or k.get("share_tilt", 0.0):  # run-vs-pass tilt (off by default)
            mix_u = rng.normal(0.0, 1.0, n_trials)
        share = np.clip(DROPBACKS / (DROPBACKS + CARRIES) - 0.004 * exp_margin + k["script"] * deficit
                        + rng.normal(0.0, k["dropback_noise"], n_trials)
                        + (k.get("share_tilt", 0.0) * mix_u if mix_u is not None else 0.0), 0.35, 0.80)
        plays = np.maximum(np.round(rng.normal((DROPBACKS + CARRIES) * np.exp(g_pace[:, gi]), 5.5)), 35)
        dropbacks = np.round(plays * share)
        carries = plays - dropbacks
        if t in vol:
            dropbacks = np.round(dropbacks * vol[t][0])
            carries = np.round(carries * vol[t][1])
        sacks = rng.binomial(dropbacks.astype(np.int64), np.clip(SACK_RATE * np.exp(-0.5 * e), 0.01, 0.2))
        att = dropbacks - sacks
        ints = rng.poisson(INT_RATE * att * np.exp(-0.3 * e))
        fum = rng.poisson(FUMBLES_LOST * plays / (DROPBACKS + CARRIES))
        box[t] = {"e": e, "e_r": e_r, "deficit": deficit, "gi": gi, "att": att, "tgts": np.round(att * TARGETS_PER_ATT),
                  "carries": carries, "sacks": sacks, "ints": ints, "fum": fum, "mix_u": mix_u}
    # 2. scoring: an opponent's turnovers hand this offense extra (and shorter) possessions
    exp_to = INT_RATE * DROPBACKS * (1 - SACK_RATE) + FUMBLES_LOST
    for t in all_teams:
        b, o = box[t], opp_of.get(t)
        ob = box.get(o) if isinstance(o, str) else None
        gift = (ob["ints"] + ob["fum"] - exp_to) if ob else np.zeros(n_trials)
        e, deficit = b["e"], b["deficit"]
        scale = imp[t] / AVG_TOTAL
        bonus = np.exp(k["turnover_pts"] * gift)
        lam_p = PASS_TDS * scale ** PASS_TD_ELASTICITY * bonus * np.exp(
            k["pass_td_load"] * e - 0.5 * (k["pass_td_load"] * k["team_eff_sd"]) ** 2)
        lam_r = RUSH_TDS * scale ** RUSH_TD_ELASTICITY * bonus * np.exp(
            k["rush_td_load"] * b["e_r"] - k["script_rush_td"] * deficit - 0.5 * (k["rush_td_load"] * k["team_eff_sd"]) ** 2)
        mx = k.get("td_mix_sd", 0.0)
        if mx and b["mix_u"] is not None:  # a passing game's TDs come through the air, a running game's on the ground
            lam_p = lam_p * np.exp(mx * b["mix_u"] - mx * mx / 2)
            lam_r = lam_r * np.exp(-mx * b["mix_u"] - mx * mx / 2)
        b["pass_td"] = rng.poisson(lam_p)
        b["rush_td"] = rng.poisson(lam_r)
        b["tds"] = b["pass_td"] + b["rush_td"]
        fga = rng.poisson(FG_ATTEMPTS * np.exp(g_pace[:, b["gi"]]) * scale ** 0.3 * bonus
                          * np.exp(k["fg_eff_load"] * e - k["fg_td_tradeoff"] * (b["tds"] - (lam_p + lam_r))))
        b["fg"] = rng.binomial(fga, FG_MAKE)
        b["fg_pts"] = (rng.multinomial(b["fg"], FG_DIST) * np.array(FG_POINTS)).sum(axis=1)
        b["xp"] = rng.binomial(b["tds"], XP_PER_TD)
    # 3. defensive / special-teams scores need the other side's turnovers
    for t in all_teams:
        o = opp_of.get(t)
        opp_box = box.get(o) if isinstance(o, str) else None
        to = (opp_box["ints"] + opp_box["fum"]) if opp_box else rng.poisson(1.27, n_trials)
        box[t]["def_td"] = rng.binomial(to, DEF_TD_PER_TURNOVER) + rng.poisson(ST_TDS, n_trials)
        box[t]["safety"] = rng.poisson(SAFETIES, n_trials)
    for t in all_teams:
        b = box[t]
        b["points"] = 6 * (b["tds"] + b["def_td"]) + b["xp"] + 3 * b["fg"] + 2 * b["safety"]

    for t in teams:
        rows = team_rows[t]
        sub = df.iloc[rows]
        b = box[t]
        use = usage[t]
        pos = sub["position"].to_numpy()
        # targets: pool receivers + the rest of the team
        exp_t = np.append(use["targets"], 0.0)
        team_exp = team_tgts * vol[t][0]
        exp_t[-1] = max(team_exp - exp_t[:-1].sum(), 0.08 * team_exp)
        tg = _dirichlet_multinomial(rng, b["tgts"], exp_t, k["target_conc"])
        exp_c = np.append(use["carries"], 0.0)
        exp_c[-1] = max(CARRIES * vol[t][1] - exp_c[:-1].sum(), 0.05 * CARRIES)
        ca = _dirichlet_multinomial(rng, b["carries"], exp_c, k["carry_conc"])
        ymult = np.exp(k["yards_load"] * b["e"])[:, None]
        ymult_r = ymult if b["e_r"] is b["e"] else np.exp(k["yards_load"] * b["e_r"])[:, None]
        rec = np.zeros_like(tg)
        rec_yds = np.zeros(tg.shape)
        rush_yds = np.zeros(ca.shape)
        ppos = list(pos) + ["other"]
        for j, p in enumerate(ppos):
            pk = _pos_key(p, CATCH)
            rec[:, j] = rng.binomial(tg[:, j], CATCH[pk])
            rec_yds[:, j] = _gamma_sum(rng, rec[:, j], YDS_PER_CATCH[pk], CATCH_CV[pk], ymult[:, 0])
            ck = _pos_key(p, YDS_PER_CARRY)
            rush_yds[:, j] = _gamma_sum(rng, ca[:, j], YDS_PER_CARRY[ck], CARRY_CV[ck], ymult_r[:, 0])
        # TDs go to whoever got the looks this game
        rec_rate = np.array([REC_TD_PER_TGT[_pos_key(p, REC_TD_PER_TGT)] for p in ppos])
        rush_rate = np.array([RUSH_TD_PER_CARRY[_pos_key(p, RUSH_TD_PER_CARRY)] for p in ppos])
        own_rec, own_rush = np.append(use["rec_td_rate"], np.nan), np.append(use["rush_td_rate"], np.nan)
        rec_rate = np.where(np.isfinite(own_rec), np.clip(own_rec, 0.2 * rec_rate, 4 * rec_rate), rec_rate)
        rush_rate = np.where(np.isfinite(own_rush), np.clip(own_rush, 0.2 * rush_rate, 4 * rush_rate), rush_rate)
        w_rec = tg * rec_rate[None, :]           # red-zone roles: a player's own projected TD rate if known
        w_rush = ca * rush_rate[None, :]
        rec_td = _alloc(rng, b["pass_td"], w_rec)
        rush_td = _alloc(rng, b["rush_td"], w_rush)
        touches = rec + ca
        fum = _alloc(rng, b["fum"], touches.astype(float) + 1e-9)
        pts = (rec + 0.1 * rec_yds + 6 * rec_td + 0.1 * rush_yds + 6 * rush_td - fum
               + 3 * (rec_yds >= 100) + 3 * (rush_yds >= 100))
        pass_yds = rec_yds.sum(axis=1)
        qb_line = None
        if use["passer"] is not None:  # the starting QB's line; with several candidates, one takes it per trial
            j0 = use["passer"]
            qb_line = (0.04 * pass_yds + 4 * b["pass_td"] - b["ints"] + 3 * (pass_yds >= 300)
                       + 0.1 * rush_yds[:, j0] + 6 * rush_td[:, j0] - fum[:, j0] + 3 * (rush_yds[:, j0] >= 100))
            starter = np.asarray(use["starters"])[rng.choice(len(use["starters"]), n_trials, p=use["start_p"])]
        for j, p in enumerate(pos):
            i = rows[j]
            if p == "QB":
                out[:, i] = np.where(starter == j, qb_line, 0.0) if qb_line is not None else 0.0
            elif p in ("WR", "TE", "RB"):
                out[:, i] = pts[:, j]
            elif p == "K":
                out[:, i] = b["fg_pts"] + b["xp"]
            elif p == "DST":
                o = opp_of.get(t)
                ob = box.get(o) if isinstance(o, str) else None
                if ob is None:
                    out[:, i] = 0.0
                    continue
                out[:, i] = _dst_points(ob["sacks"], ob["ints"], ob["fum"], b["def_td"], b["safety"], ob["points"])
    return out


def _alloc(rng, totals, weights):
    """Split each trial's integer total across columns in proportion to that trial's weights."""
    w = np.asarray(weights, dtype=float)
    s = w.sum(axis=1, keepdims=True)
    p = np.where(s > 0, w / np.where(s > 0, s, 1.0), 1.0 / w.shape[1])
    return rng.multinomial(np.asarray(totals, dtype=np.int64), p)
