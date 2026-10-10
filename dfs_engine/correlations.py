"""
Player-to-player fantasy-score correlations measured from real NFL games.

Source: nflverse weekly player stats (`stats_player_week_<season>.csv`,
https://github.com/nflverse/nflverse-data/releases/tag/stats_player).
Each player-game gets DraftKings points; each player gets a *role* for the
season (QB1, RB1, RB2, WR1..WR3, TE1) by ranking teammates on average DK
points. We then correlate the DK scores of role pairs over team-games
(same team) and game pairings (opponents), e.g. QB1~WR1, RB1~RB2,
QB1~opp QB1.

This is the "historical pairs" idea from Bergman, Cardonha, Imbrogno &
Lozano (UConn) -- estimate how real teammates/opponents co-move -- in its
simplest role-based form. `fit_simulation` then tunes the simulation's
correlation knobs so its role correlations match the measured ones.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

ROLES = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
SAME_TEAM_PAIRS = [("QB1", "WR1"), ("QB1", "WR2"), ("QB1", "TE1"), ("QB1", "RB1"),
                   ("RB1", "RB2"), ("WR1", "WR2"), ("WR1", "TE1"), ("RB1", "WR1"), ("WR2", "WR3")]
ROLE_NAMES = [f"{p}{i}" for p, n in ROLES.items() for i in range(1, n + 1)]
OPP_PAIRS = [("QB1", "QB1"), ("QB1", "WR1"), ("WR1", "WR1"), ("RB1", "RB1"), ("RB1", "QB1")]


def dk_points(s: pd.DataFrame) -> pd.Series:
    """DraftKings NFL Classic offensive scoring from nflverse weekly columns."""
    g = lambda c: s[c].fillna(0) if c in s else 0.0  # noqa: E731
    pts = (0.04 * g("passing_yards") + 4 * g("passing_tds") - 1 * g("passing_interceptions")
           + 0.1 * g("rushing_yards") + 6 * g("rushing_tds")
           + 1 * g("receptions") + 0.1 * g("receiving_yards") + 6 * g("receiving_tds")
           + 2 * (g("passing_2pt_conversions") + g("rushing_2pt_conversions") + g("receiving_2pt_conversions"))
           - 1 * (g("rushing_fumbles_lost") + g("receiving_fumbles_lost") + g("sack_fumbles_lost"))
           + 6 * g("special_teams_tds"))
    pts = pts + 3 * (g("passing_yards") >= 300) + 3 * (g("rushing_yards") >= 100) + 3 * (g("receiving_yards") >= 100)
    return pts


def load_weekly(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_csv(p, low_memory=False)
        d = d[(d["season_type"] == "REG") & d["position"].isin(list(ROLES))]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True).copy()
    d["dk"] = dk_points(d)
    d["team_tds"] = (d.groupby(["season", "week", "team"])[["passing_tds", "rushing_tds"]]
                     .transform("sum").sum(axis=1))  # passing TDs already include the catches
    return d


def assign_roles(d: pd.DataFrame, min_games: int = 4) -> pd.DataFrame:
    """Season roles by average DK points among teammates at the position."""
    avg = (d.groupby(["season", "team", "position", "player_id"])["dk"]
           .agg(["mean", "size"]).reset_index())
    avg = avg[avg["size"] >= min_games]
    avg["rank"] = avg.groupby(["season", "team", "position"])["mean"].rank(ascending=False, method="first")
    avg = avg[avg["rank"] <= avg["position"].map(ROLES)]
    avg["role"] = avg["position"] + avg["rank"].astype(int).astype(str)
    return d.merge(avg[["season", "team", "player_id", "role"]], on=["season", "team", "player_id"])


def role_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game, one column per role (DK points; NaN if that role didn't play)."""
    t = d.pivot_table(index=["season", "week", "team", "opponent_team"], columns="role",
                      values="dk", aggfunc="first").reset_index()
    tds = d.groupby(["season", "week", "team"])["team_tds"].first()
    return t.merge(tds.rename("team_tds").reset_index(), on=["season", "week", "team"])


def measure(paths: list[str]) -> dict:
    d = assign_roles(load_weekly(paths))
    t = role_table(d)
    out = {"seasons": sorted(int(s) for s in d["season"].unique()), "team_games": int(len(t)),
           "same_team": {}, "opponent": {},
           "role_means": {r: round(float(t[r].mean()), 2) for r in ROLE_NAMES if r in t}}
    for a, b in SAME_TEAM_PAIRS:
        if a not in t or b not in t:
            continue
        x = t[[a, b]].dropna()
        out["same_team"][f"{a}~{b}"] = {"corr": round(float(x[a].corr(x[b])), 3), "n": int(len(x))}
    o = t.merge(t, left_on=["season", "week", "opponent_team"], right_on=["season", "week", "team"],
                suffixes=("", "_opp"))
    for a, b in OPP_PAIRS:
        if a not in o or b + "_opp" not in o:
            continue
        x = o[[a, b + "_opp"]].dropna()
        out["opponent"][f"{a}~opp {b}"] = {"corr": round(float(x[a].corr(x[b + "_opp"])), 3), "n": int(len(x))}
    out["heterogeneity"] = measure_heterogeneity(d, t)
    prof = player_profiles(d).merge(d[["season", "team", "player_id", "role"]].drop_duplicates(),
                                    on=["season", "team", "player_id"])
    out["role_profile"] = {r: {"tgt_pg": round(float(g["tgt_pg"].mean()), 2),
                               "rush_share": round(float(g["rush_share"].mean()), 3)}
                           for r, g in prof.groupby("role")}
    # Backup-RB question: how RB1 and RB2 do when the offense explodes.
    rb = t[["RB1", "RB2", "team_tds"]].dropna()
    big = rb["team_tds"] >= 5
    out["rb_blowouts"] = {
        "share_games_5plus_tds": round(float(big.mean()), 3),
        "rb2_mean_normal": round(float(rb.loc[~big, "RB2"].mean()), 2),
        "rb2_mean_5plus_tds": round(float(rb.loc[big, "RB2"].mean()), 2),
        "rb1_mean_normal": round(float(rb.loc[~big, "RB1"].mean()), 2),
        "rb1_mean_5plus_tds": round(float(rb.loc[big, "RB1"].mean()), 2),
        "rb1_rb2_corr_5plus_tds": round(float(rb.loc[big, "RB1"].corr(rb.loc[big, "RB2"])), 3),
        "p_both_rbs_15plus_5plus_tds": round(float(((rb.loc[big, "RB1"] >= 15) & (rb.loc[big, "RB2"] >= 15)).mean()), 3),
        "p_both_rbs_15plus_normal": round(float(((rb.loc[~big, "RB1"] >= 15) & (rb.loc[~big, "RB2"] >= 15)).mean()), 3),
    }
    return out


HETERO_PAIRS = [  # (pair, profile column, role it describes)
    (("QB1", "WR1"), "rush_share", "QB1"),
    (("QB1", "TE1"), "rush_share", "QB1"),
    (("QB1", "WR1"), "tgt_pg", "WR1"),
    (("QB1", "TE1"), "tgt_pg", "TE1"),
    (("QB1", "RB1"), "tgt_pg", "RB1"),
]


def player_profiles(d: pd.DataFrame) -> pd.DataFrame:
    """Season profile per player: targets/game, carries/game and share of DK points from rushing."""
    col = lambda c: d[c].fillna(0) if c in d else 0.0  # noqa: E731
    rush_dk = 0.1 * col("rushing_yards") + 6 * col("rushing_tds")
    g = d.assign(rush_dk=rush_dk, tgt=col("targets"), car=col("carries")).groupby(
        ["season", "team", "player_id"])
    out = g.agg(rush_dk=("rush_dk", "sum"), dk=("dk", "sum"), tgt_pg=("tgt", "mean"), car_pg=("car", "mean"))
    out["rush_share"] = (out["rush_dk"] / out["dk"].clip(lower=1)).clip(0, 1)
    return out.drop(columns=["rush_dk", "dk"]).reset_index()


def measure_heterogeneity(d: pd.DataFrame, t: pd.DataFrame) -> dict:
    """Pair correlations within terciles of a player's profile (`HETERO_PAIRS`), with each tercile's mean."""
    prof = player_profiles(d)
    roles = d[["season", "week", "team", "player_id", "role"]].merge(prof, on=["season", "team", "player_id"])
    out = {}
    for (a, b), col, who in HETERO_PAIRS:
        r = roles[roles["role"] == who][["season", "week", "team", col]]
        x = t.merge(r, on=["season", "week", "team"])[[a, b, col]].dropna()
        x["q"] = pd.qcut(x[col].rank(method="first"), 3, labels=False)  # ties split evenly
        out[f"{a}~{b} by {who} {col}"] = [
            {"mean": round(float(g[col].mean()), 3), "corr": round(float(g[a].corr(g[b])), 3), "n": int(len(g))}
            for _, g in x.groupby("q")]
    return out


def save(profile: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=1)


# --- matching the simulation to the measured correlations --------------------

SALARY = {"QB1": 6500, "RB1": 7000, "RB2": 4800, "WR1": 7200, "WR2": 5600, "WR3": 4300, "TE1": 4800}


def synthetic_slate(role_means: dict, n_games: int = 8, game_total: float = 45.0) -> pd.DataFrame:
    """A neutral slate: every team has one player per role at that role's real average."""
    rows = []
    for g in range(n_games):
        a, b = f"A{g}", f"B{g}"
        for team, opp in ((a, b), (b, a)):
            for role in ROLE_NAMES:
                rows.append({"name": f"{team} {role}", "team": team, "opp": opp, "position": role[:-1],
                             "role": role, "salary": SALARY[role], "proj": role_means[role],
                             "team_total": game_total / 2, "game_total": game_total, "spread": 0.0})
            rows.append({"name": f"{team} DST", "team": team, "opp": opp, "position": "DST", "role": "DST",
                         "salary": 3000, "proj": 7.0, "team_total": game_total / 2,
                         "game_total": game_total, "spread": 0.0})
    return pd.DataFrame(rows)


def simulated_correlations(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """Role-pair correlations in simulated scores, averaged over teams (same keys as `measure`)."""
    role = df["role"].to_numpy()
    teams = df["team"].unique()

    def col(team, r):
        i = np.flatnonzero((df["team"].to_numpy() == team) & (role == r))
        return scores[:, i[0]] if len(i) else None

    out = {"same_team": {}, "opponent": {}}
    for a, b in SAME_TEAM_PAIRS:
        cs = [np.corrcoef(col(t, a), col(t, b))[0, 1] for t in teams]
        out["same_team"][f"{a}~{b}"] = {"corr": round(float(np.mean(cs)), 3)}
    opp = df.groupby("team")["opp"].first()
    for a, b in OPP_PAIRS:
        cs = [np.corrcoef(col(t, a), col(opp[t], b))[0, 1] for t in teams]
        out["opponent"][f"{a}~opp {b}"] = {"corr": round(float(np.mean(cs)), 3)}
    return out


def hetero_slate(profile: dict, n_games: int = 9) -> pd.DataFrame:
    """
    `synthetic_slate` plus player profiles: each team's QB1 rush share and
    WR1 / TE1 / RB1 targets per game take the measured tercile means
    (cycled independently across teams); other roles get role averages.
    """
    df = synthetic_slate(profile["role_means"], n_games=n_games)
    rp, het = profile["role_profile"], profile["heterogeneity"]
    df["tgt_pg"] = df["role"].map(lambda r: rp.get(r, {}).get("tgt_pg", np.nan))
    df["rush_share"] = np.where(df["role"] == "QB1", rp["QB1"]["rush_share"], np.nan)
    teams = list(dict.fromkeys(df["team"]))
    plan = {"QB1": ("rush_share", "QB1~WR1 by QB1 rush_share", 0), "WR1": ("tgt_pg", "QB1~WR1 by WR1 tgt_pg", 1),
            "TE1": ("tgt_pg", "QB1~TE1 by TE1 tgt_pg", 2), "RB1": ("tgt_pg", "QB1~RB1 by RB1 tgt_pg", 3)}
    for role, (col, key, shift) in plan.items():
        means = [b["mean"] for b in het[key]]
        for i, t in enumerate(teams):
            df.loc[(df["team"] == t) & (df["role"] == role), col] = means[(i // (1 + shift) + shift) % 3]
    return df


def simulated_heterogeneity(df: pd.DataFrame, scores: np.ndarray, profile: dict) -> dict:
    """Simulated pair correlations per profile tercile, keyed like `measure_heterogeneity`."""
    out = {}
    role, team = df["role"].to_numpy(), df["team"].to_numpy()
    for (a, b), col, who in HETERO_PAIRS:
        key = f"{a}~{b} by {who} {col}"
        means = [x["mean"] for x in profile["heterogeneity"][key]]
        per = [[] for _ in means]
        for t in dict.fromkeys(team):
            ia = np.flatnonzero((team == t) & (role == a))
            ib = np.flatnonzero((team == t) & (role == b))
            iw = np.flatnonzero((team == t) & (role == who))
            if len(ia) and len(ib) and len(iw):
                v = df[col].to_numpy()[iw[0]]
                per[int(np.argmin([abs(v - m) for m in means]))].append(
                    np.corrcoef(scores[:, ia[0]], scores[:, ib[0]])[0, 1])
        out[key] = [{"mean": m, "corr": round(float(np.mean(c)), 3) if c else float("nan")}
                    for m, c in zip(means, per)]
    return out


def heterogeneity_error(profile: dict, simulated: dict) -> float:
    gaps = [s["corr"] - m["corr"] for k, v in profile["heterogeneity"].items() if k in simulated
            for m, s in zip(v, simulated[k]) if np.isfinite(s["corr"])]
    return float(np.sqrt(np.mean(np.square(gaps)))) if gaps else 0.0


def correlation_error(measured: dict, simulated: dict) -> float:
    """Root-mean-square gap between measured and simulated role-pair correlations."""
    gaps = [simulated[k][pair]["corr"] - v["corr"]
            for k in ("same_team", "opponent") for pair, v in measured[k].items() if pair in simulated[k]]
    return float(np.sqrt(np.mean(np.square(gaps))))


def fit_simulation(profile: dict, grid: dict, n_trials: int = 4000, seed: int = 0,
                   simulate=None) -> tuple[dict, float, list]:
    """
    Grid-search the simulation's correlation knobs (`grid`: {kwarg: [values]})
    to minimize `correlation_error` against a measured profile on a
    `synthetic_slate` -- or, when the profile has heterogeneity terciles, on
    a `hetero_slate`, averaging in `heterogeneity_error`. Returns (best
    kwargs, best error, all results).
    """
    if simulate is None:
        from .simulate import simulate_player_scores as simulate
    hetero = "heterogeneity" in profile and "role_profile" in profile
    df = hetero_slate(profile) if hetero else synthetic_slate(profile["role_means"])
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, combo))
        sc = simulate(df, n_trials=n_trials, seed=seed, **kw)
        sim = simulated_correlations(df, sc)
        err = correlation_error(profile, sim)
        if hetero:  # role pairs and profile terciles count equally
            sim["heterogeneity"] = simulated_heterogeneity(df, sc, profile)
            err = 0.5 * err + 0.5 * heterogeneity_error(profile, sim["heterogeneity"])
        results.append((kw, err, sim))
    best = min(results, key=lambda r: r[1])
    return best[0], best[1], results
