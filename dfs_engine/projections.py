"""
Our own projections from free data (nflverse box scores + Vegas lines).

Skill players: per-position ridge regression of DK points on pre-game
features, computed only from games before the week being projected
(`asof_features`):

* stab_dk / stab_targets / stab_carries / stab_attempts -- season-to-date
  averages shrunk toward last season ((n * this + 3 * last) / (n + 3))
* l3_dk / l3_targets / l3_carries -- last three games played
* share_x_team -- the player's share of his team's DK points times the
  team's Vegas-implied DK output (26.3 + 2.73 x implied total, fitted on
  2021-25: each implied point is worth ~2.7 DK points to the offense)
* implied, spread, dk_x_implied -- Vegas team total, expected margin, and
  season average scaled by the implied total

DSTs: linear model on the opponent's implied total (the main driver),
own spread, season sack rate and home field.

Backtest: trained 2021-23, tested 2024-25, within-position correlation QB
0.50 / RB 0.67 / WR 0.60 / TE 0.58 (season average alone: 0.46 / 0.65 /
0.59 / 0.56). Head-to-head with the public projections on 556 2026 Week 1-4
skill-player-weeks: MAE 6.04 vs 5.97; a 25% blend beat the public number
(5.93) in three weeks and tied the fourth. DSTs: corr 0.24 vs 0.11 for the
public DST projections. Opponent-defense adjustments added nothing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .correlations import dk_points
from .sportsgameodds import normalize_abbr, normalize_name

SKILL = ("QB", "RB", "WR", "TE")
FEATURES = ["stab_dk", "l3_dk", "stab_targets", "stab_carries", "l3_targets", "l3_carries", "stab_attempts",
            "share_x_team", "implied", "spread", "dk_x_implied"]
DST_FEATURES = ["opp_implied", "spread", "std_sacks", "home"]
TEAM_DK = (26.3, 2.73)  # team offensive DK points ~ a + b * implied team total (2021-25)
PRIOR_GAMES = 3.0       # weight of last season in the shrunk season averages
_USAGE = ("dk", "targets", "carries", "attempts", "share")
_DEF = ["def_sacks", "def_interceptions", "fumble_recovery_opp", "def_safeties", "def_tds", "special_teams_tds",
        "def_fg_blocks", "def_punt_blocks", "def_pat_blocks"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def player_weeks(weekly: pd.DataFrame) -> pd.DataFrame:
    """Regular-season QB/RB/WR/TE rows with DK points and share of the team's DK points that week."""
    p = weekly[(weekly["season_type"] == "REG") & weekly["position"].isin(SKILL)].copy()
    p["dk"] = dk_points(p).to_numpy()
    p["team"] = p["team"].map(normalize_abbr)
    p["key"] = p["player_display_name"].map(normalize_name)
    for c in ("targets", "carries", "attempts"):
        p[c] = p[c].fillna(0) if c in p else 0.0
    p["share"] = p["dk"] / p.groupby(["season", "week", "team"])["dk"].transform("sum").clip(lower=1)
    return p[["season", "week", "key", "team", "position", "dk", "targets", "carries", "attempts", "share"]]


def team_lines(games: pd.DataFrame) -> pd.DataFrame:
    """Per team-week: opp, implied total, spread (negative = favored), home flag, points scored/allowed."""
    g = games[(games["game_type"] == "REG") & games["total_line"].notna()]
    rows = []
    for r in g.itertuples(index=False):
        for team, opp, sign, home, pts, opp_pts in ((r.home_team, r.away_team, 1, 1, r.home_score, r.away_score),
                                                     (r.away_team, r.home_team, -1, 0, r.away_score, r.home_score)):
            rows.append({"season": r.season, "week": r.week, "team": normalize_abbr(team), "opp": normalize_abbr(opp),
                         "implied": r.total_line / 2 + sign * r.spread_line / 2, "spread": -sign * r.spread_line,
                         "home": home, "pts": pts, "opp_pts": opp_pts})
    return pd.DataFrame(rows)


def asof_features(pw: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Pre-game usage features by player key, from games before `week` of `season` (+ last season)."""
    cur = pw[(pw["season"] == season) & (pw["week"] < week)].sort_values("week")
    prv = pw[pw["season"] == season - 1]
    f = prv.groupby("key")[list(_USAGE)].mean().add_prefix("prev_").join(
        cur.groupby("key")[list(_USAGE)].mean().add_prefix("std_"), how="outer")
    f["n"] = cur.groupby("key").size().reindex(f.index).fillna(0)
    l3 = cur.groupby("key").tail(3).groupby("key")[["dk", "targets", "carries"]].mean().add_prefix("l3_")
    f = f.join(l3)
    w = f["n"] / (f["n"] + PRIOR_GAMES)
    for c in _USAGE:
        a, b = f[f"std_{c}"], f[f"prev_{c}"]
        f[f"stab_{c}"] = np.where(a.notna() & b.notna(), w * a.fillna(0) + (1 - w) * b.fillna(0), a.fillna(b))
    for c in ("dk", "targets", "carries"):
        f[f"l3_{c}"] = f[f"l3_{c}"].fillna(f[f"stab_{c}"])
    pos = pd.concat([prv, cur]).groupby("key")["position"].last()
    return f.assign(position=pos)


def _with_vegas(f: pd.DataFrame, implied, spread) -> pd.DataFrame:
    f = f.assign(implied=implied, spread=spread)
    f["share_x_team"] = f["stab_share"] * (TEAM_DK[0] + TEAM_DK[1] * f["implied"])
    f["dk_x_implied"] = f["stab_dk"] * f["implied"] / 22.3
    return f


def dst_weeks(weekly: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Team-week DST DK points (sacks, takeaways, TDs, blocks, points-allowed tiers) and pre-game features."""
    x = weekly[weekly["season_type"] == "REG"].copy()
    for c in _DEF:
        x[c] = x[c].fillna(0) if c in x else 0.0
    x["team"] = x["team"].map(normalize_abbr)
    t = x.groupby(["season", "week", "team"])[_DEF].sum().reset_index().merge(lines, on=["season", "week", "team"])
    pa = t["opp_pts"]
    tier = np.select([pa == 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34], [10, 7, 4, 1, 0, -1], -4)
    t["dst"] = (t.def_sacks + 2 * (t.def_interceptions + t.fumble_recovery_opp + t.def_safeties)
                + 6 * (t.def_tds + t.special_teams_tds) + 2 * (t.def_fg_blocks + t.def_punt_blocks + t.def_pat_blocks)
                + tier)
    opp = lines[["season", "week", "team", "implied"]].rename(columns={"team": "opp", "implied": "opp_implied"})
    t = t.merge(opp, on=["season", "week", "opp"], how="left").sort_values(["team", "season", "week"])
    t["std_sacks"] = t.groupby(["team", "season"])["def_sacks"].transform(lambda s: s.shift(1).expanding().mean())
    t["std_sacks"] = t["std_sacks"].fillna(t.groupby("season")["def_sacks"].transform("mean"))
    return t


def training_rows(weekly: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(skill rows, DST rows) with pre-game features and the actual DK points of that week."""
    pw = player_weeks(weekly)
    lines = team_lines(games)
    out = []
    for (season, week), wk in pw.groupby(["season", "week"]):
        f = asof_features(pw, season, week)
        f = f.drop(columns="position").join(wk.set_index("key")[["team", "position", "dk"]], how="inner")
        ln = lines[(lines["season"] == season) & (lines["week"] == week)].set_index("team")
        f = _with_vegas(f, f["team"].map(ln["implied"]), f["team"].map(ln["spread"]))
        out.append(f.assign(season=season, week=week))
    skill = pd.concat(out).dropna(subset=FEATURES + ["dk"])
    return skill, dst_weeks(weekly, lines).dropna(subset=DST_FEATURES + ["dst"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    pen = np.eye(Z.shape[1]) * lam
    pen[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + pen, Z.T @ y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist()}


def _apply(m: dict, X: np.ndarray) -> np.ndarray:
    Z = np.column_stack([np.ones(len(X)), (X - np.array(m["mu"])) / np.array(m["sd"])])
    return Z @ np.array(m["beta"])


def fit(skill: pd.DataFrame, dst: pd.DataFrame, lam: float = 10.0) -> dict:
    """Per-position ridge models + the DST model, JSON-serializable."""
    model = {"features": FEATURES, "dst_features": DST_FEATURES, "lam": lam,
             "seasons": sorted(int(s) for s in skill["season"].unique()), "positions": {}}
    for pos in SKILL:
        g = skill[skill["position"] == pos]
        model["positions"][pos] = _ridge(g[FEATURES].to_numpy(float), g["dk"].to_numpy(float), lam) | {"n": int(len(g))}
    model["dst"] = _ridge(dst[DST_FEATURES].to_numpy(float), dst["dst"].to_numpy(float), 1.0) | {"n": int(len(dst))}
    return model


def save(model: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


MODEL_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "projection_model.json"


def fetch_weekly(seasons: list[int], out_dir: str) -> list[str]:
    """Download nflverse weekly player stats for `seasons` (skips files already there)."""
    import os
    import urllib.request
    from .ownership_model import NFLVERSE

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for y in seasons:
        path = f"{out_dir}/stats_player_week_{y}.csv"
        if not os.path.exists(path):
            with urllib.request.urlopen(NFLVERSE["weekly"].format(season=y), timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
        paths.append(path)
    return paths


def project(pool: pd.DataFrame, nfl: dict[str, pd.DataFrame], season: int, week: int, model: dict) -> pd.Series:
    """
    Our projection for every pool player with history (NaN otherwise, e.g.
    rookies before their first game). Uses the pool's team_total / spread
    when present, else the nflverse schedule.
    """
    out = pd.Series(np.nan, index=pool.index)
    weekly, games = nfl["weekly"], nfl["games"]
    if "prior_weekly" in nfl:
        weekly = pd.concat([nfl["prior_weekly"], weekly], ignore_index=True)
    lines = team_lines(games)
    ln = lines[(lines["season"] == season) & (lines["week"] == week)].set_index("team")
    team = pool["team"].map(normalize_abbr)
    implied = pool["team_total"] if "team_total" in pool and pool["team_total"].notna().any() else team.map(ln["implied"])
    implied = implied.fillna(team.map(ln["implied"]))
    spread = pool["spread"] if "spread" in pool and pool["spread"].notna().any() else team.map(ln["spread"])
    spread = spread.fillna(team.map(ln["spread"]))

    f = asof_features(player_weeks(weekly), season, week)
    keys = pool["name"].map(normalize_name)
    sk = pool["position"].isin(SKILL)
    x = _with_vegas(f.reindex(keys[sk]).set_index(pool.index[sk]), implied[sk].to_numpy(), spread[sk].to_numpy())
    for pos in SKILL:
        sel = (pool.loc[sk, "position"] == pos).to_numpy() & x[FEATURES].notna().all(axis=1).to_numpy()
        if sel.any():
            out[x.index[sel]] = _apply(model["positions"][pos], x.loc[x.index[sel], FEATURES].to_numpy(float))

    dmask = pool["position"] == "DST"
    if dmask.any():
        d = dst_weeks(weekly[weekly["season"] == season], lines)
        d = d[d["week"] < week]
        sacks = d.groupby("team")["def_sacks"].mean()
        lg = float(d["def_sacks"].mean()) if len(d) else 2.4
        opp = pool.loc[dmask, "opp"].map(normalize_abbr)
        opp_imp = opp.map(implied.groupby(team).first()).fillna(opp.map(ln["implied"]))
        home = team[dmask].map(ln["home"]).fillna(0.5)
        X = np.column_stack([opp_imp, spread[dmask], team[dmask].map(sacks).fillna(lg), home]).astype(float)
        ok = np.isfinite(X).all(axis=1)
        out[pool.index[dmask][ok]] = _apply(model["dst"], X[ok])
    return out.clip(lower=0.0)


def blend(pool: pd.DataFrame, ours: pd.Series, weight: float, dst_weight: float) -> pd.Series:
    """proj <- (1 - w) * proj + w * ours where we have a projection (DSTs use `dst_weight`)."""
    w = np.where(pool["position"] == "DST", dst_weight, weight)
    has = ours.notna() & pool["proj"].notna()
    return pool["proj"].where(~has, (1 - w) * pool["proj"] + w * ours)
