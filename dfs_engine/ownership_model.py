"""
Our own ownership projections, without a paid ownership source.

The field's ownership is mostly predictable from what every player can
see before lock: salary, the popular projections, Vegas lines, the
lobby's points-per-game, last week's box score and the injury report.
This module turns those into features, then fits a per-position *softmax*
(conditional logit) model to real contest %Drafted:

    share_i = exp(x_i . b_pos) / sum_j exp(x_j . b_pos)   (within a slate + position)
    own_i   = share_i * position total (base slots + its FLEX share)

The softmax is fitted by cross-entropy against the actual ownership
shares, so it is judged on how ownership is *split* within a position,
which is what matters, and it weights the chalk most.

Features (z-scored within position unless noted):

* value      -- points per $1k
* proj, salary
* value_rank -- -log(rank of value at the position): the field piles onto
                the top few values, not linearly
* team_total, favorite -- Vegas implied total and expected margin
                (z-scored across the slate)
* last_dk    -- last week's DK points (the field chases recent box scores)
* avg_value  -- season-to-date DK points per game per $1k (the lobby's FPPG)
* vacated    -- DK points/game of same-group teammates ruled Out/Doubtful
                (per 10 pts; not z-scored)

Lines, box scores and injury reports come from nflverse (`fetch_nflverse`,
`prepare`). Leave-one-week-out on 2026 Weeks 1-4 Millionaire Makers
(`cross_validate`): corr 0.73 / mean abs error 2.9 pts with this model,
vs 0.88 / 2.05 for GoingFor2's published projection (the older
log-linear fallback: 0.65 / 3.5 on Weeks 1-3). A 10% blend of ours into
GoingFor2 averaged 0.883 / 2.03 but lost in Week 3, so a published
projection stays first choice; this is the fallback and leverage scan,
and it gets better with each week of contest results.
(Optimal% from our sim was tried as a feature and added nothing.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .correlations import dk_points
from .sportsgameodds import normalize_abbr, normalize_name

FEATURES = ("value", "proj", "salary", "value_rank", "team_total", "favorite", "last_dk",
            "avg_value", "vacated")
POSITIONS = ("QB", "RB", "WR", "TE", "DST")


# ---------------------------------------------------------------------------
# Inputs: nflverse schedule/lines, box scores, injury reports
# ---------------------------------------------------------------------------

NFLVERSE = {
    "games": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "weekly": "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv",
    "injuries": "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv",
}


def nflverse_paths(out_dir: str, season: int) -> dict[str, str]:
    return {"games": f"{out_dir}/games.csv", "weekly": f"{out_dir}/stats_player_week_{season}.csv",
            "prior_weekly": f"{out_dir}/stats_player_week_{season - 1}.csv",
            "injuries": f"{out_dir}/injuries_{season}.csv"}


def fetch_nflverse(season: int, out_dir: str) -> dict[str, str]:
    """Download the schedule (with lines), this and last season's weekly stats, and injury reports."""
    import os
    import urllib.request

    os.makedirs(out_dir, exist_ok=True)
    paths = nflverse_paths(out_dir, season)
    urls = {"games": NFLVERSE["games"], "weekly": NFLVERSE["weekly"].format(season=season),
            "prior_weekly": NFLVERSE["weekly"].format(season=season - 1),
            "injuries": NFLVERSE["injuries"].format(season=season)}
    for k, url in urls.items():
        with urllib.request.urlopen(url, timeout=60) as r, open(paths[k], "wb") as f:
            f.write(r.read())
    return paths


def load_nflverse(out_dir: str, season: int) -> dict[str, pd.DataFrame]:
    """The files `fetch_nflverse` wrote (missing ones are skipped)."""
    import os

    out = {}
    for k, p in nflverse_paths(out_dir, season).items():
        if os.path.exists(p):
            out[k] = pd.read_csv(p, low_memory=False)
    return out


def attach_profiles(df: pd.DataFrame, weekly: pd.DataFrame, season: int, week: int,
                    prior_weekly: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Season-to-date `tgt_pg` (targets/game), `car_pg` (carries/game) and
    `rush_share` (share of DK points from rushing) per player -- the
    profiles the simulation's QB-receiver links use. Week 1 uses last season.
    """
    out = df.copy()
    past = weekly[(weekly["season"] == season) & (weekly["week"] < week)]
    if past.empty and prior_weekly is not None:
        past = prior_weekly[prior_weekly["season_type"] == "REG"]
    if past.empty:
        return out
    k = past["player_display_name"].map(normalize_name)
    col = lambda c: past[c].fillna(0) if c in past else pd.Series(0.0, index=past.index)  # noqa: E731
    rush = 0.1 * col("rushing_yards") + 6 * col("rushing_tds")
    g = pd.DataFrame({"k": k, "tgt": col("targets"), "car": col("carries"),
                      "rush": rush, "dk": dk_points(past).to_numpy()}).groupby("k")
    prof = pd.DataFrame({"tgt_pg": g["tgt"].mean(), "car_pg": g["car"].mean(),
                         "rush_share": (g["rush"].sum() / g["dk"].sum().clip(lower=1)).clip(0, 1)})
    key = out["name"].map(normalize_name)
    for c in prof.columns:
        out[c] = key.map(prof[c])
    out.loc[out["position"] == "DST", list(prof.columns)] = np.nan
    return out


def prepare(df: pd.DataFrame, nfl: dict[str, pd.DataFrame], season: int, week: int) -> pd.DataFrame:
    """
    Add every input nflverse can supply: lines (where missing), last_dk,
    avg_dk, vacated (ownership model) and tgt_pg / car_pg / rush_share
    (simulation profiles).
    """
    out = df
    if "games" in nfl:
        out = attach_schedule(out, nfl["games"], season, week)
    if "weekly" in nfl:
        out = attach_recency(out, nfl["weekly"], season, week)
        out = attach_usage(out, nfl["weekly"], season, week, injuries=nfl.get("injuries"),
                           prior_weekly=nfl.get("prior_weekly"))
        out = attach_profiles(out, nfl["weekly"], season, week, prior_weekly=nfl.get("prior_weekly"))
    return out


def schedule_lines(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Per-team opp / team_total / game_total / spread (negative = favored) from nflverse games.csv."""
    g = games[(games["season"] == season) & (games["week"] == week) & (games["game_type"] == "REG")]
    g = g.dropna(subset=["spread_line", "total_line"])
    rows = []
    for r in g.itertuples(index=False):
        home, away = normalize_abbr(r.home_team), normalize_abbr(r.away_team)
        sp, tot = float(r.spread_line), float(r.total_line)  # spread_line > 0: home favored
        rows.append({"team": home, "opp": away, "team_total": tot / 2 + sp / 2, "game_total": tot,
                     "spread": -sp})
        rows.append({"team": away, "opp": home, "team_total": tot / 2 - sp / 2, "game_total": tot,
                     "spread": sp})
    return pd.DataFrame(rows)


def attach_schedule(df: pd.DataFrame, games: pd.DataFrame, season: int, week: int,
                    overwrite: bool = False) -> pd.DataFrame:
    """Fill opp / team_total / game_total / spread from the schedule (keeps existing values unless overwrite)."""
    lines = schedule_lines(games, season, week).set_index("team")
    out = df.copy()
    out["team"] = out["team"].map(normalize_abbr)
    for c in ("opp", "team_total", "game_total", "spread"):
        new = out["team"].map(lines[c])
        out[c] = new if overwrite or c not in out else out[c].where(out[c].notna(), new)
    return out


def attach_recency(df: pd.DataFrame, weekly: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """`last_dk`: each player's DK points in the previous week (NaN if he didn't play / week 1)."""
    out = df.copy()
    prev = weekly[(weekly["season"] == season) & (weekly["week"] == week - 1)]
    if prev.empty:
        out["last_dk"] = np.nan
        return out
    pts = pd.Series(dk_points(prev).to_numpy(), index=prev["player_display_name"].map(normalize_name))
    pts = pts[~pts.index.duplicated()]
    out["last_dk"] = out["name"].map(normalize_name).map(pts)
    out.loc[out["position"] == "DST", "last_dk"] = np.nan
    return out


GROUP = {"QB": "QB", "RB": "RB", "WR": "REC", "TE": "REC"}


def attach_usage(df: pd.DataFrame, weekly: pd.DataFrame, season: int, week: int,
                 injuries: pd.DataFrame | None = None, prior_weekly: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    `avg_dk`: season-to-date DK points per game before this week (the
    FPPG the DraftKings lobby shows; Week 1 uses last season's). With an
    injury report, `vacated`: season DK points per game of same-group
    teammates (QB / RB / WR+TE) listed Out or Doubtful this week -- the
    volume a backup inherits, which the field piles onto.
    """
    out = df.copy()
    past = weekly[(weekly["season"] == season) & (weekly["week"] < week)]
    if past.empty and prior_weekly is not None:
        past = prior_weekly[prior_weekly["season_type"] == "REG"]
    if past.empty:
        out["avg_dk"], out["vacated"] = np.nan, 0.0
        return out
    past = past.assign(dk=dk_points(past).to_numpy(), _k=past["player_display_name"].map(normalize_name))
    avg = past.groupby("_k")["dk"].mean()
    out["avg_dk"] = out["name"].map(normalize_name).map(avg)
    out.loc[out["position"] == "DST", "avg_dk"] = np.nan
    out["vacated"] = 0.0
    if injuries is not None:
        inj = injuries[(injuries["season"] == season) & (injuries["week"] == week)
                       & injuries["report_status"].isin(["Out", "Doubtful"])
                       & injuries["position"].isin(list(GROUP))]
        inj = inj.assign(_k=inj["full_name"].map(normalize_name), _g=inj["position"].map(GROUP),
                         team=inj["team"].map(normalize_abbr))
        inj["pts"] = inj["_k"].map(avg).fillna(0.0)
        lost = inj.groupby(["team", "_g"])["pts"].sum()
        keys = list(zip(out["team"], out["position"].map(GROUP)))
        out["vacated"] = [float(lost.get(k, 0.0)) if k[1] is not None else 0.0 for k in keys]
    return out


# ---------------------------------------------------------------------------
# Features and model
# ---------------------------------------------------------------------------

def _z(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    sd = x.std(ddof=0)
    if not sd or not np.isfinite(sd):
        return x * 0.0
    return ((x - x.mean()) / sd).fillna(0.0)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix (see module docstring); missing inputs give a zero column."""
    X = pd.DataFrame(0.0, index=df.index, columns=list(FEATURES))
    value = df["proj"] / (df["salary"] / 1000.0)
    for _, g in df.groupby("position"):
        i = g.index
        X.loc[i, "value"] = _z(value[i])
        X.loc[i, "proj"] = _z(g["proj"])
        X.loc[i, "salary"] = _z(g["salary"])
        X.loc[i, "value_rank"] = _z(-np.log(value[i].rank(ascending=False, method="min")))
        if "last_dk" in g and g["last_dk"].notna().any():
            X.loc[i, "last_dk"] = _z(g["last_dk"])
        if "avg_dk" in g and g["avg_dk"].notna().any():
            X.loc[i, "avg_value"] = _z(g["avg_dk"] / (g["salary"] / 1000.0))
        if "vacated" in g:
            X.loc[i, "vacated"] = g["vacated"].fillna(0.0) / 10.0  # per 10 DK pts/game freed up
    if "team_total" in df and df["team_total"].notna().any():
        X["team_total"] = _z(df["team_total"])
    if "spread" in df and df["spread"].notna().any():
        X["favorite"] = _z(-df["spread"])
    return X


# Players per game at each position in the training files (the realistic
# candidates: starting QBs/RBs/DSTs, top WRs/TEs). `predict` scores the same
# number of top projections so its shares are on the same footing; deeper
# players split a small tail (real fields gave them ~4% of ownership).
UNIVERSE_PER_GAME = {"QB": 2.0, "RB": 2.0, "WR": 4.2, "TE": 1.6, "DST": 2.0}
TAIL_SHARE = 0.04


def universe(df: pd.DataFrame) -> pd.DataFrame:
    """Projected players a model scores: the top `UNIVERSE_PER_GAME` x games by proj at each position."""
    d = df[df["proj"].notna() & (df["proj"] > 0)]
    n_games = max(d["team"].nunique() / 2.0, 1.0)
    keep = [g.nlargest(max(int(round(UNIVERSE_PER_GAME.get(p, 2.0) * n_games)), 1), "proj")
            for p, g in d.groupby("position")]
    return pd.concat(keep) if keep else d


def fit(weeks: list[pd.DataFrame], l2: float = 0.3, feature_names=FEATURES, fit_scale: bool = True) -> dict:
    """
    Per-position softmax fit. Each frame is one slate with the columns
    `features` needs plus `actual_own`. Returns
    {position: {feature: coef, ..., "n": players}}.
    """
    names = list(feature_names)
    frames = []
    for s, w in enumerate(weeks):
        w = universe(w[w["salary"].notna()])
        X = features(w)[names]
        frames.append(pd.concat([w[["position", "actual_own"]], X], axis=1).assign(_slate=s))
    d = pd.concat(frames, ignore_index=True)
    model = {}
    for p, g in d.groupby("position"):
        X = g[names].to_numpy(float)
        groups = []
        for _, gg in g.groupby("_slate"):
            ix = g.index.get_indexer(gg.index)
            y = gg["actual_own"].clip(lower=0).to_numpy(float)
            if y.sum() > 0:
                groups.append((ix, y / y.sum()))

        def loss(b):
            total, grad = 0.0, np.zeros_like(b)
            for ix, y in groups:
                z = X[ix] @ b
                z = z - z.max()
                pr = np.exp(z) / np.exp(z).sum()
                total -= float(y @ np.log(pr + 1e-12))
                grad += X[ix].T @ (pr - y)
            k = max(len(groups), 1)
            return total / k + l2 * b @ b, grad / k + 2 * l2 * b

        b = minimize(loss, np.zeros(len(names)), jac=True, method="L-BFGS-B").x
        if fit_scale and b.any():
            # The L2 penalty picks a stable direction but also flattens the
            # shares; re-fit one unpenalized scale so chalk is as
            # concentrated as in real fields.
            ce = lambda t: loss(t * b)[0] - l2 * (t * b) @ (t * b)  # noqa: E731
            b = b * minimize_scalar(ce, bounds=(0.2, 20.0), method="bounded").x
        model[p] = {f: float(c) for f, c in zip(names, b)} | {"n": int(len(g))}
    return model


def predict(df: pd.DataFrame, model: dict, totals: dict[str, float] | None = None) -> pd.Series:
    """Fraction-owned per player: softmax share within position x the position's total."""
    if totals is None:
        from .ownership import position_totals
        totals = position_totals()
    own = pd.Series(0.0, index=df.index)
    d = df[df["proj"].notna() & (df["proj"] > 0)]
    top_all = universe(d)
    for p, g in d.groupby("position"):
        c = model.get(p)
        if c is None:
            continue
        top = top_all[top_all["position"] == p]
        rest = g.index.difference(top.index)
        names = [f for f in FEATURES if f in c]
        z = features(top)[names].to_numpy(float) @ np.array([c[f] for f in names])
        e = np.exp(z - z.max())
        tail = TAIL_SHARE if len(rest) else 0.0
        own[top.index] = e / e.sum() * totals.get(p, 1.0) * (1 - tail)
        if len(rest):
            own[rest] = g.loc[rest, "proj"] / g.loc[rest, "proj"].sum() * totals.get(p, 1.0) * tail
    return own.clip(upper=0.95)


def score(pred: pd.Series, actual: pd.Series, position: pd.Series) -> dict:
    """corr / MAE overall and per position, plus how many of the 10 chalkiest players are in our top 10."""
    out = {"corr": float(np.corrcoef(pred, actual)[0, 1]), "mae": float((pred - actual).abs().mean()),
           "top10_hit": int(len(set(actual.nlargest(10).index) & set(pred.nlargest(10).index)))}
    out["by_position"] = {p: float(np.corrcoef(pred[position == p], actual[position == p])[0, 1])
                          for p in POSITIONS if (position == p).sum() > 2}
    return out


def cross_validate(weeks: list[pd.DataFrame], l2: float = 0.3, feature_names=FEATURES,
                   baseline_col: str | None = "own", fit_scale: bool = True) -> list[dict]:
    """
    Leave-one-week-out: fit on the other weeks, score on the held-out
    one. If the frames carry a published projection (`baseline_col`, in %
    or fraction) it is scored on the same players for comparison.
    """
    out = []
    for k, test in enumerate(weeks):
        model = fit([w for j, w in enumerate(weeks) if j != k], l2=l2, feature_names=feature_names,
                    fit_scale=fit_scale)
        t = test[test["proj"].notna() & (test["proj"] > 0)]
        pred = predict(t, model)
        r = {"week": k, "ours": score(pred, t["actual_own"], t["position"])}
        if baseline_col and baseline_col in t and t[baseline_col].notna().all():
            b = t[baseline_col] / (100.0 if t[baseline_col].max() > 1.5 else 1.0)
            r["published"] = score(b, t["actual_own"], t["position"])
            r["blend"] = score(0.5 * b + 0.5 * pred, t["actual_own"], t["position"])
        out.append(r)
    return out
