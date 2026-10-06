"""
Correlated Monte Carlo player-score simulation.

This replicates the core structure of how pro-level tools (e.g. SaberSim)
actually simulate NFL slates, rather than drawing independent per-player
noise:

1. Each GAME gets a simulated total + a split between the two teams,
   drawn around the Vegas-implied team totals / spread.
2. Each TEAM's offensive output that trial becomes an "offense_factor"
   that correlates every one of that team's players together (a big
   game for the team lifts its QB/WRs/RBs/TE together), and
   anti-correlates with the opposing DST.
3. Within a team, same-position-group pass-catchers (WR/TE) compete for
   a fixed share of targets: one player's good trial comes partly at
   the expense of his teammates (negative within-group correlation),
   which stops the model from over-stacking a team's whole receiving
   corps as if they were independent.
4. Game script is asymmetric: a team that out-scores its opponent (by
   more than the spread implied) runs more (RB volume up, pass-catcher
   volume down); a trailing team passes more (QB/WR/TE up, RB down).
   Both teams' passers also share a per-game "shootout" shock.
5. Residual player-level noise is right-skewed / fat-tailed (blend of
   Gaussian + shifted-exponential) to reflect real NFL boom risk
   (long TDs, garbage-time scores) better than a pure Gaussian.
6. QB variance is scaled up for lower-salary (backup/game-manager-risk)
   quarterbacks relative to top-salary-tier starters.
7. Ceiling calibration: each player's simulated deviation from his
   projection is then stretched/shrunk (and re-centered) so his
   `ceiling_pct` percentile (default 85th) lands on the `ceiling` column
   while his mean stays on `proj` -- the way Stokastic/SaberSim fit each
   distribution to a median + ceiling. A per-player affine rescale leaves
   pairwise correlations unchanged, so the game/team structure is kept.
   Players with no usable ceiling get a position-default ceiling
   (`fill_default_ceilings`, scaled by `spread_scale`), fitted so the
   simulated field's score spread matches real Millionaire Makers; pass
   `spread_scale=None` to leave them on the raw spread from steps 5-6.

The output is a (n_trials, n_players) array of simulated fantasy points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def skewed_noise(n: int, rng: np.random.Generator, skew_weight: float = 0.35) -> np.ndarray:
    """
    Zero-mean, unit-ish-scale noise blending Gaussian with a shifted
    exponential tail, to produce right skew (boom risk) without
    shifting the mean.
    """
    gauss = rng.normal(0.0, 1.0, n)
    exp_raw = rng.exponential(1.0, n)
    exp_centered = exp_raw - 1.0  # zero-mean shifted exponential
    return (1 - skew_weight) * gauss + skew_weight * exp_centered


def qb_variance_multiplier(salary: pd.Series) -> pd.Series:
    """
    Tier QBs into salary terciles and scale variance up for cheaper/
    less-proven starters (proxy for backup / game-manager bust risk).
    """
    if salary.nunique() < 3:  # can't tier (e.g. every QB at one salary)
        return pd.Series(1.0, index=salary.index)
    terciles = pd.qcut(salary, 3, labels=["low", "mid", "high"], duplicates="drop")
    mult = terciles.map({"low": 1.35, "mid": 1.12, "high": 1.00}).astype(float)
    return mult.fillna(1.00)


# Default ceiling (85th percentile) as a multiple of projection, for players
# whose `ceiling` is missing: proj * (1 + spread_scale * (mult - 1)). The
# scale 0.80 makes a simulated, ownership-sampled field's score spread
# (top 0.1% / 1% / 20% vs median) match three real 2026 Millionaire Makers
# (data/field_profile.json); `contest` / `dk-run` re-fit it per slate.
BASE_CEILING_MULT = {"QB": 1.55, "RB": 1.85, "WR": 1.90, "TE": 1.90, "DST": 2.00}
DEFAULT_SPREAD_SCALE = 0.80


def fill_default_ceilings(ceiling: np.ndarray, proj: np.ndarray, position: pd.Series,
                          spread_scale: float = DEFAULT_SPREAD_SCALE) -> np.ndarray:
    """Replace missing / non-usable ceilings (not above proj) with the scaled position default."""
    mult = position.map(BASE_CEILING_MULT).fillna(1.85).to_numpy(dtype=float)
    default = proj * (1.0 + spread_scale * (mult - 1.0))
    usable = np.isfinite(ceiling) & (ceiling > proj)
    return np.where(usable, ceiling, default)


def simulate_player_scores(
    df: pd.DataFrame,
    n_trials: int = 50_000,
    seed: int | None = None,
    game_total_sd_frac: float = 0.10,
    team_split_sd: float = 2.0,
    target_competition_weight: float = 0.60,
    game_script_weight: float = 0.40,
    residual_sd_frac: float = 0.90,
    rb_competition_weight: float = 0.30,
    qb_catcher_link: float = 0.85,
    shootout_sd: float = 0.20,
    calibrate_ceiling: bool = True,
    ceiling_pct: float = 0.85,
    spread_scale: float | None = DEFAULT_SPREAD_SCALE,
) -> np.ndarray:
    """
    Simulate fantasy points for every player in `df` across `n_trials`
    correlated Monte Carlo trials.

    The correlation knobs (game_total_sd_frac .. shootout_sd) default to a
    fit (`correlations.fit_simulation`) against role-pair correlations of
    real DK scores, 2021-25 regular seasons (data/correlation_profile.json):
    QB1~WR1 +0.38, WR1~WR2 +0.09, RB1~RB2 -0.07, QB1~opp QB1 +0.19.

    Returns
    -------
    scores : np.ndarray, shape (n_trials, len(df))
    """
    rng = np.random.default_rng(seed)
    n_players = len(df)

    games = df.groupby(["team", "opp"]).ngroup()
    game_ids = df.apply(lambda r: tuple(sorted([r["team"], r["opp"]])), axis=1)
    unique_games = game_ids.unique()
    game_index = {g: i for i, g in enumerate(unique_games)}
    game_id_arr = game_ids.map(game_index).to_numpy()
    n_games = len(unique_games)

    # --- 1. Game-level total draws ---
    base_game_total = df.groupby(game_id_arr)["game_total"].first().reindex(range(n_games)).to_numpy()
    game_total_draws = rng.normal(
        loc=base_game_total[None, :],
        scale=(base_game_total * game_total_sd_frac)[None, :],
        size=(n_trials, n_games),
    )

    # --- 2. Team-level offense factor (team share of game total + noise) ---
    team_keys = list(df[["team", "opp"]].drop_duplicates().itertuples(index=False, name=None))
    team_index = {t: i for i, t in enumerate(sorted(set(df["team"])))}
    n_teams = len(team_index)

    team_total_base = df.groupby("team")["team_total"].first()
    team_total_base = team_total_base.reindex(sorted(team_index, key=lambda k: team_index[k])).to_numpy()

    team_game = df.groupby("team").apply(
        lambda g: game_index[tuple(sorted([g.name, g["opp"].iloc[0]]))],
        include_groups=False,
    )
    team_game_arr = team_game.reindex(sorted(team_index, key=lambda k: team_index[k])).to_numpy()

    team_split_noise = rng.normal(0.0, team_split_sd, size=(n_trials, n_teams))
    team_game_total_draw = game_total_draws[:, team_game_arr]
    team_share = team_total_base / base_game_total[team_game_arr]
    team_points_draw = team_game_total_draw * team_share + team_split_noise
    offense_factor = team_points_draw / team_total_base  # ~1.0 centered multiplier

    # --- 3. Leading/trailing game-script signal: the team's simulated margin
    # over its opponent vs the expected margin, per point of team total
    # (falls back to over/under-performance when the opponent isn't in the pool) ---
    team_names = sorted(team_index, key=lambda k: team_index[k])
    opp_of = df.groupby("team")["opp"].first().reindex(team_names)
    opp_idx = np.array([team_index.get(o, -1) for o in opp_of])
    has_opp = opp_idx >= 0
    opp_points = team_points_draw[:, np.clip(opp_idx, 0, None)]
    opp_base = team_total_base[np.clip(opp_idx, 0, None)]
    margin_signal = ((team_points_draw - opp_points) - (team_total_base - opp_base)) / team_total_base
    script_signal = np.where(has_opp[None, :], margin_signal, offense_factor - 1.0)

    player_team_idx = df["team"].map(team_index).to_numpy()
    player_offense_factor = offense_factor[:, player_team_idx]
    player_script_signal = script_signal[:, player_team_idx]

    # --- 4. Opponent DST anti-correlation ---
    player_opp_team_idx = df["opp"].map(lambda t: team_index.get(t, -1)).to_numpy()
    dst_mask = (df["position"] == "DST").to_numpy()
    opp_offense_factor = np.where(
        player_opp_team_idx[None, :] >= 0,
        offense_factor[:, np.clip(player_opp_team_idx, 0, n_teams - 1)],
        1.0,
    )

    # --- 5. Target-share competition among same-team pass catchers ---
    pass_catcher_mask = df["position"].isin(["WR", "TE"]).to_numpy()
    idiosyncratic = rng.normal(0.0, 1.0, size=(n_trials, n_players))
    team_group_key = df["team"].astype(str) + "_" + df["position"].astype(str).where(pass_catcher_mask, "")
    for team in df["team"].unique():
        grp_mask = pass_catcher_mask & (df["team"] == team).to_numpy()
        if grp_mask.sum() > 1:
            grp_mean = idiosyncratic[:, grp_mask].mean(axis=1, keepdims=True)
            idiosyncratic[:, grp_mask] = (
                idiosyncratic[:, grp_mask]
                - target_competition_weight * grp_mean
            )

    # --- 5b. Same-team RBs split carries the same way ---
    rb_pos = (df["position"] == "RB").to_numpy()
    if rb_competition_weight:
        for team in df["team"].unique():
            grp_mask = rb_pos & (df["team"] == team).to_numpy()
            if grp_mask.sum() > 1:
                grp_mean = idiosyncratic[:, grp_mask].mean(axis=1, keepdims=True)
                idiosyncratic[:, grp_mask] -= rb_competition_weight * grp_mean

    # --- 5c. A QB's passing line is his receivers' production: tie his
    # player-level noise to the (sqrt-)projection-weighted noise of his team's
    # pass catchers (`qb_catcher_link` = share of his noise variance) ---
    if qb_catcher_link:
        proj_arr = df["proj"].to_numpy(dtype=float)
        for team in df["team"].unique():
            on_team = (df["team"] == team).to_numpy()
            qbs = np.flatnonzero(on_team & (df["position"] == "QB").to_numpy())
            catchers = on_team & pass_catcher_mask
            if not len(qbs) or not catchers.any():
                continue
            w = np.sqrt(proj_arr[catchers].clip(min=0.1))
            link = idiosyncratic[:, catchers] @ w
            link = (link - link.mean()) / (link.std() or 1.0)
            idiosyncratic[:, qbs] = (np.sqrt(1 - qb_catcher_link) * idiosyncratic[:, qbs]
                                     + np.sqrt(qb_catcher_link) * link[:, None])

    # --- 6. Residual skewed noise ---
    residual = np.column_stack([
        skewed_noise(n_trials, rng) for _ in range(n_players)
    ])

    # --- 7. QB variance scaling ---
    qb_mask = (df["position"] == "QB").to_numpy()
    var_mult = np.ones(n_players)
    if qb_mask.sum() > 1:
        var_mult[qb_mask] = qb_variance_multiplier(df.loc[qb_mask, "salary"]).to_numpy()

    # --- Assemble final scores ---
    proj = df["proj"].to_numpy()
    rb_mask = (df["position"] == "RB").to_numpy()
    pass_mask = df["position"].isin(["QB", "WR", "TE"]).to_numpy()

    game_script_adj = np.zeros((n_trials, n_players))
    game_script_adj[:, rb_mask] = -game_script_weight * player_script_signal[:, rb_mask]
    game_script_adj[:, pass_mask] = game_script_weight * player_script_signal[:, pass_mask]

    # Shootouts: a game-level passing-volume shock shared by both teams'
    # QBs and pass catchers (real QB1~opp QB1 corr ~0.19 while same-team
    # RB1~WR1 is ~0, so it can't all come through the team factor).
    if shootout_sd:
        shootout = rng.normal(0.0, shootout_sd, size=(n_trials, n_games))
        game_script_adj[:, pass_mask] += shootout[:, game_id_arr[pass_mask]]

    mean_component = proj[None, :] * (player_offense_factor + game_script_adj)
    mean_component[:, dst_mask] = proj[None, dst_mask] * (2.0 - opp_offense_factor[:, dst_mask])

    noise_sd = (proj * residual_sd_frac * var_mult)[None, :]
    noise_component = (
        0.5 * idiosyncratic + 0.5 * residual
    ) * noise_sd

    scores = mean_component + noise_component
    if calibrate_ceiling:
        ceiling = (df["ceiling"].to_numpy(dtype=float) if "ceiling" in df.columns
                   else np.full(n_players, np.nan))
        if spread_scale is not None:
            ceiling = fill_default_ceilings(ceiling, proj, df["position"], spread_scale)
        scale, shift = ceiling_calibration(scores, ceiling, proj, ceiling_pct)
        scores = proj[None, :] + shift[None, :] + scale[None, :] * (scores - proj[None, :])

    scores = np.clip(scores, 0.0, None)
    return scores


def ceiling_calibration(
    raw_scores: np.ndarray,
    ceiling: np.ndarray,
    proj: np.ndarray,
    ceiling_pct: float = 0.85,
    calib_trials: int = 4000,
    scale_bounds: tuple[float, float] = (0.2, 6.0),
    n_outer: int = 6,
    n_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-player (scale, shift) such that
        x = max(proj + shift + scale * (raw - proj), 0)
    has its `ceiling_pct` percentile at `ceiling` and its mean at `proj`.

    Alternates a vectorized bisection on `scale` (the percentile is
    monotone in it) with a mean-correcting update of `shift` (needed
    because clipping at 0 lifts the mean of wide distributions). Fit on a
    subsample of trials. Players whose ceiling is missing or not above
    their projection get (1, 0), i.e. the uncalibrated spread.
    """
    dev = raw_scores[:calib_trials] - proj[None, :]
    n_players = dev.shape[1]
    q = ceiling_pct * 100.0
    valid = np.isfinite(ceiling) & (ceiling > proj) & (proj > 0)

    def draw(scale, shift):
        return np.clip(proj[None, :] + shift[None, :] + scale[None, :] * dev, 0.0, None)

    shift = np.zeros(n_players)
    scale = np.ones(n_players)
    for _ in range(n_outer):
        lo = np.full(n_players, scale_bounds[0])
        hi = np.full(n_players, scale_bounds[1])
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            too_high = np.percentile(draw(mid, shift), q, axis=0) > ceiling
            hi = np.where(too_high, mid, hi)
            lo = np.where(too_high, lo, mid)
        scale = 0.5 * (lo + hi)
        shift = shift + (proj - draw(scale, shift).mean(axis=0))

    return np.where(valid, scale, 1.0), np.where(valid, shift, 0.0)
