import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dfs_engine.data import load_player_pool
from dfs_engine.simulate import simulate_player_scores
from dfs_engine.optimize import solve_classic, StackRules
from dfs_engine.outcomes import player_outcomes
from dfs_engine.portfolio import build_portfolio
from dfs_engine.diagnostics import run_optimal_pct_chunk

POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_classic_pool.csv")


def _multi_game_pool(seed=0):
    """Synthetic 2-game / 4-team pool deep enough for stacking rules."""
    rng = np.random.default_rng(seed)
    games = [("AAA", "BBB", 48.0, -3.0), ("CCC", "DDD", 44.0, 2.5)]
    depth = {"QB": 2, "RB": 3, "WR": 5, "TE": 2, "DST": 1}
    base = {"QB": (5500, 8000, 2.8), "RB": (4000, 8500, 2.2), "WR": (3500, 8500, 2.0),
            "TE": (2800, 6500, 1.8), "DST": (2200, 3600, 2.3)}
    rows = []
    for home, away, total, spread in games:
        for team, opp, sp in ((home, away, spread), (away, home, -spread)):
            tt = total / 2 - sp / 2
            for pos, n in depth.items():
                lo, hi, ppk = base[pos]
                for i in range(n):
                    sal = int(round(rng.uniform(lo, hi) / 100) * 100)
                    proj = round(sal / 1000 * ppk * rng.uniform(0.8, 1.2), 1)
                    rows.append({
                        "name": f"{team}_{pos}{i}", "team": team, "opp": opp, "position": pos,
                        "salary": sal, "proj": proj, "own": round(rng.uniform(0.01, 0.3), 3),
                        "ceiling": round(proj * rng.uniform(1.6, 2.1), 1),
                        "team_total": tt, "game_total": total, "spread": sp,
                    })
    df = pd.DataFrame(rows)
    df["player_id"] = df.index
    return df


# --- ceiling calibration ---------------------------------------------------

def test_ceiling_calibration_hits_ceiling_and_keeps_mean():
    pool = load_player_pool(POOL_PATH)
    scores = simulate_player_scores(pool, n_trials=20000, seed=11)
    p85 = np.percentile(scores, 85, axis=0)
    assert np.allclose(p85 / pool["ceiling"].to_numpy(), 1.0, atol=0.06)
    assert np.allclose(scores.mean(axis=0) / pool["proj"].to_numpy(), 1.0, atol=0.04)


def test_ceiling_calibration_preserves_correlation_structure():
    pool = load_player_pool(POOL_PATH)
    raw = simulate_player_scores(pool, n_trials=20000, seed=12, calibrate_ceiling=False)
    cal = simulate_player_scores(pool, n_trials=20000, seed=12)
    qb, wr = 0, 5  # Josh Allen, Stefon Diggs
    c_raw = np.corrcoef(raw[:, qb], raw[:, wr])[0, 1]
    c_cal = np.corrcoef(cal[:, qb], cal[:, wr])[0, 1]
    assert abs(c_raw - c_cal) < 0.08


def test_ceiling_calibration_skips_missing_ceiling_without_default():
    pool = load_player_pool(POOL_PATH)
    pool.loc[0, "ceiling"] = np.nan
    a = simulate_player_scores(pool, n_trials=3000, seed=13, spread_scale=None)
    b = simulate_player_scores(pool, n_trials=3000, seed=13, calibrate_ceiling=False)
    assert np.allclose(a[:, 0], b[:, 0])


def test_missing_ceiling_gets_position_default():
    from dfs_engine.simulate import BASE_CEILING_MULT
    pool = load_player_pool(POOL_PATH)
    pool["ceiling"] = np.nan
    for scale in (0.5, 1.0):
        sc = simulate_player_scores(pool, n_trials=20000, seed=14, spread_scale=scale)
        expect = pool["proj"] * (1 + scale * (pool["position"].map(BASE_CEILING_MULT) - 1))
        assert np.allclose(np.percentile(sc, 85, axis=0) / expect, 1.0, atol=0.06)


# --- provided percentiles (SaberSim) ---------------------------------------

SKEW = {"p25": 0.50, "p50": 0.88, "p75": 1.40, "p85": 1.74, "p95": 2.40, "p99": 3.25}  # x proj


def _with_percentiles(df, rows=None):
    df = df.copy()
    rows = df.index if rows is None else rows
    for c, m in SKEW.items():
        df[c] = np.nan
        df.loc[rows, c] = df.loc[rows, "proj"] * m
    return df


def test_percentiles_set_the_shape_and_keep_the_mean():
    df = _with_percentiles(_multi_game_pool())
    sc = simulate_player_scores(df, n_trials=20000, seed=21)
    proj = df["proj"].to_numpy()
    for c, m in SKEW.items():
        got = np.percentile(sc, float(c[1:]), axis=0) / (proj * m)
        assert np.median(np.abs(got - 1)) < 0.05, c  # mean rescale nudges each point a little
    assert np.allclose(sc.mean(axis=0) / proj, 1.0, atol=0.01)


def test_percentiles_keep_correlations_and_leave_other_players_alone():
    base = _multi_game_pool()
    qb = base.index[base["name"] == "AAA_QB0"][0]
    wr = base[(base["team"] == "AAA") & (base["position"] == "WR")]["proj"].idxmax()
    other = base.index[base["name"] == "CCC_WR0"][0]
    df = _with_percentiles(base, rows=base.index != other)
    plain = simulate_player_scores(df, n_trials=20000, seed=22, use_percentiles=False)
    mapped = simulate_player_scores(df, n_trials=20000, seed=22)
    rank = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable")  # noqa: E731 (ties at 0)
    assert np.array_equal(rank(plain[:, qb]), rank(mapped[:, qb]))  # same trial order per player
    c0 = np.corrcoef(plain[:, qb], plain[:, wr])[0, 1]
    c1 = np.corrcoef(mapped[:, qb], mapped[:, wr])[0, 1]
    assert c0 > 0.2 and abs(c0 - c1) < 0.06
    assert np.allclose(plain[:, other], mapped[:, other])  # no percentiles -> untouched


# --- shared touchdowns (TD layer) ------------------------------------------

def test_touchdown_points_are_centered_and_shared_by_qb_and_receivers():
    from dfs_engine.simulate import touchdown_points
    df = _multi_game_pool()
    teams = sorted(df["team"].unique())
    idx = df["team"].map({t: i for i, t in enumerate(teams)}).to_numpy()
    team_pts = np.full((20000, len(teams)), 22.5)
    td = touchdown_points(df, team_pts, idx, teams, np.random.default_rng(3))
    assert np.abs(td.mean(axis=0)).max() < 1e-9
    a = df[df["team"] == "AAA"]
    qb = a[a["position"] == "QB"]["proj"].idxmax()
    rec = a.index[a["position"].isin(["WR", "TE"])]
    assert np.corrcoef(td[:, qb], td[:, rec].sum(axis=1))[0, 1] > 0.6  # his TD passes are their catches
    backup = a[(a["position"] == "QB") & (a.index != qb)].index
    assert np.allclose(td[:, backup], 0.0)                                 # backups don't share the TDs
    dst = df.index[df["position"] == "DST"]
    assert np.allclose(td[:, dst], 0.0)


def test_td_model_makes_qb_booms_come_with_receiver_booms():
    from dfs_engine.simulate import TD_MODEL
    df = _multi_game_pool()
    a = df[df["team"] == "AAA"]
    qb = a[a["position"] == "QB"]["proj"].idxmax()
    rec = a.index[a["position"].isin(["WR", "TE"])]

    def qb_boom_with_receiver(sc):
        boom = sc[:, qb] >= np.quantile(sc[:, qb], 0.95)
        hit = np.zeros(len(sc), dtype=bool)
        for j in rec:
            hit |= sc[:, j] >= np.quantile(sc[:, j], 0.90)
        return hit[boom].mean()

    base = simulate_player_scores(df, n_trials=30000, seed=31)
    td = simulate_player_scores(df, n_trials=30000, seed=31, **TD_MODEL)
    assert qb_boom_with_receiver(td) > qb_boom_with_receiver(base) + 0.03
    assert np.allclose(td.mean(axis=0) / df["proj"].to_numpy(), 1.0, atol=0.06)


def test_qb_noise_mult_ties_opposing_qbs_together():
    from dfs_engine.simulate import TD_MODEL
    df = _multi_game_pool()
    qb = df[df["position"] == "QB"].sort_values("proj").groupby("team").tail(1)
    a = qb[qb["team"] == "AAA"].index[0]
    b = qb[qb["team"] == qb.loc[a, "opp"]].index[0]
    corr = lambda sc: np.corrcoef(sc[:, a], sc[:, b])[0, 1]  # noqa: E731
    loose = simulate_player_scores(df, n_trials=30000, seed=32, **dict(TD_MODEL, qb_noise_mult=1.0))
    tight = simulate_player_scores(df, n_trials=30000, seed=32, **TD_MODEL)
    assert TD_MODEL["qb_noise_mult"] < 1 and corr(tight) > corr(loose) + 0.04
    assert np.allclose(tight.mean(axis=0) / df["proj"].to_numpy(), 1.0, atol=0.06)


def test_run_pass_and_dst_knobs_default_off_and_shape_real_patterns():
    from dfs_engine.simulate import TD_MODEL
    df = _multi_game_pool()
    a = df[df["team"] == "AAA"]
    qb = a[a["position"] == "QB"]["proj"].idxmax()
    rb = a[a["position"] == "RB"]["proj"].idxmax()
    opp_dst = df[(df["team"] == a["opp"].iloc[0]) & (df["position"] == "DST")].index[0]
    base = simulate_player_scores(df, n_trials=20000, seed=33, **TD_MODEL)
    same = simulate_player_scores(df, n_trials=20000, seed=33, **dict(TD_MODEL, run_pass_sd=0.0, run_pass_yards=0.0,
                                                                       run_pass_script=0.0, dst_td_cost=0.0))
    assert np.array_equal(base, same)  # all off = the old sim, draw for draw

    def co_boom(sc, i, j, q=0.9):
        return ((sc[:, i] >= np.quantile(sc[:, i], q)) & (sc[:, j] >= np.quantile(sc[:, j], q))).mean() / (1 - q) ** 2
    tilt = simulate_player_scores(df, n_trials=20000, seed=33, **dict(TD_MODEL, run_pass_sd=0.8, run_pass_yards=0.15))
    assert co_boom(tilt, qb, rb) < co_boom(base, qb, rb) - 0.2      # big RB games come with smaller passing games
    dst = simulate_player_scores(df, n_trials=20000, seed=33, **dict(TD_MODEL, dst_td_cost=2.0))
    assert np.corrcoef(dst[:, qb], dst[:, opp_dst])[0, 1] < np.corrcoef(base[:, qb], base[:, opp_dst])[0, 1] - 0.1
    for sc in (tilt, dst):
        assert np.allclose(sc.mean(axis=0) / df["proj"].to_numpy(), 1.0, atol=0.06)


# --- outcomes report -------------------------------------------------------

def test_player_outcomes_columns_and_ranges():
    pool = load_player_pool(POOL_PATH)
    scores = simulate_player_scores(pool, n_trials=3000, seed=14)
    counts = run_optimal_pct_chunk(pool, scores[:100])
    rep = player_outcomes(pool, scores, optimal_counts=counts, optimal_trials=100)
    for col in ("sim_median", "sim_p85", "sim_p99", "boom_pct", "bust_pct", "optimal_pct", "leverage"):
        assert col in rep.columns
    assert rep["boom_pct"].between(0, 1).all() and rep["bust_pct"].between(0, 1).all()
    assert (rep["sim_p99"] >= rep["sim_p85"]).all() and (rep["sim_p85"] >= rep["sim_median"]).all()
    assert rep["leverage"].is_monotonic_decreasing


# --- stacking rules --------------------------------------------------------

def _check_rules(df, ids, rules):
    lu = df.loc[ids]
    qb = lu[lu["position"] == "QB"].iloc[0]
    dst = lu[lu["position"] == "DST"].iloc[0]
    stackers = lu[(lu["team"] == qb["team"]) & lu["position"].isin(rules.stack_positions)]
    assert len(stackers) >= rules.qb_stack
    assert ((lu["team"] == qb["opp"]) & (lu["position"] != "DST")).sum() >= rules.bring_back
    if rules.max_vs_dst is not None:
        assert ((lu["team"] == dst["opp"]) & (lu["position"] != "DST")).sum() <= rules.max_vs_dst
    if rules.max_per_team is not None:
        assert lu["team"].value_counts().max() <= rules.max_per_team


def test_stack_rules_enforced_across_random_trials():
    df = _multi_game_pool()
    rules = StackRules(qb_stack=2, bring_back=1, max_vs_dst=0, max_per_team=4)
    scores = simulate_player_scores(df, n_trials=25, seed=15)
    solved = 0
    for pts in scores:
        res = solve_classic(df, pts, stack_rules=rules)
        assert res is not None
        _check_rules(df, res.player_ids, rules)
        solved += 1
    assert solved == 25


def test_stack_rules_change_the_unconstrained_optimum():
    df = _multi_game_pool()
    pts = df["proj"].to_numpy()
    free = solve_classic(df, pts)
    free_max = df.loc[free.player_ids, "team"].value_counts().max()
    cap = free_max - 1  # force the solver off its unconstrained answer
    capped = solve_classic(df, pts, stack_rules=StackRules(max_per_team=cap))
    assert capped is not None
    assert df.loc[capped.player_ids, "team"].value_counts().max() <= cap
    assert capped.projected_points <= free.projected_points + 1e-9


def test_stack_rules_flow_through_portfolio_builder():
    df = _multi_game_pool()
    rules = StackRules(qb_stack=2, bring_back=1)
    scores = simulate_player_scores(df, n_trials=200, seed=16)
    lineups = build_portfolio(df, scores, n_lineups=5, min_unique=1, seed=16, stack_rules=rules)
    assert len(lineups) == 5
    for lu in lineups:
        _check_rules(df, lu.player_ids, rules)


def test_max_per_game_caps_single_game_stacks():
    from dfs_engine.optimize import _game_keys
    df = _multi_game_pool()
    df["salary"] = 3000
    pts = np.where(df["team"].isin(["AAA", "BBB"]), 30.0, 1.0)
    free = solve_classic(df, pts)
    capped = solve_classic(df, pts, stack_rules=StackRules(max_per_game=5))
    count = lambda lu: np.unique(_game_keys(df)[lu.player_ids], return_counts=True)[1].max()  # noqa: E731
    assert count(free) == 8 and count(capped) == 5
