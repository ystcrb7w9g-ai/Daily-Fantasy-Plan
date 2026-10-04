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
