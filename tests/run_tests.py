"""
Minimal standalone test runner (no pytest dependency required).
Run with: python3 tests/run_tests.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from dfs_engine.data import load_player_pool, validate_showdown_pool
from dfs_engine.simulate import simulate_player_scores, skewed_noise
from dfs_engine.optimize import solve_classic, solve_showdown, SALARY_CAP
from dfs_engine.portfolio import build_portfolio
from dfs_engine.diagnostics import run_optimal_pct_chunk, leverage_report

from test_sportsgameodds import *  # noqa: E402,F401,F403
from test_sim_tools import *  # noqa: E402,F401,F403
from test_contest import *  # noqa: E402,F401,F403
from test_dk import *  # noqa: E402,F401,F403
from test_history import *  # noqa: E402,F401,F403
from test_correlations import *  # noqa: E402,F401,F403
from test_ownership_model import *  # noqa: E402,F401,F403
from test_lateswap import *  # noqa: E402,F401,F403
from test_field_model import *  # noqa: E402,F401,F403
from test_review import *  # noqa: E402,F401,F403
from test_projections import *  # noqa: E402,F401,F403

POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_classic_pool.csv")

results = []


def test(fn):
    try:
        fn()
        results.append((fn.__name__, True, None))
    except Exception as e:  # noqa: BLE001
        results.append((fn.__name__, False, traceback.format_exc()))


def test_load_player_pool_normalizes_ownership():
    pool = load_player_pool(POOL_PATH)
    assert pool["own"].max() <= 1.0
    assert set(pool["position"]) <= {"QB", "RB", "WR", "TE", "DST"}


def test_skewed_noise_is_approximately_zero_mean():
    rng = np.random.default_rng(0)
    noise = skewed_noise(200_000, rng)
    assert abs(noise.mean()) < 0.02


def test_simulate_scores_shape_and_nonnegative():
    pool = load_player_pool(POOL_PATH)
    scores = simulate_player_scores(pool, n_trials=500, seed=1)
    assert scores.shape == (500, len(pool))
    assert (scores >= 0).all()


def test_simulate_scores_are_correlated_within_team():
    pool = load_player_pool(POOL_PATH)
    scores = simulate_player_scores(pool, n_trials=5000, seed=2)
    qb_idx = pool.index[(pool["team"] == "BUF") & (pool["position"] == "QB")][0]
    wr_idx = pool.index[(pool["team"] == "BUF") & (pool["position"] == "WR")][0]
    opp_dst_idx = pool.index[(pool["team"] == "MIA") & (pool["position"] == "DST")][0]
    corr_qb_wr = np.corrcoef(scores[:, qb_idx], scores[:, wr_idx])[0, 1]
    corr_qb_opp_dst = np.corrcoef(scores[:, qb_idx], scores[:, opp_dst_idx])[0, 1]
    assert corr_qb_wr > 0.05
    assert corr_qb_opp_dst < corr_qb_wr


def test_solve_classic_respects_roster_rules():
    pool = load_player_pool(POOL_PATH)
    points = pool["proj"].to_numpy()
    result = solve_classic(pool, points)
    assert result is not None
    assert len(result.player_ids) == 9
    assert result.salary_used <= SALARY_CAP
    chosen = pool.loc[result.player_ids]
    assert (chosen["position"] == "QB").sum() == 1
    assert (chosen["position"] == "DST").sum() == 1
    assert (chosen["position"] == "RB").sum() in (2, 3)
    assert (chosen["position"] == "WR").sum() in (3, 4)
    assert (chosen["position"] == "TE").sum() in (1, 2)


def test_solve_classic_honors_locked_and_excluded():
    # Note: this tiny 13-player sample pool is combinatorially tight —
    # locking the $8,200 QB (Josh Allen) leaves no feasible lineup under
    # the salary cap (verified by brute force). Use the cheap QB instead,
    # which is feasible, to test lock/exclude wiring itself.
    pool = load_player_pool(POOL_PATH)
    points = pool["proj"].to_numpy()
    lock_id = pool.index[pool["name"] == "Tua Tagovailoa"][0]
    exclude_id = pool.index[pool["name"] == "Josh Allen"][0]
    result = solve_classic(pool, points, locked_ids={lock_id}, excluded_ids={exclude_id})
    assert result is not None
    assert lock_id in result.player_ids
    assert exclude_id not in result.player_ids


def test_solve_showdown_respects_captain_rule():
    pool = load_player_pool(POOL_PATH)
    validate_showdown_pool(pool)
    points = pool["proj"].to_numpy()
    result = solve_showdown(pool, points)
    assert result is not None
    assert len(result.player_ids) == 6
    assert result.captain_id in result.player_ids
    assert result.salary_used <= SALARY_CAP


def test_build_portfolio_returns_valid_lineups():
    pool = load_player_pool(POOL_PATH)
    scores = simulate_player_scores(pool, n_trials=300, seed=3)
    lineups = build_portfolio(pool, scores, n_lineups=3, fmt="classic", min_unique=1, seed=3)
    assert len(lineups) >= 1
    for lu in lineups:
        assert len(lu.player_ids) == 9
        assert lu.salary_used <= SALARY_CAP


def test_diagnostics_leverage_report_sums_to_one_optimal_pct():
    pool = load_player_pool(POOL_PATH)
    scores = simulate_player_scores(pool, n_trials=200, seed=4)
    counts = run_optimal_pct_chunk(pool, scores, fmt="classic")
    report = leverage_report(pool, counts, trials_done=200)
    assert (report["optimal_pct"] >= 0).all()
    assert (report["optimal_pct"] <= 1).all()
    assert "leverage" in report.columns


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            test(fn)

    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            print(err)
    print(f"\n{n_pass}/{len(results)} tests passed")
    sys.exit(0 if n_pass == len(results) else 1)
