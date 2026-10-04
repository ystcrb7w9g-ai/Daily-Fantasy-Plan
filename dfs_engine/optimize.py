"""
MILP exact-optimal lineup solver using scipy.optimize.milp (HiGHS).

Supports:
    - Classic: QB(1) RB(2-3) WR(3-4) TE(1-2) FLEX(RB/WR/TE) DST(1), 9 total, $50,000 cap
    - Showdown: 1 Captain (1.5x salary + 1.5x points) + 5 FLEX, $50,000 cap,
      single combined pool across both teams.

Classic lineups can additionally enforce SaberSim/Stokastic-style
stacking rules via `StackRules` (QB stack, bring-back, no offense vs your
DST, max players per team).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, Bounds, milp

SALARY_CAP = 50_000
CLASSIC_ROSTER_SIZE = 9


@dataclass
class StackRules:
    """
    Classic-only correlation/stacking constraints. Every rule is optional.

    qb_stack        : min same-team players from `stack_positions` paired
                      with the chosen QB (e.g. 2 -> "QB + 2").
    stack_positions : positions that count toward the QB stack.
    bring_back      : min players (non-DST) from the QB's opponent.
    max_vs_dst      : max offensive players facing your own DST (0 = none).
    max_per_team    : max players from any single team (DST included).
    """
    qb_stack: int = 0
    stack_positions: tuple[str, ...] = ("WR", "TE")
    bring_back: int = 0
    max_vs_dst: int | None = None
    max_per_team: int | None = None

    def is_active(self) -> bool:
        return bool(self.qb_stack or self.bring_back
                    or self.max_vs_dst is not None or self.max_per_team is not None)


def stack_constraint_rows(df: pd.DataFrame, rules: StackRules) -> tuple[list, list, list]:
    """Linear rows (A, lb, ub) over the per-player binary vector implementing `rules`."""
    n = len(df)
    pos = df["position"].to_numpy()
    team = df["team"].to_numpy()
    opp = df["opp"].to_numpy()
    offense = pos != "DST"
    big = float(CLASSIC_ROSTER_SIZE)
    A_rows, lb, ub = [], [], []

    for q in np.where(pos == "QB")[0]:
        # sum(stackers on QB's team) - k * x_qb >= 0
        if rules.qb_stack:
            row = ((team == team[q]) & np.isin(pos, rules.stack_positions)).astype(float)
            row[q] = -rules.qb_stack
            A_rows.append(row); lb.append(0); ub.append(np.inf)
        # sum(opponent offense) - k * x_qb >= 0
        if rules.bring_back:
            row = ((team == opp[q]) & offense).astype(float)
            row[q] = -rules.bring_back
            A_rows.append(row); lb.append(0); ub.append(np.inf)

    if rules.max_vs_dst is not None:
        # sum(offense facing DST d) + (big - k) * x_d <= big
        for d in np.where(pos == "DST")[0]:
            row = ((team == opp[d]) & offense).astype(float)
            row[d] = big - rules.max_vs_dst
            A_rows.append(row); lb.append(-np.inf); ub.append(big)

    if rules.max_per_team is not None:
        for t in np.unique(team):
            A_rows.append((team == t).astype(float)); lb.append(0); ub.append(rules.max_per_team)

    return A_rows, lb, ub


@dataclass
class LineupResult:
    player_ids: list[int]
    captain_id: int | None  # Showdown only
    salary_used: int
    projected_points: float


def solve_classic(
    df: pd.DataFrame,
    points: np.ndarray,
    locked_ids: set[int] | None = None,
    excluded_ids: set[int] | None = None,
    max_exposure_mask: np.ndarray | None = None,
    stack_rules: StackRules | None = None,
) -> LineupResult | None:
    """
    Solve one optimal Classic DK lineup given a points vector (one
    simulated trial, or projections for a baseline solve).
    """
    n = len(df)
    locked_ids = locked_ids or set()
    excluded_ids = excluded_ids or set()

    c = -points  # milp minimizes; we want to maximize points
    salary = df["salary"].to_numpy()
    pos = df["position"].to_numpy()

    def pos_mask(p):
        return (pos == p).astype(float)

    A_rows = []
    lb = []
    ub = []

    # total roster size == 9
    A_rows.append(np.ones(n)); lb.append(CLASSIC_ROSTER_SIZE); ub.append(CLASSIC_ROSTER_SIZE)
    # salary cap
    A_rows.append(salary.astype(float)); lb.append(0); ub.append(SALARY_CAP)
    # QB exactly 1
    A_rows.append(pos_mask("QB")); lb.append(1); ub.append(1)
    # DST exactly 1
    A_rows.append(pos_mask("DST")); lb.append(1); ub.append(1)
    # RB between 2 and 3 (2 base + up to 1 flex)
    A_rows.append(pos_mask("RB")); lb.append(2); ub.append(3)
    # WR between 3 and 4
    A_rows.append(pos_mask("WR")); lb.append(3); ub.append(4)
    # TE between 1 and 2
    A_rows.append(pos_mask("TE")); lb.append(1); ub.append(2)

    for pid in locked_ids:
        row = np.zeros(n); row[pid] = 1
        A_rows.append(row); lb.append(1); ub.append(1)
    for pid in excluded_ids:
        row = np.zeros(n); row[pid] = 1
        A_rows.append(row); lb.append(0); ub.append(0)

    if stack_rules is not None and stack_rules.is_active():
        s_rows, s_lb, s_ub = stack_constraint_rows(df, stack_rules)
        A_rows += s_rows; lb += s_lb; ub += s_ub

    A = np.vstack(A_rows)
    constraints = LinearConstraint(A, lb, ub)

    ub_var = np.ones(n)
    if max_exposure_mask is not None:
        ub_var = np.where(max_exposure_mask, 1.0, 0.0)
    bounds = Bounds(lb=np.zeros(n), ub=ub_var)
    integrality = np.ones(n)

    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    if not res.success:
        return None

    chosen = np.where(res.x > 0.5)[0].tolist()
    return LineupResult(
        player_ids=chosen,
        captain_id=None,
        salary_used=int(salary[chosen].sum()),
        projected_points=float(points[chosen].sum()),
    )


def solve_showdown(
    df: pd.DataFrame,
    points: np.ndarray,
    locked_ids: set[int] | None = None,
    excluded_ids: set[int] | None = None,
) -> LineupResult | None:
    """
    Showdown solve: each player gets two decision variables, FLEX and
    CAPTAIN (1.5x salary, 1.5x points), mutually exclusive, with exactly
    1 captain + 5 flex chosen from the combined pool.
    """
    n = len(df)
    locked_ids = locked_ids or set()
    excluded_ids = excluded_ids or set()

    salary = df["salary"].to_numpy().astype(float)
    cpt_salary = salary * 1.5
    cpt_points = points * 1.5

    # Variable layout: [flex_0..flex_{n-1}, cpt_0..cpt_{n-1}]
    c = np.concatenate([-points, -cpt_points])

    A_rows, lb, ub = [], [], []

    # total selected == 6 (1 captain + 5 flex)
    row = np.concatenate([np.ones(n), np.ones(n)])
    A_rows.append(row); lb.append(6); ub.append(6)

    # exactly 1 captain
    row = np.concatenate([np.zeros(n), np.ones(n)])
    A_rows.append(row); lb.append(1); ub.append(1)

    # salary cap
    row = np.concatenate([salary, cpt_salary])
    A_rows.append(row); lb.append(0); ub.append(SALARY_CAP)

    # mutual exclusivity: flex_i + cpt_i <= 1 for each player
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = 1
        row[n + i] = 1
        A_rows.append(row); lb.append(0); ub.append(1)

    for pid in locked_ids:
        row = np.zeros(2 * n); row[pid] = 1; row[n + pid] = 1
        A_rows.append(row); lb.append(1); ub.append(1)
    for pid in excluded_ids:
        row = np.zeros(2 * n); row[pid] = 1; row[n + pid] = 1
        A_rows.append(row); lb.append(0); ub.append(0)

    A = np.vstack(A_rows)
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(lb=np.zeros(2 * n), ub=np.ones(2 * n))
    integrality = np.ones(2 * n)

    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
    if not res.success:
        return None

    x = res.x
    flex_chosen = np.where(x[:n] > 0.5)[0].tolist()
    cpt_chosen = np.where(x[n:] > 0.5)[0].tolist()
    captain_id = cpt_chosen[0]
    all_ids = flex_chosen + [captain_id]

    total_salary = int(salary[flex_chosen].sum() + cpt_salary[captain_id])
    total_points = float(points[flex_chosen].sum() + cpt_points[captain_id])

    return LineupResult(
        player_ids=all_ids,
        captain_id=captain_id,
        salary_used=total_salary,
        projected_points=total_points,
    )
