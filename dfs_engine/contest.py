"""
Contest simulation + portfolio selection (SaberSim/Stokastic-style).

Pipeline
--------
1. **Candidates** -- build a large pool of distinct lineups by solving the
   MILP against many simulated outcomes, each blended with the median
   projection by a random amount, so the pool spans "safe" through
   "boom-or-bust" builds. Stacking rules apply.
2. **Field** -- sample opponent lineups from projected ownership
   (weighted sampling per roster slot, last slot restricted to a legal
   salary, QB-teammate stacking boost), then rescale the sampling weights
   a few rounds so the field's realized ownership matches the `own`
   column. Duplication emerges naturally from chalk.
3. **Contest sim** -- on a *separate* set of simulated outcomes (so
   candidates aren't graded on the same trials they were optimized for),
   score every candidate and field lineup, place each candidate against
   the field, and pay it from the contest payout curve. A sampled field of
   F lineups stands in for the full contest: each field lineup represents
   (contest_size - 1) / F real entries, so a placement covers a window of
   ranks and is paid the average prize over it. Ties and duplicates widen
   the window, which splits prizes the way DraftKings does.
4. **Portfolio** -- greedily choose N lineups by expected payout ("roi")
   or by coverage of top-1% finishes ("top1"), under exposure caps and
   a minimum-uniqueness rule.

Known simplifications: the field is ownership-sampled, not a model of
real opponents' skill; our own entries don't compete with each other;
and the payout curve is stylized unless you load the real one with
`load_payout_csv`.
"""
from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .optimize import SALARY_CAP, LineupResult, ShowdownRules, StackRules, solve_classic, solve_showdown
from .portfolio import default_exposure_cap
from .simulate import percentile_mask

CLASSIC_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}  # + 1 FLEX (RB/WR/TE)
FLEX_POSITIONS = ("RB", "WR", "TE")


# ---------------------------------------------------------------------------
# Payouts
# ---------------------------------------------------------------------------

@dataclass
class PayoutCurve:
    """Prize for each finishing rank: payouts[r - 1] is paid to rank r."""
    payouts: np.ndarray
    entry_fee: float

    def __post_init__(self):
        self.payouts = np.asarray(self.payouts, dtype=float)
        self._cum = np.concatenate([[0.0], np.cumsum(self.payouts)])

    @property
    def contest_size(self) -> int:
        return len(self.payouts)

    @property
    def paid_places(self) -> int:
        return int((self.payouts > 0).sum())

    def average_payout(self, start: np.ndarray, length: np.ndarray) -> np.ndarray:
        """Mean prize over the (fractional) rank window [start, start + length)."""
        x = np.arange(len(self._cum), dtype=float)
        a = np.clip(start, 0, self.contest_size)
        b = np.clip(start + length, 0, self.contest_size)
        return (np.interp(b, x, self._cum) - np.interp(a, x, self._cum)) / length


def gpp_payout_curve(
    contest_size: int,
    entry_fee: float,
    rake: float = 0.15,
    paid_frac: float = 0.22,
    first_frac: float = 0.20,
    min_cash_mult: float = 2.0,
    prize_pool: float | None = None,
) -> PayoutCurve:
    """
    Stylized top-heavy GPP curve (Milly Maker-like): the top `paid_frac`
    of entries cash, min-cash is `min_cash_mult` x entry fee, 1st place
    takes `first_frac` of the prize pool, and prizes decay as a power law
    in rank between them. Pass `prize_pool` when it's known (it then
    overrides size x fee x (1 - rake)); freerolls (fee 0) need it.
    """
    pool = prize_pool if prize_pool is not None else contest_size * entry_fee * (1 - rake)
    paid = max(1, int(contest_size * paid_frac))
    # Freerolls have no fee to multiply; give min-cash a quarter of an even split.
    min_cash = min_cash_mult * entry_fee if entry_fee > 0 else 0.25 * pool / paid
    if paid * min_cash >= pool:
        raise ValueError("min-cash x paid places exceeds the prize pool; lower paid_frac/min_cash_mult")
    r = np.arange(1, paid + 1, dtype=float)

    def curve(alpha):
        shape = r ** -alpha - paid ** -alpha
        a = (pool - paid * min_cash) / shape.sum()
        return min_cash + a * shape

    lo, hi = 0.01, 5.0  # bisection on the decay exponent to hit first_frac
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if curve(mid)[0] / pool < first_frac:
            lo = mid
        else:
            hi = mid
    payouts = np.zeros(contest_size)
    payouts[:paid] = curve(0.5 * (lo + hi))
    return PayoutCurve(payouts, entry_fee)


def load_payout_csv(path: str, entry_fee: float, contest_size: int | None = None) -> PayoutCurve:
    """
    Load a real payout table with columns `rank_min, rank_max, payout`
    (one row per prize tier, payout per entry).
    """
    t = pd.read_csv(path)
    t.columns = [c.strip().lower() for c in t.columns]
    size = contest_size or int(t["rank_max"].max())
    payouts = np.zeros(size)
    for row in t.itertuples():
        payouts[int(row.rank_min) - 1:int(row.rank_max)] = float(row.payout)
    return PayoutCurve(payouts, entry_fee)


# ---------------------------------------------------------------------------
# Lineup <-> matrix helpers
# ---------------------------------------------------------------------------

def lineup_key(player_ids, captain_id=None) -> tuple:
    return (captain_id, tuple(sorted(int(p) for p in player_ids)))


def lineups_to_weights(lineups: list[LineupResult], n_players: int) -> np.ndarray:
    """(n_lineups, n_players) scoring weights: 1 per rostered player, 1.5 for a Showdown captain."""
    w = np.zeros((len(lineups), n_players), dtype=np.float32)
    for i, lu in enumerate(lineups):
        w[i, lu.player_ids] = 1.0
        if lu.captain_id is not None:
            w[i, lu.captain_id] = 1.5
    return w


# ---------------------------------------------------------------------------
# 1. Candidate generation
# ---------------------------------------------------------------------------

def flexible_stack_rules(rules: StackRules) -> list[StackRules]:
    """
    Every stack structure up to `rules`' QB stack and bring-back, with its
    caps (max per game / team, max vs DST) kept on all of them. E.g. QB+2
    with a 1-player bring-back expands to QB+1, QB+1+1, QB+2, QB+2+1.
    """
    from dataclasses import replace
    top = max(rules.qb_stack, 1)
    out = []
    for k in range(1, top + 1):
        for b in sorted({0, rules.bring_back}):
            out.append(replace(rules, qb_stack=k, bring_back=b))
    return out


def max_game_stack(df: pd.DataFrame, lineup: LineupResult) -> int:
    """Most players this lineup has from any single game."""
    from .optimize import _game_keys
    _, counts = np.unique(_game_keys(df)[lineup.player_ids], return_counts=True)
    return int(counts.max())


def has_same_team_rbs(df: pd.DataFrame, lineup: LineupResult) -> bool:
    """True if the lineup rosters two or more RBs from one team."""
    d = df.iloc[lineup.player_ids]
    return bool(d.loc[d["position"] == "RB", "team"].duplicated().any())


# Showdown default: at most one DST, nothing else. The correlation rules threw
# out the lineups that won: in the Week 5 SNF Showdown (78k entries) only
# 3.4% of top-0.1% lineups (39% of the top 1%) met all of them -- 72% had a
# QB without one of his own WR/TE (the winner: CPT Huntley with Henry and
# London), 45% two RBs from one team. The sims carry the correlations.
LIGHT_SHOWDOWN_RULES = ShowdownRules(max_dst=1, no_same_team_rbs=False, qb_needs_catcher=False,
                                     catcher_cpt_needs_qb=False)


def soft_showdown_rules(rules: ShowdownRules | None = None) -> list[ShowdownRules]:
    """
    Strict Showdown rules plus a copy allowing same-team RB pairs. Real
    games (data/correlation_profile.json, 2021-25) put RB1/RB2 scores at
    corr -0.07, but in the ~7% of games where an offense scores 5+ TDs
    both RBs reach 15 DK points 18% of the time (vs 4% otherwise) -- the
    Bijan + Brian Robinson Jr. win. Rotating the two keeps those builds in
    the candidate pool; `select_portfolio(max_rb_pair_share=...)` then
    limits how many get played.
    """
    strict = rules or ShowdownRules()
    return [strict, replace(strict, no_same_team_rbs=False)]


# Salary caps for leftover-salary candidates. Real fields spend nearly all of it (Millys: 85% at
# $49,800+; Week 5 SNF Showdown: 57% at $49,500+), so a lineup at $49,000 rarely has twins.
LEFTOVER_CAPS = {"classic": (49_600, 49_200, 48_800, 48_400), "showdown": (49_000, 48_400, 47_800, 47_000)}


def generate_candidates(
    df: pd.DataFrame,
    scores: np.ndarray,
    n_candidates: int,
    fmt: str = "classic",
    stack_rules: StackRules | ShowdownRules | list | None = None,
    mix_range: tuple[float, float] = (0.3, 1.0),
    seed: int | None = None,
    max_solves: int | None = None,
    progress: bool = False,
    locked_ids: set[int] | None = None,
    excluded_ids: set[int] | None = None,
    leftover_share: float = 0.0,
    leftover_caps: tuple[int, ...] | None = None,
    own_penalty_share: float = 0.0,
    own_penalty_max: float = 0.6,
) -> list[LineupResult]:
    """
    Distinct lineups, each optimal for `(1 - m) * proj + m * trial` with a
    random simulated trial and a random mix m ~ U(mix_range). `locked_ids`
    / `excluded_ids` (pool row ids) are forced in / kept out of every one.

    Two optional kinds of contrarian candidates, as the pros build them
    (the contest sim then decides whether they're worth playing):
    - `leftover_share` of the solves cap salary below the max, at one of
      `leftover_caps` (default `LEFTOVER_CAPS`): real fields spend nearly
      all of it, so leaving some is a cheap way to be unique.
    - `own_penalty_share` of the solves score players at
      points x (1 - lam x own), lam ~ U(0, own_penalty_max): leverage
      builds that pay for fading the chalk.

    `stack_rules` may be a list of rule sets (see `flexible_stack_rules`):
    each solve then uses one of them in turn, so the pool spans stack
    structures and the contest sim decides which ones are worth playing.
    For Showdown pass `ShowdownRules` (or a list, e.g. `soft_showdown_rules()`).
    """
    rule_sets = stack_rules if isinstance(stack_rules, list) else [stack_rules]
    rng = np.random.default_rng(seed)
    proj = df["proj"].to_numpy(dtype=float)
    own = pd.to_numeric(df["own"], errors="coerce").fillna(0.0).to_numpy(dtype=float) if "own" in df else np.zeros(len(df))
    own = own / 100.0 if own.max() > 1.5 else own
    if fmt == "showdown":
        cpt = captain_ownership(df)
        own = own + (cpt if cpt is not None else 0.0)  # how often the field rosters him at all
    caps = leftover_caps or LEFTOVER_CAPS[fmt]
    max_solves = max_solves or 3 * n_candidates
    seen: set[tuple] = set()
    out: list[LineupResult] = []
    for solve_i in range(max_solves):
        if len(out) >= n_candidates:
            break
        m = rng.uniform(*mix_range)
        pts = (1 - m) * proj + m * scores[rng.integers(len(scores))]
        cap = caps[rng.integers(len(caps))] if leftover_share and rng.random() < leftover_share else SALARY_CAP
        if own_penalty_share and rng.random() < own_penalty_share:
            pts = pts * (1.0 - rng.uniform(0.0, own_penalty_max) * np.clip(own, 0.0, 1.0))
        if fmt == "classic":
            lu = solve_classic(df, pts, locked_ids=locked_ids, excluded_ids=excluded_ids,
                               stack_rules=rule_sets[solve_i % len(rule_sets)], max_salary=cap)
        else:
            lu = solve_showdown(df, pts, locked_ids=locked_ids, excluded_ids=excluded_ids,
                                rules=rule_sets[solve_i % len(rule_sets)], max_salary=cap)
        if lu is None:
            continue
        key = lineup_key(lu.player_ids, lu.captain_id)
        if key in seen:
            continue
        seen.add(key)
        lu.projected_points = float(proj[lu.player_ids].sum()
                                    + (0.5 * proj[lu.captain_id] if lu.captain_id is not None else 0.0))
        out.append(lu)
        if progress and len(out) % 250 == 0:
            print(f"  {len(out)}/{n_candidates} candidates ({solve_i + 1} solves)")
    return out


def add_locked_candidates(
    df: pd.DataFrame, scores: np.ndarray, candidates: list[LineupResult], player_rows, n_each: int,
    fmt: str = "classic", stack_rules=None, seed: int | None = None, excluded_ids: set[int] | None = None,
) -> int:
    """
    Append up to `n_each` new candidates built around each player in
    `player_rows` (locked in), so a min-exposure target has lineups to pick
    from. Returns how many were added.
    """
    seen = {lineup_key(c.player_ids, c.captain_id) for c in candidates}
    added = 0
    for k, row in enumerate(player_rows):
        new = generate_candidates(df, scores, n_each, fmt=fmt, stack_rules=stack_rules,
                                  seed=None if seed is None else seed + 7919 * (k + 1),
                                  locked_ids={int(row)}, excluded_ids=excluded_ids)
        for lu in new:
            key = lineup_key(lu.player_ids, lu.captain_id)
            if key not in seen:
                seen.add(key)
                candidates.append(lu)
                added += 1
    return added


def satisfies_showdown_rules(df: pd.DataFrame, rows, captain: int, rules: ShowdownRules | None) -> bool:
    """Whether a Showdown lineup (pool rows incl. the captain) meets `rules` (None = always)."""
    if rules is None:
        return True
    d = df.iloc[list(rows)]
    pos, team = d["position"].to_numpy(), d["team"].to_numpy()
    if rules.max_dst is not None and (pos == "DST").sum() > rules.max_dst:
        return False
    if rules.max_k is not None and (pos == "K").sum() > rules.max_k:
        return False
    for t in np.unique(team):
        on = team == t
        if rules.no_same_team_rbs and ((pos == "RB") & on).sum() > 1:
            return False
        has_catcher = (np.isin(pos, ("WR", "TE")) & on).any()
        if rules.qb_needs_catcher and ((pos == "QB") & on).any() and not has_catcher:
            return False
    cp = df.iloc[captain]
    if rules.catcher_cpt_needs_qb and cp["position"] in ("WR", "TE"):
        if (df["position"].eq("QB") & df["team"].eq(cp["team"])).any() and \
                not ((pos == "QB") & (team == cp["team"])).any():
            return False
    return True


def _showdown_from_ids(df, lineups, stack_rules, existing):
    rule_sets = stack_rules if isinstance(stack_rules, list) else [stack_rules]
    flex_of = {str(d): i for i, d in enumerate(df["dk_id"].astype(str))}
    cpt_of = ({str(int(float(d))): i for i, d in enumerate(df["cpt_dk_id"]) if pd.notna(d)}
              if "cpt_dk_id" in df.columns else {})
    proj = df["proj"].to_numpy(dtype=float)
    salary = df["salary"].to_numpy(dtype=int)
    team = df["team"].to_numpy()
    seen = {lineup_key(c.player_ids, c.captain_id) for c in (existing or [])}
    kept, counts = [], {"read": len(lineups), "not_in_pool": 0, "invalid": 0, "breaks_rules": 0, "duplicate": 0}
    for ids in lineups:
        cap = cpt_of.get(str(ids[0]), flex_of.get(str(ids[0])))
        rows = [cap] + [flex_of.get(str(d)) for d in ids[1:]]
        if any(r is None for r in rows):
            counts["not_in_pool"] += 1
            continue
        sal = int(salary[rows].sum() + 0.5 * salary[cap])
        if len(set(rows)) != 6 or sal > SALARY_CAP or len(set(team[rows])) < 2:
            counts["invalid"] += 1
            continue
        if not any(satisfies_showdown_rules(df, rows, cap, r) for r in rule_sets):
            counts["breaks_rules"] += 1
            continue
        key = lineup_key(rows, cap)
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(LineupResult(sorted(rows), cap, sal, float(proj[rows].sum() + 0.5 * proj[cap])))
    counts["kept"] = len(kept)
    return kept, counts


def candidates_from_ids(df: pd.DataFrame, lineups: list[list[str]], stack_rules=None,
                        existing: list[LineupResult] | None = None) -> tuple[list[LineupResult], dict]:
    """
    Lineups given as DK player ids (another optimizer's export, e.g.
    SaberSim's) -> candidates over `df` rows. Kept only if every player is
    in the pool (projected) and the lineup meets `stack_rules` (any one of a
    list); duplicates of `existing` are dropped. Returns (kept, counts).
    Six-id lineups are Showdown: captain first, by its captain id
    (`cpt_dk_id`) or FLEX id.
    """
    if lineups and len(lineups[0]) == 6:
        return _showdown_from_ids(df, lineups, stack_rules, existing)
    from .optimize import satisfies_stack_rules
    rule_sets = stack_rules if isinstance(stack_rules, list) else [stack_rules]
    row_of = {str(d): i for i, d in enumerate(df["dk_id"].astype(str))}
    proj = df["proj"].to_numpy(dtype=float)
    salary = df["salary"].to_numpy(dtype=int)
    seen = {lineup_key(c.player_ids, c.captain_id) for c in (existing or [])}
    pos = df["position"].to_numpy()
    kept, counts = [], {"read": len(lineups), "not_in_pool": 0, "invalid": 0, "breaks_rules": 0, "duplicate": 0}
    for ids in lineups:
        rows = [row_of.get(str(d)) for d in ids]
        if any(r is None for r in rows):
            counts["not_in_pool"] += 1
            continue
        n = {p: int((pos[rows] == p).sum()) for p in ("QB", "RB", "WR", "TE", "DST")}
        if (len(set(rows)) != 9 or salary[rows].sum() > SALARY_CAP or n["QB"] != 1 or n["DST"] != 1
                or not (2 <= n["RB"] <= 3 and 3 <= n["WR"] <= 4 and 1 <= n["TE"] <= 2)):
            counts["invalid"] += 1
            continue
        if not any(satisfies_stack_rules(df, rows, r) for r in rule_sets):
            counts["breaks_rules"] += 1
            continue
        key = lineup_key(rows, None)
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(LineupResult(sorted(rows), None, int(salary[rows].sum()), float(proj[rows].sum())))
    counts["kept"] = len(kept)
    return kept, counts


# ---------------------------------------------------------------------------
# 2. Field generation
# ---------------------------------------------------------------------------

def _gumbel_top_k(logw: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling of k columns without replacement, per row (Gumbel-top-k)."""
    keys = logw + rng.gumbel(size=logw.shape)
    if k == 1:
        return np.argmax(keys, axis=1)[:, None]
    return np.argpartition(-keys, k - 1, axis=1)[:, :k]


def _fill_last(cols, logw_rows, picked, used_salary, salary, min_salary, rng):
    """
    Draw the final roster spot from `cols`, excluding players already
    picked and anyone whose salary would put the lineup outside
    [min_salary, SALARY_CAP]. Returns (player ids, ok mask).
    """
    lw = logw_rows.copy()
    lw[(cols[None, :, None] == picked[:, None, :]).any(axis=2)] = -np.inf
    sal = salary[cols][None, :]
    room_hi = (SALARY_CAP - used_salary)[:, None]
    room_lo = (min_salary - used_salary)[:, None]
    lw[(sal > room_hi) | (sal < room_lo)] = -np.inf
    ok = np.isfinite(lw).any(axis=1)
    lw[~ok] = 0.0  # placeholder row; discarded by the caller
    return cols[_gumbel_top_k(lw, 1, rng)[:, 0]], ok


def _team_arrays(df):
    """Team code, opponent's team code (-1 if not in pool) and position per player."""
    codes, uniq = pd.factorize(df["team"].to_numpy())
    code_of = {t: i for i, t in enumerate(uniq)}
    opp = np.array([code_of.get(o, -1) for o in df["opp"].to_numpy()])
    return codes, opp, df["position"].to_numpy()


def _sample_classic(df, logw, n, rng, stack_boost, min_salary, boosts=None):
    """
    Ownership-weighted classic lineups: (ids (n, 9), captain None, ok mask).
    `boosts` (see `FIELD_STRUCTURE_KEYS`) multiply the sampling weight of
    players that create a structure: a bring-back (QB's opponent), the QB's
    own RB, the DST's own RB, or a player facing the lineup's own DST.
    """
    pos = df["position"].to_numpy()
    team_codes, opp_codes, _ = _team_arrays(df)
    salary = df["salary"].to_numpy(dtype=float)
    idx = {p: np.where(pos == p)[0] for p in CLASSIC_SLOTS}
    lb = {k: np.log(v) for k, v in (boosts or {}).items()}

    qb = idx["QB"][_gumbel_top_k(np.broadcast_to(logw[idx["QB"]], (n, len(idx["QB"]))), 1, rng)[:, 0]]
    qb_team = team_codes[qb][:, None]
    qb_opp = opp_codes[qb][:, None]
    log_boost = np.log(stack_boost)

    def slot_logw(cols, boost, dst_team=None, rb=False):
        lw = np.broadcast_to(logw[cols], (n, len(cols))).copy()
        t = team_codes[cols][None, :]
        if boost:  # QB's own pass catchers get a stacking boost
            lw += np.where(t == qb_team, log_boost, 0.0)
        if lb:
            lw += np.where(t == qb_opp, lb.get("bring_back", 0.0), 0.0)
            if dst_team is not None:
                lw += np.where(opp_codes[cols][None, :] == dst_team, lb.get("vs_dst", 0.0), 0.0)
            if rb is not False:
                rbm = rb[None, :]
                lw += np.where(rbm & (t == qb_team), lb.get("qb_rb", 0.0), 0.0)
                if dst_team is not None:
                    lw += np.where(rbm & (t == dst_team), lb.get("rb_dst", 0.0), 0.0)
        return lw

    dst_cols = idx["DST"]
    dst_lw = np.broadcast_to(logw[dst_cols], (n, len(dst_cols))).copy()
    if lb:  # a DST facing the QB puts the QB against it
        dst_lw += np.where(opp_codes[dst_cols][None, :] == qb_team, lb.get("vs_dst", 0.0), 0.0)
    dst = dst_cols[_gumbel_top_k(dst_lw, 1, rng)[:, 0]]
    dst_team = team_codes[dst][:, None]
    picks = [qb[:, None], dst[:, None]]
    for p, k in (("TE", 1), ("RB", 2), ("WR", 3)):
        cols = idx[p]
        if len(cols) < k:
            raise ValueError(f"Not enough {p} in pool to fill a lineup")
        lw = slot_logw(cols, p in ("WR", "TE"), dst_team, rb=np.full(len(cols), p == "RB"))
        picks.append(cols[_gumbel_top_k(lw, k, rng)])
    picked = np.concatenate(picks, axis=1)

    flex_cols = np.where(np.isin(pos, FLEX_POSITIONS))[0]
    flex_lw = slot_logw(flex_cols, False, dst_team, rb=pos[flex_cols] == "RB")
    flex_lw += np.where(
        (team_codes[flex_cols][None, :] == qb_team) & np.isin(pos[flex_cols], ("WR", "TE"))[None, :],
        log_boost, 0.0,
    )
    flex, ok = _fill_last(flex_cols, flex_lw, picked, salary[picked].sum(axis=1), salary, min_salary, rng)
    return np.concatenate([picked, flex[:, None]], axis=1), None, ok


# Real-field lineup structure (2026 Week 1-5 Millionaire Makers, ~40k lineups
# a week; `field_model.structure_summary`, data/field_structure.json): share
# of lineups with at least one bring-back (player from the QB's opponent), the
# QB's own RB, an RB on the DST's team, and an offensive player facing the
# lineup's own DST, recency-weighted (`field_targets`, half-life 2 weeks):
# fields stack more as the season goes (bring-backs 44/44/41/44/58% in Weeks
# 1-5). Our raw ownership sampler gave 0.24 / 0.10 / 0.08 / 0.22 -- too few
# correlated builds, 3x too many players against their own DST.
# Measured on 12-13 game main slates, so applied only to slates with at
# least STRUCTURE_MIN_GAMES games (on tiny slates you can hardly avoid
# facing your own DST).
DEFAULT_FIELD_STRUCTURE = {"bring_back": 0.483, "qb_rb": 0.164, "rb_dst": 0.164, "vs_dst": 0.073}
FIELD_STRUCTURE_KEYS = tuple(DEFAULT_FIELD_STRUCTURE)
STRUCTURE_MIN_GAMES = 6


def structure_rates(field_w: np.ndarray, df: pd.DataFrame) -> dict[str, float]:
    """Shares of classic field lineups with each `FIELD_STRUCTURE_KEYS` structure (+ 4+ from one game)."""
    codes, opp, pos = _team_arrays(df)
    n_t = codes.max() + 1
    T = np.zeros((len(df), n_t), dtype=np.float32)
    T[np.arange(len(df)), codes] = 1.0
    O = np.zeros((n_t, n_t), dtype=np.float32)
    for t, o in zip(codes, opp):
        if o >= 0:
            O[t, o] = 1.0
    W = (field_w > 0).astype(np.float32)
    by = lambda mask: W[:, mask] @ T[mask]  # noqa: E731  (n, teams) counts
    qb = by(pos == "QB")
    dst = by(pos == "DST")
    off = by(pos != "DST")
    rb = by(pos == "RB")
    game = np.minimum(codes, np.where(opp >= 0, opp, codes))
    G = np.zeros((len(df), game.max() + 1), dtype=np.float32)
    G[np.arange(len(df)), game] = 1.0
    return {
        "bring_back": float(((qb @ O) * off).sum(axis=1).astype(bool).mean()),
        "qb_rb": float((qb * rb).sum(axis=1).astype(bool).mean()),
        "rb_dst": float((dst * rb).sum(axis=1).astype(bool).mean()),
        "vs_dst": float(((dst @ O) * off).sum(axis=1).astype(bool).mean()),
        "game_stack_4plus": float(((W @ G).max(axis=1) >= 4).mean()),
    }


def _sample_showdown(df, logw, n, rng, min_salary, cpt_logw=None):
    """Ownership-weighted showdown lineups: (ids (n, 6), captain ids, ok mask). The captain
    is drawn from `cpt_logw` (captain ownership) when given, else from the FLEX weights."""
    salary = df["salary"].to_numpy(dtype=float)
    all_cols = np.arange(len(df))
    lw = np.broadcast_to(logw, (n, len(df))).copy()
    cpt_lw = lw if cpt_logw is None else np.broadcast_to(cpt_logw, (n, len(df)))
    cpt = _gumbel_top_k(cpt_lw, 1, rng)[:, 0]
    lw[np.arange(n), cpt] = -np.inf
    flex4 = _gumbel_top_k(lw, 4, rng)
    picked = np.concatenate([cpt[:, None], flex4], axis=1)
    used = 1.5 * salary[cpt] + salary[flex4].sum(axis=1)
    last, ok = _fill_last(all_cols, np.broadcast_to(logw, (n, len(df))).copy(),
                          picked, used, salary, min_salary, rng)
    return np.concatenate([picked, last[:, None]], axis=1), cpt, ok


def captain_ownership(df: pd.DataFrame) -> np.ndarray | None:
    """Showdown captain ownership (fraction) from an optional `cpt_own` column, or None.
    With it, `own` is FLEX ownership (summing to 500%) and `cpt_own` sums to 100%."""
    if "cpt_own" not in df.columns:
        return None
    c = pd.to_numeric(df["cpt_own"], errors="coerce")
    if not (c > 0).any():
        return None
    c = c / 100.0 if c.max() > 1.5 else c
    return c.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)


def scale_ownership(df: pd.DataFrame, scales: dict[str, float], column: str = "own") -> np.ndarray:
    """`column` with each listed position's ownership times its factor (e.g. {"QB": 0.85}) and every
    other player rescaled so the column's total stays the same. Meant for Showdown, where no slot is
    tied to a position, so a position's share of the field can move."""
    v = pd.to_numeric(df[column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    f = df["position"].map(scales).fillna(1.0).to_numpy(dtype=float)
    out = v * f
    moved = f != 1.0
    rest = out[~moved].sum()
    if rest > 0:
        out[~moved] *= max(v.sum() - out[moved].sum(), 0.0) / rest
    return out


def field_shares(field_w: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """(share of field lineups rostering each player, share captaining him -- None for Classic)."""
    if not (field_w == 1.5).any():
        return (field_w > 0).mean(axis=0), None
    return (field_w == 1.0).mean(axis=0), (field_w == 1.5).mean(axis=0)


def _ids_to_weights(ids, captain, n_players):
    w = np.zeros((len(ids), n_players), dtype=np.float32)
    w[np.arange(len(ids))[:, None], ids] = 1.0
    if captain is not None:
        w[np.arange(len(ids)), captain] = 1.5
    return w


# Share of field lineups pairing the QB with 0 / 1 / 2+ of his own WR/TE,
# recency-weighted over 2026 Week 1-5 Millionaire Makers (`field_targets`;
# QB+2 was 27/27/31/31/42% by week -- real fields stack deliberately: even
# sampled from *actual* ownership, independent draws give only ~18%).
DEFAULT_STACK_MIX = (0.148, 0.511, 0.341)

# Real Milly lineups spend nearly all the salary (2026 Weeks 4-5: 50% at
# exactly $50,000, 85% at $49,800+, 96% at $49,500+, 99% at $49,000+). Each
# sampled Classic lineup draws its floor from these (floor, share) pairs.
CLASSIC_SALARY_FLOORS = ((50_000, 0.40), (49_800, 0.40), (49_500, 0.15), (49_000, 0.05))
# Showdown fields spend a little less (2026 Week 5 SNF, 78k entries: 6% at
# $50k, 57% at $49.5k+, 76% at $49k+, 93% at $48k+); these floors, with the
# optimizer slice below, reproduce that.
SHOWDOWN_SALARY_FLOORS = ((49_500, 0.33), (49_000, 0.27), (48_000, 0.22), (47_000, 0.10), (45_000, 0.08))

# Optimizer slice of the simulated field: share of entries and projection
# noise chosen so a 165k-entry simulated Milly duplicates like the real
# ones (6.7% of entries in duplicated lineups, top lineup ~60-140 copies,
# vs 5.7-10.2% and 69-346 in 2026 Weeks 1-3; data/field_profile.json).
DEFAULT_OPTIMIZER_SHARE = 0.06
DEFAULT_OPTIMIZER_NOISE = 0.18
# Showdown fields duplicate far more: in the Week 5 SNF Showdown (78k
# entries) 94% of entries shared their exact lineup, the top lineup had 415
# copies, ours 206 and the user's 63. A 25% slice from 2,000 solves (plus
# captain ownership and the floors above) gives 600-680 / ~160 / 61-70
# copies and a collision probability of 5.6e-4 (real 5.3e-4); with no slice
# and captains drawn from FLEX ownership, the field had 22 / 7 / 5.
SHOWDOWN_OPTIMIZER_SHARE = 0.25
SHOWDOWN_OPTIMIZER_SOLVES = 2000
# Share of Showdown lineups with 1 / 2 / 3 / 4 / 5 players from the captain's
# team, mean of the 2026 Week 5 SNF and MNF and Week 6 TNF fields (78-88k
# entries each; 5-1 builds 12.6-20.9%, 2-4 9.5-12.3%). Sampled from ownership
# alone, our fields had 11-13% 5-1 and ~16% 2-4. Matching it (fit on two
# contests, tested on the third) brought the field's copy counts of the
# lineups we'd play closer to the real ones in all three: log-scale error
# 1.07 -> 0.96, rank correlation 0.51 -> 0.56.
SHOWDOWN_CPT_TEAM_MIX = (0.013, 0.111, 0.349, 0.357, 0.170)
# `blend` objective default: weight on the field-free ceiling (how often a
# lineup is the best candidate in a sim) vs the top-1% chance against the field.
DEFAULT_CEILING_WEIGHT = 0.5
# Real Showdown fields pair the two starting QBs less often than independent
# picks would: Week 5 SNF, 15.1% of 78k lineups played both, vs 45.6% x 39.5%
# = 18.0% if independent -- an odds ratio of 0.61 (both x neither over
# one-only x other-only), where our sampled field had ~0.67. The odds ratio
# carries over to QBs owned 70-80%, where "both" can't fall far below
# independence (MNF: 79% and 69% rostered force 48%+ to play both).
SHOWDOWN_QB_PAIR_ODDS = 0.61

# Real Milly field scores at the top 0.1% / top 1% / top 20%, divided by
# the median score: mean of 2026 Weeks 1-5 (week to week 1.62-1.76 /
# 1.47-1.58 / 1.17-1.21). Used to fit the simulation's spread.
HISTORICAL_SCORE_RATIOS = (1.675, 1.514, 1.184)


def field_targets(structure: dict, half_life: float = 2.0) -> tuple[tuple[float, float, float], dict]:
    """
    (stack_mix, structure rates) from a field_structure.json dict, weighting
    week k of n by 0.5 ** ((n - k) / half_life) -- the recent weeks count most.
    """
    weeks = sorted(structure["weeks"], key=lambda k: int(re.sub(r"\D", "", k) or 0))
    w = np.array([0.5 ** ((len(weeks) - 1 - i) / half_life) for i in range(len(weeks))])
    w = w / w.sum()
    rows = [structure["weeks"][k]["all"] for k in weeks]
    q = np.array([r["qb_stack_0_1_2_3plus"] for r in rows])
    mix = (float(w @ q[:, 0]), float(w @ q[:, 1]), float(w @ (q[:, 2] + q[:, 3])))
    rates = {"bring_back": float(w @ np.array([1 - r["bring_back_0_1_2plus"][0] for r in rows])),
             "qb_rb": float(w @ np.array([r["qb_with_own_rb"] for r in rows])),
             "rb_dst": float(w @ np.array([r["rb_with_own_dst"] for r in rows])),
             "vs_dst": float(w @ np.array([r["any_vs_own_dst"] for r in rows]))}
    return mix, rates


def _qb_stack_counts(ids: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Number of the QB's own WR/TE in each classic lineup (QB is column 0 of `ids`)."""
    team_codes = pd.factorize(df["team"].to_numpy())[0]
    catcher = np.isin(df["position"].to_numpy(), ("WR", "TE"))
    same = team_codes[ids[:, 1:]] == team_codes[ids[:, :1]]
    return (same & catcher[ids[:, 1:]]).sum(axis=1)


def _auto_stack_boost(df, own, mix, rng, probe=2048,
                      options=(1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 15.0)) -> float:
    """
    Boost whose raw draws best cover the target stack mix, i.e. maximize
    min_k(raw_share_k / target_k): the scarcest bucket sets how much
    oversampling the resampler needs.
    """
    best, best_cov = options[0], -1.0
    for b in options:
        ids, _, ok = _sample_classic(df, np.log(own), probe, rng, b, 0)
        counts = np.minimum(_qb_stack_counts(ids[ok], df), 2)
        share = np.bincount(counts, minlength=3) / max(len(counts), 1)
        cov = float(np.min(share / np.maximum(mix, 1e-9)))
        if cov > best_cov:
            best, best_cov = b, cov
    return best


def generate_field(
    df: pd.DataFrame,
    n_field: int,
    fmt: str = "classic",
    min_salary: int | None = None,
    stack_mix: tuple[float, float, float] | None = DEFAULT_STACK_MIX,
    stack_boost: float | None = None,
    calibration_rounds: int = 5,
    seed: int | None = None,
    batch: int = 8192,
    optimizer_share: float = 0.0,
    optimizer_solves: int = 500,
    optimizer_noise: float = DEFAULT_OPTIMIZER_NOISE,
    structure: dict[str, float] | None = DEFAULT_FIELD_STRUCTURE,
    qb_pair_odds: float | None = SHOWDOWN_QB_PAIR_ODDS,
    cpt_team_mix: tuple[float, ...] | None = SHOWDOWN_CPT_TEAM_MIX,
) -> np.ndarray:
    """
    Simulated opponent lineups as a (n_field, n_players) weight matrix
    (same layout as `lineups_to_weights`).

    - Classic `structure`: target shares of lineups with a bring-back, the
      QB's own RB, an RB with his own DST, and a player facing the
      lineup's own DST (`DEFAULT_FIELD_STRUCTURE`, measured in real
      Millys). Sampling boosts for those players are calibrated alongside
      ownership so the field matches them; None turns this off. Used on
      slates of `STRUCTURE_MIN_GAMES`+ games.

    - `optimizer_share` of the field is an "optimizer slice": lineups that
      are MILP-optimal for the projections times lognormal noise
      (`optimizer_noise`, standing in for different users' projections),
      solved `optimizer_solves` times, each appearing in proportion to how
      often it came out optimal. Popular builds therefore repeat, which is
      how real fields duplicate (see `DEFAULT_OPTIMIZER_SHARE`). The
      ownership-sampled remainder is calibrated to the ownership left over,
      so the whole field still matches `own`.

    - Every slot is drawn with probability proportional to a per-player
      weight; the last slot is restricted to players that keep the lineup
      between a salary floor and the cap. Floors are drawn per lineup from
      `CLASSIC_SALARY_FLOORS` / `SHOWDOWN_SALARY_FLOORS` (real fields spend
      nearly all of it) unless `min_salary` is given. If almost no sampled
      lineup can reach the floor (tiny pools) it is lowered, with a warning.
    - Showdown captains: with a `cpt_own` column (SaberSim's captain rows)
      captains are drawn from captain ownership and the rest from `own`
      (then FLEX-only, summing to 500%), each calibrated to its own target.
      Without it, captains follow `own` like any other slot.
    - Showdown QB pairing: lineups with both starting QBs are thinned (or
      boosted) until the whole field, optimizer slice included, pairs them
      at the odds ratio `qb_pair_odds` (`SHOWDOWN_QB_PAIR_ODDS`, measured in
      a real Showdown); None = as drawn.
    - Showdown team split: `cpt_team_mix` gives the target share of lineups
      with 1..5 players from the captain's team (`SHOWDOWN_CPT_TEAM_MIX`,
      measured in real Showdowns: they stack the captain's side harder than
      independent draws do). The ownership-sampled lineups are resampled so
      the whole field, optimizer slice included, matches it; lineups from
      one team only (not allowed on DK) are dropped. None = as drawn.
    - Classic stacking: QB teammates' WR/TE get a `stack_boost` sampling
      weight, then lineups are resampled so the shares with 0 / 1 / 2+
      QB-stacked pass catchers match `stack_mix` (None = no resampling).
      `stack_boost=None` auto-picks the boost whose raw draws come closest
      to `stack_mix` on this slate, which keeps the resampling cheap on
      both short and full slates.
    - Finally the weights are rescaled over a few rounds so the field's
      realized ownership matches the `own` column (rescaled to sum to the
      roster size if it doesn't). Field strength therefore
      comes from the ownership projections: chalk tracks value, so a field
      that matches realistic ownership is already near-optimal on median
      projection.
    """
    rng = np.random.default_rng(seed)
    n_players = len(df)
    # Field ownership must sum to the roster size (900% Classic, 600%
    # Showdown); rescale the targets if they don't, then cap each at 95%
    # since nobody can be in more than every lineup. With Showdown captain
    # ownership (`cpt_own`), `own` is FLEX-only (500%) and both are matched:
    # the targets are then [FLEX shares..., captain shares...].
    cpt_own = captain_ownership(df) if fmt == "showdown" else None
    roster = 9 if fmt == "classic" else (5 if cpt_own is not None else 6)
    own = np.clip(df["own"].to_numpy(dtype=float), 1e-4, None)
    if abs(own.sum() / roster - 1.0) > 0.25:
        warnings.warn(f"Projected ownership sums to {own.sum():.0%}, not {roster:.0%}; "
                      "rescaling it for the field.")
    own = np.clip(own * roster / own.sum(), 1e-4, 0.95)
    if cpt_own is not None:
        c = np.clip(cpt_own, 1e-4, None)
        if abs(c.sum() - 1.0) > 0.25:
            warnings.warn(f"Captain ownership sums to {c.sum():.0%}, not 100%; rescaling it for the field.")
        own = np.concatenate([own, np.clip(c / c.sum(), 1e-4, 0.95)])
    salary = df["salary"].to_numpy(dtype=float)

    def realized_shares(w):
        flex, cpt = field_shares(w, df) if cpt_own is not None else ((w > 0).mean(axis=0), None)
        return flex if cpt_own is None else np.concatenate([flex, cpt])

    opt_block = np.zeros((0, n_players), dtype=np.float32)
    if optimizer_share > 0:
        opt_w, opt_p = optimizer_slice(df, fmt, optimizer_solves, optimizer_noise, rng)
        n_opt = int(round(optimizer_share * n_field))
        counts = _largest_remainder(opt_p, n_opt)
        opt_block = np.repeat(opt_w, counts, axis=0)
        # The rest of the field gets whatever ownership the slice didn't use.
        share = len(opt_block) / n_field
        if cpt_own is None:
            opt_own = (opt_p[:, None] * (opt_w > 0)).sum(axis=0)
        else:
            opt_own = np.concatenate([(opt_p[:, None] * (opt_w == 1.0)).sum(axis=0),
                                      (opt_p[:, None] * (opt_w == 1.5)).sum(axis=0)])
        own = np.clip((own - share * opt_own) / (1 - share), 1e-4, 0.95)
        n_field = n_field - len(opt_block)

    # Showdown: each team's top-projected QB, and how the optimizer slice pairs them.
    qb_pair = None
    if fmt == "showdown" and qb_pair_odds is not None:
        qbs = df[(df["position"] == "QB") & (pd.to_numeric(df["proj"], errors="coerce") > 0)]
        qb_rows = [g["proj"].idxmax() for _, g in qbs.groupby("team")]
        if len(qb_rows) == 2:
            qb_pair = np.array([df.index.get_loc(i) for i in qb_rows])
    opt_frac = len(opt_block) / (n_field + len(opt_block))
    if qb_pair is not None and len(opt_block):
        on = opt_w[:, qb_pair] > 0
        opt_pair = (opt_p @ on.all(axis=1), opt_p @ on[:, 0], opt_p @ on[:, 1])
    else:
        opt_pair = (0.0, 0.0, 0.0)
    pair_accept = 1.0

    def both_qbs(ids):
        return (ids == qb_pair[0]).any(axis=1) & (ids == qb_pair[1]).any(axis=1)

    def pair_odds(w):
        """Whole-field odds ratio of rostering both starting QBs."""
        on = w[:, qb_pair] > 0
        both = (1 - opt_frac) * on.all(axis=1).mean() + opt_frac * opt_pair[0]
        r1 = (1 - opt_frac) * on[:, 0].mean() + opt_frac * opt_pair[1]
        r2 = (1 - opt_frac) * on[:, 1].mean() + opt_frac * opt_pair[2]
        return both * max(1 - r1 - r2 + both, 1e-6) / max((r1 - both) * (r2 - both), 1e-6)

    team_codes = pd.factorize(df["team"].to_numpy())[0]
    use_split = fmt == "showdown" and cpt_team_mix is not None
    if use_split:
        split_mix = np.asarray(cpt_team_mix, dtype=float)
        split_mix = split_mix / split_mix.sum()
        if len(opt_block):  # the slice's own split, so the whole field hits the target
            opt_cpt = np.argmax(opt_w == 1.5, axis=1)
            k = ((opt_w > 0) & (team_codes[None, :] == team_codes[opt_cpt][:, None])).sum(axis=1)
            opt_mix = np.bincount(np.clip(k, 1, 5) - 1, weights=opt_p, minlength=5)
            split_mix = np.clip(split_mix - opt_frac * opt_mix, 0.0, None) / (1 - opt_frac)
            split_mix = split_mix / split_mix.sum()

    use_mix = fmt == "classic" and stack_mix is not None
    if use_mix:
        mix = np.asarray(stack_mix, dtype=float)
        mix = mix / mix.sum()

    use_structure = (fmt == "classic" and structure is not None
                     and df["team"].nunique() / 2 >= STRUCTURE_MIN_GAMES)
    boosts = {k: 1.0 for k in structure} if use_structure else None

    def draw(logw, floor):
        if fmt == "classic":
            return _sample_classic(df, logw, batch, rng, stack_boost, floor, boosts)
        if cpt_own is not None:
            return _sample_showdown(df, logw[:n_players], batch, rng, floor, cpt_logw=logw[n_players:])
        return _sample_showdown(df, logw, batch, rng, floor)

    def update_boosts(sample_w):
        rates = structure_rates(sample_w, df)
        for k, target in structure.items():
            boosts[k] = float(np.clip(boosts[k] * (target / max(rates[k], 1e-3)) ** 0.8, 0.02, 50.0))

    if stack_boost is None:
        stack_boost = _auto_stack_boost(df, own, mix, rng) if use_mix else 2.5

    if use_structure:  # warm-start the structure boosts on raw draws
        for _ in range(4):
            ids, _, ok = draw(np.log(own), 0)
            update_boosts(_ids_to_weights(ids[ok], None, n_players))

    floors = None
    if min_salary is None:
        # real fields spend nearly everything: per-lineup floors
        floors = CLASSIC_SALARY_FLOORS if fmt == "classic" else SHOWDOWN_SALARY_FLOORS
        min_salary = min(f for f, _ in floors)
    ids, cpt, ok = draw(np.log(own), 0)
    probe = _ids_to_weights(ids[ok], None if cpt is None else cpt[ok], n_players) @ salary
    if len(probe) and (probe >= min_salary).mean() < 0.01:
        new_floor = int(np.quantile(probe, 0.8) // 100 * 100)
        warnings.warn(f"Few sampled lineups reach ${min_salary:,}; using field min salary ${new_floor:,}.")
        min_salary, floors = new_floor, None
    elif floors is not None and len(probe) and (probe >= floors[0][0] - 1).mean() < 0.01:
        floors = None  # a pool that can't spend to the cap (tiny / synthetic slates): single floor

    def floor_for(n):
        if floors is None:
            return min_salary
        levels = np.array([f for f, _ in floors], dtype=float)
        p = np.array([s for _, s in floors], dtype=float)
        return rng.choice(levels, size=n, p=p / p.sum())

    def sample(weights, n):
        logw = np.log(weights)
        if use_mix:  # buckets sum to exactly n
            need = _largest_remainder(mix, n)
        elif use_split:
            need = _largest_remainder(split_mix, n)
        else:
            need = np.array([n])
        buckets = [[] for _ in need]
        have = np.zeros(len(need), dtype=int)
        for i in range(500):
            if (have >= need).all():
                break
            if i == 40 and have.sum() < 0.05 * need.sum():
                break  # hopeless (salary floor / stack mix unreachable); fail fast
            if use_split and i and i % 60 == 0:  # a split this pool won't reach in time: hand its share to the others
                short = have + have / i * (500 - i) < need
                if short.any() and not short.all():
                    extra = int((need - have)[short].sum())
                    need = np.where(short, have, need)
                    need[~short] += _largest_remainder(split_mix[~short], extra)
            ids, cpt, ok = draw(logw, floor_for(batch))
            if qb_pair is not None and pair_accept != 1.0:
                keep_p = np.where(both_qbs(ids), min(1.0, pair_accept), min(1.0, 1.0 / pair_accept))
                ok &= rng.random(len(ids)) < keep_p
            ids, cpt = ids[ok], (None if cpt is None else cpt[ok])
            if len(ids) == 0:
                continue
            w = _ids_to_weights(ids, cpt, n_players)
            if use_mix:
                bucket = np.minimum(_qb_stack_counts(ids, df), 2)
            elif use_split:  # players from the captain's team, 1..5 -> 0..4 (6 = one team: dropped)
                bucket = (team_codes[ids] == team_codes[ids[:, :1]]).sum(axis=1) - 1
            else:
                bucket = np.zeros(len(w), dtype=int)
            for k in range(len(need)):
                if have[k] < need[k]:
                    sel = w[bucket == k][: need[k] - have[k]]
                    buckets[k].append(sel)
                    have[k] += len(sel)
        if (have < need).any():
            raise RuntimeError(
                "Field sampler can't fill the field (salary floor or stack mix unreachable); "
                "lower min_salary or adjust stack_mix."
            )
        out = np.concatenate([np.concatenate(bk) for bk in buckets if bk])
        return out[rng.permutation(len(out))]

    # Calibration can chase targets the salary floor makes unreachable (a
    # tiny pool's punt plays), so bound each weight to 10x either side of
    # its ownership and stop early if the sampler stops finding lineups.
    weights, last_ok = own.copy(), own.copy()
    for _ in range(calibration_rounds):
        try:
            probe_w = sample(weights, min(n_field, 5000))
        except RuntimeError:
            weights = last_ok
            break
        realized = realized_shares(probe_w)
        if use_structure:
            update_boosts(probe_w)
        if qb_pair is not None:
            pair_accept = float(np.clip(pair_accept * qb_pair_odds / max(pair_odds(probe_w), 1e-3), 0.05, 20.0))
        last_ok = weights
        weights = weights * np.clip(own / np.maximum(realized, 1e-4), 0.2, 5.0) ** 0.8
        weights = np.clip(weights, own / 10.0, own * 10.0)

    for w in (weights, last_ok, own):
        try:
            rest = sample(w, n_field)
            break
        except RuntimeError:
            warnings.warn("Field ownership calibration hit an unsampleable state; backing off.")
    else:
        raise RuntimeError("Field sampler can't fill the field; lower min_salary or adjust stack_mix.")
    if not len(opt_block):
        return rest
    out = np.concatenate([rest, opt_block])
    return out[rng.permutation(len(out))]


FIELD_OWN_DRAWS = 4


def generate_field_mixture(df: pd.DataFrame, n_field: int, own_sd: float, draws: int = FIELD_OWN_DRAWS,
                           seed: int | None = None, **kwargs) -> np.ndarray:
    """
    A field built from `draws` ownership scenarios: each multiplies every
    player's projected ownership (and captain ownership) by lognormal(0,
    `own_sd`) before sampling its share of the lineups. Projections miss
    (real Millys: ~2-3 points per player, more on chalk), so a pick that only
    looks good against the exact projected field is fragile.
    """
    rng = np.random.default_rng(seed)
    parts = []
    sizes = _largest_remainder(np.ones(draws), n_field)
    for j, size in enumerate(sizes):
        d = df.copy()
        d["own"] = pd.to_numeric(d["own"], errors="coerce").fillna(0.0) * np.exp(
            rng.normal(0.0, own_sd, len(d)) - own_sd ** 2 / 2)
        if "cpt_own" in d:
            d["cpt_own"] = pd.to_numeric(d["cpt_own"], errors="coerce").fillna(0.0) * np.exp(
                rng.normal(0.0, own_sd, len(d)) - own_sd ** 2 / 2)
        parts.append(generate_field(d, int(size), seed=int(rng.integers(1 << 31)), **kwargs))
    out = np.concatenate(parts)
    return out[rng.permutation(len(out))]


def _largest_remainder(p: np.ndarray, n: int) -> np.ndarray:
    """Integer counts summing to n, proportional to p (largest-remainder rounding)."""
    raw = p / p.sum() * n
    counts = np.floor(raw).astype(int)
    short = n - counts.sum()
    if short > 0:
        counts[np.argsort(-(raw - counts))[:short]] += 1
    return counts


def optimizer_slice(df: pd.DataFrame, fmt: str, n_solves: int, noise: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Distinct lineups that are optimal for projections x lognormal(0, noise),
    with the share of `n_solves` solves each won: (weights (M, P), probs (M)).
    """
    proj = df["proj"].to_numpy(dtype=float)
    counts: Counter = Counter()
    lineups: dict = {}
    for _ in range(n_solves):
        pts = proj * rng.lognormal(0.0, noise, len(proj))
        lu = solve_classic(df, pts) if fmt == "classic" else solve_showdown(df, pts)
        if lu is None:
            continue
        key = lineup_key(lu.player_ids, lu.captain_id)
        counts[key] += 1
        lineups.setdefault(key, lu)
    if not counts:
        raise RuntimeError("Optimizer slice found no feasible lineups")
    keys = list(counts)
    w = lineups_to_weights([lineups[k] for k in keys], len(df))
    p = np.array([counts[k] for k in keys], dtype=float)
    return w, p / p.sum()


def field_score_ratios(field_w: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Mean over sims of the field's top 0.1% / 1% / 20% score divided by its median."""
    fs = scores.astype(np.float32) @ field_w.T
    q = np.quantile(fs, [0.999, 0.99, 0.8, 0.5], axis=1)
    return (q[:3] / q[3]).mean(axis=1)


def calibrate_spread(
    df: pd.DataFrame,
    field_w: np.ndarray,
    simulate,
    target: tuple[float, float, float] = HISTORICAL_SCORE_RATIOS,
    n_trials: int = 300,
    bounds: tuple[float, float] = (0.2, 2.5),
    n_iter: int = 8,
    seed: int | None = 0,
) -> tuple[float | None, np.ndarray, np.ndarray]:
    """
    Fit `spread_scale` (the default-ceiling width for players without a
    ceiling) so the simulated field's score spread matches real
    Millionaire Makers. `simulate(df, n_trials, seed, spread_scale)` must
    return a score matrix. Returns (spread_scale, ratios at that scale,
    target); spread_scale is None when every player has his own ceiling or
    percentiles (nothing to fit -- the ratios are then just a diagnostic).
    """
    target = np.asarray(target, dtype=float)
    proj = df["proj"].to_numpy(dtype=float)
    ceil = df["ceiling"].to_numpy(dtype=float) if "ceiling" in df.columns else np.full(len(df), np.nan)
    if ((np.isfinite(ceil) & (ceil > proj)) | percentile_mask(df)).all():
        return None, field_score_ratios(field_w, simulate(df, n_trials, seed, None)), target

    def err(k):
        r = field_score_ratios(field_w, simulate(df, n_trials, seed, k))
        return float(np.mean(r / target) - 1.0), r

    lo, hi = bounds
    e_lo, r_lo = err(lo)
    e_hi, r_hi = err(hi)
    if e_lo >= 0:   # even the narrowest default is too wide (user ceilings dominate)
        return lo, r_lo, target
    if e_hi <= 0:
        return hi, r_hi, target
    r_mid = r_lo
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        e_mid, r_mid = err(mid)
        if e_mid < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), r_mid, target


# ---------------------------------------------------------------------------
# 3. Contest simulation
# ---------------------------------------------------------------------------

@dataclass
class ContestResult:
    payouts: np.ndarray      # (n_sims, n_candidates) prize won per sim
    metrics: pd.DataFrame    # per-candidate summary
    top1: np.ndarray         # (n_sims, n_candidates) bool: finished in top 1%
    best: np.ndarray | None = None  # (n_sims, k) the k highest-scoring candidates per sim


@dataclass
class FieldRanks:
    """
    Contest-independent placement of candidates against the sampled field:
    for each sim and candidate, how many field lineups scored strictly
    higher / exactly equal. Computed once, then priced for any number of
    contests (they differ only in size and payouts).
    """
    n_above: np.ndarray      # (n_sims, n_candidates) int32
    n_equal: np.ndarray      # (n_sims, n_candidates) int32
    n_field: int
    cand_mean_pts: np.ndarray
    field_dupe_counts: np.ndarray  # exact copies of each candidate in the field sample
    best: np.ndarray | None = None  # (n_sims, k) int32: the k highest-scoring candidates per sim


def rank_against_field(
    cand_w: np.ndarray, field_w: np.ndarray, scores: np.ndarray, batch: int = 256,
    cand_offset: np.ndarray | None = None, field_offset: np.ndarray | None = None,
    best_k: int = 1,
) -> FieldRanks:
    """
    `cand_offset` / `field_offset` add fixed points per lineup (e.g. late
    swap: points already scored by rostered players outside the pool).
    `best_k`: also record each sim's `best_k` highest-scoring candidates
    (field-free ceiling: how often a lineup is the best one, like
    SaberSim's "Sim Optimals").
    """
    n_sims = len(scores)
    n_cand, n_field = len(cand_w), len(field_w)
    n_above = np.zeros((n_sims, n_cand), dtype=np.int32)
    n_equal = np.zeros((n_sims, n_cand), dtype=np.int32)
    best_k = int(min(max(best_k, 1), n_cand))
    best = np.zeros((n_sims, best_k), dtype=np.int32)
    cand_pts = np.zeros(n_cand)
    for s0 in range(0, n_sims, batch):
        sc = scores[s0:s0 + batch].astype(np.float32)
        b = len(sc)
        # Round so identical lineups tie exactly despite BLAS summation order.
        cs = sc @ cand_w.T
        fs = sc @ field_w.T
        if cand_offset is not None:
            cs = cs + np.asarray(cand_offset, dtype=np.float32)[None, :]
        if field_offset is not None:
            fs = fs + np.asarray(field_offset, dtype=np.float32)[None, :]
        cs = np.round(cs, 2).astype(np.float64)
        fs = np.sort(np.round(fs, 2).astype(np.float64), axis=1)
        # One flattened searchsorted across the batch via per-row offsets.
        offset = (np.arange(b) * (fs.max() + cs.max() + 10.0))[:, None]
        flat = (fs + offset).ravel()
        q = (cs + offset).ravel()
        row_base = np.arange(b)[:, None] * n_field
        lo = np.searchsorted(flat, q, side="left").reshape(b, n_cand) - row_base
        hi = np.searchsorted(flat, q, side="right").reshape(b, n_cand) - row_base
        n_above[s0:s0 + b] = n_field - hi
        n_equal[s0:s0 + b] = hi - lo
        cand_pts += cs.sum(axis=0)
        best[s0:s0 + b] = (np.argmax(cs, axis=1)[:, None] if best_k == 1
                           else np.argpartition(-cs, best_k - 1, axis=1)[:, :best_k])

    field_keys = Counter(map(bytes, (field_w * 2).astype(np.uint8)))
    dupes = np.array([field_keys.get(bytes(r), 0) for r in (cand_w * 2).astype(np.uint8)])
    return FieldRanks(n_above, n_equal, n_field, cand_pts / n_sims, dupes, best)


def score_contest(ranks: FieldRanks, payout: PayoutCurve, top_frac: float = 0.01) -> ContestResult:
    """Price precomputed field placements under one contest's size and payouts.
    `top_frac` sets the finish that counts in `top1` / `top1_pct` (default the top 1%)."""
    scale = (payout.contest_size - 1) / ranks.n_field
    top1_rank = max(1.0, top_frac * payout.contest_size)

    # Each sampled field lineup stands for `scale` real entries, so a
    # candidate with k sampled lineups above it really sits somewhere in
    # ranks [k * scale, (k + 1) * scale); duplicates/ties widen the window
    # further. Pay the average prize across that window (paying the
    # window's best rank would hand out 1st place every time a candidate
    # beats the whole sample).
    start = ranks.n_above * scale
    length = max(scale, 1.0) + ranks.n_equal * scale
    pay = payout.average_payout(start, length).astype(np.float32)
    mid = start + 0.5 * length
    top1 = mid < top1_rank

    mean_pay = pay.mean(axis=0)
    fee = payout.entry_fee
    metrics = pd.DataFrame({
        "sim_mean_pts": ranks.cand_mean_pts,
        "exp_payout": mean_pay,
        "roi": mean_pay / fee - 1.0 if fee > 0 else np.full(len(mean_pay), np.nan),
        "top1_pct": top1.mean(axis=0),
        "cash_pct": (mid < payout.paid_places).mean(axis=0),
        "win_pct": ((ranks.n_above == 0) / length).mean(axis=0),
        "exp_dupes": ranks.field_dupe_counts * scale,
    })
    if ranks.best is not None:  # share of sims in which it is (one of) the best candidate(s)
        metrics["opt_pct"] = np.bincount(ranks.best.ravel(), minlength=len(mean_pay)) / len(ranks.best)
    return ContestResult(pay, metrics, top1, ranks.best)


def simulate_contest(
    cand_w: np.ndarray,
    field_w: np.ndarray,
    scores: np.ndarray,
    payout: PayoutCurve,
    batch: int = 256,
) -> ContestResult:
    """
    Place every candidate against the sampled field in every simulated
    outcome and pay it from `payout`. Returns per-sim payouts plus a
    metrics table (roi, top1_pct, cash_pct, win_pct, exp_dupes...).
    """
    return score_contest(rank_against_field(cand_w, field_w, scores, batch), payout)


# ---------------------------------------------------------------------------
# 4. Portfolio selection
# ---------------------------------------------------------------------------

def player_exposure_limits(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-player (min, max) exposure fractions from optional `min_exp` / `max_exp` pool columns
    (e.g. SaberSim's Min/Max Exp); defaults 0 and 1 mean no limit of the player's own."""
    def col(name, default):
        v = pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)
        v = v / 100.0 if v.max() > 1.0 else v  # given as percentages
        return v.fillna(default).clip(0.0, 1.0).to_numpy(dtype=float)
    return col("min_exp", 0.0), col("max_exp", 1.0)


def _share_count(share: float | None, n_lineups: int) -> int:
    """Lineups a share cap allows. Any positive share allows at least one, like the player caps:
    flooring 0.5 x 1 entry to zero would ban the structure outright in single-entry contests."""
    if share is None or share <= 0:
        return 0
    return max(1, int(np.floor(share * n_lineups)))


def select_portfolio(
    df: pd.DataFrame,
    candidates: list[LineupResult],
    result: ContestResult,
    n_lineups: int,
    objective: str = "roi",
    min_unique: int = 2,
    dst_cap: float = 0.22,
    max_exposure: float = 1.0,
    tiered_caps: bool = True,
    exclude: set[int] | None = None,
    max_game_stack_share: float | None = None,
    game_stack_size: int = 4,
    cheap_salary: float | None = None,
    cheap_cap: float = 0.4,
    max_rb_pair_share: float | None = None,
    ceiling_weight: float = DEFAULT_CEILING_WEIGHT,
    prior_counts: np.ndarray | None = None,
    prior_n: int = 0,
) -> list[int]:
    """
    Greedy portfolio pick; returns candidate indices in selection order.

    objective="roi"  : highest expected payout first.
    objective="top1" : each pick maximizes the number of simulated
                       outcomes in which *some* portfolio lineup finishes
                       top 1% (rewards lineups that win in different
                       worlds), ties broken by expected payout.
    objective="emax" : each pick maximizes E[best payout among the picks]
                       (Bergman et al.: entries should play off each other
                       -- one 1st and one last beats two mid-pack cashes).
    objective="ceiling": each pick maximizes the sims in which some pick is
                       among the best candidates of that sim
                       (`result.best`; field-free, like SaberSim's "Sim
                       Optimals").
    objective="blend": (1 - ceiling_weight) x the top1 gain + ceiling_weight
                       x the ceiling gain, each scaled by its best
                       candidate's, so 0 is "top1" and 1 is "ceiling".
    Exposure caps are the same ownership-tiered caps as `build`, plus a
    global `max_exposure`; lineups must differ from every pick by at least
    `min_unique` players. `tiered_caps=False` drops the ownership tiers
    (keeping `dst_cap` / `max_exposure`); `exclude` skips candidate indices.
    `max_game_stack_share` caps the share of picks with `game_stack_size`+
    players from one game (big game stacks only where they earn a spot).
    Players salaried under `cheap_salary` are capped at `cheap_cap` of the
    picks, so one punt that scores zero can't sink every lineup;
    `max_rb_pair_share` caps the share of picks with two RBs from one team.
    Optional `min_exp` / `max_exp` pool columns set a player's own limits
    (they override the default caps; max 0 bans him). Minimums are filled
    first, each with the best eligible lineup containing the player -- the
    candidate pool must hold enough of them (`add_locked_candidates`).
    `prior_counts` / `prior_n`: lineups already picked for other contests
    (per-player counts, how many); the exposure caps then apply to all of
    them together, so one player can't take over the whole slate's entries
    contest by contest.
    """
    if objective not in ("roi", "top1", "emax", "ceiling", "blend"):
        raise ValueError("objective must be 'roi', 'top1', 'emax', 'ceiling' or 'blend'")
    if objective == "ceiling":
        objective, ceiling_weight = "blend", 1.0
    if objective == "blend" and result.best is None:
        raise ValueError("the ceiling / blend objectives need ContestResult.best (rank_against_field)")
    n_players = len(df)
    is_dst = (df["position"] == "DST").to_numpy()
    tiered = df["own"].apply(default_exposure_cap).to_numpy() if tiered_caps else np.ones(n_players)
    caps = np.minimum(np.where(is_dst, dst_cap, tiered), max_exposure)
    if cheap_salary is not None:
        cheap = df["salary"].to_numpy(dtype=float) < cheap_salary
        caps = np.where(cheap, np.minimum(caps, cheap_cap), caps)
    min_exp, max_exp = player_exposure_limits(df)
    caps = np.where(max_exp < 1.0, max_exp, caps)    # a player's own limit beats the defaults
    caps = np.maximum(caps, min_exp)
    need = np.ceil(min_exp * n_lineups - 1e-9).astype(int)
    # The cap rounds down and the minimum up (60% of 3 = cap 1, need 2): the minimum wins.
    max_count = np.maximum(np.maximum(1, np.floor(caps * n_lineups)), need).astype(int)
    if prior_counts is not None and prior_n > 0:  # caps over every entry so far, this contest's included
        total_cap = np.maximum(1, np.floor(caps * (prior_n + n_lineups))).astype(int)
        max_count = np.minimum(max_count, np.maximum(total_cap - np.asarray(prior_counts, dtype=int), need))
        max_count = np.maximum(max_count, 0)
    max_count[max_exp <= 0] = 0                       # max 0% = banned

    ev = result.payouts.mean(axis=0).astype(np.float64)
    is_big = np.array([max_game_stack(df, c) >= game_stack_size for c in candidates]) \
        if max_game_stack_share is not None else np.zeros(len(candidates), dtype=bool)
    max_big = _share_count(max_game_stack_share, n_lineups)
    n_big = 0
    is_pair = np.array([has_same_team_rbs(df, c) for c in candidates]) \
        if max_rb_pair_share is not None else np.zeros(len(candidates), dtype=bool)
    max_pair = _share_count(max_rb_pair_share, n_lineups)
    n_pair = 0
    sets = [set(c.player_ids) for c in candidates]
    counts = np.zeros(n_players, dtype=int)
    chosen: list[int] = []
    available = np.ones(len(candidates), dtype=bool)
    if exclude:
        available[list(exclude)] = False
    covered = np.zeros(result.top1.shape[0], dtype=np.float32)
    top1_f = result.top1.astype(np.float32) if objective in ("top1", "blend") else None
    best_pay = np.zeros(result.payouts.shape[0], dtype=np.float32)
    w_ceil = float(np.clip(ceiling_weight, 0.0, 1.0)) if objective == "blend" else 0.0
    opt_covered = np.zeros(result.top1.shape[0], dtype=bool)

    def scaled(x):
        top = x[available].max() if available.any() else 0.0
        return x / top if top > 0 else np.zeros_like(x)

    while len(chosen) < n_lineups and available.any():
        if objective == "roi":
            score = ev.copy()
        elif objective == "top1":
            score = (1.0 - covered) @ top1_f + 1e-6 * ev / max(ev.max(), 1e-9)
        elif objective == "blend":
            t_gain = (1.0 - covered) @ top1_f if w_ceil < 1 else np.zeros(len(candidates))
            o_gain = np.bincount(result.best[~opt_covered].ravel(), minlength=len(candidates)).astype(float)
            score = ((1 - w_ceil) * scaled(t_gain) + w_ceil * scaled(o_gain)
                     + 1e-6 * ev / max(ev.max(), 1e-9))
        else:
            gain = np.zeros(len(candidates))
            live = np.flatnonzero(available)
            for b0 in range(0, len(live), 512):
                cols = live[b0:b0 + 512]
                gain[cols] = (np.maximum(result.payouts[:, cols], best_pay[:, None])
                              - best_pay[:, None]).mean(axis=0)
            score = gain + 1e-6 * ev / max(ev.max(), 1e-9)
        score[~available] = -np.inf
        order = np.argsort(-score)

        # Min exposures first: while players are short, the pick is the best lineup with all of
        # them, else with the one furthest short (ties: higher minimum), else any lineup.
        short = need - counts
        passes = [set()]
        if (short > 0).any():
            lead = {int(np.argmax(short + 1e-3 * min_exp))}
            everyone = set(np.flatnonzero(short > 0).tolist())
            passes = ([everyone] if len(everyone) > 1 else []) + [lead, set()]
        picked = None
        for require in passes:
            for idx in order:
                if score[idx] == -np.inf:
                    break
                if not available[idx]:  # ruled out earlier in this pick
                    continue
                if require and not require <= sets[idx]:
                    continue
                ids = candidates[idx].player_ids
                if (is_big[idx] and n_big >= max_big) or (is_pair[idx] and n_pair >= max_pair):
                    available[idx] = False
                    continue
                ok = (counts[ids] < max_count[ids]).all() and all(
                    len(sets[idx] ^ sets[j]) >= 2 * min_unique for j in chosen
                )
                if ok:
                    picked = idx
                    break
                available[idx] = False  # it can only get less eligible from here
            if picked is not None:
                break
        if picked is None:
            break
        chosen.append(int(picked))
        n_big += int(is_big[picked])
        n_pair += int(is_pair[picked])
        available[picked] = False
        counts[candidates[picked].player_ids] += 1
        if objective in ("top1", "blend"):
            covered = np.maximum(covered, top1_f[:, picked])
            if objective == "blend":
                opt_covered |= (result.best == picked).any(axis=1)
        elif objective == "emax":
            best_pay = np.maximum(best_pay, result.payouts[:, picked])
    return chosen


def portfolio_summary(result: ContestResult, chosen: list[int], entry_fee: float) -> dict:
    """Portfolio-level ROI and the chance at least one entry finishes top 1%."""
    if not chosen:
        return {"n_lineups": 0}
    pay = result.payouts[:, chosen].sum(axis=1)
    cost = entry_fee * len(chosen)
    return {
        "n_lineups": len(chosen),
        "exp_payout": float(pay.mean()),
        "exp_profit": float(pay.mean() - cost),
        "roi": float(pay.mean() / cost - 1.0) if cost > 0 else float("nan"),
        "p_any_top1": float(result.top1[:, chosen].any(axis=1).mean()),
        "p_profit": float((pay > cost).mean()),
    }
