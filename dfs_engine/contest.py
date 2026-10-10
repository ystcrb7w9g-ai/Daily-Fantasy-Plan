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

import warnings
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .optimize import SALARY_CAP, LineupResult, ShowdownRules, StackRules, solve_classic, solve_showdown
from .portfolio import default_exposure_cap

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
) -> list[LineupResult]:
    """
    Distinct lineups, each optimal for `(1 - m) * proj + m * trial` with a
    random simulated trial and a random mix m ~ U(mix_range).

    `stack_rules` may be a list of rule sets (see `flexible_stack_rules`):
    each solve then uses one of them in turn, so the pool spans stack
    structures and the contest sim decides which ones are worth playing.
    For Showdown pass `ShowdownRules` (or a list, e.g. `soft_showdown_rules()`).
    """
    rule_sets = stack_rules if isinstance(stack_rules, list) else [stack_rules]
    rng = np.random.default_rng(seed)
    proj = df["proj"].to_numpy(dtype=float)
    max_solves = max_solves or 3 * n_candidates
    seen: set[tuple] = set()
    out: list[LineupResult] = []
    for solve_i in range(max_solves):
        if len(out) >= n_candidates:
            break
        m = rng.uniform(*mix_range)
        pts = (1 - m) * proj + m * scores[rng.integers(len(scores))]
        if fmt == "classic":
            lu = solve_classic(df, pts, stack_rules=rule_sets[solve_i % len(rule_sets)])
        else:
            lu = solve_showdown(df, pts, rules=rule_sets[solve_i % len(rule_sets)])
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


def _sample_classic(df, logw, n, rng, stack_boost, min_salary):
    """Ownership-weighted classic lineups: (ids (n, 9), captain None, ok mask)."""
    pos = df["position"].to_numpy()
    team_codes = pd.factorize(df["team"].to_numpy())[0]
    salary = df["salary"].to_numpy(dtype=float)
    idx = {p: np.where(pos == p)[0] for p in CLASSIC_SLOTS}

    qb = idx["QB"][_gumbel_top_k(np.broadcast_to(logw[idx["QB"]], (n, len(idx["QB"]))), 1, rng)[:, 0]]
    qb_team = team_codes[qb][:, None]
    log_boost = np.log(stack_boost)

    def slot_logw(cols, boost):
        lw = np.broadcast_to(logw[cols], (n, len(cols))).copy()
        if boost:  # QB's own pass catchers get a stacking boost
            lw += np.where(team_codes[cols][None, :] == qb_team, log_boost, 0.0)
        return lw

    picks = [qb[:, None]]
    for p, k in (("DST", 1), ("TE", 1), ("RB", 2), ("WR", 3)):
        cols = idx[p]
        if len(cols) < k:
            raise ValueError(f"Not enough {p} in pool to fill a lineup")
        picks.append(cols[_gumbel_top_k(slot_logw(cols, p in ("WR", "TE")), k, rng)])
    picked = np.concatenate(picks, axis=1)

    flex_cols = np.where(np.isin(pos, FLEX_POSITIONS))[0]
    flex_lw = slot_logw(flex_cols, False)
    flex_lw += np.where(
        (team_codes[flex_cols][None, :] == qb_team) & np.isin(pos[flex_cols], ("WR", "TE"))[None, :],
        log_boost, 0.0,
    )
    flex, ok = _fill_last(flex_cols, flex_lw, picked, salary[picked].sum(axis=1), salary, min_salary, rng)
    return np.concatenate([picked, flex[:, None]], axis=1), None, ok


def _sample_showdown(df, logw, n, rng, min_salary):
    """Ownership-weighted showdown lineups: (ids (n, 6), captain ids, ok mask)."""
    salary = df["salary"].to_numpy(dtype=float)
    all_cols = np.arange(len(df))
    lw = np.broadcast_to(logw, (n, len(df))).copy()
    cpt = _gumbel_top_k(lw, 1, rng)[:, 0]
    lw[np.arange(n), cpt] = -np.inf
    flex4 = _gumbel_top_k(lw, 4, rng)
    picked = np.concatenate([cpt[:, None], flex4], axis=1)
    used = 1.5 * salary[cpt] + salary[flex4].sum(axis=1)
    last, ok = _fill_last(all_cols, np.broadcast_to(logw, (n, len(df))).copy(),
                          picked, used, salary, min_salary, rng)
    return np.concatenate([picked, last[:, None]], axis=1), cpt, ok


def _ids_to_weights(ids, captain, n_players):
    w = np.zeros((len(ids), n_players), dtype=np.float32)
    w[np.arange(len(ids))[:, None], ids] = 1.0
    if captain is not None:
        w[np.arange(len(ids)), captain] = 1.5
    return w


# Share of field lineups pairing the QB with 0 / 1 / 2+ of his own WR/TE,
# measured on three 2026 Millionaire Makers (data/field_profile.json).
DEFAULT_STACK_MIX = (0.19, 0.53, 0.28)

# Optimizer slice of the simulated field: share of entries and projection
# noise chosen so a 165k-entry simulated Milly duplicates like the real
# ones (6.7% of entries in duplicated lineups, top lineup ~60-140 copies,
# vs 5.7-10.2% and 69-346 in 2026 Weeks 1-3; data/field_profile.json).
DEFAULT_OPTIMIZER_SHARE = 0.06
DEFAULT_OPTIMIZER_NOISE = 0.18

# Real Milly field scores at the top 0.1% / top 1% / top 20%, divided by
# the median score (same source). Used to fit the simulation's spread.
HISTORICAL_SCORE_RATIOS = (1.664, 1.512, 1.187)


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
) -> np.ndarray:
    """
    Simulated opponent lineups as a (n_field, n_players) weight matrix
    (same layout as `lineups_to_weights`).

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
      between `min_salary` and the cap. `min_salary` defaults to $49,000
      (Classic) / $47,000 (Showdown); if almost no sampled lineup can reach
      it (tiny pools) it is lowered, with a warning.
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
    # since nobody can be in more than every lineup.
    roster = 9 if fmt == "classic" else 6
    own = np.clip(df["own"].to_numpy(dtype=float), 1e-4, None)
    if abs(own.sum() / roster - 1.0) > 0.25:
        warnings.warn(f"Projected ownership sums to {own.sum():.0%}, not {roster:.0%}; "
                      "rescaling it for the field.")
    own = np.clip(own * roster / own.sum(), 1e-4, 0.95)
    salary = df["salary"].to_numpy(dtype=float)

    opt_block = np.zeros((0, n_players), dtype=np.float32)
    if optimizer_share > 0:
        opt_w, opt_p = optimizer_slice(df, fmt, optimizer_solves, optimizer_noise, rng)
        n_opt = int(round(optimizer_share * n_field))
        counts = _largest_remainder(opt_p, n_opt)
        opt_block = np.repeat(opt_w, counts, axis=0)
        # The rest of the field gets whatever ownership the slice didn't use.
        share = len(opt_block) / n_field
        opt_own = (opt_p[:, None] * (opt_w > 0)).sum(axis=0)
        own = np.clip((own - share * opt_own) / (1 - share), 1e-4, 0.95)
        n_field = n_field - len(opt_block)

    use_mix = fmt == "classic" and stack_mix is not None
    if use_mix:
        mix = np.asarray(stack_mix, dtype=float)
        mix = mix / mix.sum()

    def draw(logw, floor):
        if fmt == "classic":
            return _sample_classic(df, logw, batch, rng, stack_boost, floor)
        return _sample_showdown(df, logw, batch, rng, floor)

    if stack_boost is None:
        stack_boost = _auto_stack_boost(df, own, mix, rng) if use_mix else 2.5

    if min_salary is None:
        min_salary = 49_000 if fmt == "classic" else 47_000
    ids, cpt, ok = draw(np.log(own), 0)
    probe = _ids_to_weights(ids[ok], None if cpt is None else cpt[ok], n_players) @ salary
    if len(probe) and (probe >= min_salary).mean() < 0.01:
        new_floor = int(np.quantile(probe, 0.8) // 100 * 100)
        warnings.warn(f"Few sampled lineups reach ${min_salary:,}; using field min salary ${new_floor:,}.")
        min_salary = new_floor

    def sample(weights, n):
        logw = np.log(weights)
        need = np.round(mix * n).astype(int) if use_mix else np.array([n])
        buckets = [[] for _ in need]
        have = np.zeros(len(need), dtype=int)
        for i in range(500):
            if (have >= need).all():
                break
            if i == 40 and have.sum() < 0.05 * need.sum():
                break  # hopeless (salary floor / stack mix unreachable); fail fast
            ids, cpt, ok = draw(logw, min_salary)
            ids, cpt = ids[ok], (None if cpt is None else cpt[ok])
            if len(ids) == 0:
                continue
            w = _ids_to_weights(ids, cpt, n_players)
            bucket = np.minimum(_qb_stack_counts(ids, df), 2) if use_mix else np.zeros(len(w), dtype=int)
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
            realized = (sample(weights, min(n_field, 5000)) > 0).mean(axis=0)
        except RuntimeError:
            weights = last_ok
            break
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
    target); spread_scale is None when every player has his own ceiling
    (nothing to fit -- the ratios are then just a diagnostic).
    """
    target = np.asarray(target, dtype=float)
    proj = df["proj"].to_numpy(dtype=float)
    ceil = df["ceiling"].to_numpy(dtype=float) if "ceiling" in df.columns else np.full(len(df), np.nan)
    if (np.isfinite(ceil) & (ceil > proj)).all():
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


def rank_against_field(
    cand_w: np.ndarray, field_w: np.ndarray, scores: np.ndarray, batch: int = 256,
    cand_offset: np.ndarray | None = None, field_offset: np.ndarray | None = None,
) -> FieldRanks:
    """
    `cand_offset` / `field_offset` add fixed points per lineup (e.g. late
    swap: points already scored by rostered players outside the pool).
    """
    n_sims = len(scores)
    n_cand, n_field = len(cand_w), len(field_w)
    n_above = np.zeros((n_sims, n_cand), dtype=np.int32)
    n_equal = np.zeros((n_sims, n_cand), dtype=np.int32)
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

    field_keys = Counter(map(bytes, (field_w * 2).astype(np.uint8)))
    dupes = np.array([field_keys.get(bytes(r), 0) for r in (cand_w * 2).astype(np.uint8)])
    return FieldRanks(n_above, n_equal, n_field, cand_pts / n_sims, dupes)


def score_contest(ranks: FieldRanks, payout: PayoutCurve) -> ContestResult:
    """Price precomputed field placements under one contest's size and payouts."""
    scale = (payout.contest_size - 1) / ranks.n_field
    top1_rank = max(1.0, 0.01 * payout.contest_size)

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
    return ContestResult(pay, metrics, top1)


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
) -> list[int]:
    """
    Greedy portfolio pick; returns candidate indices in selection order.

    objective="roi"  : highest expected payout first.
    objective="top1" : each pick maximizes the number of simulated
                       outcomes in which *some* portfolio lineup finishes
                       top 1% (rewards lineups that win in different
                       worlds), ties broken by expected payout.
    Exposure caps are the same ownership-tiered caps as `build`, plus a
    global `max_exposure`; lineups must differ from every pick by at least
    `min_unique` players. `tiered_caps=False` drops the ownership tiers
    (keeping `dst_cap` / `max_exposure`); `exclude` skips candidate indices.
    `max_game_stack_share` caps the share of picks with `game_stack_size`+
    players from one game (big game stacks only where they earn a spot).
    Players salaried under `cheap_salary` are capped at `cheap_cap` of the
    picks, so one punt that scores zero can't sink every lineup;
    `max_rb_pair_share` caps the share of picks with two RBs from one team.
    """
    if objective not in ("roi", "top1"):
        raise ValueError("objective must be 'roi' or 'top1'")
    n_players = len(df)
    is_dst = (df["position"] == "DST").to_numpy()
    tiered = df["own"].apply(default_exposure_cap).to_numpy() if tiered_caps else np.ones(n_players)
    caps = np.minimum(np.where(is_dst, dst_cap, tiered), max_exposure)
    if cheap_salary is not None:
        cheap = df["salary"].to_numpy(dtype=float) < cheap_salary
        caps = np.where(cheap, np.minimum(caps, cheap_cap), caps)
    max_count = np.maximum(1, np.floor(caps * n_lineups)).astype(int)

    ev = result.payouts.mean(axis=0).astype(np.float64)
    is_big = np.array([max_game_stack(df, c) >= game_stack_size for c in candidates]) \
        if max_game_stack_share is not None else np.zeros(len(candidates), dtype=bool)
    max_big = int(np.floor(max_game_stack_share * n_lineups)) if max_game_stack_share is not None else 0
    n_big = 0
    is_pair = np.array([has_same_team_rbs(df, c) for c in candidates]) \
        if max_rb_pair_share is not None else np.zeros(len(candidates), dtype=bool)
    max_pair = int(np.floor(max_rb_pair_share * n_lineups)) if max_rb_pair_share is not None else 0
    n_pair = 0
    sets = [set(c.player_ids) for c in candidates]
    counts = np.zeros(n_players, dtype=int)
    chosen: list[int] = []
    available = np.ones(len(candidates), dtype=bool)
    if exclude:
        available[list(exclude)] = False
    covered = np.zeros(result.top1.shape[0], dtype=np.float32)
    top1_f = result.top1.astype(np.float32) if objective == "top1" else None

    while len(chosen) < n_lineups and available.any():
        if objective == "roi":
            score = ev.copy()
        else:
            score = (1.0 - covered) @ top1_f + 1e-6 * ev / max(ev.max(), 1e-9)
        score[~available] = -np.inf

        picked = None
        for idx in np.argsort(-score):
            if not available[idx]:
                break
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
        if picked is None:
            break
        chosen.append(int(picked))
        n_big += int(is_big[picked])
        n_pair += int(is_pair[picked])
        available[picked] = False
        counts[candidates[picked].player_ids] += 1
        if objective == "top1":
            covered = np.maximum(covered, top1_f[:, picked])
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
