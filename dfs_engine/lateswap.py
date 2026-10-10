"""
Late swap: once the early games are under way, re-optimize the slots of
each entry whose players haven't kicked off yet.

What we know mid-slate that nobody knew at the first lock:

* every locked player's points so far (DK contest-standings export, FPTS);
* the field's *actual* ownership (%Drafted), not a projection;
* every opponent's lineup (the same export), so we can rank against the
  real contest instead of a sampled one.

Each entry keeps its locked players. For the open slots we generate
completions (MILP-optimal for a blend of projection and one simulated
outcome of the late games, like `contest.generate_candidates`), always
including "keep the current lineup", and score every completion against
the field: locked players count their actual points plus, for games still
in progress, a simulated remainder; late players are simulated. Each
entry gets the completion with the best expected payout (or top-1% odds).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from . import contest as cs
from .history import parse_lineup, player_table
from .optimize import SALARY_CAP
from .sportsgameodds import normalize_name

SLOT_OK = {"QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "FLEX": {"RB", "WR", "TE"},
           "DST": {"DST"}}
GAME_MINUTES = 195  # kickoff to final whistle, roughly
_ID_RE = re.compile(r"\((\d{5,})\)")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_current_lineups(path: str, slots: list[str]) -> pd.DataFrame:
    """
    Current lineups from a DK entry file downloaded mid-slate: one row per
    entry with `slot_0..slot_8` DK IDs and `locked_0..locked_8` flags (DK
    marks started players "(LOCKED)"). Cells may be "Name (ID)" or a bare ID.
    """
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if not r or not r[0].strip().isdigit():
                continue
            row = {"entry_id": r[0].strip(), "contest_id": r[2].strip()}
            for k, cell in enumerate(r[4:4 + len(slots)]):
                cell = cell.strip()
                m = _ID_RE.search(cell)
                row[f"slot_{k}"] = m.group(1) if m else (cell if cell.isdigit() else None)
                row[f"locked_{k}"] = "LOCKED" in cell.upper()
            rows.append(row)
    return pd.DataFrame(rows)


def add_rostered_players(pool: pd.DataFrame, current: pd.DataFrame, dk_players: pd.DataFrame) -> pd.DataFrame:
    """
    Add players on our current lineups who aren't in the projected pool
    (e.g. a DST the projection source missed), projected at their DK
    season average, so "keep the current lineup" can always be scored.
    They are marked `keep_only`: never swapped *in*, since nothing but
    FPPG backs their projection.
    """
    have = set(pool["dk_id"].astype(str))
    ids = {str(v) for c in current.columns if c.startswith("slot_") for v in current[c].dropna()}
    add = dk_players[dk_players["dk_id"].astype(str).isin(ids - have)].copy()
    if add.empty:
        return pool
    lines = pool.drop_duplicates("team").set_index("team")
    rows = []
    for r in add.itertuples(index=False):
        row = {"name": r.name, "team": r.team, "opp": r.opp, "position": r.position, "salary": r.salary,
               "proj": float(r.avg_ppg) if pd.notna(r.avg_ppg) and r.avg_ppg > 0 else 1.0, "own": 0.0,
               "dk_id": str(r.dk_id), "game_time": r.game_time, "keep_only": True}
        for c in ("team_total", "game_total", "spread"):
            if r.team in lines.index:
                row[c] = lines.at[r.team, c]
            elif r.opp in lines.index:  # mirror the opponent's line
                row[c] = (lines.at[r.opp, "game_total"] - lines.at[r.opp, "team_total"] if c == "team_total"
                          else -lines.at[r.opp, "spread"] if c == "spread" else lines.at[r.opp, c])
        rows.append(row)
    out = pd.concat([pool.assign(keep_only=False), pd.DataFrame(rows)], ignore_index=True)
    out["keep_only"] = out["keep_only"].fillna(False).astype(bool)
    return out


def _key(name: str, position: str | None = None) -> str:
    k = normalize_name(name)
    return "DST|" + k.split()[-1] if str(position).upper() == "DST" and k else k


def remaining_fraction(game_time: pd.Series, now: pd.Timestamp, assume_final: bool = False) -> np.ndarray:
    """Share of each player's game still to play: 1 before kickoff, 0 once it's (assumed) over."""
    t = pd.to_datetime(game_time, errors="coerce")
    elapsed = (now - t).dt.total_seconds().to_numpy() / 60.0
    frac = np.clip(1.0 - elapsed / GAME_MINUTES, 0.0, 1.0)
    frac = np.where(np.isnan(elapsed), 1.0, frac)
    if assume_final:
        frac = np.where(elapsed >= 0, 0.0, frac)
    return frac


@dataclass
class LiveSlate:
    pool: pd.DataFrame          # projected players (with dk_id, game_time); `own` = actual %Drafted when known
    started: np.ndarray         # bool per pool player: game has kicked off
    remaining: np.ndarray       # share of game left per pool player
    actual: np.ndarray          # points so far per pool player (0 if not started / unknown)
    points_by_key: dict         # every drafted player's points so far, by name key (incl. non-pool players)
    unknown_started: list       # started pool players we have no points for


def live_slate(pool: pd.DataFrame, standings: list[pd.DataFrame], now: pd.Timestamp,
               assume_final: bool = False) -> LiveSlate:
    pool = pool.copy()
    remaining = remaining_fraction(pool["game_time"], now, assume_final)
    started = remaining < 1.0
    pts, own = {}, {}
    for raw in standings:
        t = player_table(raw)
        for r in t.itertuples(index=False):
            k = _key(r.name, r.position)
            if pd.notna(r.fpts):
                pts[k] = float(r.fpts)
            own[k] = max(own.get(k, 0.0), float(r.own))
    keys = [_key(n, p) for n, p in zip(pool["name"], pool["position"])]
    actual = np.array([pts.get(k, 0.0) for k in keys]) * started
    unknown = [n for n, k, s in zip(pool["name"], keys, started) if s and k not in pts]
    if own:
        pool["own"] = [own.get(k, 0.0) for k in keys]
    return LiveSlate(pool, started, remaining, actual, pts, unknown)


def live_scores(slate: LiveSlate, sim: np.ndarray) -> np.ndarray:
    """Full-game sims -> live outcomes: points so far + the simulated share of the game still to play."""
    return slate.actual[None, :] + slate.remaining[None, :] * sim


def field_from_standings(raw: pd.DataFrame, slate: LiveSlate, exclude_entry_ids: set[str],
                         max_lineups: int = 30_000, seed: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Opponents' lineups as weight rows over the pool, plus a per-lineup
    offset for rostered players outside the pool (their points so far).
    Slots the export doesn't show are filled with an unstarted player drawn
    by actual ownership. Returns (weights, offsets, n_real_entries).
    """
    rng = np.random.default_rng(seed)
    e = raw[raw["EntryId"].notna() & raw["Lineup"].notna()]
    e = e[~e["EntryId"].astype("int64").astype(str).isin(exclude_entry_ids)]
    n_real = len(e)
    if n_real > max_lineups:
        e = e.sample(max_lineups, random_state=int(rng.integers(1 << 31)))
    pool = slate.pool
    index = {_key(n, p): i for i, (n, p) in enumerate(zip(pool["name"], pool["position"]))}
    pos = pool["position"].to_numpy()
    open_by_pos = {p: np.flatnonzero((pos == p) & ~slate.started) for p in SLOT_OK if p != "FLEX"}
    W = np.zeros((len(e), len(pool)), dtype=np.float32)
    off = np.zeros(len(e))
    for i, text in enumerate(e["Lineup"].astype(str)):
        lu = parse_lineup(text)
        for slot, name in lu:
            j = index.get(_key(name, slot))
            if j is not None:
                W[i, j] = 1.0
            else:
                off[i] += slate.points_by_key.get(_key(name, slot), 0.0)
        for slot in _missing_slots([s for s, _ in lu]):
            choices = np.concatenate([open_by_pos[p] for p in SLOT_OK[slot]])
            choices = choices[W[i, choices] == 0]
            if len(choices):
                w = pool["own"].to_numpy(dtype=float)[choices] + 1e-4
                W[i, rng.choice(choices, p=w / w.sum())] = 1.0
    return W, off, n_real


def _missing_slots(shown: list[str]) -> list[str]:
    need = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
    for s in shown:
        if s in need:
            need.remove(s)
    return need


# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------

def complete_lineup(pool: pd.DataFrame, points: np.ndarray, fixed: list[int], open_slots: list[str],
                    open_players: np.ndarray, salary_cap: int = SALARY_CAP) -> list[int] | None:
    """Best players for `open_slots` (pool indices, in slot order) given the `fixed` ones; None if infeasible."""
    if not open_slots:
        return []
    pos = pool["position"].to_numpy()
    salary = pool["salary"].to_numpy(dtype=float)
    cand = [j for j in open_players if j not in set(fixed)]
    pairs = [(j, s) for s, slot in enumerate(open_slots) for j in cand if pos[j] in SLOT_OK[slot]]
    if not pairs:
        return None
    n = len(pairs)
    c = -np.array([points[j] for j, _ in pairs])
    rows, lb, ub = [], [], []
    for s in range(len(open_slots)):
        rows.append([1.0 if ps == s else 0.0 for _, ps in pairs]); lb.append(1); ub.append(1)
    for j in cand:
        r = [1.0 if pj == j else 0.0 for pj, _ in pairs]
        if sum(r) > 1:
            rows.append(r); lb.append(0); ub.append(1)
    rows.append([salary[j] for j, _ in pairs]); lb.append(0); ub.append(salary_cap - salary[fixed].sum())
    res = milp(c, constraints=LinearConstraint(np.array(rows), lb, ub), integrality=np.ones(n),
               bounds=Bounds(0, 1))
    if res.x is None:
        return None
    pick = {ps: pj for (pj, ps), x in zip(pairs, res.x) if x > 0.5}
    return [pick[s] for s in range(len(open_slots))]


@dataclass
class EntryPlan:
    entry_id: str
    contest_id: str
    dk_ids: list[str | None]    # current DK ID per slot
    locked: list[bool]
    options: list[list]         # full lineups: pool index per slot (None only for a locked non-pool
                                # player); options[0] is the current lineup when `keeps_current`
    keeps_current: bool
    offset: float               # points so far of locked players outside the pool


def plan_entries(current: pd.DataFrame, slate: LiveSlate, gen_scores: np.ndarray,
                 dk_players: pd.DataFrame | None = None, n_solves: int = 40,
                 mix: tuple[float, float] = (0.3, 1.0), seed: int | None = None) -> list[EntryPlan]:
    """
    Options for every entry: the current lineup plus distinct completions of
    its open slots. `dk_players` (the entry file's player list) supplies
    name/salary for rostered players outside the projected pool.
    """
    pool = slate.pool
    rng = np.random.default_rng(seed)
    by_id = {str(d): i for i, d in enumerate(pool["dk_id"].astype("int64").astype(str))}
    info = dk_players.set_index("dk_id") if dk_players is not None else None
    proj = pool["proj"].to_numpy(dtype=float)
    keep_only = pool["keep_only"].to_numpy(bool) if "keep_only" in pool else np.zeros(len(pool), bool)
    open_players = np.flatnonzero(~slate.started & ~keep_only)
    plans = []
    for r in current.itertuples(index=False):
        d = r._asdict()
        ids = [d[f"slot_{k}"] for k in range(9)]
        idx = [by_id.get(str(i)) if i else None for i in ids]
        # A slot is fixed if DK says LOCKED or its player's game has started.
        locked = [bool(d[f"locked_{k}"]) or (idx[k] is not None and bool(slate.started[idx[k]]))
                  for k in range(9)]
        fixed = [idx[k] for k in range(9) if locked[k] and idx[k] is not None]
        offset, extra_salary = 0.0, 0.0
        for k in range(9):
            if locked[k] and idx[k] is None and ids[k] and info is not None and ids[k] in info.index:
                row = info.loc[ids[k]]
                offset += slate.points_by_key.get(_key(row["name"], row["position"]), 0.0)
                extra_salary += float(row["salary"])
        open_k = [k for k in range(9) if not locked[k]]
        options, seen = [], set()
        keeps = all(i is not None or locked[k] for k, i in enumerate(idx))
        if keeps:
            options.append(list(idx)); seen.add(tuple(idx))
        for _ in range(n_solves if open_k else 0):
            m = rng.uniform(*mix)
            pts = (1 - m) * proj + m * gen_scores[rng.integers(len(gen_scores))]
            fill = complete_lineup(pool, pts, fixed, [dk_slot(k) for k in open_k], open_players,
                                   SALARY_CAP - extra_salary)
            if fill is None:
                continue
            full = list(idx)
            for k, j in zip(open_k, fill):
                full[k] = j
            if tuple(full) not in seen:
                seen.add(tuple(full)); options.append(full)
        plans.append(EntryPlan(d["entry_id"], d["contest_id"], ids, locked, options, keeps, offset))
    return plans


def dk_slot(k: int) -> str:
    return ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"][k]


def choose(plans: list[EntryPlan], slate: LiveSlate, eval_scores: np.ndarray,
           fields: dict[str, tuple[np.ndarray, np.ndarray]], default_field: tuple[np.ndarray, np.ndarray],
           payouts: dict[str, cs.PayoutCurve], objective: str = "roi", min_gain: float = 0.02) -> list[dict]:
    """
    Score every option of every entry against its contest's field and pick
    the best; two entries in one contest never end on the same lineup. A
    swap must beat keeping the lineup by `min_gain` x entry fee (or 1% in
    top-1% odds), so sim noise alone doesn't trigger swaps.
    Returns one result dict per entry.
    """
    n = len(slate.pool)
    out, taken = [], {}
    for p in plans:
        if not p.options:
            out.append({"entry_id": p.entry_id, "contest_id": p.contest_id, "choice": None})
            continue
        W = np.zeros((len(p.options), n), dtype=np.float32)
        for i, lu in enumerate(p.options):
            W[i, [j for j in lu if j is not None]] = 1.0
        fw, foff = fields.get(p.contest_id, default_field)
        ranks = cs.rank_against_field(W, fw, eval_scores, cand_offset=np.full(len(W), p.offset),
                                      field_offset=foff)
        res = cs.score_contest(ranks, payouts[p.contest_id])
        key = res.metrics["exp_payout"] if objective == "roi" else res.metrics["top1_pct"]
        used = taken.setdefault(p.contest_id, set())
        best = None
        for i in np.argsort(-key.to_numpy(), kind="stable"):
            if _lineup_key(p, i) not in used:
                best = int(i)
                break
        best = 0 if best is None else best
        if p.keeps_current and best != 0 and _lineup_key(p, 0) not in used:
            k = key.to_numpy()
            fee = payouts[p.contest_id].entry_fee
            margin = min_gain * fee if objective == "roi" else 0.01 * max(k[0], 1e-9)
            if k[best] - k[0] <= max(margin, 0.01 if objective == "roi" else 0.0):
                best = 0
        used.add(_lineup_key(p, best))
        cur = res.metrics.iloc[0] if p.keeps_current else None
        new = res.metrics.iloc[best]
        out.append({"entry_id": p.entry_id, "contest_id": p.contest_id, "choice": p.options[best],
                    "changed": not (p.keeps_current and best == 0),
                    "n_options": len(p.options), "open_slots": sum(not x for x in p.locked),
                    "cur_exp_payout": float(cur["exp_payout"]) if cur is not None else np.nan,
                    "new_exp_payout": float(new["exp_payout"]),
                    "cur_top1": float(cur["top1_pct"]) if cur is not None else np.nan,
                    "new_top1": float(new["top1_pct"]),
                    "points_so_far": float(slate.actual[[j for j in p.options[best] if j is not None]].sum()
                                           + p.offset)})
    return out


def _lineup_key(p: EntryPlan, i: int) -> tuple:
    return tuple(sorted(str(j) if j is not None else f"id{p.dk_ids[k]}" for k, j in enumerate(p.options[i])))


def upload_ids(p: EntryPlan, choice: list, pool: pd.DataFrame) -> list[str]:
    """DK IDs in slot order for a chosen option (locked non-pool players keep their original ID)."""
    dk_id = pool["dk_id"].astype("int64").astype(str).to_numpy()
    return [p.dk_ids[k] if j is None else dk_id[j] for k, j in enumerate(choice)]
