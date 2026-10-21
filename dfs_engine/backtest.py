"""
Backtest the engine on past contests: replay `dk-run` with the inputs you had
at lock, then grade everything against the real DraftKings standings.

The sites tune on hundreds of slates; a change that helps one game proves
little. Every past contest you keep (pool + standings export) becomes a test.

A slate manifest (CSV) lists one past contest per row:

    label, pool, standings, entry_fee, prize_pool, top_prizes, extra_lineups, actuals

- `pool`: the player pool you ran (`dk-pool` output: proj, own, ...).
- `standings`: the contest's standings export (CSV or ZIP).
- `entry_fee`, `prize_pool`: the contest's. `top_prizes`: known prizes for
  1st, 2nd, ... ("1000000;100000;60000"); winnings are estimated from a
  DK-style table anchored on them (22% paid, 2x min cash).
- `extra_lineups` (optional): another tool's lineups for the same contest
  (e.g. SaberSim's export, DK IDs), graded head to head and offered to the
  engine as candidates like `--extra-lineups`. Several: separate with ';'.
- `actuals` (optional): a name,actual CSV of DK points for a contest
  without standings; finishes are then estimated against a simulated field.
- `contest_size` (optional): defaults to the standings' entry count.

Per slate:
1. ranking: our sims rank the contest's *real* entries. AUC for a real top-1%
   finish, and how many real top-1% entries land in the 1% / 5% we rate best.
   "Predicted top-1%" is how often real entries beat our simulated field's
   top-1% line; real entries beat the real field's 1% of the time, so above
   1% means our field is weaker than the real one.
2. field: our simulated field vs the real field (projection, ownership,
   salary, stacking, duplication, the top-1% score line in our sims).
3. replay: the engine's picks for 1 / 20 / 150 entries scored on the real
   results: finishes, cashes, top-1% finishes, estimated winnings, sim vs
   real ROI.
4. head to head: the extra lineups' real results and where our sim ranks them.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from collections import Counter
from dataclasses import dataclass, field as dc_field

import numpy as np
import pandas as pd

from . import contest as cs
from . import dk
from .data import load_player_pool
from .history import parse_lineup, read_standings
from .review import _key


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class SlateSpec:
    label: str
    pool: str
    standings: str | None = None
    entry_fee: float = 20.0
    prize_pool: float | None = None
    top_prizes: list[float] = dc_field(default_factory=list)
    extra_lineups: list[str] = dc_field(default_factory=list)
    actuals: str | None = None
    contest_size: int | None = None


def read_manifest(path: str) -> list[SlateSpec]:
    """Slate specs from a manifest CSV; relative paths resolve against the manifest's folder."""
    t = pd.read_csv(path, dtype=str).fillna("")
    base = os.path.dirname(os.path.abspath(path))

    def resolve(p):
        p = p.strip()
        return p if not p or os.path.isabs(p) else os.path.join(base, p)

    def num(v, cast=float):
        v = str(v).strip().replace(",", "").replace("$", "")
        return cast(float(v)) if v else None
    out = []
    for r in t.to_dict("records"):
        out.append(SlateSpec(
            label=r["label"].strip(), pool=resolve(r["pool"]),
            standings=resolve(r.get("standings", "")) or None,
            entry_fee=num(r.get("entry_fee", "")) or 0.0,
            prize_pool=num(r.get("prize_pool", "")),
            top_prizes=[float(x.replace(",", "")) for x in str(r.get("top_prizes", "")).split(";") if x.strip()],
            extra_lineups=[resolve(x) for x in str(r.get("extra_lineups", "")).split(";") if x.strip()],
            actuals=resolve(r.get("actuals", "")) or None,
            contest_size=num(r.get("contest_size", ""), int),
        ))
    return out


@dataclass
class Slate:
    spec: SlateSpec
    df: pd.DataFrame
    fmt: str
    points: np.ndarray | None        # every real entry's points (None without standings)
    lineups: list | None             # per real entry: (captain row or -1, sorted rows), None if unmapped
    actual: np.ndarray               # actual DK points per pool row (non-captain)
    size: int
    real_own: np.ndarray | None = None   # real roster share per pool row (FLEX + CPT)

    @property
    def label(self) -> str:
        return self.spec.label


def _row_lookup(df: pd.DataFrame) -> dict[str, int]:
    rows = {}
    for i, (n, p) in enumerate(zip(df["name"], df["position"])):
        rows.setdefault(_key(n, p), i)
        rows.setdefault(_key(n, "X"), i)
    return rows


def _actual_points(raw: pd.DataFrame, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(actual DK points, real roster share) per pool row from a standings export's player table.
    Showdown exports list a captain row at 1.5x: the FLEX row's points are used (or CPT / 1.5)."""
    t = raw[["Player", "Roster Position", "%Drafted", "FPTS"]].dropna(subset=["Player"]).copy()
    t["Player"] = t["Player"].astype(str).str.strip()
    t["FPTS"] = pd.to_numeric(t["FPTS"], errors="coerce")
    t["own"] = pd.to_numeric(t["%Drafted"].astype(str).str.rstrip("%"), errors="coerce") / 100.0
    pts = {}
    for name, g in t.groupby("Player"):
        flex = g.loc[g["Roster Position"] == "FLEX", "FPTS"]
        cpt = g.loc[g["Roster Position"] == "CPT", "FPTS"]
        pts[name] = float(flex.iloc[0]) if len(flex) and len(cpt) else (
            float(cpt.iloc[0]) / 1.5 if len(cpt) else float(g["FPTS"].iloc[0]))
    own = t.groupby("Player")["own"].sum()
    by_key = {}
    for name in pts:
        for k in (_key(name, "X"), _key(name, "DST")):
            by_key.setdefault(k, (pts[name], float(own.get(name, 0.0))))
    a = np.zeros(len(df))
    o = np.zeros(len(df))
    for i, (n, p) in enumerate(zip(df["name"], df["position"])):
        hit = by_key.get(_key(n, p)) or by_key.get(_key(n, "X"))
        if hit:
            a[i], o[i] = hit
    return a, o


def _parse_entry(text: str, rows: dict[str, int], fmt: str):
    text = str(text).strip()
    if fmt == "showdown":
        if not text.startswith("CPT "):
            return None
        names = [x.strip() for x in text[4:].split(" FLEX ")]
        ids = [rows.get(_key(n, "X"), rows.get(_key(n, "DST"))) for n in names]
        if len(ids) != 6 or any(i is None for i in ids):
            return None
        return ids[0], tuple(sorted(ids))
    slots = parse_lineup(text)
    ids = [rows.get(_key(n, "DST" if sl == "DST" else "X")) for sl, n in slots]
    if len(ids) != 9 or any(i is None for i in ids):
        return None
    return -1, tuple(sorted(ids))


def load_slate(spec: SlateSpec) -> Slate:
    df = load_player_pool(spec.pool)
    df = df.reset_index(drop=True)
    df["player_id"] = df.index
    for col in ("dk_id", "cpt_dk_id"):  # DK IDs as plain strings, the way dk-run keeps them
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df[col] = v.map(lambda x: str(int(x)) if pd.notna(x) else None)
    fmt = "showdown" if cs.captain_ownership(df) is not None or df.get("cpt_dk_id") is not None else "classic"
    points = lineups = real_own = None
    if spec.standings:
        raw = read_standings(spec.standings)
        ent = raw[raw["EntryId"].notna()].copy()
        ent["Points"] = pd.to_numeric(ent["Points"], errors="coerce")
        ent = ent[ent["Points"].notna()]
        sample = ent["Lineup"].dropna().astype(str).head(50)
        if (sample.str.startswith("CPT ")).mean() > 0.5:
            fmt = "showdown"
        elif len(sample):
            fmt = "classic"
        rows = _row_lookup(df)
        points = ent["Points"].to_numpy(dtype=float)
        lineups = [_parse_entry(s, rows, fmt) for s in ent["Lineup"].fillna("")]
        actual, real_own = _actual_points(raw, df)
    elif spec.actuals:
        a = pd.read_csv(spec.actuals)
        look = {}
        for n, v in zip(a.iloc[:, 0], a.iloc[:, 1]):
            for k in (_key(n, "X"), _key(n, "DST")):
                look.setdefault(k, float(v))
        actual = np.array([look.get(_key(n, p), look.get(_key(n, "X"), 0.0))
                           for n, p in zip(df["name"], df["position"])])
    else:
        raise ValueError(f"{spec.label}: needs standings or actuals")
    size = spec.contest_size or (len(points) if points is not None else 0)
    if not size:
        raise ValueError(f"{spec.label}: contest_size is required without standings")
    return Slate(spec, df, fmt, points, lineups, actual, size, real_own)


# ---------------------------------------------------------------------------
# Payouts and grading
# ---------------------------------------------------------------------------

def anchored_payouts(size: int, fee: float, pool: float, top: list[float], paid_frac: float = 0.22,
                     min_mult: float = 2.0, next_ratio: float = 2 / 3) -> np.ndarray:
    """DK-style payout table: the known top prizes, then max(min cash, B * rank^-beta) down to the
    last paid place, with the first unknown place at `next_ratio` x the last known prize and beta
    solved so the table sums to the prize pool."""
    paid = max(1, int(round(size * paid_frac)))
    min_cash = min_mult * fee
    k = len(top)
    p = np.zeros(size)
    p[:k] = top
    r = np.arange(k + 1, paid + 1, dtype=float)
    anchor = next_ratio * top[-1]
    budget = pool - sum(top)
    if budget < (paid - k) * min_cash:
        raise ValueError("min cash x paid places exceeds what's left of the prize pool")

    def tail(beta):
        return np.maximum(min_cash, anchor * (r / (k + 1)) ** -beta)
    lo, hi = 0.05, 6.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tail(mid).sum() > budget else (lo, mid)
    p[k:paid] = tail(0.5 * (lo + hi))
    return p


def payout_tables(slate: Slate) -> tuple[cs.PayoutCurve, np.ndarray]:
    """(the curve the engine optimizes against, the table used to estimate real winnings)."""
    s = slate.spec
    pool = s.prize_pool if s.prize_pool else slate.size * s.entry_fee * 0.85
    first = s.top_prizes[0] / pool if s.top_prizes else 0.20
    engine = cs.gpp_payout_curve(slate.size, s.entry_fee, prize_pool=pool, paid_frac=0.22,
                                 first_frac=min(max(first, 0.01), 0.6))
    real = anchored_payouts(slate.size, s.entry_fee, pool, s.top_prizes) if s.top_prizes else engine.payouts
    return engine, real


def place_in_contest(real_points: np.ndarray, ours: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each of our lineups' finish and prize if entered alone into the real field (ties split
    their places evenly, as DraftKings does). Returns (ranks, prizes)."""
    srt = np.sort(real_points)
    ranks, prizes = [], []
    for s in ours:
        above = len(srt) - np.searchsorted(srt, s + 1e-3, side="left")
        tied = np.searchsorted(srt, s + 1e-3, side="left") - np.searchsorted(srt, s - 1e-3, side="left")
        ranks.append(above + 1)
        window = table[above:above + tied + 1]
        prizes.append(float(window.mean()) if len(window) else 0.0)
    return np.array(ranks), np.array(prizes)


def lineup_points(lineups, actual: np.ndarray) -> np.ndarray:
    return np.array([actual[list(lu.player_ids)].sum()
                     + (0.5 * actual[lu.captain_id] if lu.captain_id is not None else 0.0) for lu in lineups])


# ---------------------------------------------------------------------------
# 1-2. Ranking the real entries, and our field vs the real one
# ---------------------------------------------------------------------------

def _auc(score: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def _weights(keys, n_players: int) -> np.ndarray:
    w = np.zeros((len(keys), n_players), dtype=np.float32)
    for i, (cpt, rows) in enumerate(keys):
        w[i, list(rows)] = 1.0
        if cpt >= 0:
            w[i, cpt] = 1.5
    return w


def _real_sample(slate: Slate, n_sample: int, seed: int):
    """Mapped real entries (a sample): distinct-lineup weights, entry -> lineup index, points."""
    ok = np.array([lu is not None for lu in slate.lineups])
    idx = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    if len(idx) > n_sample:
        idx = np.sort(rng.choice(idx, n_sample, replace=False))
    keys = [slate.lineups[i] for i in idx]
    uniq = {k: j for j, k in enumerate(dict.fromkeys(keys))}
    inv = np.array([uniq[k] for k in keys])
    return _weights(list(uniq), len(slate.df)), inv, slate.points[idx], ok.mean()


# Proper scores of our top-1% chances: each lineup's sim rate is shrunk toward the 1% base rate by
# PRIOR_SIMS pseudo-sims, and floored, so a lineup we never saw finish top 1% isn't scored as impossible.
BASE_RATE = 0.01
PRIOR_SIMS = 100
P_FLOOR = 1e-4


def logistic_fit(X: np.ndarray, y: np.ndarray, iters: int = 100) -> np.ndarray:
    """Logistic regression by Newton's method from the base rate (first column = intercept), halving
    any step that lowers the likelihood, so a forecast far off calibration can't send it diverging."""
    yf = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    w[0] = np.log(max(yf.mean(), 1e-6) / max(1 - yf.mean(), 1e-6))

    def loglik(v):
        z = np.clip(X @ v, -30, 30)
        return float(yf @ z - np.logaddexp(0.0, z).sum())
    ll = loglik(w)
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (yf - mu))
        for _ in range(30):
            ll_new = loglik(w + step)
            if ll_new >= ll - 1e-12:
                break
            step = step / 2
        w, ll = w + step, ll_new
        if np.abs(step).max() < 1e-9:
            break
    return w


def _calibration_fit(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Logistic regression of the real outcome on logit(predicted chance): (intercept, slope).
    Calibrated forecasts give (0, 1); a slope below 1 means we spread our chances too far apart."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    x = np.log(p / (1 - p))
    w = logistic_fit(np.column_stack([np.ones_like(x), x]), y)
    return float(w[0]), float(w[1])


def proper_scores(hits: np.ndarray, n_sims: int, y: np.ndarray) -> dict:
    """How well our predicted top-1% chances (top-1% finishes in `n_sims` sims) forecast which real
    entries finished top 1% (`y`). Skill scores compare with forecasting the base rate for every
    entry: above 0 is better than that, 1 is perfect. Log skill rewards putting probability on the
    lineups that actually won; Brier skill is the gentler squared-error version."""
    y = np.asarray(y, dtype=bool)
    p = np.clip((np.asarray(hits, dtype=float) + PRIOR_SIMS * BASE_RATE) / (n_sims + PRIOR_SIMS),
                P_FLOOR, 1 - P_FLOOR)
    base = float(np.clip(y.mean(), P_FLOOR, 1 - P_FLOOR))
    ll = float(np.where(y, np.log(p), np.log1p(-p)).mean())
    ll0 = float(np.where(y, np.log(base), np.log1p(-base)).mean())
    br, br0 = float(((p - y) ** 2).mean()), float(((base - y) ** 2).mean())
    a, b = _calibration_fit(p, y)
    return {"log_score": ll, "log_skill": 1 - ll / ll0 if ll0 else float("nan"),
            "brier_skill": 1 - br / br0 if br0 else float("nan"),
            "calibration_intercept": a, "calibration_slope": b}


def rank_real_entries(slate: Slate, field: np.ndarray, scores: np.ndarray, n_sample: int = 40_000,
                      seed: int = 7, batch: int = 250) -> dict:
    """How well our sims pick out the real entries that finished top 1%, and how strong our
    simulated field is next to the real one."""
    if slate.points is None:
        return {}
    U, inv, pts, mapped = _real_sample(slate, n_sample, seed)
    line1 = np.quantile(slate.points, 0.99)
    y = pts >= line1
    hits = np.zeros(len(U))
    lines_field, lines_real = [], []
    for b in range(0, len(scores), batch):
        s = scores[b:b + batch].astype(np.float32)
        fs = s @ field.T
        us = s @ U.T
        q = np.quantile(fs, 0.99, axis=1, keepdims=True)
        hits += (us >= q).sum(axis=0)
        lines_field.append(q.ravel())
        lines_real.append(np.quantile(us[:, inv], 0.99, axis=1))
    p = hits[inv] / len(scores)
    proj = slate.df["proj"].to_numpy(dtype=float)
    own = pd.to_numeric(slate.df["own"], errors="coerce").fillna(0).to_numpy(dtype=float)
    own = own / 100.0 if own.max() > 1.5 else own
    lineup_proj = (U @ proj)[inv]
    lineup_own = ((U > 0).astype(np.float32) @ own)[inv]
    out = {"mapped_share": float(mapped), "sampled": int(len(pts)), "real_top1": int(y.sum()),
           "auc_top1": _auc(p, y), "auc_projection": _auc(lineup_proj, y), "auc_low_ownership": _auc(-lineup_own, y),
           "spearman_points": float(pd.Series(p).corr(pd.Series(pts), method="spearman"))}
    out.update(proper_scores(hits[inv], len(scores), y))
    for frac in (0.01, 0.05):
        best = p >= np.quantile(p, 1 - frac)
        out[f"real_top1_in_best_{int(frac * 100)}pct"] = float(y[best].mean())
    out["predicted_top1_mean"] = float(p.mean())       # vs our field; the real field gives 1%
    out["top1_line_field"] = float(np.mean(np.concatenate(lines_field)))
    out["top1_line_real"] = float(np.mean(np.concatenate(lines_real)))
    # Calibration by ownership: do we over- or under-rate popular lineups?
    q = pd.qcut(pd.Series(lineup_own).rank(method="first"), 5, labels=False)
    out["by_ownership_quintile"] = [
        {"quintile": int(k), "own_sum": float(lineup_own[q == k].mean()), "predicted": float(p[q == k].mean()),
         "real": float(y[q == k].mean())} for k in range(5)]
    lineup_sal = (U @ slate.df["salary"].to_numpy(dtype=float))[inv]
    out["_entry_scores"] = {"p": p, "y": y, "own": lineup_own, "proj": lineup_proj, "salary": lineup_sal,
                            "points": pts}
    return out


def _salary(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    return w @ df["salary"].to_numpy(dtype=float)


def _dupe_stats(keys: list, size: int) -> dict:
    """Size-free duplication: the chance two random entries are the same lineup (x1e4), and the
    most-played lineup's copies scaled to the contest size. (The share of entries in duplicated
    lineups grows with field size, so it can't compare a 20k sample with a 160k contest.)"""
    cnt = Counter(keys)
    v = np.array(list(cnt.values()), dtype=float)
    n = len(keys)
    return {"collision_x1e4": float((v * (v - 1)).sum() / (n * (n - 1)) * 1e4),
            "top_lineup_copies": float(v.max() * size / n)}


def compare_fields(slate: Slate, field: np.ndarray, n_sample: int = 40_000, seed: int = 7) -> dict:
    """Our simulated field next to the real one: projection, ownership, salary, structure, duplication."""
    if slate.points is None:
        return {}
    df = slate.df
    U, inv, _, _ = _real_sample(slate, n_sample, seed)
    proj = df["proj"].to_numpy(dtype=float)
    own = pd.to_numeric(df["own"], errors="coerce").fillna(0).to_numpy(dtype=float)
    own = own / 100.0 if own.max() > 1.5 else own
    real_w = U[inv]
    out = {}
    for lab, w in (("real", real_w), ("ours", field)):
        lp = w @ proj
        sal = _salary(df, w)
        out[lab] = {"proj_mean": float(lp.mean()), "proj_p90": float(np.quantile(lp, 0.9)),
                    "proj_p99": float(np.quantile(lp, 0.99)),
                    "own_sum": float(((w > 0).astype(np.float32) @ own).mean()),
                    "salary_mean": float(sal.mean()),
                    "at_cap": float((sal >= cs.SALARY_CAP - 1e-6).mean()),
                    "within_500": float((sal >= cs.SALARY_CAP - 500).mean())}
        if slate.fmt == "classic":
            out[lab].update({k: float(v) for k, v in cs.structure_rates(w, df).items()})
            ids = np.array([np.flatnonzero(r) for r in w])
            stacked = np.minimum(cs._qb_stack_counts(ids, df), 2)
            out[lab]["qb_stack_0_1_2plus"] = [float((stacked == k).mean()) for k in range(3)]
        else:
            qbs = df[(df["position"] == "QB") & (df["proj"] > 0)].sort_values("proj").groupby("team").tail(1).index
            on = w[:, list(qbs)] > 0
            out[lab]["both_qbs"] = float(on.all(axis=1).mean()) if len(qbs) == 2 else float("nan")
    real_keys = [lu for lu in slate.lineups if lu is not None]
    out["real"].update(_dupe_stats(real_keys, slate.size))
    if slate.real_own is not None:
        out["own_sharpen_fit"] = fit_sharpen(df, slate.real_own)
    field_keys = [bytes(r) for r in (field * 2).astype(np.uint8)]
    out["ours"].update(_dupe_stats(field_keys, slate.size))
    return out


def fit_sharpen(df: pd.DataFrame, real_own: np.ndarray, grid=np.round(np.arange(0.6, 1.81, 0.05), 2)) -> dict:
    """The power on projected ownership (within position) that best matches this contest's real
    roster shares: above 1 the contest was chalkier than projected, below 1 flatter. Set it as the
    contest's `field_sharpen` for similar contests."""
    from .ownership import sharpen_ownership
    own = pd.to_numeric(df["own"], errors="coerce").fillna(0.0)
    own = own / 100.0 if own.max() > 1.5 else own
    cpt = cs.captain_ownership(df)
    errs = {}
    for g in grid:
        s = sharpen_ownership(own, df["position"], float(g)).to_numpy()
        if cpt is not None:
            c = np.clip(cpt, 0, None)
            s = s + (c ** g / (c ** g).sum() * c.sum() if c.sum() > 0 else c)
        errs[float(g)] = float(np.sqrt(np.mean((s - real_own) ** 2)))
    best = min(errs, key=errs.get)
    return {"best": best, "rmse_best": errs[best], "rmse_at_1": errs.get(1.0, float("nan"))}


# ---------------------------------------------------------------------------
# 3-4. Replay the engine, and grade other tools' lineups
# ---------------------------------------------------------------------------

def default_args(extra_cli: list[str] | None = None):
    """dk-run's options (defaults plus `extra_cli`, e.g. the stacking flags you always use)."""
    from .cli import build_parser
    return build_parser().parse_args(["dk-run", "--pool", "-", "--entries", "-", "--upload-out", "-"]
                                     + list(extra_cli or []))


def replay(slate: Slate, args, n_lineups=(1, 20, 150), quiet: bool = True) -> dict:
    """Run dk-run's model on this slate and grade its picks for each entry count."""
    from .cli import model_slate, _pick_for_contest, _objective_text
    args.fmt = slate.fmt
    args.extra_lineups = list(slate.spec.extra_lineups)
    if args.seed is None:
        args.seed = 100
    df = slate.df.copy()
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        m = model_slate(df, args, total=max(n_lineups))
    engine_pay, real_table = payout_tables(slate)
    result = cs.score_contest(m.ranks, engine_pay)
    hold = cs.score_contest(m.hold_ranks, engine_pay)
    cand_pts = lineup_points(m.candidates, slate.actual)
    real_pts = slate.points
    estimated = real_pts is None
    if estimated:  # no standings: stand our simulated field, scored on the real results, in for the real one
        real_pts = np.sort(m.field @ slate.actual)
        real_pts = np.interp(np.linspace(0, len(real_pts) - 1, slate.size), np.arange(len(real_pts)), real_pts)
    paid = int((real_table > 0).sum())
    # The same table with every prize above 11th place cut to 11th's: winnings that one lucky
    # 1st place can't swing, for comparing settings on a handful of contests.
    capped_table = np.minimum(real_table, real_table[min(10, len(real_table) - 1)])
    out = {"objective": _objective_text(args, engine_pay.contest_size), "candidates": len(m.candidates),
           "finishes_estimated": estimated, "picks": {}, "model": m, "candidate_points": cand_pts}
    for n in n_lineups:
        with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
            chosen = _pick_for_contest(df, m.candidates, result, n, args, contest_size=engine_pay.contest_size)
        pts = cand_pts[chosen]
        rec = {"n": len(chosen), "points": pts.tolist(), "median": float(np.median(pts)), "best": float(pts.max()),
               "sim_roi_holdout": float(cs.portfolio_summary(hold, chosen, slate.spec.entry_fee)["roi"])}
        if real_pts is not None:
            ranks, prizes = place_in_contest(real_pts, pts, real_table)
            _, capped = place_in_contest(real_pts, pts, capped_table)
            rec.update({"won_capped": float(capped.sum()), "median_finish_pct": float(np.median(ranks) / len(real_pts))})
            fees = slate.spec.entry_fee * len(chosen)
            rec.update({"best_finish": int(ranks.min()), "cashes": int((ranks <= paid).sum()),
                        "top1": int((ranks <= 0.01 * slate.size).sum()), "won": float(prizes.sum()),
                        "fees": float(fees), "real_roi": float(prizes.sum() / fees - 1) if fees else float("nan")})
        rec["lineups"] = [_lineup_text(df, m.candidates[i]) for i in chosen]
        rec["chosen"] = chosen
        out["picks"][n] = rec
    best = int(np.argmax(cand_pts))
    out["pool_best"] = {"points": float(cand_pts[best]), "lineup": _lineup_text(df, m.candidates[best])}
    if real_pts is not None:
        out["pool_best"]["finish"] = int(place_in_contest(real_pts, cand_pts[[best]], real_table)[0][0])
    out["extra"] = grade_extra(slate, m, result, real_table, real_pts)
    return out


def _lineup_text(df: pd.DataFrame, lu) -> str:
    names = df["name"].to_numpy()
    if lu.captain_id is not None:
        return f"CPT {names[lu.captain_id]} | " + ", ".join(names[j] for j in lu.player_ids if j != lu.captain_id)
    return ", ".join(names[j] for j in lu.player_ids)


def grade_extra(slate: Slate, m, result: cs.ContestResult, real_table: np.ndarray,
                real_pts: np.ndarray | None) -> list[dict]:
    """Other tools' lineups (manifest `extra_lineups`): real points / finish, and our sim's rank of them."""
    out = []
    top1 = result.metrics["top1_pct"].to_numpy()
    index = {cs.lineup_key(c.player_ids, c.captain_id): i for i, c in enumerate(m.candidates)}
    for path in slate.spec.extra_lineups:
        lineups, _ = cs.candidates_from_ids(slate.df, dk.read_lineup_file(path), stack_rules=None)
        if not lineups:
            continue
        pts = lineup_points(lineups, slate.actual)
        rec = {"file": os.path.basename(path), "n": len(lineups), "median": float(np.median(pts)),
               "best": float(pts.max()), "points": pts.tolist()}
        if real_pts is not None:
            ranks, prizes = place_in_contest(real_pts, pts, real_table)
            paid = int((real_table > 0).sum())
            rec.update({"best_finish": int(ranks.min()), "median_finish": float(np.median(ranks)),
                        "cashes": int((ranks <= paid).sum()), "top1": int((ranks <= 0.01 * slate.size).sum()),
                        "won": float(prizes.sum())})
        # where our sim ranks the ones that made it into the candidate pool (rank 1 = our favorite)
        order = pd.Series(top1).rank(ascending=False).to_numpy()
        found = [order[index[k]] for k in (cs.lineup_key(lu.player_ids, lu.captain_id) for lu in lineups) if k in index]
        rec["in_our_pool"] = len(found)
        rec["our_rank_median"] = float(np.median(found)) if found else float("nan")
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_slate(spec: SlateSpec, args, n_lineups=(1, 20, 150), n_sample: int = 40_000, quiet: bool = True,
              keep: str | None = None) -> dict:
    """Backtest one slate. `keep`: a path to pickle the loaded slate and the fitted model to
    (field, sims, candidates, placements), for trying other settings on them later."""
    slate = load_slate(spec)
    rep = replay(slate, copy.deepcopy(args), n_lineups=n_lineups, quiet=quiet)
    m = rep.pop("model")
    rep.pop("candidate_points")
    if keep:
        import pickle
        with open(keep, "wb") as fh:
            pickle.dump({"slate": slate, "model": m}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    out = {"label": spec.label, "fmt": slate.fmt, "contest_size": slate.size, "replay": rep}
    out["ranking"] = rank_real_entries(slate, m.field, m.eval_scores, n_sample=n_sample)
    out["ranking"].pop("_entry_scores", None)
    out["field"] = compare_fields(slate, m.field, n_sample=n_sample)
    return out


def summarize(results: list[dict], n_lineups=(1, 20, 150)) -> dict[str, pd.DataFrame]:
    rank_rows, field_rows, pick_rows, extra_rows = [], [], [], []
    for r in results:
        g = r.get("ranking") or {}
        if g:
            rank_rows.append({"slate": r["label"], "real entries mapped": g["mapped_share"],
                              "AUC top-1%": g["auc_top1"], "AUC projection": g["auc_projection"],
                              "real top-1% in best 1%": g["real_top1_in_best_1pct"],
                              "real top-1% in best 5%": g["real_top1_in_best_5pct"],
                              "log skill": g.get("log_skill"), "Brier skill": g.get("brier_skill"),
                              "calibration slope": g.get("calibration_slope"),
                              "predicted top-1% (real is 1%)": g["predicted_top1_mean"],
                              "top-1% line: our field / real": g["top1_line_field"] / g["top1_line_real"]})
        f = r.get("field") or {}
        if f:
            field_rows.append({"slate": r["label"],
                               "own power (best fit)": (f.get("own_sharpen_fit") or {}).get("best"),
                               **{f"{k} real": f["real"][k] for k in ("proj_mean", "own_sum", "salary_mean", "collision_x1e4")},
                               **{f"{k} ours": f["ours"][k] for k in ("proj_mean", "own_sum", "salary_mean", "collision_x1e4")}})
        for n in n_lineups:
            p = r["replay"]["picks"].get(n) or r["replay"]["picks"].get(str(n))
            if not p:
                continue
            pick_rows.append({"slate": r["label"], "entries": p["n"], "median pts": p["median"], "best pts": p["best"],
                              "best finish": p.get("best_finish"), "cashes": p.get("cashes"), "top-1%": p.get("top1"),
                              "fees": p.get("fees"), "won": p.get("won"), "real ROI": p.get("real_roi"),
                              "won (capped at 11th)": p.get("won_capped"), "median finish %": p.get("median_finish_pct"),
                              "sim ROI (holdout)": p["sim_roi_holdout"]})
        for e in r["replay"].get("extra", []):
            extra_rows.append({"slate": r["label"], "file": e["file"], "lineups": e["n"], "median pts": e["median"],
                               "best pts": e["best"], "best finish": e.get("best_finish"), "cashes": e.get("cashes"),
                               "top-1%": e.get("top1"), "won": e.get("won"), "in our pool": e["in_our_pool"],
                               "our rank (median)": e["our_rank_median"]})
    return {"ranking": pd.DataFrame(rank_rows), "field": pd.DataFrame(field_rows),
            "picks": pd.DataFrame(pick_rows), "extra": pd.DataFrame(extra_rows)}


def load_results(dirs) -> list[dict]:
    """The per-slate results saved by `save` in one or more backtest output folders."""
    out = []
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        with open(os.path.join(d, "backtest.json")) as fh:
            out += json.load(fh)
    return out


# (key, label, +1 if higher is better / -1 if lower is better)
RANK_METRICS = [("log_skill", "log skill", 1), ("brier_skill", "Brier skill", 1), ("auc_top1", "AUC top-1%", 1),
                ("real_top1_in_best_5pct", "real top-1% in best 5%", 1)]
PICK_METRICS = [("cashes", "cashes", 1), ("top1", "top-1% finishes", 1), ("capped_roi", "ROI, prizes capped at 11th", 1),
                ("median_finish_pct", "median finish %", -1)]


def _pick_metric(r: dict, n: int, key: str):
    picks = r["replay"]["picks"]
    p = picks.get(n) or picks.get(str(n))
    if not p:
        return None
    if key == "capped_roi":
        return p["won_capped"] / p["fees"] - 1 if p.get("won_capped") is not None and p.get("fees") else None
    return p.get(key)


def compare(a: list[dict], b: list[dict], n_lineups=(1, 20, 150), n_boot: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """Variant B minus baseline A on every slate both ran: per-slate differences, their mean, a 90%
    interval from resampling whole slates (a contest is the independent unit, not a lineup), how
    many slates went each way, and a verdict that is "unclear" unless the interval excludes zero."""
    A, B = {r["label"]: r for r in a}, {r["label"]: r for r in b}
    labels = [lab for lab in A if lab in B]
    getters = [(name, sign, lambda r, k=key: (r.get("ranking") or {}).get(k)) for key, name, sign in RANK_METRICS]
    getters += [(f"{n} entries: {name}", sign, lambda r, k=key, n=n: _pick_metric(r, n, k))
                for n in n_lineups for key, name, sign in PICK_METRICS]
    rng = np.random.default_rng(seed)
    rows = []
    for name, sign, get in getters:
        d = {}
        for lab in labels:
            va, vb = get(A[lab]), get(B[lab])
            if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
                d[lab] = float(vb) - float(va)
        if not d:
            continue
        diffs = np.array(list(d.values()))
        boot = diffs[rng.integers(0, len(diffs), (n_boot, len(diffs)))].mean(axis=1)
        lo, hi = (float(x) for x in np.quantile(boot, [0.05, 0.95]))
        good, bad = (lo > 0, hi < 0) if sign > 0 else (hi < 0, lo > 0)
        rows.append({"metric": name, "slates": len(d), "mean B-A": float(diffs.mean()), "90% low": lo, "90% high": hi,
                     "B better in": int((sign * diffs > 0).sum()), "B worse in": int((sign * diffs < 0).sum()),
                     "verdict": "B better" if good else "B worse" if bad else "unclear", **d})
    return pd.DataFrame(rows)


def save(results: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    clean = json.loads(json.dumps(results, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    with open(os.path.join(out_dir, "backtest.json"), "w") as fh:
        json.dump(clean, fh, indent=1)
    for name, t in summarize(results).items():
        if len(t):
            t.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
