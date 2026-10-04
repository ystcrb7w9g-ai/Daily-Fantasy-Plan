"""
Learn field behavior from past DraftKings contest results.

DraftKings' contest-standings export (Contest page -> Export Lineups,
`contest-standings-<id>.csv`, often zipped) has two parts:

* every entry: `Rank, EntryId, EntryName, TimeRemaining, Points, Lineup`,
  where Lineup is "QB Josh Allen RB ... FLEX ... DST Bills ";
* a player table to the right: `Player, Roster Position, %Drafted, FPTS`
  (FLEX ownership is listed on separate FLEX rows).

`profile_standings` turns one file into aggregate field statistics:
score-distribution ratios (how far the top 0.1% / 1% / 20% sit above the
median -- a scale-free check of simulation variance), QB stacking mix,
FLEX position split, ownership concentration by position, duplication
and multi-entry rates. Those feed the engine's defaults
(`HISTORICAL_PROFILE`) and the `field-study` command. No salaries or
pre-lock projections are in the export, so this can't tell *why* players
were owned -- only how the field is built.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

from .sportsgameodds import normalize_name

_SLOT_RE = re.compile(r"(?:^|\s)(QB|RB|WR|TE|FLEX|DST)\s+")
POSITIONS = ("QB", "RB", "WR", "TE", "DST")
OWN_CURVE_LEN = {"QB": 20, "RB": 30, "WR": 50, "TE": 20, "DST": 20}


def read_standings(path: str) -> pd.DataFrame:
    """Read a contest-standings CSV, or the first CSV inside a .zip."""
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            data = z.read(name)
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", low_memory=False)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def parse_lineup(text: str) -> list[tuple[str, str]]:
    """'QB Josh Allen RB James Cook ...' -> [('QB', 'Josh Allen'), ('RB', 'James Cook'), ...]"""
    parts = _SLOT_RE.split(" " + str(text))
    return [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def player_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per player: name, position, own (fraction, FLEX included), fpts."""
    t = raw[["Player", "Roster Position", "%Drafted", "FPTS"]].dropna(subset=["Player"]).copy()
    t["Player"] = t["Player"].str.strip()
    t["own"] = t["%Drafted"].astype(str).str.rstrip("%").astype(float) / 100.0
    pos = t[t["Roster Position"] != "FLEX"].drop_duplicates("Player").set_index("Player")["Roster Position"]
    out = t.groupby("Player").agg(own=("own", "sum"), fpts=("FPTS", "first"))
    out["position"] = out.index.map(pos)
    flex = t[t["Roster Position"] == "FLEX"].assign(position=lambda d: d["Player"].map(pos))
    out.attrs["flex_split"] = flex.groupby("position")["own"].sum().to_dict()
    return out.reset_index().rename(columns={"Player": "name"})


def profile_standings(path: str, team_of: dict[str, str] | None = None,
                      stack_sample: int = 60_000, seed: int = 0) -> dict:
    """
    Aggregate field statistics for one contest. `team_of` maps normalized
    player name -> team (e.g. from a current DK player list); it's needed
    only for the QB-stack mix, which is measured on lineups whose QB is
    mapped.
    """
    raw = read_standings(path)
    entries = raw[raw["EntryId"].notna() & raw["Lineup"].notna()]
    players = player_table(raw)
    pos_of = players.set_index("name")["position"].to_dict()
    n = len(entries)

    pts = np.sort(entries["Points"].astype(float).to_numpy())[::-1]
    med = float(np.median(pts))
    prof = {
        "file": path.split("/")[-1], "entries": n, "players": len(players),
        "score_first": float(pts[0]), "score_top0.1": float(pts[int(n * 0.001)]),
        "score_top1": float(pts[int(n * 0.01)]), "score_top20": float(pts[int(n * 0.2)]),
        "score_median": med,
        "ratio_top0.1": float(pts[int(n * 0.001)] / med), "ratio_top1": float(pts[int(n * 0.01)] / med),
        "ratio_top20": float(pts[int(n * 0.2)] / med),
        "flex_split": {k: round(v, 4) for k, v in players.attrs["flex_split"].items()},
    }

    curves = {}
    for p in POSITIONS:
        s = np.sort(players.loc[players["position"] == p, "own"].to_numpy())[::-1]
        k = OWN_CURVE_LEN[p]
        curves[p] = np.pad(s[:k], (0, max(0, k - len(s)))).round(4).tolist()
    prof["own_curves"] = curves

    keys = entries["Lineup"].map(lambda s: tuple(sorted(nm for _, nm in parse_lineup(s))))
    counts = keys.value_counts()
    prof["unique_lineup_frac"] = float(len(counts) / n)
    prof["entries_in_duplicates"] = float(counts[counts >= 2].sum() / n)
    prof["max_duplicates"] = int(counts.max())
    per_user = entries["EntryName"].astype(str).str.extract(r"\((\d+)/(\d+)\)")[1]
    max_entries = pd.to_numeric(per_user, errors="coerce").fillna(1)
    prof["entries_from_150_max"] = float((max_entries >= 150).mean())
    prof["single_entry_frac"] = float((max_entries == 1).mean())

    if team_of:
        mix = Counter()
        sample = entries["Lineup"].sample(min(n, stack_sample), random_state=seed)
        for text in sample:
            lu = parse_lineup(text)
            qb = next((nm for sl, nm in lu if sl == "QB"), None)
            qb_team = team_of.get(normalize_name(qb)) if qb else None
            if qb_team is None:
                continue
            k = sum(1 for sl, nm in lu
                    if sl not in ("QB", "DST") and pos_of.get(nm) in ("WR", "TE")
                    and team_of.get(normalize_name(nm)) == qb_team)
            mix[min(k, 2)] += 1
        total = sum(mix.values())
        if total:
            prof["qb_stack_mix"] = [round(mix[k] / total, 4) for k in range(3)]
            prof["qb_stack_sample"] = total
    return prof


def combine_profiles(profiles: list[dict]) -> dict:
    """Average the per-contest profiles (each contest weighted equally)."""
    avg = lambda key: float(np.mean([p[key] for p in profiles]))  # noqa: E731
    out = {
        "contests": [p["file"] for p in profiles],
        "ratio_top0.1": avg("ratio_top0.1"), "ratio_top1": avg("ratio_top1"),
        "ratio_top20": avg("ratio_top20"),
        "entries_in_duplicates": avg("entries_in_duplicates"),
        "flex_split": {p: float(np.mean([pr["flex_split"].get(p, 0) for pr in profiles]))
                       for p in ("RB", "WR", "TE")},
        "own_curves": {p: np.mean([pr["own_curves"][p] for pr in profiles], axis=0).round(4).tolist()
                       for p in POSITIONS},
    }
    mixes = [p["qb_stack_mix"] for p in profiles if "qb_stack_mix" in p]
    if mixes:
        out["qb_stack_mix"] = np.mean(mixes, axis=0).round(4).tolist()
    return out


def save_profile(profile: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=1)


def load_profile(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
