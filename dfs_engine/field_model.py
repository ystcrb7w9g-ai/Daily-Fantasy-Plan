"""
How real fields build lineups, measured from DraftKings contest-standings
exports (every entry's full lineup), versus our simulated field.

`lineup_structure` turns lineups into per-lineup structure features (QB
stack size, bring-backs, biggest game stack, RB + own DST, players facing
your DST); `structure_summary` turns those into the rates the field
generator should reproduce. Players are mapped to teams with nflverse
weekly rosters (`team_map`), DSTs by nickname.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .history import parse_lineup
from .sportsgameodds import TEAM_ID_TO_ABBR, normalize_abbr, normalize_name

NICKNAME_TO_ABBR = {k.rsplit("_", 2)[-2].lower(): v for k, v in TEAM_ID_TO_ABBR.items()}


def team_map(weekly: pd.DataFrame, season: int, week: int) -> dict[str, str]:
    """Normalized player name -> team that week (nflverse weekly stats), plus 'DST|nickname' keys."""
    w = weekly[(weekly["season"] == season) & (weekly["week"] == week)]
    m = {normalize_name(n): normalize_abbr(t) for n, t in zip(w["player_display_name"], w["team"])}
    m.update({f"DST|{nick}": abbr for nick, abbr in NICKNAME_TO_ABBR.items()})
    return m


def _team(name: str, slot: str, teams: dict[str, str]) -> str | None:
    if slot == "DST":
        k = normalize_name(name).split()
        return teams.get(f"DST|{k[-1]}") if k else None
    return teams.get(normalize_name(name))


def lineup_structure(lineups: pd.Series, teams: dict[str, str], opp: dict[str, str],
                     pos: dict[str, str]) -> pd.DataFrame:
    """
    One row per lineup (DK 'Lineup' strings) with: qb_stack (QB's own
    WR/TE), qb_rb (QB's own RBs), bring_back (non-DST players from the QB's
    opponent), max_game (most players, DST included, from one game),
    rb_dst (RBs on the DST's team), vs_dst (offensive players facing the
    DST) and mapped (share of the 9 players mapped to a team).
    """
    rows = []
    for text in lineups:
        lu = parse_lineup(text)
        t = [(_team(nm, sl, teams), pos.get(nm, sl) if sl == "FLEX" else sl) for sl, nm in lu]
        qb = next((tm for tm, p in t if p == "QB"), None)
        dst = next((tm for tm, p in t if p == "DST"), None)
        games = Counter(frozenset((tm, opp.get(tm))) for tm, _ in t if tm)
        rows.append({
            "qb_stack": sum(1 for tm, p in t if p in ("WR", "TE") and tm and tm == qb),
            "qb_rb": sum(1 for tm, p in t if p == "RB" and tm and tm == qb),
            "bring_back": sum(1 for tm, p in t if p != "DST" and tm and qb and tm == opp.get(qb)),
            "max_game": max(games.values()) if games else 0,
            "rb_dst": sum(1 for tm, p in t if p == "RB" and tm and tm == dst),
            "vs_dst": sum(1 for tm, p in t if p not in ("DST",) and tm and dst and tm == opp.get(dst)),
            "mapped": sum(1 for tm, _ in t if tm) / max(len(t), 1),
        })
    return pd.DataFrame(rows)


def structure_summary(f: pd.DataFrame) -> dict:
    """Rates the field generator should match (lineups with a fully mapped QB/DST only)."""
    f = f[f["mapped"] >= 8 / 9]
    def dist(col, cuts):
        v = f[col].clip(upper=cuts[-1])
        return [round(float((v == c).mean()), 4) for c in cuts]
    return {
        "n": int(len(f)),
        "qb_stack_0_1_2_3plus": dist("qb_stack", [0, 1, 2, 3]),
        "bring_back_0_1_2plus": dist("bring_back", [0, 1, 2]),
        "max_game_le3_4_5_6plus": [round(float((f["max_game"] <= 3).mean()), 4)]
                                  + [round(float(v), 4) for v in
                                     [(f["max_game"] == 4).mean(), (f["max_game"] == 5).mean(),
                                      (f["max_game"] >= 6).mean()]],
        "qb_with_own_rb": round(float((f["qb_rb"] >= 1).mean()), 4),
        "rb_with_own_dst": round(float((f["rb_dst"] >= 1).mean()), 4),
        "any_vs_own_dst": round(float((f["vs_dst"] >= 1).mean()), 4),
    }


def weights_to_lineups(field_w: np.ndarray, df: pd.DataFrame) -> pd.Series:
    """Simulated field rows -> DK-style 'Lineup' strings (FLEX slot labelled by position)."""
    names, posn = df["name"].to_numpy(), df["position"].to_numpy()
    return pd.Series([" ".join(f"{posn[j]} {names[j]}" for j in np.flatnonzero(r)) for r in field_w])


def measure_standings(raw: pd.DataFrame, weekly: pd.DataFrame, games: pd.DataFrame, season: int, week: int,
                      sample: int = 40_000, seed: int = 1) -> dict:
    """Structure summary of a contest's lineups: all entries, casual (1-3 entries) and 150-max accounts."""
    from .history import player_table
    from .ownership_model import schedule_lines

    e = raw[raw["EntryId"].notna() & raw["Lineup"].notna()].copy()
    e["n_entries"] = pd.to_numeric(e["EntryName"].astype(str).str.extract(r"\(\d+/(\d+)\)")[0],
                                   errors="coerce").fillna(1)
    teams = team_map(weekly, season, week)
    lines = schedule_lines(games, season, week)
    opp = dict(zip(lines["team"], lines["opp"]))
    pt = player_table(raw)
    pos = dict(zip(pt["name"], pt["position"]))
    out = {}
    for label, sub in (("all", e), ("casual", e[e["n_entries"] <= 3]), ("max_entry", e[e["n_entries"] >= 150])):
        if len(sub):
            smp = sub["Lineup"].sample(min(sample, len(sub)), random_state=seed)
            out[label] = structure_summary(lineup_structure(smp, teams, opp, pos))
    return out
