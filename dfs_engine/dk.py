"""
DraftKings entry-file support (Classic).

DraftKings' "Edit Entries" download (DKEntries.csv) holds two things:

* your entries -- `Entry ID, Contest Name, Contest ID, Entry Fee` followed
  by one column per roster slot (QB, RB, RB, WR, WR, WR, TE, FLEX, DST);
* the slate's full player list, embedded to the right of the entries
  (`Position, Name + ID, Name, ID, Roster Position, Salary, Game Info,
  TeamAbbrev, AvgPointsPerGame`).

This module reads both, turns the player list into an engine player pool
(with `dk_id` and `game_time` columns), merges in your projections,
estimates each contest's size/payouts from its name, assigns optimized
lineups to DraftKings slots, and writes the upload file (player IDs only,
which is what DraftKings asks for).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sportsgameodds import normalize_abbr, normalize_name

CLASSIC_SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
REPORT_SLOT_LABELS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
ENTRY_COLUMNS = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee"]
BASE_COUNTS = {"RB": 2, "WR": 3, "TE": 1}

POOL_COLUMNS = [
    "name", "team", "opp", "position", "salary", "proj", "own", "ceiling",
    "team_total", "game_total", "spread", "dk_id", "game_time", "avg_ppg",
]


@dataclass
class DKEntryFile:
    header: list[str]          # original header row
    slots: list[str]           # roster-slot column names, e.g. CLASSIC_SLOTS
    entries: pd.DataFrame      # entry_id, contest_name, contest_id, entry_fee, + raw strings
    players: pd.DataFrame      # dk_id, name, position, roster_position, salary, team, opp, game, game_time, avg_ppg


def _money(text: str) -> float:
    return float(str(text).replace("$", "").replace(",", "").strip() or 0)


_GAME_RE = re.compile(r"^(\w+)@(\w+)\s+(\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}[AP]M)")


def read_entry_file(path: str) -> DKEntryFile:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    if header[:4] != ENTRY_COLUMNS:
        raise ValueError(f"Not a DraftKings entry file (header starts {header[:4]})")
    slots = []
    for col in header[4:]:
        if not col.strip():
            break
        slots.append(col.strip())
    if slots != CLASSIC_SLOTS:
        raise ValueError(f"Only NFL Classic entry files are supported (slots: {slots})")

    entries = []
    for r in rows[1:]:
        if r and r[0].strip().isdigit():
            entries.append({
                "entry_id": r[0].strip(), "contest_name": r[1], "contest_id": r[2].strip(),
                "entry_fee_raw": r[3], "entry_fee": _money(r[3]),
            })

    start = col = None
    for i, r in enumerate(rows):
        if "Name + ID" in r:
            start, col = i, r.index("Name + ID") - 1
            break
    if start is None:
        raise ValueError("No player list found in the entry file")
    players = []
    for r in rows[start + 1:]:
        if len(r) <= col + 7 or not r[col].strip():
            continue
        pos, _name_id, name, dk_id, roster_pos, salary, game_info, team, avg = (r[col:col + 9] + [""])[:9]
        m = _GAME_RE.match(game_info.strip())
        away, home, when = (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)
        team = team.strip()
        players.append({
            "dk_id": dk_id.strip(), "name": name.strip(), "position": pos.strip().upper(),
            "roster_position": roster_pos.strip(), "salary": int(salary),
            "team": team, "opp": home if team == away else away,
            "game": f"{away}@{home}" if m else None,
            "game_time": pd.to_datetime(when, format="%m/%d/%Y %I:%M%p") if when else pd.NaT,
            "avg_ppg": pd.to_numeric(avg, errors="coerce"),
        })
    return DKEntryFile(header, slots, pd.DataFrame(entries), pd.DataFrame(players))


# ---------------------------------------------------------------------------
# Player pool
# ---------------------------------------------------------------------------

def _match_key(name: str, position: str, team: str | None) -> str:
    # DSTs are named inconsistently across sources ("Bills", "Buffalo Bills",
    # "BUF DST"), so match them on team instead of name.
    if position == "DST" and team:
        return f"DST|{normalize_abbr(team)}"
    return normalize_name(name)


_DST_WORDS = re.compile(r"\b(dst|d st|def|defense|defence|special teams)\b")


def match_players(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    """
    For each row of `left` (name, position?, team?), the index of the
    matching row in `right` (the DK player list), or NaN. Names are
    normalized (accents, punctuation, Jr./III suffixes); a team match
    breaks ties; DSTs match on team.
    """
    r = right.assign(_key=[_match_key(n, p, t) for n, p, t in
                           zip(right["name"], right["position"], right["team"])])
    by_key = {k: g for k, g in r.groupby("_key")}
    out = []
    for _, row in left.iterrows():
        pos = str(row.get("position", "") or "").upper().strip()
        team = row.get("team") if isinstance(row.get("team"), str) else None
        is_dst = pos in ("DST", "DEF", "D/ST", "D")
        cand = by_key.get(_match_key(row["name"], "DST" if is_dst else pos, team))
        if cand is None and is_dst and not team:
            # Name-only DST row ("Buffalo Bills", "Bills D/ST", "BUF"): match on the
            # team abbreviation, or a DK DST name the row's name ends with.
            raw = str(row["name"]).strip().upper()
            if normalize_abbr(raw) in set(r["team"]):
                cand = by_key.get(f"DST|{normalize_abbr(raw)}")
            else:
                nm = _DST_WORDS.sub("", normalize_name(row["name"])).strip()
                hits = r[(r["position"] == "DST") & r["name"].map(lambda n: nm.endswith(normalize_name(n)))]
                cand = hits if len(hits) else None
        if cand is not None and len(cand) > 1 and team:
            same = cand[cand["team"] == normalize_abbr(team)]
            cand = same if len(same) else cand
        out.append(cand.index[0] if cand is not None and len(cand) == 1 else np.nan)
    return pd.Series(out, index=left.index, dtype="float")


# Header aliases for projection exports / copied web tables (e.g. GoingFor2,
# which shows player, salary, opponent, projected points, points per
# salary and blended ownership). Headers are compared lowercased with
# everything but letters and digits removed.
PROJECTION_ALIASES = {
    "name": ["name", "player", "playername", "players", "fullname"],
    "position": ["pos", "position", "dkpos", "dkposition", "rosterposition"],
    "team": ["team", "tm", "teamabbrev", "teamabbr"],
    "proj": ["proj", "projection", "projections", "projected", "projectedpoints", "projectedpts",
             "projpts", "projpoints", "projectedfpts", "projfpts", "fpts", "fantasypoints",
             "points", "pts", "dkproj", "dkprojection", "dkfpts", "median", "fppg"],
    "own": ["own", "ownership", "projown", "projectedown", "projectedownership", "blendedownership",
            "blendedown", "blended", "pown", "ownpct", "ownershippct", "ownershippercent", "dkown",
            "dkownership"],
    "ceiling": ["ceiling", "ceil", "ceilingpts", "projceiling"],
    "salary": ["salary", "sal", "dksalary", "dksal"],
}


def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(h).lower())


def standardize_projection_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Rename a projections table's columns to name / position / team / proj /
    own / ceiling / salary using `PROJECTION_ALIASES`, falling back to
    "contains own" / "starts with proj" for unlisted headers. Value
    columns accept strings like "18.5%", "$7,700". Raises with the
    headers it saw if it can't find a name and projection column.
    """
    lookup = {alias: std for std, aliases in PROJECTION_ALIASES.items() for alias in aliases}
    rename, taken = {}, set()
    for col in raw.columns:
        std = lookup.get(_norm_header(col))
        if std and std not in taken:
            rename[col] = std
            taken.add(std)
    for col in raw.columns:
        if col in rename:
            continue
        h = _norm_header(col)
        # "...own", "...ownership", "...ownpct" -- but not e.g. "touchdowns"
        if "own" not in taken and (h.startswith("own") or re.search(r"own(ership)?(pct|percent)?$", h)):
            rename[col] = "own"
            taken.add("own")
        elif "proj" not in taken and h.startswith("proj") and not re.search(r"sal|val|per|own", h):
            rename[col] = "proj"
            taken.add("proj")
    out = raw.rename(columns=rename)[list(rename.values())].copy()
    if not {"name", "proj"} <= set(out.columns):
        raise ValueError(f"Couldn't find player-name and projection columns in: {list(raw.columns)}")
    for col in ("proj", "own", "ceiling", "salary"):
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].astype(str).str.replace(r"[%$,\s]", "", regex=True), errors="coerce")
    out["name"] = out["name"].astype(str).str.strip()
    return out[out["name"].ne("") & out["name"].ne("nan")]


def read_projections(path: str) -> pd.DataFrame:
    """Read a projections CSV/TSV (or a table pasted into a text file) and standardize it."""
    raw = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    return standardize_projection_columns(raw)


def read_lines(path: str, players: pd.DataFrame) -> pd.DataFrame:
    """
    Vegas lines typed in by hand: one row per team (at least one team per
    game) with `team`, `spread` (negative = favored) and `total` (or
    `game_total`), optionally `team_total`. The opponent's row is inferred
    from the slate. Returns team, game_total, spread, team_total.
    """
    raw = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    raw.columns = [_norm_header(c) for c in raw.columns]
    raw = raw.rename(columns={"gametotal": "total", "ou": "total", "overunder": "total",
                              "teamtotal": "team_total", "tm": "team"})
    if not {"team", "spread", "total"} <= set(raw.columns):
        raise ValueError("Lines file needs columns: team, spread, total (optional team_total)")
    opp_of = players.drop_duplicates("team").set_index("team")["opp"].to_dict()
    rows = {}
    for r in raw.itertuples():
        team = normalize_abbr(r.team)
        total, spread = float(r.total), float(r.spread)
        tt = getattr(r, "team_total", np.nan)
        tt = float(tt) if pd.notna(tt) else total / 2 - spread / 2
        rows[team] = {"team": team, "game_total": total, "spread": spread, "team_total": tt}
        opp = opp_of.get(team)
        if opp and opp not in rows:
            rows[opp] = {"team": opp, "game_total": total, "spread": -spread,
                         "team_total": total - tt}
    return pd.DataFrame(rows.values())


def apply_lines(pool: pd.DataFrame, lines: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fill game_total / spread / team_total per team; returns (pool, teams with no line)."""
    out = pool.copy()
    by_team = lines.set_index("team")
    for col in ("game_total", "spread", "team_total"):
        out[col] = out["team"].map(by_team[col])
    missing = sorted(set(out["team"]) - set(by_team.index))
    return out, missing


def pool_from_entry_file(ef: DKEntryFile, projections: pd.DataFrame | None = None
                         ) -> tuple[pd.DataFrame, list[str]]:
    """
    Engine player pool built from the entry file's player list. With
    `projections` (name + projection columns, optionally team, position,
    ownership, ceiling -- header names are flexible, see
    `PROJECTION_ALIASES`), those values are merged in; returns (pool, unmatched names
    from `projections`). Vegas-line columns are left blank for sgo-enrich.
    """
    p = ef.players
    pool = pd.DataFrame({
        "name": p["name"], "team": p["team"], "opp": p["opp"], "position": p["position"],
        "salary": p["salary"], "proj": np.nan, "own": np.nan, "ceiling": np.nan,
        "team_total": np.nan, "game_total": np.nan, "spread": np.nan,
        "dk_id": p["dk_id"], "game_time": p["game_time"], "avg_ppg": p["avg_ppg"],
    })[POOL_COLUMNS]
    unmatched: list[str] = []
    if projections is not None:
        proj = standardize_projection_columns(projections)
        idx = match_players(proj, p)
        unmatched = proj.loc[idx.isna(), "name"].astype(str).tolist()
        for col in ("proj", "own", "ceiling"):
            if col in proj.columns:
                vals = pd.to_numeric(proj[col], errors="coerce")
                hit = idx.notna()
                pool.loc[idx[hit].astype(int).to_numpy(), col] = vals[hit].to_numpy()
    return pool, unmatched


def attach_dk_ids(pool: pd.DataFrame, ef: DKEntryFile) -> tuple[pd.DataFrame, list[str]]:
    """Add `dk_id` / `game_time` to a pool that lacks them; returns (pool, unmatched names)."""
    idx = match_players(pool, ef.players)
    out = pool.copy()
    hit = idx.notna()
    out["dk_id"] = None
    out.loc[hit, "dk_id"] = ef.players.loc[idx[hit].astype(int), "dk_id"].to_numpy()
    if "game_time" not in out.columns:
        out["game_time"] = pd.NaT
        out.loc[hit, "game_time"] = ef.players.loc[idx[hit].astype(int), "game_time"].to_numpy()
    return out, out.loc[~hit, "name"].astype(str).tolist()


# ---------------------------------------------------------------------------
# Contests
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)\s*([KMB]?)\b", re.I)
_FIRST_RE = re.compile(r"\[\s*\$([\d,]+(?:\.\d+)?)\s*([KMB]?)\s+to\s+1st", re.I)
_MULT = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}


def _parse_amount(m) -> float:
    return float(m.group(1).replace(",", "")) * _MULT[m.group(2).upper()]


def estimate_contests(entries: pd.DataFrame, rake: float = 0.15,
                      default_first_frac: float = 0.20, freeroll_size: int = 50_000) -> pd.DataFrame:
    """
    One row per contest with editable assumptions. The prize pool (and a
    "[$X to 1st]" prize, when present) is parsed from the contest name;
    size is estimated as prize_pool / (fee x (1 - rake)). Freerolls have no
    fee, so their size is a placeholder guess -- check the DK lobby and
    edit `contest_size`.
    """
    rows = []
    for (cid, name), g in entries.groupby(["contest_id", "contest_name"], sort=False):
        fee = float(g["entry_fee"].iloc[0])
        m = _MONEY_RE.search(name)
        pool = _parse_amount(m) if m else np.nan
        f = _FIRST_RE.search(name)
        first = _parse_amount(f) if f else np.nan
        if fee > 0 and np.isfinite(pool):
            size, source = int(round(pool / (fee * (1 - rake)))), "estimated from prize pool"
        else:
            size, source = freeroll_size, "GUESS - check lobby"
        rows.append({
            "contest_id": cid, "contest_name": name.strip(), "entry_fee": fee,
            "our_entries": len(g), "prize_pool": pool, "contest_size": size,
            "first_prize": first,
            "first_frac": round(first / pool, 4) if np.isfinite(first) and pool else default_first_frac,
            "paid_frac": 0.22, "payouts_file": "", "size_source": source,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Slots + upload
# ---------------------------------------------------------------------------

def assign_slots(player_ids: list[int], pool: pd.DataFrame) -> list[str]:
    """
    DK IDs in slot order QB, RB, RB, WR, WR, WR, TE, FLEX, DST. The FLEX
    goes to the latest-kickoff player of the position with an extra body,
    which keeps late-swap options open.
    """
    lu = pool.loc[list(player_ids)]
    times = pd.to_datetime(lu["game_time"], errors="coerce") if "game_time" in lu else pd.Series(pd.NaT, index=lu.index)
    times = times.fillna(pd.Timestamp.min)
    by_pos = {p: list(lu.index[lu["position"] == p]) for p in ("QB", "RB", "WR", "TE", "DST")}
    if len(by_pos["QB"]) != 1 or len(by_pos["DST"]) != 1:
        raise ValueError("Lineup needs exactly one QB and one DST")
    extra = [p for p, base in BASE_COUNTS.items() if len(by_pos[p]) == base + 1]
    if len(extra) != 1 or any(len(by_pos[p]) < b for p, b in BASE_COUNTS.items()):
        raise ValueError(f"Not a legal Classic roster: { {p: len(v) for p, v in by_pos.items()} }")
    flex_group = sorted(by_pos[extra[0]], key=lambda i: times[i])
    flex = flex_group[-1]
    by_pos[extra[0]] = flex_group[:-1]
    order = by_pos["QB"] + by_pos["RB"] + by_pos["WR"] + by_pos["TE"] + [flex] + by_pos["DST"]
    return [str(pool.at[i, "dk_id"]) for i in order]


def write_upload(ef: DKEntryFile, slot_ids: dict[str, list[str]], path: str) -> int:
    """
    Write the DraftKings upload file for the entries in `slot_ids`
    (entry_id -> 9 DK IDs in slot order). Entry columns are copied
    verbatim from the downloaded file. Returns the number of rows written.
    """
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(ENTRY_COLUMNS + ef.slots)
        for e in ef.entries.itertuples():
            ids = slot_ids.get(e.entry_id)
            if ids is None:
                continue
            if len(ids) != len(ef.slots):
                raise ValueError(f"Entry {e.entry_id}: expected {len(ef.slots)} players, got {len(ids)}")
            w.writerow([e.entry_id, e.contest_name, e.contest_id, e.entry_fee_raw] + list(ids))
            n += 1
    return n


def validate_upload_lineup(ids: list[str], players: pd.DataFrame, salary_cap: int = 50_000) -> None:
    """Re-check a finished lineup against DK's Classic rules using the DK player list."""
    p = players.set_index("dk_id").loc[ids]
    if len(set(ids)) != 9:
        raise ValueError("Duplicate player in lineup")
    if p["salary"].sum() > salary_cap:
        raise ValueError(f"Salary {p['salary'].sum()} over cap")
    if p["game"].nunique() < 2:
        raise ValueError("Players must come from at least 2 games")
    need = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", None, "DST"]
    for slot_pos, (_, row) in zip(need, p.iterrows()):
        if slot_pos and row["position"] != slot_pos:
            raise ValueError(f"{row['name']} ({row['position']}) in {slot_pos} slot")
    if p.iloc[7]["position"] not in ("RB", "WR", "TE"):
        raise ValueError("FLEX must be RB/WR/TE")
