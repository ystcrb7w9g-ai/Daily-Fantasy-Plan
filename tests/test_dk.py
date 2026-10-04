import csv
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dfs_engine import dk
from dfs_engine.data import load_player_pool
from dfs_engine.optimize import solve_classic, solve_showdown
from test_contest import _value_owned_pool

ROOT = os.path.join(os.path.dirname(__file__), "..")
POOL_PATH = os.path.join(ROOT, "data", "sample_classic_pool.csv")
DST_NAMES = {"AAA": "Aardvarks", "BBB": "Bobcats", "CCC": "Cougars", "DDD": "Dingos"}
KICKOFF = {"AAA": "01:00PM", "BBB": "01:00PM", "CCC": "04:25PM", "DDD": "04:25PM"}
CONTESTS = [
    ("NFL $2.75M Fantasy Football Millionaire [$1M to 1st]", "111", "$20", 1),
    ("$2K Free Football Takeover", "222", "$0", 1),
    ("NFL $20K Quarter Jukebox [Just $0.25!] ", "333", "$0.25", 3),
]


def _write_entry_file(path, df):
    """Synthetic DKEntries.csv in DraftKings' layout, built from a test pool."""
    header = dk.ENTRY_COLUMNS + dk.CLASSIC_SLOTS + ["", "Instructions"]
    entries = []
    eid = 9000
    for name, cid, fee, n in CONTESTS:
        for _ in range(n):
            eid += 1
            entries.append([str(eid), name, cid, fee] + [""] * 9 + [""])
    players = []
    for i, r in df.iterrows():
        home, away = (r["team"], r["opp"]) if r["team"] in ("AAA", "CCC") else (r["opp"], r["team"])
        name = DST_NAMES[r["team"]] + " " if r["position"] == "DST" else r["name"]
        roster = r["position"] if r["position"] in ("QB", "DST") else f"{r['position']}/FLEX"
        players.append([r["position"], f"{name} ({50000 + i})", name, str(50000 + i), roster,
                        str(r["salary"]), f"{away}@{home} 10/04/2026 {KICKOFF[r['team']]} ET",
                        r["team"], "12.5"])
    pad = [""] * 14
    rows = [header]
    n_rows = max(len(entries), 7 + 1 + len(players))
    for k in range(n_rows):
        left = entries[k] if k < len(entries) else pad
        left = (left + [""] * 14)[:14]
        if k < 6:
            right = [f"{k + 1}. instructions"]
        elif k == 6:
            right = ["Position", "Name + ID", "Name", "ID", "Roster Position", "Salary",
                     "Game Info", "TeamAbbrev", "AvgPointsPerGame"]
        elif k - 7 < len(players):
            right = players[k - 7]
        else:
            right = []
        rows.append(left + right)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def _alpha(i):
    s = ""
    i += 26
    while i:
        i, r = divmod(i, 26)
        s = chr(97 + r) + s
    return s.capitalize()


def _fixture():
    df = _value_owned_pool()
    # Distinct alphabetic names (the matcher drops digits/punctuation).
    df["name"] = [f"{_alpha(i)} {df.at[i, 'position'].title()}man" for i in df.index]
    d = tempfile.mkdtemp()
    path = os.path.join(d, "DKEntries.csv")
    _write_entry_file(path, df)
    return df, path, d


# --- parsing ---------------------------------------------------------------

def test_read_entry_file_parses_entries_and_players():
    df, path, _ = _fixture()
    ef = dk.read_entry_file(path)
    assert len(ef.entries) == 5
    assert ef.entries["entry_fee"].tolist() == [20.0, 0.0, 0.25, 0.25, 0.25]
    assert len(ef.players) == len(df)
    row = ef.players[ef.players["team"] == "BBB"].iloc[0]
    assert row["opp"] == "AAA" and row["game"] == "BBB@AAA"
    assert ef.players.loc[ef.players["team"] == "CCC", "game_time"].iloc[0].hour == 16
    assert (ef.players.loc[ef.players["position"] == "DST", "name"] == "Aardvarks").sum() == 1


def test_estimate_contests_parses_names():
    _, path, _ = _fixture()
    t = dk.estimate_contests(dk.read_entry_file(path).entries).set_index("contest_id")
    milly = t.loc["111"]
    assert milly["prize_pool"] == 2_750_000 and milly["first_prize"] == 1_000_000
    assert abs(milly["first_frac"] - 1 / 2.75) < 1e-3
    assert milly["contest_size"] == round(2_750_000 / (20 * 0.85))
    assert t.loc["222", "size_source"].startswith("GUESS")
    assert t.loc["333", "our_entries"] == 3 and t.loc["333", "prize_pool"] == 20_000


def test_match_players_handles_suffixes_punctuation_and_dsts():
    right = pd.DataFrame({
        "name": ["Kenneth Walker III", "C.J. Stroud", "Ja'Marr Chase", "Bills", "Aaron Jones Sr."],
        "position": ["RB", "QB", "WR", "DST", "RB"], "team": ["KC", "HOU", "CIN", "BUF", "MIN"],
    })
    left = pd.DataFrame({
        "name": ["Kenneth Walker", "CJ Stroud", "JaMarr Chase", "Buffalo Bills", "BUF D/ST", "Nobody"],
        "position": ["RB", "QB", "WR", "DST", "DST", "WR"], "team": [None, None, None, None, "BUF", None],
    })
    idx = dk.match_players(left, right)
    assert idx.tolist()[:5] == [0, 1, 2, 3, 3]
    assert np.isnan(idx.iloc[5])


def test_pool_from_entry_file_merges_projections():
    df, path, _ = _fixture()
    ef = dk.read_entry_file(path)
    proj = df[["name", "team", "position", "proj", "own"]].copy()
    proj.loc[proj["position"] == "DST", "name"] = "whatever"  # DSTs match on team
    proj = pd.concat([proj, pd.DataFrame([{"name": "Not On Slate", "proj": 9.0}])])
    pool, unmatched = dk.pool_from_entry_file(ef, proj)
    assert unmatched == ["Not On Slate"]
    assert pool["proj"].notna().sum() == len(df)
    assert pool["dk_id"].str.match(r"^5\d{4}$").all()


# --- optimizer DK rules ----------------------------------------------------

def test_classic_requires_two_games():
    df = _value_owned_pool()
    df["salary"] = 3000
    pts = np.where(df["team"].isin(["AAA", "BBB"]), 30.0, 1.0)
    lu = solve_classic(df, pts)
    assert df.loc[lu.player_ids, "team"].isin(["AAA", "BBB"]).sum() == 8


def test_showdown_requires_both_teams():
    df = load_player_pool(POOL_PATH)
    df["salary"] = 1000
    pts = np.where(df["team"] == "BUF", 30.0, 1.0)
    lu = solve_showdown(df, pts)
    assert (df.loc[lu.player_ids, "team"] == "BUF").sum() == 5


# --- slots + upload --------------------------------------------------------

def test_assign_slots_orders_and_puts_latest_kickoff_in_flex():
    df, path, _ = _fixture()
    ef = dk.read_entry_file(path)
    pool, _ = dk.pool_from_entry_file(ef, df[["name", "team", "position", "proj", "own"]])
    pool = pool.reset_index(drop=True)
    def pick(team, pos, k):  # cheapest k, so the lineup stays under the cap
        rows = pool[(pool["team"] == team) & (pool["position"] == pos)].sort_values("salary")
        return rows.index[:k].tolist()
    lineup = (pick("AAA", "QB", 1) + pick("AAA", "RB", 2) + pick("CCC", "RB", 1)
              + pick("AAA", "WR", 3) + pick("BBB", "TE", 1) + pick("BBB", "DST", 1))
    ids = dk.assign_slots(lineup, pool)
    by_id = pool.set_index("dk_id")
    assert [by_id.loc[i, "position"] for i in ids] == ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB", "DST"]
    assert by_id.loc[ids[7], "team"] == "CCC"  # the 4:25 RB is the FLEX
    dk.validate_upload_lineup(ids, ef.players)


def test_validate_upload_lineup_rejects_bad_rosters():
    df, path, _ = _fixture()
    ef = dk.read_entry_file(path)
    p = ef.players
    one_game = (p[(p["game"] == "BBB@AAA") & (p["position"] == "QB")]["dk_id"][:1].tolist()
                + p[(p["game"] == "BBB@AAA") & (p["position"] == "RB")]["dk_id"][:2].tolist()
                + p[(p["game"] == "BBB@AAA") & (p["position"] == "WR")]["dk_id"][:3].tolist()
                + p[(p["game"] == "BBB@AAA") & (p["position"] == "TE")]["dk_id"][:1].tolist()
                + p[(p["game"] == "BBB@AAA") & (p["position"] == "WR")]["dk_id"][3:4].tolist()
                + p[(p["game"] == "BBB@AAA") & (p["position"] == "DST")]["dk_id"][:1].tolist())
    p = p.assign(salary=1000)
    try:
        dk.validate_upload_lineup(one_game, p)
        raise AssertionError("single-game lineup accepted")
    except ValueError as e:
        assert "2 games" in str(e)


def test_write_upload_copies_entry_columns_verbatim():
    df, path, d = _fixture()
    ef = dk.read_entry_file(path)
    ids = [str(x) for x in range(9)]
    out = os.path.join(d, "up.csv")
    first = ef.entries["entry_id"].iloc[-1]
    assert dk.write_upload(ef, {first: ids}, out) == 1
    rows = list(csv.reader(open(out, encoding="utf-8")))
    assert rows[0] == dk.ENTRY_COLUMNS + dk.CLASSIC_SLOTS
    assert rows[1][:4] == [first, "NFL $20K Quarter Jukebox [Just $0.25!] ", "333", "$0.25"]
    assert rows[1][4:] == ids


# --- loader ----------------------------------------------------------------

def test_load_player_pool_drops_unprojected_and_requires_lines():
    d = tempfile.mkdtemp()
    raw = pd.read_csv(POOL_PATH)
    raw.loc[0, "proj"] = np.nan
    raw.loc[1, "own"] = np.nan
    raw.to_csv(os.path.join(d, "a.csv"), index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pool = load_player_pool(os.path.join(d, "a.csv"))
    assert len(pool) == len(raw) - 1 and pool["own"].notna().all()
    assert (pool["player_id"] == np.arange(len(pool))).all()
    raw.loc[2, "team_total"] = np.nan
    raw.to_csv(os.path.join(d, "b.csv"), index=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_player_pool(os.path.join(d, "b.csv"))
        raise AssertionError("missing Vegas lines accepted")
    except ValueError as e:
        assert "team_total" in str(e)


# --- end to end ------------------------------------------------------------

def test_dk_run_end_to_end():
    df, path, d = _fixture()
    ef = dk.read_entry_file(path)
    pool, _ = dk.pool_from_entry_file(ef, df[["name", "team", "position", "proj", "own"]])
    lines = df.set_index("team")[["team_total", "game_total", "spread"]].groupby(level=0).first()
    for col in lines.columns:
        pool[col] = pool["team"].map(lines[col])
    pool_path = os.path.join(d, "pool.csv")
    pool.to_csv(pool_path, index=False)
    up = os.path.join(d, "upload.csv")
    cmd = [sys.executable, "-W", "ignore", "-m", "dfs_engine.cli", "dk-run", "--pool", pool_path,
           "--entries", path, "--candidates", "25", "--gen-trials", "200", "--eval-trials", "300",
           "--holdout-trials", "200", "--field-size", "1500", "--min-unique", "1", "--seed", "3",
           "--upload-out", up, "--report-out", os.path.join(d, "report.csv")]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    rows = list(csv.reader(open(up, encoding="utf-8")))
    assert len(rows) == 1 + len(ef.entries)
    for r in rows[1:]:
        dk.validate_upload_lineup(r[4:13], ef.players)
    jukebox = [frozenset(r[4:13]) for r in rows[1:] if r[2] == "333"]
    assert len(set(jukebox)) == 3

    res = subprocess.run(cmd + ["--unique-across-contests"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    rows = list(csv.reader(open(up, encoding="utf-8")))
    lineups = [frozenset(r[4:13]) for r in rows[1:]]
    assert len(set(lineups)) == len(lineups)  # no lineup reused across contests
