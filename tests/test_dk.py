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

    # a chalkier field for one contest, one exposure budget for the slate, ownership uncertainty
    contests = dk.estimate_contests(ef.entries)
    contests["field_sharpen"] = np.where(contests["contest_id"] == "333", 1.3, 1.0)
    contests_path = os.path.join(d, "contests.csv")
    contests.to_csv(contests_path, index=False)
    res = subprocess.run(cmd + ["--contests", contests_path, "--global-exposure", "--max-exposure", "0.6",
                                "--field-own-sd", "0.3"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "ownership sharpened to the power 1.3" in res.stdout
    assert "ownership uncertainty: 4 ownership draws" in res.stdout
    rows = list(csv.reader(open(up, encoding="utf-8")))
    for r in rows[1:]:
        dk.validate_upload_lineup(r[4:13], ef.players)
    if "note: only" not in res.stdout:  # caps held without relaxing: no player in more than 60% of all 5
        counts = pd.Series([p for r in rows[1:] for p in r[4:13]]).value_counts()
        assert counts.max() <= 3


# --- projections / lines import -------------------------------------------

def test_standardize_projection_columns_handles_web_table_headers():
    raw = pd.DataFrame({
        "Player": ["Josh Allen", "Bills D/ST"], "Pos": ["QB", "DST"], "Salary": ["$7,700", "$3,000"],
        "Projected Points": ["24.1", "8.0"], "Pts/Salary": ["3.13", "2.67"],
        "Blended Ownership": ["18.5%", "9%"], "Rush Touchdowns": ["0.4", "0"],
    })
    out = dk.standardize_projection_columns(raw)
    assert set(out.columns) == {"name", "position", "salary", "proj", "own"}
    assert out["proj"].tolist() == [24.1, 8.0]
    assert out["own"].tolist() == [18.5, 9.0]
    assert out["salary"].tolist() == [7700, 3000]


def test_standardize_projection_columns_reports_unknown_headers():
    try:
        dk.standardize_projection_columns(pd.DataFrame({"Who": ["x"], "How many": [1]}))
        raise AssertionError("accepted a table with no name/projection columns")
    except ValueError as e:
        assert "Who" in str(e)


def test_read_projections_reads_pasted_tab_separated_table():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "paste.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Player\tPos\tProj\tOwn%\nJosh Allen\tQB\t24.1\t18.5%\nCJ Stroud\tQB\t17.2\t6%\n")
    out = dk.read_projections(path)
    assert out["name"].tolist() == ["Josh Allen", "CJ Stroud"] and out["own"].tolist() == [18.5, 6.0]


def test_dst_name_variants_match():
    right = pd.DataFrame({"name": ["Bills", "Ravens"], "position": ["DST", "DST"], "team": ["BUF", "BAL"]})
    left = pd.DataFrame({"name": ["Bills D/ST", "BUF", "Baltimore Ravens Defense", "Ravens DST"],
                         "position": ["DST", "DEF", "D", "DST"]})
    assert dk.match_players(left, right).tolist() == [0, 0, 1, 1]


def test_read_lines_infers_opponent_and_team_totals():
    df, path, d = _fixture()
    ef = dk.read_entry_file(path)
    lp = os.path.join(d, "lines.csv")
    with open(lp, "w") as f:
        f.write("team,spread,total\nAAA,-6.5,47.5\nDDD,3,41\n")
    lines = dk.read_lines(lp, ef.players).set_index("team")
    assert lines.loc["AAA", "team_total"] == 27.0 and lines.loc["BBB", "team_total"] == 20.5
    assert lines.loc["BBB", "spread"] == 6.5 and lines.loc["CCC", "spread"] == -3.0
    pool, _ = dk.pool_from_entry_file(ef)
    pool, missing = dk.apply_lines(pool, lines.reset_index())
    assert missing == [] and pool["game_total"].notna().all()


def test_cli_field_defaults_track_engine_defaults():
    from dfs_engine import contest as cs
    out = subprocess.run([sys.executable, "-m", "dfs_engine.cli", "dk-run", "--help"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    assert ",".join(map(str, cs.DEFAULT_STACK_MIX)) in out.replace("\n", "").replace(" ", "")


def test_reads_dff_cheatsheet_layout():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "DFF_NFL_cheatsheet.csv")
    pd.DataFrame({"first_name": ["Jahmyr", "Texans", "Hurt"], "last_name": ["Gibbs", None, "Guy"],
                  "position": ["RB", "DST", "WR"], "injury_status": [None, None, "O"], "team": ["DET", "HOU", "DET"],
                  "salary": [9800, 3400, 4000], "ppg_projection": [28.6, 8.2, 9.0],
                  "ownership_projection": [None, None, None]}).to_csv(path, index=False)
    p = dk.read_projections(path)
    assert list(p["name"]) == ["Jahmyr Gibbs", "Texans", "Hurt Guy"]
    assert p.loc[0, "proj"] == 28.6 and p.loc[1, "position"] == "DST"
    assert np.isnan(p.loc[2, "proj"])  # ruled out


# --- SaberSim exports -------------------------------------------------------

PCT_MULT = {25: 0.5, 50: 0.9, 75: 1.4, 85: 1.7, 95: 2.3, 99: 3.1}


def _sabersim_export(df, ef, path):
    """SaberSim-layout DK projections export for the fixture slate."""
    p = ef.players.set_index("name")
    rows = []
    for i, r in df.iterrows():
        dk_name = DST_NAMES[r["team"]] if r["position"] == "DST" else r["name"]
        pos = r["position"] if r["position"] in ("QB", "DST") else f"{r['position']}/FLEX"
        row = {"DFS ID": p.loc[dk_name, "dk_id"], "Name": dk_name, "Pos": pos, "Team": r["team"], "Opp": r["opp"],
               "Status": "", "Salary": r["salary"], "Actual": 0, "SS Proj": r["proj"], "Live Proj": None,
               "My Proj": r["proj"], "Value": r["proj"] / r["salary"] * 1000, "My Own": r["own"] * 100,
               "Adj Own": r["own"] * 105, "Min Exp": 0, "Max Exp": 100,
               "Saber Team": r["team_total"] + 0.5, "Saber Total": r["game_total"] + 1.0, "dk_points": r["proj"]}
        for q, m in PCT_MULT.items():
            row[f"dk_{q}_percentile"] = r["proj"] * m
        row["dk_std"] = r["proj"] * 0.7
        row.update({"Pass Att": 34.0 if r["position"] == "QB" else 0.0, "Rec": 4.0 if r["position"] in ("WR", "TE", "RB") else 0.0,
                    "Rec TD": 0.3 if r["position"] in ("WR", "TE") else 0.1, "Rush Att": 14.0 if r["position"] == "RB" else 0.5,
                    "Rush TD": 0.4 if r["position"] == "RB" else 0.05})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.loc[0, "Name"] = "Renamed On SaberSim"         # matched by DFS ID anyway
    out.loc[1, ["SS Proj", "My Proj"]] = 0.0            # SaberSim expects him inactive
    out.loc[2, "Status"] = "O"
    out.loc[3, ["Min Exp", "Max Exp"]] = [20, 60]
    out.to_csv(path, index=False)
    return out


def test_reads_sabersim_export_into_pool():
    df, path, d = _fixture()
    ef = dk.read_entry_file(path)
    ss = _sabersim_export(df, ef, os.path.join(d, "NFL_DK_Main.csv"))
    p = dk.read_projections(os.path.join(d, "NFL_DK_Main.csv"))
    assert {"dk_id", "p25", "p99", "min_exp", "max_exp", "team_total", "game_total"} <= set(p.columns)
    assert p["position"].isin(["QB", "RB", "WR", "TE", "DST"]).all()
    assert np.isnan(p.loc[1, "proj"]) and np.isnan(p.loc[2, "proj"]) and np.isnan(p.loc[1, "own"])
    assert p.loc[3, "min_exp"] == 0.2 and p.loc[3, "max_exp"] == 0.6
    assert p.loc[4, "ceiling"] == p.loc[4, "p85"]

    pool, unmatched = dk.pool_from_entry_file(ef, p)
    assert unmatched == []
    by_id = pool.set_index("dk_id")
    first = by_id.loc[str(ss.loc[0, "DFS ID"])]
    assert first["proj"] == ss.loc[0, "My Proj"]       # renamed player still matched by id
    assert pool["proj"].notna().sum() == len(df) - 2
    team_tt = df.groupby("team")["team_total"].first() + 0.5
    assert np.allclose(pool["team_total"], pool["team"].map(team_tt))
    assert np.allclose(pool["spread"], pool["game_total"] - 2 * pool["team_total"])
    sim_ready = load_player_pool_from(pool, d)
    assert {"p25", "p99", "min_exp"} <= set(sim_ready.columns)
    # SaberSim's stat projections ride along for the stat engine
    assert {"proj_pass_att", "proj_rec", "proj_rush_att", "proj_rec_td", "proj_rush_td"} <= set(pool.columns)
    rb = pool[(pool["position"] == "RB") & pool["proj"].notna()].iloc[0]
    assert rb["proj_rush_att"] == 14.0 and rb["proj_rec"] == 4.0


def load_player_pool_from(pool, d):
    path = os.path.join(d, "pool_from_ss.csv")
    pool.to_csv(path, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_player_pool(path)


def test_read_lineup_file_reads_ids_and_name_id_cells():
    d = tempfile.mkdtemp()
    a = os.path.join(d, "saber.csv")
    with open(a, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(dk.ENTRY_COLUMNS + dk.CLASSIC_SLOTS)
        w.writerow(["1", "C", "9", "$1"] + [str(10000 + i) for i in range(9)])
        w.writerow(["2", "C", "9", "$1"] + [""] * 9)  # reservation: skipped
    b = os.path.join(d, "DKEntries.csv")
    with open(b, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(dk.ENTRY_COLUMNS + dk.CLASSIC_SLOTS + ["", "Instructions"])
        w.writerow(["1", "C", "9", "$1"] + [f"Player {i} ({20000 + i})" for i in range(9)] + ["", "1. text"])
    assert dk.read_lineup_file(a) == [[str(10000 + i) for i in range(9)]]
    assert dk.read_lineup_file(b) == [[str(20000 + i) for i in range(9)]]


def test_sabersim_pool_to_dk_run_with_extra_lineups_and_exposure_flags():
    df, path, d = _fixture()
    ef = dk.read_entry_file(path)
    ss = os.path.join(d, "NFL_DK_Main.csv")
    _sabersim_export(df, ef, ss)
    pool_path = os.path.join(d, "pool.csv")
    res = subprocess.run([sys.executable, "-W", "ignore", "-m", "dfs_engine.cli", "dk-pool", "--entries", path,
                          "--projections", ss, "--out", pool_path], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    pool = pd.read_csv(pool_path)
    assert pool["p99"].notna().sum() == len(df) - 2 and pool["spread"].notna().all()
    assert pool["min_exp"].max() == 0.2  # SaberSim's Min Exp carried into the pool ...
    pool[["min_exp", "max_exp"]] = np.nan  # ... cleared so only the flags below apply
    pool.to_csv(pool_path, index=False)

    base = [sys.executable, "-W", "ignore", "-m", "dfs_engine.cli", "dk-run", "--pool", pool_path,
            "--entries", path, "--candidates", "25", "--gen-trials", "200", "--eval-trials", "300",
            "--holdout-trials", "200", "--field-size", "1500", "--min-unique", "1", "--seed", "3"]
    first = os.path.join(d, "first.csv")
    res = subprocess.run(base + ["--upload-out", first], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "carry their own p25-p99 percentiles" in res.stdout

    proj = pool[pool["proj"].notna() & (pool["position"] != "DST")]
    ban = proj.sort_values("proj").iloc[-1]["name"]                  # the best player: banned
    must = proj[proj["name"] != ban].sort_values("proj").iloc[0]["name"]  # a scrub: forced in
    second = os.path.join(d, "second.csv")
    res = subprocess.run(base + ["--upload-out", second, "--extra-lineups", first,
                                 "--max-exp", f"{ban}=0", "--min-exp", f"{must}=60%"],
                         cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "lineups from first.csv" in res.stdout and "Player exposure limits (2)" in res.stdout
    assert "WARNING" not in res.stdout
    ids = pool.set_index("name")["dk_id"].astype(str)
    rows = list(csv.reader(open(second, encoding="utf-8")))[1:]
    for r in rows:
        dk.validate_upload_lineup(r[4:13], ef.players)
        assert ids[ban] not in r[4:13]
    by_contest = {}
    for r in rows:
        by_contest.setdefault(r[2], []).append(ids[must] in r[4:13])
    assert all(sum(v) >= np.ceil(0.6 * len(v)) for v in by_contest.values())  # 1 of 1, 2 of 3


# --- Showdown --------------------------------------------------------------

SD_DST = {"ATL": "Falcons", "NO": "Saints"}


def _showdown_fixture():
    """A synthetic DK Showdown entries file (each player as a CPT and a FLEX row) and a
    SaberSim-layout export with both rows, as for a real single-game slate."""
    from test_contest import _showdown_pool
    df = _showdown_pool()
    df["name"] = [SD_DST[t] if p == "DST" else f"{_alpha(i)} {p.title()}man"
                  for i, (t, p) in enumerate(zip(df["team"], df["position"]))]
    d = tempfile.mkdtemp()
    path = os.path.join(d, "DKEntries_SD.csv")
    header = dk.ENTRY_COLUMNS + dk.SHOWDOWN_SLOTS + ["", "Instructions"]
    name = "NFL Showdown $500K Monday Night Showdown [$100K to 1st] (NO @ ATL)"
    entries = [[str(7001 + k), name, "444", "$5"] + [""] * 6 + [""] for k in range(2)]
    players = []
    for i, r in df.iterrows():
        for slot, pid, sal in (("CPT", 60000 + i, int(r["salary"] * 1.5)), ("FLEX", 50000 + i, int(r["salary"]))):
            players.append([r["position"], f"{r['name']} ({pid})", r["name"], str(pid), slot, str(sal),
                            "NO@ATL 10/12/2026 08:15PM ET", r["team"], "11.0"])
    rows = [header]
    for k in range(max(len(entries), 8 + len(players))):
        left = (entries[k] if k < len(entries) else []) + [""] * 11
        left = left[:11]
        if k < 6:
            right = [f"{k + 1}. instructions"]
        elif k == 6:
            right = ["Position", "Name + ID", "Name", "ID", "Roster Position", "Salary", "Game Info",
                     "TeamAbbrev", "AvgPointsPerGame"]
        elif k - 7 < len(players):
            right = players[k - 7]
        else:
            right = []
        rows.append(left + right)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    cpt_own = np.full(len(df), 0.70 / 12)
    cpt_own[[0, 7]] = [0.25, 0.05]
    ss_rows = []
    for i, r in df.iterrows():
        for pid, mult, own in ((60000 + i, 1.5, cpt_own[i] * 100), (50000 + i, 1.0, 500 / len(df))):
            row = {"DFS ID": pid, "Name": r["name"], "Pos": r["position"], "Team": r["team"], "Opp": r["opp"],
                   "Status": "", "Salary": int(r["salary"] * mult), "SS Proj": r["proj"] * mult,
                   "My Proj": r["proj"] * mult, "My Own": own, "Adj Own": own, "Min Exp": 0, "Max Exp": 100,
                   "Saber Team": 22.5, "Saber Total": 45.0}
            for q, m in PCT_MULT.items():
                row[f"dk_{q}_percentile"] = r["proj"] * mult * m
            ss_rows.append(row)
    ss_path = os.path.join(d, "NFL_DK_SHOWDOWN_NO--ATL.csv")
    pd.DataFrame(ss_rows).to_csv(ss_path, index=False)
    return df, path, ss_path, d


def test_reads_showdown_entry_file_and_sabersim_captain_rows():
    df, path, ss_path, d = _showdown_fixture()
    ef = dk.read_entry_file(path)
    assert ef.fmt == "showdown" and ef.slots == dk.SHOWDOWN_SLOTS and len(ef.entries) == 2
    assert len(ef.players) == 2 * len(df)
    pool, unmatched = dk.pool_from_entry_file(ef, dk.read_projections(ss_path))
    assert unmatched == [] and len(pool) == len(df)
    assert (pool["dk_id"].astype(int) < 60000).all() and (pool["cpt_dk_id"].astype(int) >= 60000).all()
    assert np.isclose(pool["cpt_own"].sum(), 100) and np.isclose(pool["own"].sum(), 500)
    assert pool.loc[0, "cpt_own"] == 25.0 and pool.loc[0, "proj"] == df.loc[0, "proj"]  # FLEX row's projection
    lu = solve_showdown(load_player_pool_from(pool, d), pool["proj"].to_numpy())
    ids = dk.assign_showdown_slots(lu.player_ids, lu.captain_id, pool)
    dk.validate_showdown_upload(ids, ef.players)
    assert ids[0] == str(pool.loc[lu.captain_id, "cpt_dk_id"])
    bad = [str(pool.loc[lu.captain_id, "dk_id"])] + ids[1:]          # FLEX id in the CPT slot
    with np.testing.assert_raises(ValueError):
        dk.validate_showdown_upload(bad, ef.players)


def test_showdown_pool_to_dk_run_with_sabersim_lineups():
    df, path, ss_path, d = _showdown_fixture()
    pool_path = os.path.join(d, "pool.csv")
    res = subprocess.run([sys.executable, "-W", "ignore", "-m", "dfs_engine.cli", "dk-pool", "--entries", path,
                          "--projections", ss_path, "--out", pool_path], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    assert "captain 100%" in res.stdout
    ef = dk.read_entry_file(path)
    pool = pd.read_csv(pool_path)
    flex_of = dict(zip(pool["name"], pool["dk_id"].astype(str)))
    cpt_of = dict(zip(pool["name"], pool["cpt_dk_id"].astype(str)))
    names = pool["name"].tolist()
    saber = os.path.join(d, "lineups_dk_nfl_showdown.csv")
    with open(saber, "w", newline="") as f:  # SaberSim's lineups export: CPT + 5 FLEX ids, then its metrics
        w = csv.writer(f)
        w.writerow(["CPT", "FLEX", "FLEX", "FLEX", "FLEX", "FLEX", "", "Proj Score", "Saber Score"])
        w.writerow([cpt_of[names[0]], flex_of[names[3]], flex_of[names[7]], flex_of[names[10]],
                    flex_of[names[5]], flex_of[names[12]], "", 80.0, 50.0])
    up = os.path.join(d, "upload.csv")
    res = subprocess.run([sys.executable, "-W", "ignore", "-m", "dfs_engine.cli", "dk-run", "--pool", pool_path,
                          "--entries", path, "--candidates", "30", "--gen-trials", "200", "--eval-trials", "300",
                          "--holdout-trials", "200", "--field-size", "1500", "--field-optimizer-solves", "100",
                          "--min-unique", "1", "--seed", "4", "--upload-out", up, "--extra-lineups", saber,
                          "--qb-stack", "2", "--max-game-stack-share", "0.5"],
                         cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    out = res.stdout
    assert "Showdown: ATL vs NO" in out and "field captains:" in out and "25% of the field" in out
    assert "stat engine: simulated box scores" in out  # --engine auto: box scores for Showdown
    assert "Classic only" in out and "+1 of 1 lineups from lineups_dk_nfl_showdown.csv" in out
    assert out.count("    CPT ") == 2  # each pick printed with its teams
    with open(up, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == dk.ENTRY_COLUMNS + dk.SHOWDOWN_SLOTS and len(rows) == 3
    for r in rows[1:]:
        dk.validate_showdown_upload(r[4:10], ef.players)
    assert rows[1][4:10] != rows[2][4:10]
