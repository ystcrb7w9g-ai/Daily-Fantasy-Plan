import io
import json
import os
import sys
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dfs_engine import sportsgameodds as sgo
from dfs_engine.data import load_player_pool

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
POOL_PATH = os.path.join(DATA_DIR, "sample_classic_pool.csv")
EVENTS_PATH = os.path.join(DATA_DIR, "sample_sgo_events.json")


def _events():
    return sgo.load_events(EVENTS_PATH)


def test_sgo_american_odds_conversion():
    assert abs(sgo.american_to_prob("-110") - 110 / 210) < 1e-9
    assert abs(sgo.american_to_prob("+150") - 0.4) < 1e-9
    assert sgo.american_to_prob(None) is None


def test_sgo_extract_game_lines_maps_home_away():
    lines = sgo.extract_game_lines(_events()).set_index("team")
    assert set(lines.index) == {"BUF", "MIA"}
    assert lines.loc["BUF", "spread"] == -6.0
    assert lines.loc["MIA", "spread"] == 6.0
    assert lines.loc["BUF", "game_total"] == 49.5
    assert lines.loc["BUF", "team_total"] == 27.5
    assert lines.loc["MIA", "opp"] == "BUF"


def test_sgo_game_lines_derive_team_totals_when_missing():
    ev = json.loads(json.dumps(_events()[0]))
    for k in [k for k in ev["odds"] if k.startswith(("points-home-game-ou", "points-away-game-ou"))]:
        del ev["odds"][k]
    lines = sgo.extract_game_lines([ev]).set_index("team")
    assert lines.loc["BUF", "team_total"] == 49.5 / 2 + 3.0
    assert lines.loc["MIA", "team_total"] == 49.5 / 2 - 3.0


def test_sgo_props_ignore_non_game_periods_and_team_markets():
    props = sgo.extract_player_props(_events())
    assert set(props["stat_id"]) <= set(sgo.DK_POINTS)
    allen_py = props[(props["name"] == "Josh Allen") & (props["stat_id"] == "passing_yards")]
    assert len(allen_py) == 1 and allen_py["line"].iloc[0] == 252.5
    assert (props["p_over"].between(0, 1)).all()


def test_sgo_count_and_yardage_models_invert_line():
    lam = sgo.expected_count(0.5, 0.5)
    assert abs(lam - np.log(2)) < 1e-6  # P(X>=1) = 0.5
    mu, sigma = sgo.lognormal_params(80.0, 0.5, 0.5)
    assert abs(np.exp(mu) - 80.0) < 1e-6  # median sits on the line at even odds
    # Higher over probability -> higher implied mean.
    assert sgo.expected_count(1.5, 0.6) > sgo.expected_count(1.5, 0.4)


def test_sgo_baseline_is_reasonable_and_flags_imputation():
    base = sgo.props_to_baseline(sgo.extract_player_props(_events())).set_index("name")
    assert 15 < base.loc["Josh Allen", "sgo_proj"] < 35
    assert 8 < base.loc["Stefon Diggs", "sgo_proj"] < 25
    assert "passing_interceptions" in base.loc["Tua Tagovailoa", "sgo_imputed"]
    assert base.loc["Josh Allen", "sgo_imputed"] == ""


def test_sgo_apply_blends_proj_and_updates_lines():
    events = _events()
    raw = pd.read_csv(POOL_PATH)
    df, report = sgo.apply_sgo_baseline(
        raw, sgo.extract_game_lines(events),
        sgo.props_to_baseline(sgo.extract_player_props(events)), weight=0.5,
    )
    allen = df[df["name"] == "Josh Allen"].iloc[0]
    assert allen["proj_source"] == "blend"
    assert abs(allen["proj"] - (0.5 * 24.5 + 0.5 * allen["sgo_proj"])) < 0.01
    assert (df.loc[df["team"] == "BUF", "spread"] == -6.0).all()
    # DST and players without props keep their input projection.
    assert df.loc[df["name"] == "Bills DST", "proj"].iloc[0] == 7.5
    assert "Durham Smythe" in report.players_unmatched
    assert report.players_matched == 10


def test_sgo_apply_fills_blank_proj_and_ceiling():
    events = _events()
    raw = pd.read_csv(POOL_PATH)
    raw.loc[raw["name"] == "Tyreek Hill", ["proj", "ceiling"]] = np.nan
    df, _ = sgo.apply_sgo_baseline(
        raw, None, sgo.props_to_baseline(sgo.extract_player_props(events)),
    )
    hill = df[df["name"] == "Tyreek Hill"].iloc[0]
    assert hill["proj_source"] == "sgo"
    assert hill["proj"] == hill["sgo_proj"]
    assert hill["ceiling"] > hill["proj"]


def test_sgo_enriched_pool_loads_into_engine():
    import tempfile
    events = _events()
    df, _ = sgo.apply_sgo_baseline(
        pd.read_csv(POOL_PATH), sgo.extract_game_lines(events),
        sgo.props_to_baseline(sgo.extract_player_props(events)),
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pool.csv")
        df.to_csv(path, index=False)
        pool = load_player_pool(path)
    assert len(pool) == 13 and pool["proj"].notna().all()


def test_sgo_name_normalization():
    assert sgo.normalize_name("De'Von Achane") == sgo.normalize_name("DeVon Achane")
    assert sgo.normalize_name("Marvin Harrison Jr.") == "marvin harrison"
    assert sgo.normalize_abbr("jac") == "JAX"


def test_sgo_fetch_follows_cursor_pagination():
    pages = [
        {"success": True, "data": [{"eventID": "A"}], "nextCursor": "c1"},
        {"success": True, "data": [{"eventID": "B"}], "nextCursor": None},
    ]
    seen_urls, seen_headers = [], []

    def fake_urlopen(req, timeout=None):
        seen_urls.append(req.full_url)
        seen_headers.append(dict(req.header_items()))
        return io.BytesIO(json.dumps(pages[len(seen_urls) - 1]).encode())

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        events = sgo.fetch_events(api_key="KEY")
    assert [e["eventID"] for e in events] == ["A", "B"]
    assert "leagueID=NFL" in seen_urls[0] and "cursor=c1" in seen_urls[1]
    assert seen_headers[0].get("X-api-key") == "KEY"
