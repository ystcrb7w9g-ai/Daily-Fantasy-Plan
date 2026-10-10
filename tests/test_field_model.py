import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dfs_engine import field_model as fm


def test_lineup_structure_counts_stacks_bring_backs_and_dst_conflicts():
    teams = {"josh allen": "BUF", "khalil shakir": "BUF", "dalton kincaid": "BUF", "james cook": "BUF",
             "tyreek hill": "MIA", "devon achane": "MIA", "jamarr chase": "CIN", "joe mixon": "HOU",
             "nico collins": "HOU", "DST|bengals": "CIN", "DST|bills": "BUF"}
    opp = {"BUF": "MIA", "MIA": "BUF", "CIN": "HOU", "HOU": "CIN"}
    pos = {"Tyreek Hill": "WR"}
    lineups = pd.Series([
        # QB + 2 own catchers + own RB, 2 bring-backs, Bengals DST faces Mixon/Collins (HOU)
        "QB Josh Allen RB James Cook RB Joe Mixon WR Khalil Shakir WR Ja'Marr Chase WR Nico Collins "
        "TE Dalton Kincaid FLEX Tyreek Hill DST Bengals",
        # no stack; Bills DST with Bills RB
        "QB Josh Allen RB James Cook RB De'Von Achane WR Ja'Marr Chase WR Nico Collins WR Tyreek Hill "
        "TE Dalton Kincaid FLEX Joe Mixon DST Bills",
    ])
    f = fm.lineup_structure(lineups, teams, opp, pos)
    a, b = f.iloc[0], f.iloc[1]
    assert (a.qb_stack, a.qb_rb, a.bring_back, a.vs_dst, a.rb_dst) == (2, 1, 1, 2, 0)
    assert a.max_game == 5  # Allen, Cook, Shakir, Kincaid, Hill
    assert (b.rb_dst, b.bring_back) == (1, 2) and b.vs_dst == 2  # Achane, Hill face the Bills DST
    s = fm.structure_summary(f)
    assert s["n"] == 2 and s["rb_with_own_dst"] == 0.5 and s["any_vs_own_dst"] == 1.0
