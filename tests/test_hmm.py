"""Unit tests for the Viterbi decoder on a synthetic two-arc network.

Arc 0 and arc 1 run in parallel 20 m apart (path distance between them is
large because the only connection is a long U-turn). A vehicle driving
straight should stay on one arc even when the classifier briefly prefers
the parallel one.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from src.config import CFG
from src.hmm import viterbi_decode


class FakeNetwork:
    """Minimal RoadNetwork stand-in: only path_distance is used."""

    def path_distance(self, a, b, cutoff):
        if a == b:
            return 0.0
        # U-turn detour between the parallel arcs: 500 m
        return 500.0 if 500.0 <= cutoff else None


def _straight_track(n=10, step_m=15.0):
    # drive due east ~15 m/s along the equator
    dlon = step_m / 111_195.0
    return pd.DataFrame({
        "lat": np.zeros(n),
        "lon": np.arange(n) * dlon,
        "timestamp": np.arange(n, dtype=float),
    })


def _emissions(n, wrong_at):
    out = []
    for i in range(n):
        p = np.array([0.8, 0.2])
        if i in wrong_at:
            p = p[::-1]  # classifier briefly prefers the parallel arc
        out.append(p)
    return out


def test_decoder_resists_single_point_classifier_flip():
    n = 10
    cands = [[0, 1]] * n
    track = _straight_track(n)
    pred = viterbi_decode(_emissions(n, wrong_at={5}), cands, track,
                          FakeNetwork(), replace(CFG, stay_bonus=0.0))
    # a one-off flip costs a 500 m detour twice; the decoder must stay on arc 0
    assert list(pred) == [0] * n


def test_decoder_follows_a_sustained_switch():
    n = 10
    cands = [[0, 1]] * n
    track = _straight_track(n)
    pred = viterbi_decode(_emissions(n, wrong_at=set(range(4, n))), cands,
                          track, FakeNetwork(), replace(CFG, stay_bonus=0.0))
    # from point 4 on the classifier consistently prefers arc 1
    assert list(pred[5:]) == [1] * (n - 5)


def test_stay_bonus_suppresses_borderline_switch():
    n = 10
    cands = [[0, 1]] * n
    track = _straight_track(n)
    # two consecutive borderline flips (0.55 vs 0.45): weak evidence
    em = []
    for i in range(n):
        em.append(np.array([0.45, 0.55]) if i in (5, 6)
                  else np.array([0.8, 0.2]))
    base = viterbi_decode(em, cands, track, FakeNetwork(),
                          replace(CFG, stay_bonus=0.0))
    blended = viterbi_decode(em, cands, track, FakeNetwork(),
                             replace(CFG, stay_bonus=CFG.stay_bonus))
    # with the persistence blend the decoder must not switch more often
    switches = lambda p: int((np.diff(p) != 0).sum())
    assert switches(blended) <= switches(base)


def test_missing_candidates_inherit_neighbour():
    n = 6
    cands = [[0, 1]] * n
    cands[3] = []  # GPS point with no candidates
    track = _straight_track(n)
    pred = viterbi_decode(_emissions(n, wrong_at=set()), cands, track,
                          FakeNetwork(), replace(CFG, stay_bonus=0.0))
    assert pred[3] == 0  # inherits the decoded arc of its neighbour
