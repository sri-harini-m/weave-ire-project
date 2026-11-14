"""Matched controls, observable signals, and privileged-reference isolation."""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_report import synthetic_world
from src.sim.artifacts import PopulationArchive, load_checkpoint, load_state, load_world
from src.sim.experiment import run_grid
from src.sim.news_users import sample_news_users
from src.sim.params import Params
from src.sim.recommenders import ContentBasedRecommender, IdealRecommender, REGISTRY
from src.sim.simulate import run_population
from src.sim.study import add_study_arguments, apply_visibility, factories, slate_sizes, write_comparisons


def test_command_controls_and_extended_mmr_weights():
    parser = argparse.ArgumentParser()
    add_study_arguments(parser)
    args = parser.parse_args(["--slate-sizes", "5", "10", "25", "50", "500",
                              "--reader-history-visible", "both"])
    assert slate_sizes(args) == [5, 10, 25, 50, 500]
    assert apply_visibility(args, {})["reader_history_visible"] == [False, True]
    assert args.snapshot_policy == "final" and not args.archive_diagnostics
    assert set(factories(["mmr"], {}, [0., .3, .6, .9, 1.])) == {
        "mmr_0", "mmr_0.3", "mmr_0.6", "mmr_0.9", "mmr_1"}


def test_matched_seeds_and_hidden_preferences_across_sizes_and_visibility(tmp_path):
    frame, _ = run_grid(partial(synthetic_world, N=10, dimensions=4),
                         {"rho": [.5], "reader_history_visible": [False, True]},
                         base={"T": 2}, U=3, n_pop=1,
                         recommenders={"content": partial(REGISTRY["content"], K=2)},
                         users_fn=partial(sample_news_users, U=3), slate_sizes=[2, 10],
                         archive_dir=tmp_path, snapshot_policy="final", diagnostics=True)
    assert len(frame) == 4 and frame["seed"].nunique() == 1
    truths = []
    for directory in sorted((tmp_path / "populations").iterdir()):
        meta = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "initial_hidden.npz", allow_pickle=False) as arrays:
            truths.append(arrays["X"])
        state = load_state(directory, 0)
        assert ("profiles" in state.extra) == meta["params"]["reader_history_visible"]
        assert ("query_texts" in state.extra) == meta["params"]["reader_history_visible"]
        assert ("category_profiles" in state.extra) == meta["params"]["reader_history_visible"]
        assert "X" not in state.extra and "V" not in state.extra
        assert [p.name for p in (directory / "models").iterdir()] == ["final.npz"]
    for truth in truths[1:]:
        np.testing.assert_array_equal(truth, truths[0])


def test_k_500_exposes_same_catalogue_as_none():
    world = synthetic_world(.5, N=500, dimensions=4)
    p = Params(T=3, phi=.5)
    baseline = run_population(world, p, 2, np.random.default_rng(6))
    for name in ("random", "popularity", "content", "mmr", "mf", "ideal", "calibrated",
                 "fatigue_popularity", "fatigue_content", "fatigue_mf"):
        actual = run_population(world, p, 2, np.random.default_rng(6), REGISTRY[name](K=500))
        np.testing.assert_array_equal(actual["seq"], baseline["seq"])


def test_coarse_boredom_and_departure_are_observable_without_latent_values(tmp_path):
    world = synthetic_world(.5, N=8, dimensions=3)

    class Inspect(ContentBasedRecommender):
        accepts_feedback_signals = True

        def __init__(self):
            super().__init__(K=4)
            self.received = []

        def observe(self, state, users, items, feedback, signals=None):
            assert set(signals) == {"boredom", "left_users"}
            assert signals["boredom"].shape == users.shape
            assert np.isin(signals["boredom"], [0., 1.]).all()
            assert set(state.extra).isdisjoint({"X", "V", "fatigue", "true_values"})
            self.received.append((users.copy(), signals["boredom"].copy(), signals["left_users"].copy()))
            super().observe(state, users, items, feedback)

    rec = Inspect()
    run_population(world, Params(T=4, phi=1., boredom_threshold=0.), 2,
                   np.random.default_rng(9), rec)
    assert len(rec.received) == 4
    assert any(boredom.any() for _, boredom, _ in rec.received)
    leaving = Inspect()
    run_population(world, Params(T=2, outside_option=1e9), 2,
                   np.random.default_rng(9), leaving)
    assert leaving.received[0][0].size == 0
    np.testing.assert_array_equal(leaving.received[0][2], [0, 1])


def test_ideal_receives_reference_through_separate_channel(tmp_path):
    world = synthetic_world(.5, N=8, dimensions=3)
    rec = IdealRecommender(K=1)
    archive = PopulationArchive(tmp_path / "ideal", snapshot_policy="final", diagnostics=False)
    result = run_population(world, Params(T=2), 2, np.random.default_rng(4), rec, recorder=archive)
    np.testing.assert_array_equal(result["seq"][:, 0], np.argmax(result["X"], axis=1))
    assert "X" not in load_state(archive.path, 0).extra
    restored = load_checkpoint(archive.path / "models" / "final.npz", load_world(archive.path))
    np.testing.assert_array_equal(restored.model_state()["reference_values"], result["X"])
    with np.load(archive.path / "run.npz", allow_pickle=False) as stored:
        assert "X" not in stored.files and "V" not in stored.files


def test_fatigue_variants_inherit_blind_model_settings():
    from src.sim.recommender_config import recommender_options
    options = recommender_options("fatigue_mf", {"mf": {"factors": 3, "iterations": 7}})
    assert options == {"factors": 3, "iterations": 7}
