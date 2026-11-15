"""Archives preserve recommendations, responses, and diagnostics without pickle."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.artifacts import (PopulationArchive, load_checkpoint, load_state,
                               load_world, save_checkpoint, save_world)
from src.sim.params import Params
from src.sim.recommenders import REGISTRY, SimState
from src.sim.world import World


def world_fixture():
    return World(N=4, D=np.ones((4, 4)) - np.eye(4), C=np.eye(4), rho=0.,
                 ids=np.array(["n0", "n1", "n2", "n3"], dtype=object),
                 cats=np.array(["a", "a", "b", "b"], dtype=object),
                 profile_vecs=np.array([[1., 0.], [.8, .6], [0., 1.], [-1., 0.]]),
                 popularity=np.array([4., 3., 2., 1.]), kind="news")


def state_fixture(world):
    return SimState(world, 0, np.ones((2, 4), bool), [[], []],
                    np.full((2, 4), np.nan), np.ones(2, bool),
                    {"profiles": np.array([[1., .1], [.1, 1.]])})


@pytest.mark.parametrize("name", list(REGISTRY))
def test_checkpoint_reproduces_recommendations_and_learning(tmp_path, name):
    world = world_fixture()
    state = state_fixture(world)
    rec = REGISTRY[name](K=3)
    rec.reset(world, 2, np.random.default_rng(42))
    rec.observe(state, np.array([0, 1]), np.array([2, 2]), np.array([-1., 2.]))
    path = tmp_path / "model.npz"
    save_checkpoint(rec, path)
    restored = load_checkpoint(path, world)
    expected = rec.recommend(state)
    actual = restored.recommend(state)
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual, expected)
    rec.observe(state, np.array([0]), np.array([3]), np.array([0.]))
    restored.observe(state, np.array([0]), np.array([3]), np.array([0.]))
    if hasattr(rec, "counts"):
        np.testing.assert_array_equal(restored.counts, rec.counts)
        restored.reset(world, 2, np.random.default_rng(42))
        assert restored.counts[3] < rec.counts[3]
    with np.load(path, allow_pickle=False) as arrays:
        assert all(not arrays[key].dtype.hasobject for key in arrays.files)


@pytest.mark.parametrize("bit_name", ["PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"])
def test_checkpoint_preserves_supported_rng_states(tmp_path, bit_name):
    world = world_fixture()
    state = state_fixture(world)
    rec = REGISTRY["random"](K=2)
    rec.reset(world, 2, np.random.Generator(getattr(np.random, bit_name)(7)))
    rec.recommend(state)
    save_checkpoint(rec, tmp_path / "random.npz")
    restored = load_checkpoint(tmp_path / "random.npz", world)
    np.testing.assert_array_equal(restored.recommend(state), rec.recommend(state))


def test_checkpoint_rejects_different_catalogue_order(tmp_path):
    world = world_fixture()
    rec = REGISTRY["popularity"]()
    rec.reset(world, 2, np.random.default_rng(42))
    save_checkpoint(rec, tmp_path / "model.npz")
    world.ids = world.ids[::-1]
    with pytest.raises(ValueError, match="catalogue"):
        load_checkpoint(tmp_path / "model.npz", world)


def test_world_roundtrip_retains_arrays_and_object_ids(tmp_path):
    world = world_fixture()
    save_world(world, tmp_path / "world.npz")
    restored = load_world(tmp_path)
    for field in ("D", "C", "L", "ids", "cats", "profile_vecs", "popularity"):
        np.testing.assert_array_equal(getattr(restored, field), getattr(world, field))
    assert restored.kind == world.kind
    assert restored.N == world.N


def test_archive_preserves_inputs_responses_and_final_arrays(tmp_path):
    world = world_fixture()
    world.article_metadata = [{"news_id": "n0", "title": "Example article"}]
    state = state_fixture(world)
    rec = REGISTRY["content"](K=2)
    rng = np.random.default_rng(42)
    rec.reset(world, 2, rng)
    path = tmp_path / "population"
    archive = PopulationArchive(path, {"seed": 42, "recommender": "content"})
    archive.start(world, Params(T=1), rec, rng, state.extra["profiles"], None,
                  {"X": np.arange(8).reshape(2, 4), "V": np.arange(4),
                   "prior_means": np.zeros((2, 4))})
    archive.before_round(state, rec, {"belief_mean": np.zeros((2, 4))})
    slates = rec.recommend(state)
    expected_slates = slates.copy()
    users, items = np.array([0, 1]), np.array([0, 2])
    feedback = np.array([1., -2.])
    state.history[0].append(0)
    state.available[0, 0] = False
    state.extra["profiles"][0] = [9., 9.]
    rec.observe(state, users, items, feedback)
    archive.after_round(0, slates, users, items, feedback, rec,
                        {"choice_scores": np.zeros((2, 4))})
    slates[:] = -1
    feedback[:] = 99
    run = {"seq": np.array([[0], [2]]), "val": np.array([[1.], [-1.]]),
           "pen": np.array([[0.], [1.]]), "left_at": np.array([1, 1]),
           "X": np.arange(8).reshape(2, 4), "V": np.arange(4)}
    archive.finish(run)
    status = json.loads((path / "status.json").read_text())
    assert status["status"] == "complete" and status["rounds_saved"] == 1
    restored_world = load_world(path)
    restored_state = load_state(path, 0, restored_world)
    assert restored_state.available[0, 0]
    assert restored_state.history == [[], []]
    np.testing.assert_array_equal(restored_state.extra["profiles"][0], [1., .1])
    restored_rec = load_checkpoint(path / "models" / "before_0000.npz", restored_world)
    np.testing.assert_array_equal(restored_rec.recommend(restored_state), expected_slates)
    event = json.loads((path / "responses.jsonl").read_text())
    assert event["slates"] == expected_slates.tolist()
    assert event["feedback"] == [1., -2.]
    assert restored_world.article_metadata[0]["title"] == "Example article"
    with np.load(path / "run.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["seq"], run["seq"])
    assert (path / "models" / "final.npz").exists()
    assert (path / "user_metrics.csv").exists()
    assert (path / "curves.csv").exists()
    with pytest.raises(FileExistsError):
        PopulationArchive(path)
    with pytest.raises(ValueError):
        archive.finish(run)


def test_no_recommender_response_is_distinguished_from_empty_slate(tmp_path):
    world = world_fixture()
    state = state_fixture(world)
    rec = REGISTRY["none"]()
    rng = np.random.default_rng(2)
    rec.reset(world, 2, rng)
    archive = PopulationArchive(tmp_path / "none")
    archive.start(world, Params(T=1), rec, rng, None, None, {})
    archive.before_round(state, rec)
    archive.after_round(0, None, np.array([], int), np.array([], int), np.array([]), rec)
    with np.load(archive.path / "rounds" / "round_0000_output.npz", allow_pickle=False) as stored:
        assert stored["slate_is_none"].item() is True
        assert stored["slates"].shape == (2, 0)
    event = json.loads((archive.path / "responses.jsonl").read_text())
    assert event["slate_is_none"] and event["slates"] is None


def test_failure_keeps_partial_round_and_error(tmp_path):
    world = world_fixture()
    state = state_fixture(world)
    rec = REGISTRY["popularity"]()
    rng = np.random.default_rng(3)
    rec.reset(world, 2, rng)
    archive = PopulationArchive(tmp_path / "failed")
    archive.start(world, Params(T=2), rec, rng, None, None, {})
    archive.before_round(state, rec)
    archive.fail(RuntimeError("reader failed"))
    status = json.loads((archive.path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == "reader failed"
    assert (archive.path / "rounds" / "round_0000_input.npz").exists()
    assert (archive.path / "models" / "before_0000.npz").exists()

