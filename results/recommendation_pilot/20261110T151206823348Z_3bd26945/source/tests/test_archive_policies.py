"""Storage choices retain report inputs while controlling model and hidden arrays."""
import errno
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.artifacts import PopulationArchive, load_checkpoint, load_state, load_world, save_world
from src.sim.params import Params
from src.sim.recommenders import REGISTRY, SimState
from src.sim.world import World


def make_inputs():
    world = World(N=3, D=np.ones((3, 3)) - np.eye(3), C=np.eye(3), rho=0.,
                  ids=np.array(["a", "b", "c"]), cats=np.array(["x", "x", "y"]),
                  profile_vecs=np.array([[1., 0.], [.8, .6], [0., 1.]]),
                  popularity=np.array([3., 2., 1.]), kind="news")
    state = SimState(world, 0, np.ones((2, 3), bool), [[], []],
                     np.full((2, 3), np.nan), np.ones(2, bool),
                     {"profiles": np.array([[1., .1], [.1, 1.]])})
    rng = np.random.default_rng(4)
    rec = REGISTRY["content"](K=2)
    rec.reset(world, 2, rng)
    hidden = {"X": np.arange(6).reshape(2, 3), "V": np.arange(3),
              "prior_means": np.zeros((2, 3))}
    run = {"seq": np.array([[0], [2]]), "val": np.array([[1.], [2.]]),
           "pen": np.array([[0.], [.5]]), "left_at": np.array([1, 1]),
           "X": hidden["X"], "V": hidden["V"]}
    return world, state, rng, rec, hidden, run


def complete_archive(path, policy="all", diagnostics=True):
    world, state, rng, rec, hidden, run = make_inputs()
    archive = PopulationArchive(path, {"recommender": "content", "seed": 4},
                                 snapshot_policy=policy, diagnostics=diagnostics)
    archive.start(world, Params(T=1), rec, rng, state.extra["profiles"], None, hidden)
    archive.before_round(state, rec, {"belief_mean": np.ones((2, 3)),
                                      "fatigue_before": np.ones((2, 3))})
    slates = rec.recommend(state)
    rec.observe(state, np.array([0, 1]), np.array([0, 2]), np.array([1., 1.5]))
    archive.after_round(0, slates, np.array([0, 1]), np.array([0, 2]),
                        np.array([1., 1.5]), rec,
                        {"choice_scores": np.ones((2, 3)), "active_after": np.ones(2, bool),
                         "recommendation_seconds": .125},
                        signals={"boredom": np.array([0, 1]), "left_users": np.array([1])})
    archive.finish(run)
    return archive, state, rec, run


@pytest.mark.parametrize("policy,expected", [
    ("all", {"initial.npz", "before_0000.npz", "after_0000.npz", "final.npz"}),
    ("final", {"final.npz"}),
    ("none", set()),
])
@pytest.mark.parametrize("diagnostics", [True, False])
def test_policy_selects_models_and_hidden_data_only(tmp_path, policy, expected, diagnostics):
    archive, state, rec, run = complete_archive(tmp_path / "population", policy, diagnostics)
    assert {p.name for p in (archive.path / "models").iterdir()} == expected
    meta = json.loads((archive.path / "metadata.json").read_text())
    assert meta["archive"] == {"snapshot_policy": policy, "diagnostics": diagnostics}
    assert (archive.path / "initial_hidden.npz").exists() is diagnostics
    with np.load(archive.path / "run.npz", allow_pickle=False) as stored:
        for key in ("seq", "val", "pen", "left_at"):
            np.testing.assert_array_equal(stored[key], run[key])
        assert ("X" in stored.files) is diagnostics
        assert ("V" in stored.files) is diagnostics
    for suffix in ("input", "output"):
        with np.load(archive.path / "rounds" / f"round_0000_{suffix}.npz", allow_pickle=False) as stored:
            hidden_diagnostics = [key for key in stored.files
                                  if key.startswith("diagnostic_")
                                  and key != "diagnostic_recommendation_seconds"]
            assert bool(hidden_diagnostics) is diagnostics
    restored_state = load_state(archive.path, 0)
    np.testing.assert_array_equal(restored_state.available, state.available)
    np.testing.assert_array_equal(restored_state.extra["profiles"], state.extra["profiles"])
    if policy != "none":
        restored = load_checkpoint(archive.path / "models" / "final.npz", load_world(archive.path))
        np.testing.assert_array_equal(restored.recommend(state), rec.recommend(state))
    if policy != "all":
        with pytest.raises(FileNotFoundError, match=f"snapshot_policy='{policy}'"):
            load_checkpoint(archive.path / "models" / "before_0000.npz", state.world)
    status = json.loads((archive.path / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["snapshot_policy"] == policy
    assert status["diagnostics"] is diagnostics


def test_observed_feedback_survives_without_hidden_diagnostics(tmp_path):
    archive, _, _, _ = complete_archive(tmp_path / "population", "none", False)
    with np.load(archive.path / "rounds" / "round_0000_output.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["feedback_boredom"], [0, 1])
        np.testing.assert_array_equal(stored["feedback_left_users"], [1])
        np.testing.assert_array_equal(stored["feedback"], [1., 1.5])
        assert stored["diagnostic_recommendation_seconds"].item() == .125
        assert "diagnostic_choice_scores" not in stored.files
    response = json.loads((archive.path / "responses.jsonl").read_text())
    assert response["signals"] == {"boredom": [0, 1], "left_users": [1]}
    assert set(response) == {"t", "slate_is_none", "slates", "users", "items", "feedback", "signals"}


@pytest.mark.parametrize("settings", [{"snapshot_policy": "latest"},
                                      {"diagnostics": "false"}])
def test_invalid_storage_settings_do_not_create_directories(tmp_path, settings):
    path = tmp_path / "invalid"
    with pytest.raises(ValueError):
        PopulationArchive(path, **settings)
    assert not path.exists()


def test_failure_and_overwrite_guards_remain_for_light_archive(tmp_path):
    world, state, rng, rec, hidden, _ = make_inputs()
    path = tmp_path / "failed"
    archive = PopulationArchive(path, snapshot_policy="final", diagnostics=False)
    archive.start(world, Params(T=1), rec, rng, None, None, hidden)
    archive.before_round(state, rec, {"choice_scores": np.ones((2, 3))})
    archive.fail(RuntimeError("reader failed"))
    status = json.loads((path / "status.json").read_text())
    assert status["status"] == "failed" and status["error"] == "reader failed"
    assert (path / "rounds" / "round_0000_input.npz").exists()
    assert list((path / "models").iterdir()) == []
    with pytest.raises(FileExistsError):
        PopulationArchive(path, snapshot_policy="none")


@pytest.mark.parametrize("signals", [{"boredom": [0]}, {"boredom": [0, .5]},
                                     {"left_users": [-1]}, {"left_users": [2]},
                                     {"left_users": [1.5]}, {"hidden_X": [1, 2]}])
def test_invalid_observed_signals_are_rejected_before_output(tmp_path, signals):
    world, state, rng, rec, hidden, _ = make_inputs()
    archive = PopulationArchive(tmp_path / "signals", diagnostics=False)
    archive.start(world, Params(T=1), rec, rng, None, None, hidden)
    archive.before_round(state, rec)
    with pytest.raises(ValueError):
        archive.after_round(0, rec.recommend(state), np.array([0, 1]), np.array([0, 2]),
                            np.array([1., 1.5]), rec, signals=signals)
    assert not (archive.path / "rounds" / "round_0000_output.npz").exists()
    assert (archive.path / "responses.jsonl").read_text() == ""


def test_populations_share_one_immutable_world_and_keep_reload_apis(tmp_path):
    world, state, rng, rec, hidden, _ = make_inputs()
    source = tmp_path / "shared_world.npz"
    save_world(world, source)
    original = source.read_bytes()
    archives = []
    for index in range(2):
        archive = PopulationArchive(tmp_path / f"population_{index}", snapshot_policy="none",
                                     diagnostics=False, shared_world_path=source)
        archive.start(world, Params(T=1), rec, rng, state.extra["profiles"], None, hidden)
        archives.append(archive)
        population_world = archive.path / "world.npz"
        assert population_world.stat().st_ino == source.stat().st_ino
        assert population_world.stat().st_dev == source.stat().st_dev
        np.testing.assert_array_equal(load_world(archive.path).profile_vecs, world.profile_vecs)
        np.testing.assert_array_equal(load_world(archive.path).L, world.L)
        assert (archive.path / "world.json").exists()
        metadata = json.loads((archive.path / "metadata.json").read_text())
        assert metadata["archive"]["shared_world_reference"] == str(source.resolve())
    assert source.read_bytes() == original
    assert source.stat().st_nlink >= 3
    with pytest.raises(ValueError, match="already started"):
        archives[0].start(world, Params(T=1), rec, rng, None, None, hidden)
    assert source.read_bytes() == original


def test_shared_world_mismatch_or_missing_file_is_rejected(tmp_path):
    world, state, rng, rec, hidden, _ = make_inputs()
    source = tmp_path / "shared_world.npz"
    save_world(world, source)
    original = source.read_bytes()
    world.ids = world.ids[::-1]
    archive = PopulationArchive(tmp_path / "wrong", shared_world_path=source)
    with pytest.raises(ValueError, match="does not match"):
        archive.start(world, Params(T=1), rec, rng, None, None, hidden)
    assert not (archive.path / "world.npz").exists()
    assert source.read_bytes() == original
    missing = PopulationArchive(tmp_path / "missing", shared_world_path=tmp_path / "absent.npz")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        missing.start(world, Params(T=1), rec, rng, None, None, hidden)


def test_cross_device_shared_world_copy_is_exclusive(tmp_path, monkeypatch):
    world, state, rng, rec, hidden, _ = make_inputs()
    source = tmp_path / "shared_world.npz"
    save_world(world, source)
    original = source.read_bytes()

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr("src.sim.artifacts.os.link", cross_device)
    archive = PopulationArchive(tmp_path / "copied", snapshot_policy="none",
                                 shared_world_path=source)
    archive.start(world, Params(T=1), rec, rng, None, None, hidden)
    assert (archive.path / "world.npz").read_bytes() == original
    np.testing.assert_array_equal(load_world(archive.path).ids, world.ids)
    assert source.read_bytes() == original
    protected = PopulationArchive(tmp_path / "protected", shared_world_path=source)
    destination = protected.path / "world.npz"
    destination.write_bytes(b"already present")
    with pytest.raises(FileExistsError):
        protected.start(world, Params(T=1), rec, rng, None, None, hidden)
    assert destination.read_bytes() == b"already present"
    assert source.read_bytes() == original

