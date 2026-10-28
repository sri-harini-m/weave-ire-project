"""Additional algorithms preserve safe checkpoints and shared input wiring."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.artifacts import (_read_npz, _write_npz, load_checkpoint, load_world,
                               save_checkpoint, save_world, world_digest)
from src.sim.recommender_config import recommender_options
from src.sim.recommenders import REGISTRY, SimState
from scripts.run_report import synthetic_world


def setup_rec(name):
    world = synthetic_world(.5, N=8, dimensions=4, pool_size=3)
    state = SimState(world, 0, np.ones((3, 8), bool), [[], [], []],
                     np.full((3, 8), np.nan), np.ones(3, bool),
                     {"query_texts": ["sports", "politics", "health"]})
    rec = REGISTRY[name](**recommender_options(name, {"K": 3, "mf": {"factors": 2, "iterations": 4}}))
    rec.reset(world, 3, np.random.default_rng(5))
    return world, state, rec


def test_config_defaults_and_per_algorithm_overrides():
    cfg = {"K": 5, "prior_weight": .3, "profile_weight": 2.,
           "bm25": {"k1": 1.8, "b": .5}, "mf": {"factors": 3, "iterations": 4}}
    assert recommender_options("bm25", cfg) == {"K": 5, "prior_weight": .3, "k1": 1.8, "b": .5}
    assert recommender_options("mf", cfg)["factors"] == 3
    assert recommender_options("random", cfg) == {"K": 5}
    with pytest.raises(ValueError, match="Unknown"):
        recommender_options("mf", {"mf": {"bad_setting": 1}})


@pytest.mark.parametrize("name", ["bm25", "mf"])
def test_nondefault_model_settings_and_future_learning_survive_reload(tmp_path, name):
    world, state, rec = setup_rec(name)
    rec.observe(state, np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([2., -1., 0.]))
    save_world(world, tmp_path / "world.npz")
    save_checkpoint(rec, tmp_path / "model.npz")
    restored_world = load_world(tmp_path)
    restored = load_checkpoint(tmp_path / "model.npz", restored_world)
    assert restored.get_config() == rec.get_config()
    np.testing.assert_array_equal(restored.recommend(state), rec.recommend(state))
    for model in (rec, restored):
        model.observe(state, np.array([0, 1, 2]), np.array([2, 0, 1]), np.array([-3., 2., 1.]))
    for key, value in rec.model_state().items():
        np.testing.assert_array_equal(restored.model_state()[key], value)
    np.testing.assert_array_equal(restored.recommend(state), rec.recommend(state))


def test_bm25_checkpoint_rejects_changed_text_even_with_identical_embeddings(tmp_path):
    world, _, rec = setup_rec("bm25")
    save_checkpoint(rec, tmp_path / "model.npz")
    before = world_digest(world)
    world.article_metadata[0]["title"] = "changed article content"
    assert world_digest(world) == before
    with pytest.raises(ValueError, match="article text"):
        load_checkpoint(tmp_path / "model.npz", world)


def test_existing_checkpoint_format_still_loads(tmp_path):
    world, state, rec = setup_rec("popularity")
    save_checkpoint(rec, tmp_path / "current.npz")
    values, metadata = _read_npz(tmp_path / "current.npz")
    # Historical checkpoints stored only constructor settings, counts and RNG.
    assert set(values) == {"counts"}
    _write_npz(tmp_path / "historical.npz", values, metadata)
    historical = load_checkpoint(tmp_path / "historical.npz", world)
    np.testing.assert_array_equal(historical.recommend(state), rec.recommend(state))


def test_visible_query_text_reaches_saved_simulator_input(tmp_path):
    from src.sim.artifacts import PopulationArchive, load_state
    from src.sim.params import Params
    from src.sim.simulate import run_population

    world, _, rec = setup_rec("bm25")
    query_texts = ["sports", "politics", "health"]
    archive = PopulationArchive(tmp_path / "bm25")
    run_population(world, Params(T=2, reader_history_visible=True), 3,
                   np.random.default_rng(6), rec, recorder=archive,
                   query_texts=query_texts)
    assert load_state(archive.path, 0).extra["query_texts"] == query_texts
    assert json.loads((archive.path / "rounds" / "round_0000_input.json").read_text())["extra"]["query_texts"] == query_texts
