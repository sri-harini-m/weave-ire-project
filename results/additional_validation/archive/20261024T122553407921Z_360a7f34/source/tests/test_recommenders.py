"""Behavior checks for the public slate interface, without MIND downloads."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.params import Params
from src.sim.recommenders import (
    REGISTRY,
    ContentBasedRecommender,
    MMRRecommender,
    PopularityRecommender,
    RandomRecommender,
    SimState,
)
from src.sim.simulate import run_population
from src.sim.world import World


def make_world(embeddings=None, popularity=None):
    if embeddings is None:
        embeddings = np.array([[1., 0.], [1., 0.], [.8, .6], [0., 1.],
                               [-1., 0.], [0., 0.]])
    embeddings = np.asarray(embeddings, dtype=float)
    n = len(embeddings)
    return World(N=n, D=np.zeros((n, n)), C=np.eye(n), rho=0.,
                 ids=np.arange(n), profile_vecs=embeddings,
                 popularity=np.zeros(n) if popularity is None else popularity,
                 cats=np.arange(n) % 2, kind="news")


def make_state(world, users=2, history=None, profiles=None):
    state = SimState(world=world, t=0,
                     available=np.ones((users, world.N), dtype=bool),
                     history=[[] for _ in range(users)] if history is None else history,
                     ratings=np.full((users, world.N), np.nan),
                     active=np.ones(users, dtype=bool))
    for user, items in enumerate(state.history):
        state.available[user, items] = False
    if profiles is not None:
        state.extra["profiles"] = profiles
    return state


def start(rec, world, users=2, seed=7):
    rec.reset(world, users, np.random.default_rng(seed))
    return rec


def assert_slate(slate, state, k):
    assert slate.shape == (len(state.active), k)
    assert np.issubdtype(slate.dtype, np.integer)
    for user, row in enumerate(slate):
        items = row[row >= 0]
        assert len(items) == len(set(items.tolist()))
        assert (items < state.world.N).all()
        assert state.available[user, items].all()
        assert (row >= -1).all()
        if not state.active[user]:
            assert len(items) == 0
        else:
            assert len(items) == min(k, state.available[user].sum())


def test_registry_exposes_required_recommenders():
    assert {"none", "random", "popularity", "content", "mmr"} <= set(REGISTRY)


def test_popularity_has_global_ranking_and_stable_ties():
    world = make_world(popularity=np.array([0., 5., 5., 1., 0., 0.]))
    state = make_state(world)
    rec = start(PopularityRecommender(K=4), world)
    slate = rec.recommend(state)
    np.testing.assert_array_equal(slate, [[1, 2, 3, 0], [1, 2, 3, 0]])
    state.available[1, 1] = False
    np.testing.assert_array_equal(rec.recommend(state)[1], [2, 3, 0, 4])


def test_popularity_counts_reads_independent_of_feedback_and_resets():
    world = make_world(popularity=np.array([1., 0., 0., 0., 0., 0.]))
    state = make_state(world)
    rec = start(PopularityRecommender(K=2, prior_weight=1.), world)
    rec.observe(state, np.array([0, 1]), np.array([3, 3]), np.array([-20., 0.]))
    np.testing.assert_array_equal(rec.recommend(state), [[3, 0], [3, 0]])
    rec.reset(world, 2, np.random.default_rng(7))
    np.testing.assert_array_equal(rec.recommend(state), [[0, 1], [0, 1]])


def test_popularity_prior_uses_normalized_pseudocount_mass():
    world = make_world(popularity=np.array([100., 0., 0., 0., 0., 0.]))
    state = make_state(world)
    rec = start(PopularityRecommender(K=2, prior_weight=.5), world)
    rec.observe(state, np.array([0]), np.array([3]), np.array([1.]))
    np.testing.assert_array_equal(rec.recommend(state)[0], [3, 0])


def test_content_uses_cosine_not_embedding_magnitude():
    world = make_world([[8., 6.], [1., 0.], [0., 12.], [-1., 0.]])
    state = make_state(world, profiles=np.array([[1., 0.], [0., 1.]]))
    rec = start(ContentBasedRecommender(K=4), world)
    np.testing.assert_array_equal(rec.recommend(state), [[1, 0, 2, 3], [2, 0, 1, 3]])


def test_content_learns_from_visible_history():
    world = make_world()
    state = make_state(world, history=[[0], [3]])
    rec = start(ContentBasedRecommender(K=3), world)
    np.testing.assert_array_equal(rec.recommend(state), [[1, 2, 3], [2, 0, 1]])


def test_content_combines_profile_with_simulated_history():
    world = make_world()
    state = make_state(world, history=[[3], [3]],
                       profiles=np.array([[1., 0.], [1., 0.]]))
    rec = start(ContentBasedRecommender(K=2, profile_weight=2.), world)
    np.testing.assert_array_equal(rec.recommend(state), [[2, 0], [2, 0]])
    rec = start(ContentBasedRecommender(K=2, profile_weight=0.), world)
    np.testing.assert_array_equal(rec.recommend(state), [[2, 0], [2, 0]])
    # Raising the profile weight makes a perfectly matching article outrank .8/.6.
    rec = start(ContentBasedRecommender(K=2, profile_weight=10.), world)
    np.testing.assert_array_equal(rec.recommend(state), [[0, 1], [0, 1]])


@pytest.mark.parametrize("rec_type", [ContentBasedRecommender, MMRRecommender])
def test_empty_and_zero_profiles_fall_back_to_learned_popularity(rec_type):
    world = make_world(popularity=np.array([0., 0., 1., 0., 0., 0.]))
    state = make_state(world, profiles=np.zeros((2, 2)))
    kwargs = {"diversity_weight": 0.} if rec_type is MMRRecommender else {}
    rec = start(rec_type(K=2, **kwargs), world)
    np.testing.assert_array_equal(rec.recommend(state), [[2, 0], [2, 0]])
    rec.observe(state, np.array([0, 1]), np.array([4, 4]), np.array([-1., -1.]))
    state.extra.clear()
    np.testing.assert_array_equal(rec.recommend(state), [[4, 2], [4, 2]])


def test_mmr_zero_matches_content_including_ties_and_padding():
    world = make_world()
    state = make_state(world, history=[[0], []],
                       profiles=np.array([[1., 0.], [0., 0.]]))
    state.active[1] = False
    content = start(ContentBasedRecommender(K=8), world)
    mmr = start(MMRRecommender(K=8, diversity_weight=0.), world)
    np.testing.assert_array_equal(mmr.recommend(state), content.recommend(state))


def test_mmr_penalizes_redundancy_after_first_choice():
    angle = np.deg2rad(20.)
    first = [np.cos(angle), np.sin(angle)]
    world = make_world([first, first, [np.sqrt(3.) / 2., -.5], [0., 1.]])
    state = make_state(world, profiles=np.array([[1., 0.], [1., 0.]]))
    content = start(ContentBasedRecommender(K=2), world)
    mmr = start(MMRRecommender(K=2, diversity_weight=1.), world)
    np.testing.assert_array_equal(content.recommend(state), [[0, 1], [0, 1]])
    np.testing.assert_array_equal(mmr.recommend(state), [[0, 2], [0, 2]])


@pytest.mark.parametrize("kwargs", [{"K": -1}, {"K": 1.5}, {"K": True},
                                     {"prior_weight": -1.},
                                     {"profile_weight": np.nan},
                                     {"diversity_weight": -1.},
                                     {"diversity_weight": 1.1}])
def test_mmr_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        MMRRecommender(**kwargs)


@pytest.mark.parametrize("rec_type", [RandomRecommender, PopularityRecommender,
                                     ContentBasedRecommender, MMRRecommender])
@pytest.mark.parametrize("k", [0, 1, 4, 9])
def test_slate_contract_with_inactive_and_exhausted_users(rec_type, k):
    world = make_world()
    state = make_state(world, users=3, history=[[0, 1, 2, 3, 4], [], []])
    state.available[1] = False
    state.active[2] = False
    rec = start(rec_type(K=k), world, users=3)
    slate = rec.recommend(state)
    assert_slate(slate, state, k)


@pytest.mark.parametrize("rec_type", [RandomRecommender, PopularityRecommender,
                                     ContentBasedRecommender, MMRRecommender])
def test_empty_catalog_is_padded(rec_type):
    world = make_world(np.empty((0, 2)))
    state = make_state(world)
    rec = start(rec_type(K=3), world)
    np.testing.assert_array_equal(rec.recommend(state), np.full((2, 3), -1))


@pytest.mark.parametrize("rec_type", [RandomRecommender, PopularityRecommender,
                                     ContentBasedRecommender, MMRRecommender])
def test_fixed_seed_repeats_after_reset(rec_type):
    world = make_world()
    state = make_state(world, history=[[0], [3]])
    rec = start(rec_type(K=4), world)
    first = rec.recommend(state)
    rec.observe(state, np.array([0]), np.array([2]), np.array([1.]))
    rec.reset(world, 2, np.random.default_rng(7))
    np.testing.assert_array_equal(rec.recommend(state), first)


@pytest.mark.parametrize("rec_type", [ContentBasedRecommender, MMRRecommender])
def test_content_requires_article_embeddings(rec_type):
    world = make_world()
    world.profile_vecs = None
    with pytest.raises(ValueError):
        start(rec_type(), world)


@pytest.mark.parametrize("profiles", [np.zeros((1, 2)), np.zeros((2, 3)),
                                      np.array([[np.nan, 0.], [0., 1.]])])
def test_content_rejects_malformed_profiles(profiles):
    world = make_world()
    state = make_state(world, profiles=profiles)
    rec = start(ContentBasedRecommender(), world)
    with pytest.raises(ValueError):
        rec.recommend(state)


@pytest.mark.parametrize("bad_index", [-1, 6, 1.5])
def test_content_rejects_invalid_history_indices(bad_index):
    world = make_world()
    state = make_state(world)
    state.history[0] = [bad_index]
    rec = start(ContentBasedRecommender(), world)
    with pytest.raises(ValueError):
        rec.recommend(state)


@pytest.mark.parametrize("rec_type", [ContentBasedRecommender, MMRRecommender])
def test_content_rejects_nonfinite_embeddings(rec_type):
    world = make_world()
    world.profile_vecs[0, 0] = np.inf
    with pytest.raises(ValueError):
        start(rec_type(), world)


@pytest.mark.parametrize("rec_type", [PopularityRecommender, ContentBasedRecommender,
                                     MMRRecommender])
def test_required_recommenders_run_in_simulator_without_repeated_reads(rec_type):
    world = make_world()
    world.profile_vecs[5] = [.6, .8]
    profiles = np.array([[1., .2], [.2, 1.]])
    seen = np.zeros((2, world.N), dtype=bool)
    seen[:, 5] = True
    rec = rec_type(K=3)
    run = run_population(world, Params(T=5, phi=.5), 2,
                         np.random.default_rng(6), rec, profiles=profiles, seen=seen)
    assert (run["seq"] >= 0).all()
    for row in run["seq"]:
        assert len(set(row.tolist())) == 5
        assert 5 not in row


def test_simulator_exposes_only_supplied_profile_to_recommender():
    world = make_world()
    world.profile_vecs[5] = [.6, .8]
    profiles = np.array([[1., .2], [.2, 1.]])

    class InspectContent(ContentBasedRecommender):
        def recommend(self, state):
            np.testing.assert_array_equal(state.extra["profiles"], profiles)
            assert "X" not in state.extra and "V" not in state.extra
            return super().recommend(state)

    run_population(world, Params(T=2), 2, np.random.default_rng(5),
                   InspectContent(K=2), profiles=profiles)
