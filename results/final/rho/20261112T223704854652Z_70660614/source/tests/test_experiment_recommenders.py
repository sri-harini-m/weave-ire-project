"""Known-value checks for calibration, privileged reference, and observable signals."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.recommenders import (
    CalibratedRecommender, ContentBasedRecommender, FatigueContentRecommender,
    FatigueMFRecommender, FatiguePopularityRecommender, IdealRecommender,
    MatrixFactorizationRecommender, PopularityRecommender, SimState,
)
from src.sim.world import World


def world_fixture():
    embeddings = np.array([[1., 0.]] * 4 + [[.8, .6]] * 4)
    distances = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=2)
    return World(N=8, D=distances, C=np.eye(8), rho=0., ids=np.arange(8),
                 cats=np.array(["a"] * 4 + ["b"] * 4), profile_vecs=embeddings,
                 popularity=np.arange(1., 9.), kind="news")


def state_fixture(world):
    return SimState(world, 0, np.ones((2, world.N), dtype=bool), [[], []],
                    np.full((2, world.N), np.nan), np.ones(2, dtype=bool),
                    {"profiles": np.array([[1., 0.], [1., 0.]])})


def start(rec, world):
    rec.reset(world, 2, np.random.default_rng(12))
    return rec


def assert_model_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_ideal_uses_isolated_values_and_respects_eligibility():
    world = world_fixture()
    state = state_fixture(world)
    rec = start(IdealRecommender(K=3), world)
    with pytest.raises(ValueError, match="set_reference_values"):
        rec.recommend(state)
    values = np.array([[1., 8., 3., 2., 7., 4., 5., 6.]] * 2)
    rec.set_reference_values(values)
    values[:] = -100
    state.available[0, 1] = False
    state.active[1] = False
    np.testing.assert_array_equal(rec.recommend(state), [[4, 7, 6], [-1, -1, -1]])
    assert "X" not in state.extra and "reference_values" not in state.extra
    saved = rec.model_state()
    restored = start(IdealRecommender(K=3), world)
    restored.restore_model_state(saved)
    np.testing.assert_array_equal(restored.recommend(state), rec.recommend(state))
    rec.reset(world, 2, np.random.default_rng(12))
    assert rec.model_state() == {}
    with pytest.raises(ValueError, match="set_reference_values"):
        rec.recommend(state)


def test_ideal_remains_explicit_reference_and_rejects_malformed_values():
    world = world_fixture()
    with pytest.raises(ValueError, match="reference_only"):
        IdealRecommender(reference_only=False)
    rec = start(IdealRecommender(), world)
    for values in (np.ones((1, 8)), np.full((2, 8), np.nan)):
        with pytest.raises(ValueError):
            rec.set_reference_values(values)


def test_calibration_balances_the_requested_category_mix():
    world = world_fixture()
    state = state_fixture(world)
    state.extra["category_profiles"] = np.array([[1., 1.], [1., 1.]])
    rec = start(CalibratedRecommender(K=4, calibration_weight=.8), world)
    slate = rec.recommend(state)
    np.testing.assert_array_equal(slate, [[0, 4, 1, 5], [0, 4, 1, 5]])
    for row in slate:
        assert np.count_nonzero(world.cats[row] == "a") == 2
        assert np.count_nonzero(world.cats[row] == "b") == 2


def test_calibration_target_combines_raw_visible_counts_and_history():
    world = world_fixture()
    state = state_fixture(world)
    state.extra["category_profiles"] = np.array([[1., 3.], [0., 0.]])
    state.history[0] = [0]
    state.available[0, 0] = False
    rec = start(CalibratedRecommender(K=2), world)
    np.testing.assert_array_equal(rec.category_labels, ["a", "b"])
    np.testing.assert_allclose(rec.target_distribution(state), [[.4, .6], [.5, .5]])


def test_pure_calibration_breaks_equal_objective_ties_by_article_index():
    world = world_fixture()
    world.profile_vecs[0] = [.8, .6]
    state = state_fixture(world)
    state.extra["category_profiles"] = np.array([[1., 0.], [1., 0.]])
    rec = start(CalibratedRecommender(K=2, calibration_weight=1), world)
    np.testing.assert_array_equal(rec.recommend(state), [[0, 1], [0, 1]])


def test_calibration_zero_and_complete_slate_match_content():
    world = world_fixture()
    state = state_fixture(world)
    state.active[1] = False
    for k, weight in ((3, 0.), (10, .8)):
        content = start(ContentBasedRecommender(K=k), world)
        calibrated = start(CalibratedRecommender(K=k, calibration_weight=weight), world)
        np.testing.assert_array_equal(calibrated.recommend(state), content.recommend(state))


def test_calibration_zero_does_not_require_category_labels():
    world = world_fixture()
    world.cats = None
    state = state_fixture(world)
    blind = start(ContentBasedRecommender(K=3), world)
    calibrated = start(CalibratedRecommender(K=3, calibration_weight=0), world)
    np.testing.assert_array_equal(calibrated.recommend(state), blind.recommend(state))


def test_calibration_rejects_invalid_visible_category_counts():
    world = world_fixture()
    state = state_fixture(world)
    rec = start(CalibratedRecommender(K=2), world)
    for profiles in (np.ones((2, 3)), np.array([[-1., 2.], [1., 0.]]), np.full((2, 2), np.inf)):
        state.extra["category_profiles"] = profiles
        with pytest.raises(ValueError, match="category_profiles"):
            rec.recommend(state)


def test_fatigue_popularity_downweights_only_observed_bored_reads():
    world = world_fixture()
    state = state_fixture(world)
    rec = start(FatiguePopularityRecommender(prior_weight=0, fatigue_weight=.5), world)
    rec.observe(state, np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([3., -2., .4]),
                signals={"boredom": [1, 0, 1], "left_users": [1]})
    np.testing.assert_array_equal(rec.counts, [1.5, .5, 0., 0., 0., 0., 0., 0.])
    assert rec.departed_users.tolist() == [False, True]
    before = rec.counts.copy()
    rec.observe(state, [], [], [], signals={"left_users": [0]})
    np.testing.assert_array_equal(rec.counts, before)
    assert rec.departed_users.all()


def test_fatigue_mf_trains_observed_adjusted_targets_and_weighted_counts():
    world = world_fixture()
    state = state_fixture(world)
    rec = start(FatigueMFRecommender(prior_weight=0, fatigue_weight=.5, factors=2), world)
    rec.observe(state, np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([3., -2., .4]),
                signals={"boredom": [1, 0, 1], "left_users": [1]})
    np.testing.assert_allclose(rec.observed_ratings[[0, 1, 0], [0, 0, 1]], [2.5, -2., -.1])
    assert np.isnan(rec.observed_ratings[:, 2:]).all()
    np.testing.assert_array_equal(rec.counts[:2], [1.5, .5])
    assert rec.departed_users.tolist() == [False, True]


@pytest.mark.parametrize("aware_type,blind_type,kwargs", [
    (FatiguePopularityRecommender, PopularityRecommender, {}),
    (FatigueMFRecommender, MatrixFactorizationRecommender, {"factors": 2}),
    (FatigueContentRecommender, ContentBasedRecommender, {}),
])
def test_zero_fatigue_weight_matches_blind_method(aware_type, blind_type, kwargs):
    world = world_fixture()
    state = state_fixture(world)
    blind = start(blind_type(K=3, **kwargs), world)
    aware = start(aware_type(K=3, fatigue_weight=0, **kwargs), world)
    for t in range(3):
        state.t = t
        np.testing.assert_array_equal(aware.recommend(state), blind.recommend(state))
        users, items, feedback = np.array([0, 1]), np.array([t, t + 4]), np.array([1., -1.])
        blind.observe(state, users, items, feedback)
        aware.observe(state, users, items, feedback, signals={"boredom": [1, 0]})
        np.testing.assert_array_equal(aware.counts, blind.counts)
        if hasattr(blind, "observed_ratings"):
            for key, value in blind.model_state().items():
                np.testing.assert_array_equal(aware.model_state()[key], value)
        for user, item in zip(users, items):
            state.history[user].append(int(item))
            state.available[user, item] = False


def test_fatigue_content_penalty_and_empty_round_decay_use_observable_history():
    world = world_fixture()
    state = state_fixture(world)
    rec = start(FatigueContentRecommender(fatigue_weight=1., decay=.5), world)
    baseline = rec.relevance(state).copy()
    rec.observe(state, np.array([0]), np.array([0]), np.array([.2]), signals={"boredom": [1]})
    np.testing.assert_array_equal(rec.recent_category_counts[0], [2., 0.])
    np.testing.assert_allclose(rec.relevance(state)[0, :4], baseline[0, :4] - 2 / 3)
    np.testing.assert_allclose(rec.relevance(state)[0, 4:], baseline[0, 4:])
    state.t = 2
    np.testing.assert_allclose(rec.relevance(state)[0, :4], baseline[0, :4] - 1 / 3)
    np.testing.assert_array_equal(rec.recent_category_counts[0], [.5, 0.])
    rec.recommend(state)
    np.testing.assert_array_equal(rec.recent_category_counts[0], [.5, 0.])
    state.t = 3
    rec.observe(state, [], [], [], signals={"left_users": [0]})
    np.testing.assert_array_equal(rec.recent_category_counts[0], [.25, 0.])
    assert rec.departed_users[0]
    saved = rec.model_state()
    restored = start(FatigueContentRecommender(fatigue_weight=1., decay=.5), world)
    restored.restore_model_state(saved)
    assert_model_equal(restored.model_state(), saved)
    np.testing.assert_array_equal(restored.recommend(state), rec.recommend(state))


@pytest.mark.parametrize("rec_type", [FatiguePopularityRecommender, FatigueMFRecommender,
                                      FatigueContentRecommender])
def test_invalid_signals_do_not_mutate_model(rec_type):
    world = world_fixture()
    state = state_fixture(world)
    rec = start(rec_type(), world)
    before = rec.model_state()
    for signals in ({"boredom": [2]}, {"boredom": []}, {"left_users": [2]}, {"latent_fatigue": [1]}):
        with pytest.raises(ValueError):
            rec.observe(state, np.array([0]), np.array([0]), np.array([1.]), signals=signals)
        assert_model_equal(rec.model_state(), before)


@pytest.mark.parametrize("weight", [-.1, 1.1, np.nan, True])
def test_new_weights_reject_invalid_values(weight):
    for rec_type, name in ((CalibratedRecommender, "calibration_weight"),
                           (FatiguePopularityRecommender, "fatigue_weight"),
                           (FatigueMFRecommender, "fatigue_weight"),
                           (FatigueContentRecommender, "fatigue_weight")):
        with pytest.raises(ValueError):
            rec_type(**{name: weight})
