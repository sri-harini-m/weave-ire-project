"""Independent lexical and explicit-feedback recommender behavior checks."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.recommenders import (BM25Recommender, MatrixFactorizationRecommender,
                                   PopularityRecommender, REGISTRY, SimState)
from src.sim.world import World


def text_world(documents=None):
    documents = ["alpha alpha beta", "alpha gamma gamma gamma", "beta beta",
                 "delta gamma"] if documents is None else documents
    n = len(documents)
    world = World(N=n, D=np.ones((n, n)) - np.eye(n), C=np.eye(n), rho=0.,
                  ids=np.array([f"n{i}" for i in range(n)]),
                  popularity=np.arange(1., n + 1), profile_vecs=None, kind="news")
    world.article_metadata = [{"news_id": f"n{i}", "title": document, "abstract": None}
                               for i, document in enumerate(documents)]
    return world


def state_for(world, users=2, query_texts=None, history=None):
    history = [[] for _ in range(users)] if history is None else history
    state = SimState(world, 0, np.ones((users, world.N), bool), history,
                     np.full((users, world.N), np.nan), np.ones(users, bool))
    for user, items in enumerate(history):
        state.available[user, items] = False
    if query_texts is not None:
        state.extra["query_texts"] = query_texts
    return state


def initialized(rec, world, users=2, seed=17):
    rec.reset(world, users, np.random.default_rng(seed))
    return rec


def test_additional_algorithms_are_registered():
    assert REGISTRY["bm25"] is BM25Recommender
    assert REGISTRY["mf"] is MatrixFactorizationRecommender


def test_bm25_scores_match_independent_known_term_statistics():
    world = text_world()
    state = state_for(world, query_texts=["ALPHA alpha", "beta"])
    rec = initialized(BM25Recommender(K=4), world)
    idf = np.log(1 + (4 - 2 + .5) / (2 + .5))
    average_length = 11 / 4
    expected = np.zeros((2, 4))
    # The four documents have lengths 3, 4, 2, 2. Both query terms occur in two.
    for user, frequencies in enumerate(([2, 1, 0, 0], [1, 0, 2, 0])):
        for article, (frequency, length) in enumerate(zip(frequencies, (3, 4, 2, 2))):
            expected[user, article] = idf * frequency * 2.2 / (
                frequency + 1.2 * (.25 + .75 * length / average_length))
    np.testing.assert_allclose(rec.relevance(state), expected, rtol=1e-12)
    np.testing.assert_array_equal(rec.recommend(state),
                                  np.argsort(-expected, axis=1, kind="stable"))


def test_bm25_term_saturation_and_length_normalization_are_separate():
    world = text_world(["alpha", "alpha alpha", "alpha beta beta beta"])
    state = state_for(world, query_texts=["alpha", "alpha"])
    no_length = initialized(BM25Recommender(K=3, b=0), world)
    scores = no_length.relevance(state)[0]
    assert scores[0] == scores[2]
    assert scores[0] < scores[1] < 2 * scores[0]
    with_length = initialized(BM25Recommender(K=3), world)
    assert with_length.relevance(state)[0, 0] > with_length.relevance(state)[0, 2]


def test_bm25_queries_combine_visible_text_with_history_and_ignore_embedding_profiles():
    world = text_world()
    state = state_for(world, query_texts=["alpha", "alpha"], history=[[2], []])
    state.extra["profiles"] = np.full((2, 3), 999.)
    rec = initialized(BM25Recommender(K=3), world)
    explicit = state_for(world, query_texts=["alpha beta", "alpha"])
    np.testing.assert_allclose(rec.relevance(state), rec.relevance(explicit))
    assert 2 not in rec.recommend(state)[0]


def test_bm25_aligns_article_text_by_external_id():
    world = text_world()
    state = state_for(world, query_texts=["alpha", "delta"])
    original = initialized(BM25Recommender(K=4), world).recommend(state)
    world.article_metadata.reverse()
    reordered = initialized(BM25Recommender(K=4), world).recommend(state)
    np.testing.assert_array_equal(reordered, original)


def test_bm25_empty_and_unknown_queries_use_current_popularity():
    world = text_world()
    state = state_for(world, query_texts=["", "unseenword"])
    bm25 = initialized(BM25Recommender(K=3), world)
    popularity = initialized(PopularityRecommender(K=3), world)
    for rec in (bm25, popularity):
        rec.observe(state, np.array([0, 1]), np.array([0, 0]), np.array([-1., 0.]))
    np.testing.assert_array_equal(bm25.recommend(state), popularity.recommend(state))


@pytest.mark.parametrize("metadata", [None, [],
    [{"news_id": "n0", "title": "alpha"}] * 4,
    [{"news_id": f"n{i}", "title": "", "abstract": None} for i in range(4)]])
def test_bm25_rejects_missing_duplicate_or_empty_article_text(metadata):
    world = text_world()
    if metadata is None:
        del world.article_metadata
    else:
        world.article_metadata = metadata
    with pytest.raises(ValueError):
        initialized(BM25Recommender(), world)


@pytest.mark.parametrize("query", ["alpha", ["alpha"], ["alpha", 3], [["alpha"], "beta"]])
def test_bm25_rejects_malformed_query_texts(query):
    world = text_world()
    rec = initialized(BM25Recommender(), world)
    with pytest.raises(ValueError):
        rec.recommend(state_for(world, query_texts=query))


@pytest.mark.parametrize("history", [[[-1], []], [[4], []], [[1.5], []]])
def test_bm25_rejects_invalid_history(history):
    world = text_world()
    state = state_for(world)
    state.history = history
    with pytest.raises(ValueError):
        initialized(BM25Recommender(), world).recommend(state)


def test_matrix_factorization_fits_observed_low_rank_utilities():
    world = text_world()
    state = state_for(world, users=3)
    utilities = np.array([[3., -2., 0., 1.], [2., -1., 0., 2.], [-3., 2., 0., -1.]])
    users, items = np.indices(utilities.shape)
    rec = initialized(MatrixFactorizationRecommender(factors=2, regularization=.01,
                                                    iterations=25), world, users=3)
    rec.observe(state, users.ravel(), items.ravel(), utilities.ravel())
    squared_error = np.mean((rec.predictions() - utilities) ** 2)
    mean_baseline_error = np.mean((utilities - utilities.mean()) ** 2)
    assert squared_error < .1 * mean_baseline_error
    assert squared_error < .05


def test_matrix_factorization_records_negative_and_zero_feedback_without_imputation():
    world = text_world()
    state = state_for(world)
    rec = initialized(MatrixFactorizationRecommender(), world)
    rec.observe(state, np.array([0, 1]), np.array([1, 2]), np.array([-5., 0.]))
    observed = rec.model_state()["observed_ratings"]
    assert observed[0, 1] == -5 and observed[1, 2] == 0
    assert np.isfinite(observed).sum() == 2
    assert np.isnan(observed[0, 0])


def test_matrix_factorization_cold_users_follow_popularity_and_unseen_items_tie():
    world = text_world()
    state = state_for(world, users=3)
    rec = initialized(MatrixFactorizationRecommender(K=3), world, users=3)
    popularity = initialized(PopularityRecommender(K=3), world, users=3)
    for model in (rec, popularity):
        model.observe(state, np.array([0, 1]), np.array([0, 0]), np.array([1., -1.]))
    np.testing.assert_array_equal(rec.recommend(state)[2], popularity.recommend(state)[2])
    state.available[0, 0] = False
    np.testing.assert_array_equal(rec.recommend(state)[0], [1, 2, 3])


@pytest.mark.parametrize("users,items,feedback", [
    ([-1], [0], [1.]), ([2], [0], [1.]), ([.5], [0], [1.]),
    ([0], [-1], [1.]), ([0], [4], [1.]), ([0], [0], [np.inf]),
    ([0], [0], [np.nan]), ([0, 1], [0], [1.]),
])
def test_matrix_factorization_rejects_bad_feedback_before_mutation(users, items, feedback):
    world = text_world()
    state = state_for(world)
    rec = initialized(MatrixFactorizationRecommender(), world)
    before = {k: np.array(v, copy=True) for k, v in rec.model_state().items()}
    with pytest.raises(ValueError):
        rec.observe(state, np.array(users), np.array(items), np.array(feedback))
    for key, value in before.items():
        np.testing.assert_array_equal(rec.model_state()[key], value)


@pytest.mark.parametrize("rec_type", [BM25Recommender, MatrixFactorizationRecommender])
@pytest.mark.parametrize("k", [0, 1, 7])
def test_new_recommenders_respect_eligibility_inactive_and_short_slates(rec_type, k):
    world = text_world()
    state = state_for(world, users=3, query_texts=["alpha", "beta", "delta"])
    state.available[0, :3] = False
    state.available[1] = False
    state.active[2] = False
    rec = initialized(rec_type(K=k), world, users=3)
    slate = rec.recommend(state)
    assert slate.shape == (3, k)
    assert np.issubdtype(slate.dtype, np.integer)
    for user, row in enumerate(slate):
        valid = row[row >= 0]
        assert len(valid) == len(np.unique(valid))
        assert state.available[user, valid].all()
        assert (row >= -1).all()
    assert (slate[1:] == -1).all()
    assert np.count_nonzero(slate[0] >= 0) == min(k, 1)


@pytest.mark.parametrize("rec_type", [BM25Recommender, MatrixFactorizationRecommender])
def test_new_recommenders_pad_an_empty_catalogue(rec_type):
    world = text_world([])
    state = state_for(world)
    rec = initialized(rec_type(K=3), world)
    np.testing.assert_array_equal(rec.recommend(state), np.full((2, 3), -1))


@pytest.mark.parametrize("rec_type", [BM25Recommender, MatrixFactorizationRecommender])
def test_new_recommenders_reset_and_restore_learned_state(rec_type):
    world = text_world()
    state = state_for(world, query_texts=["alpha", "beta"])
    rec = initialized(rec_type(K=3), world)
    initial = {k: np.array(v, copy=True) for k, v in rec.model_state().items()}
    rec.observe(state, np.array([0, 1]), np.array([1, 2]), np.array([2., -1.]))
    saved = {k: np.array(v, copy=True) for k, v in rec.model_state().items()}
    restored = initialized(rec_type(**rec.get_config()), world)
    restored.restore_model_state(saved)
    np.testing.assert_array_equal(restored.recommend(state), rec.recommend(state))
    rec.reset(world, 2, np.random.default_rng(17))
    for key, value in initial.items():
        np.testing.assert_array_equal(rec.model_state()[key], value)


@pytest.mark.parametrize("kwargs", [{"k1": -.1}, {"k1": np.inf}, {"b": -.1}, {"b": 1.1}])
def test_bm25_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        BM25Recommender(**kwargs)


@pytest.mark.parametrize("kwargs", [{"factors": 0}, {"factors": True},
    {"factors": 1.5}, {"regularization": 0}, {"regularization": np.nan},
    {"iterations": 0}, {"iterations": 1.5}])
def test_matrix_factorization_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        MatrixFactorizationRecommender(**kwargs)


def test_matrix_factorization_learning_repeats_with_the_same_seed():
    world = text_world()
    state = state_for(world)
    first = initialized(MatrixFactorizationRecommender(factors=2), world)
    second = initialized(MatrixFactorizationRecommender(factors=2), world)
    batches = [([0, 1], [1, 2], [2., -1.]), ([0, 1], [3, 1], [0., 3.])]
    for users, items, feedback in batches:
        for rec in (first, second):
            rec.observe(state, np.array(users), np.array(items), np.array(feedback))
    for key, value in first.model_state().items():
        np.testing.assert_array_equal(second.model_state()[key], value)
