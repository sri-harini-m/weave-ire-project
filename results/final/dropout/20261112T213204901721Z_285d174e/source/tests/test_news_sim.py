"""Unit tests for the news simulation and fatigue extensions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.store import MindStore
from src.sim.metrics import user_metrics, homogeneity
from src.sim.news_users import attach_user_pool, sample_news_users
from src.sim.params import Params
from src.sim.recommenders import RandomRecommender
from src.sim.simulate import run_population
from src.sim.users import Beliefs, Population
from src.sim.world import describe, news_world


@pytest.fixture(scope="module")
def store():
    return MindStore()


@pytest.fixture(scope="module")
def small_world(store):
    return news_world(store, N=50, rho=0.5, pca_dims=8, seed=42)


def test_news_world_is_valid_covariance(small_world):
    assert describe(small_world)["min_eig"] > -1e-8
    assert np.allclose(np.diag(small_world.C), 1.0)
    assert 0 <= small_world.D.min() <= small_world.D.max() <= 1.0
    assert small_world.profile_vecs.shape == (50, 384)
    assert len(small_world.cats) == 50
    assert len(small_world.ids) == 50
    assert hasattr(small_world, "article_metadata")


def test_fresh_readers_have_clean_slate(small_world, store):
    w = attach_user_pool(small_world, store, max_users=100, seed=0, fresh_readers=True)
    profiles, seen = sample_news_users(w, np.random.default_rng(0), U=10)
    assert seen.sum() == 0, "Fresh readers must not have pre-consumed catalogue items"
    assert profiles.shape == (10, 384)


def test_news_bayesian_spillover(small_world, store):
    w = attach_user_pool(small_world, store, max_users=100, seed=0)
    p = Params(info="none")
    rng = np.random.default_rng(1)
    profiles, seen = sample_news_users(w, rng, U=2)
    pop = Population(w, p, 2, rng, profiles=profiles, seen=seen)
    bel = Beliefs(w, pop, p)
    
    before_mean = bel.mean.copy()
    bel.observe(0, 5) # user 0 consumes article 5
    
    # Articles correlated with 5 should have updated means
    changed = np.flatnonzero(~np.isclose(before_mean[0], bel.mean[0], atol=1e-5))
    assert len(changed) > 1, "Reading an article must spill over to correlated articles"
    assert bel.var[0, 5] == pytest.approx(0.0, abs=1e-5), "Consumed item has 0 variance"


def test_fatigue_fade_and_buildup(small_world):
    p = Params(T=5, phi=1.0, fatigue_decay=0.5, fatigue_rho=0.8, info="none")
    run = run_population(small_world, p, 10, np.random.default_rng(42))
    assert np.nansum(run["pen"]) > 0, "Fatigue penalty must build up over time"
    # Net welfare should be strictly less than raw welfare when phi > 0
    raw_welfare = np.nanmean(run["val"])
    net_welfare = np.nanmean(run["val"] - run["pen"])
    assert net_welfare < raw_welfare


def test_news_metrics_entropy_and_diversity(small_world):
    p = Params(T=10, gamma=1.0, info="none")
    run = run_population(small_world, p, 15, np.random.default_rng(0))
    um = user_metrics(small_world, run)
    assert "cat_entropy" in um.columns
    assert "n_categories" in um.columns
    assert (um["cat_entropy"] >= 0).all()
    assert (um["n_categories"] >= 1).all()
    assert (um["diversity"] >= 0).all()
    assert 0 <= homogeneity(run) <= 1.0


def test_outside_option_dropout_under_fatigue(small_world):
    # High fatigue and strict outside option forces early dropout
    p = Params(T=20, phi=5.0, outside_option=0.0, info="none")
    run = run_population(small_world, p, 20, np.random.default_rng(0))
    assert (run["left_at"] < 20).any(), "Users should drop out when utility falls below outside option"


class MockRecommender:
    """Mock recommender to verify the shared plugin contract."""
    name = "mock"

    def __init__(self, K: int = 5):
        self.K = K
        self.observed_calls = 0

    def reset(self, world, U, rng):
        self.world, self.U, self.rng = world, U, rng
        self.slates_history = []

    def recommend(self, state):
        slates = np.full((self.U, self.K), -1, dtype=int)
        for u in range(self.U):
            avail = np.flatnonzero(state.available[u])
            if len(avail) > 0:
                chosen = self.rng.choice(avail, size=min(self.K, len(avail)), replace=False)
                slates[u, :len(chosen)] = chosen
        self.slates_history.append(slates.copy())
        return slates

    def observe(self, state, users, items, feedback):
        self.observed_calls += 1
        assert len(users) == len(items) == len(feedback)
        assert (users >= 0).all() and (users < self.U).all()
        assert (items >= 0).all() and (items < self.world.N).all()


def test_recommender_contract_and_slate_integrity(small_world):
    rec = MockRecommender(K=4)
    p = Params(T=8, info="none")
    run = run_population(small_world, p, 10, np.random.default_rng(7), recommender=rec)
    
    # 1. Slate size and no repeats across reads
    for u in range(10):
        reads = run["seq"][u][run["seq"][u] >= 0]
        assert len(reads) == len(set(reads)), "User must never consume an article twice"

    # 2. Chosen items must strictly come from the recommended slate in that round
    for t in range(8):
        slate_t = rec.slates_history[t] # [U, K]
        for u in range(10):
            chosen = run["seq"][u, t]
            if chosen >= 0:
                assert chosen in slate_t[u], f"Round {t}: User {u} consumed {chosen} not in slate {slate_t[u]}"

    # 3. Observe callback was triggered
    assert rec.observed_calls == 8


def test_history_length_sets_uncertainty(small_world, store):
    """Test A7: Prior variance scales inversely with history length via n0."""
    w = attach_user_pool(small_world, store, max_users=100, seed=0, min_history=20)
    p = Params(info="none", uncertainty_n0=20.0)
    rng = np.random.default_rng(42)
    profiles, seen = sample_news_users(w, rng, U=20)

    # Force artificial contrast in history lengths for testing
    lens = np.array([5] * 10 + [80] * 10)
    pop = Population(w, p, 20, rng, profiles=profiles, seen=seen, history_lens=lens)
    bel = Beliefs(w, pop, p)

    # Short history users (5 clicks) must have higher variance than long history users (80 clicks)
    var_short = bel.var[0, :].mean()
    var_long = bel.var[10, :].mean()
    assert var_short > var_long
    expected_ratio = (20.0 / (20.0 + 5)) / (20.0 / (20.0 + 80))
    assert pytest.approx(var_short / var_long, rel=1e-3) == expected_ratio


def test_separate_history_subsets_true_vs_believed(small_world, store):
    """Test A8: History split into expected (prior) and true taste profiles."""
    w = attach_user_pool(small_world, store, max_users=100, seed=0, min_history=20, true_history_size=10)
    assert w.user_pool["total_qualified_users"] > 0
    rng = np.random.default_rng(0)
    profiles, seen = sample_news_users(w, rng, U=10)
    true_profiles = w.sampled_true_profiles

    assert profiles.shape == (10, 384)
    assert true_profiles.shape == (10, 384)
    # Profiles should reflect real taste drift (not strictly identical)
    diff = np.abs(profiles - true_profiles).sum()
    assert diff > 0.1, "Prior profile and recent true taste profile should reflect temporal drift"

    p = Params(info="none")
    pop = Population(w, p, 10, rng, profiles=profiles, seen=seen, true_profiles=true_profiles)
    assert hasattr(pop, "v_true")
    # v_true should differ from vbar
    assert not np.allclose(pop.v_true, pop.vbar)


def test_reader_types_heterogeneity(small_world):
    """Test A5: Named reader types with custom gamma, phi, and fatigue_decay."""
    p = Params(T=5, reader_types_enabled=True,
               reader_type_proportions={"cautious": 0.5, "explorer": 0.5})
    rng = np.random.default_rng(123)
    run = run_population(small_world, p, 20, rng)

    assert len(run["reader_types"]) == 20
    assert set(run["reader_types"]) == {"cautious", "explorer"}

    cautious_idx = [i for i, t in enumerate(run["reader_types"]) if t == "cautious"]
    explorer_idx = [i for i, t in enumerate(run["reader_types"]) if t == "explorer"]

    assert len(cautious_idx) > 0 and len(explorer_idx) > 0
    assert (run["gammas"][cautious_idx] == 2.0).all()
    assert (run["gammas"][explorer_idx] == 0.0).all()

    um = user_metrics(small_world, run)
    assert "reader_type" in um.columns
    assert set(um["reader_type"].unique()) == {"cautious", "explorer"}


def test_reader_types_ranges(small_world):
    """Test drawing reader parameters from continuous uniform ranges."""
    ranges = {"gamma": (0.2, 1.8), "phi": (0.3, 1.2), "fatigue_decay": (0.6, 0.9)}
    p = Params(T=5, reader_types_enabled=True, reader_type_ranges=ranges)
    rng = np.random.default_rng(99)
    run = run_population(small_world, p, 30, rng)

    assert (run["gammas"] >= 0.2).all() and (run["gammas"] <= 1.8).all()
    assert (run["phis"] >= 0.3).all() and (run["phis"] <= 1.2).all()
    assert (run["decays"] >= 0.6).all() and (run["decays"] <= 0.9).all()
