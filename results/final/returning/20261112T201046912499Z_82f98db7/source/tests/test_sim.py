import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.metrics import homogeneity, user_metrics
from src.sim.params import Params
from src.sim.recommenders import RandomRecommender
from src.sim.simulate import run_population
from src.sim.users import Beliefs, Population
from src.sim.world import circle_world, describe


def test_circle_world_is_valid_covariance():
    for rho in (0, 0.3, 0.9):
        w = circle_world(200, rho)
        assert describe(w)["min_eig"] > -1e-8
        assert np.allclose(np.diag(w.C), 1)
    w = circle_world(10, 0.5)
    assert w.C[0, 1] == 0.5 and w.C[0, 9] == 0.5 and w.C[0, 2] == 0.25


def test_bayes_update_matches_bruteforce():
    w = circle_world(30, 0.7)
    p = Params(beta=1.0, sigma=1.0, info="none")
    rng = np.random.default_rng(1)
    pop = Population(w, p, 2, rng)
    bel = Beliefs(w, pop, p)
    for n in (3, 4, 17):
        bel.observe(0, n)
    Sig = bel.s2 * w.C
    S = np.array([3, 4, 17])
    R = np.setdiff1d(np.arange(30), S)
    mu = bel.mu0[0, R] + Sig[np.ix_(R, S)] @ np.linalg.solve(
        Sig[np.ix_(S, S)], pop.X[0, S] - bel.mu0[0, S])
    var = np.diag(Sig[np.ix_(R, R)] - Sig[np.ix_(R, S)] @ np.linalg.solve(
        Sig[np.ix_(S, S)], Sig[np.ix_(S, R)]))
    assert np.allclose(bel.mean[0, R], mu)
    assert np.allclose(bel.var[0, R], var)
    assert np.allclose(bel.mean[0, S], pop.X[0, S])
    assert np.allclose(bel.var[0, S], 0, atol=1e-6)


def test_rho_zero_no_spillover():
    w = circle_world(50, 0.0)
    p = Params(info="none")
    pop = Population(w, p, 1, np.random.default_rng(0))
    bel = Beliefs(w, pop, p)
    before = bel.mean.copy()
    bel.observe(0, 10)
    changed = np.flatnonzero(~np.isclose(before[0], bel.mean[0]))
    assert set(changed) <= {10}


def test_oracle_picks_true_best():
    w = circle_world(100, 0.5)
    p = Params(info="oracle", T=5)
    run = run_population(w, p, 3, np.random.default_rng(0))
    for i in range(3):
        best = np.argsort(-run["X"][i])[:5]
        assert list(run["seq"][i]) == list(best)


def test_no_repeats_and_metrics():
    w = circle_world(100, 0.5)
    run = run_population(w, Params(T=20, gamma=1), 20, np.random.default_rng(0))
    for i in range(20):
        assert len(set(run["seq"][i])) == 20
    um = user_metrics(w, run)
    assert um["diversity"].between(0, 0.5).all()
    assert 0 <= homogeneity(run) <= 1


def test_slate_restricts_choice():
    w = circle_world(100, 0.5)
    rec = RandomRecommender(K=5)
    run = run_population(w, Params(T=10), 10, np.random.default_rng(0), rec)
    assert (run["seq"] >= 0).all()


def test_fatigue_penalises_similar_consumption():
    w = circle_world(200, 0.5)
    base = dict(T=20, gamma=1, info="none")
    r0 = run_population(w, Params(**base), 50, np.random.default_rng(3))
    r1 = run_population(w, Params(**base, phi=1.0), 50, np.random.default_rng(3))
    d0 = user_metrics(w, r0)["local_distance"].mean()
    d1 = user_metrics(w, r1)["local_distance"].mean()
    assert d1 > d0
    assert np.nansum(r0["pen"]) == 0


def test_outside_option_lets_users_leave():
    w = circle_world(100, 0.5)
    run = run_population(w, Params(T=20, phi=3.0, outside_option=0.5), 30,
                         np.random.default_rng(0))
    assert (run["left_at"] < 20).any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
