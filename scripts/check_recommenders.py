"""Run this script needs NumPy, but no MIND data, downloads, or embedding model.
Its results are engineering checks, not research comparison findings.
"""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.params import Params
from src.sim.recommenders import REGISTRY, SimState
from src.sim.simulate import run_population
from src.sim.world import World


def synthetic_inputs(users, articles, dimensions, seed):
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((articles, dimensions))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    profiles = rng.standard_normal((users, dimensions))
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True)
    # The covariance is identity: this checks integration, not the paper model.
    distances = np.clip(1. - embeddings @ embeddings.T, 0., 2.) / 2.
    np.fill_diagonal(distances, 0.)
    world = World(N=articles, D=distances, C=np.eye(articles), rho=0.,
                  ids=np.arange(articles), profile_vecs=embeddings,
                  popularity=rng.random(articles),
                  cats=np.arange(articles) % 10, kind="news")
    world.article_metadata = [{"news_id": int(i), "title": f"topic{i % 10} news update {i}",
                               "abstract": f"Synthetic topic{i % 10} report"}
                              for i in range(articles)]
    return world, profiles


def check_slate(slate, state, k):
    if slate is None:
        return
    assert slate.shape == (len(state.active), k), "Incorrect slate shape"
    assert np.issubdtype(slate.dtype, np.integer), "Article indices must be integers"
    assert (slate >= -1).all(), "Only -1 can represent padding"
    for user, row in enumerate(slate):
        items = row[row >= 0]
        assert len(items) == len(np.unique(items)), "Duplicate slate items"
        assert (items < state.world.N).all(), "Article index outside catalogue"
        assert state.available[user, items].all(), "Already-read article recommended"
        expected = min(k, state.available[user].sum()) if state.active[user] else 0
        assert len(items) == expected, "Incorrect number of eligible recommendations"


def benchmark(world, profiles, name, rounds, k, seed, simulator=False):
    users = len(profiles)
    rec = REGISTRY[name](K=k)
    start = perf_counter()
    rec.reset(world, users, np.random.default_rng(seed))
    reset_seconds = perf_counter() - start
    state = SimState(world, 0, np.ones((users, world.N), dtype=bool),
                     [[] for _ in range(users)], np.full((users, world.N), np.nan),
                     np.ones(users, dtype=bool), {"profiles": profiles})
    feedback_rng = np.random.default_rng(seed + 1)
    recommendation_seconds, feedback_seconds = 0., 0.
    for t in range(rounds):
        state.t = t
        start = perf_counter()
        slate = rec.recommend(state)
        recommendation_seconds += perf_counter() - start
        check_slate(slate, state, k)
        chosen_users, chosen_items = [], []
        for user in np.flatnonzero(state.active):
            eligible = np.flatnonzero(state.available[user]) if slate is None \
                else slate[user][slate[user] >= 0]
            if len(eligible):
                # A deterministic synthetic reader picks the first eligible item.
                item = int(eligible[0])
                chosen_users.append(user)
                chosen_items.append(item)
                state.available[user, item] = False
                state.history[user].append(item)
        chosen_users = np.asarray(chosen_users, dtype=int)
        chosen_items = np.asarray(chosen_items, dtype=int)
        feedback = feedback_rng.standard_normal(len(chosen_users))
        state.ratings[chosen_users, chosen_items] = feedback
        start = perf_counter()
        rec.observe(state, chosen_users, chosen_items, feedback)
        feedback_seconds += perf_counter() - start
    result = {"recommender": name, "reset_seconds": reset_seconds,
              "recommendation_seconds": recommendation_seconds,
              "feedback_seconds": feedback_seconds,
              "recommender_total_seconds": reset_seconds + recommendation_seconds + feedback_seconds,
              "contract_passed": True}
    if simulator:
        start = perf_counter()
        run = run_population(world, Params(T=rounds, phi=.5), users,
                             np.random.default_rng(seed), REGISTRY[name](K=k),
                             profiles=profiles)
        result["simulator_seconds"] = perf_counter() - start
        for row in run["seq"]:
            items = row[row >= 0]
            assert len(items) == len(np.unique(items)), "Simulator repeated an article"
        result["simulator_passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--articles", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recommenders", nargs="+", choices=list(REGISTRY),
                        default=list(REGISTRY))
    parser.add_argument("--simulator", action="store_true",
                        help="Also time complete synthetic simulator runs")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    args = parser.parse_args()
    if min(args.users, args.articles, args.dimensions, args.rounds, args.k) <= 0:
        parser.error("users, articles, dimensions, rounds, and k must be positive")
    start = perf_counter()
    world, profiles = synthetic_inputs(args.users, args.articles, args.dimensions, args.seed)
    setup_seconds = perf_counter() - start
    results = [benchmark(world, profiles, name, args.rounds, args.k, args.seed,
                         simulator=args.simulator) for name in args.recommenders]
    report = {"data": "synthetic; not MIND or research comparison results",
              "environment": {"python": platform.python_version(),
                              "numpy": np.__version__, "platform": platform.platform()},
              "settings": {"users": args.users, "articles": args.articles,
                           "rounds": args.rounds, "dimensions": args.dimensions,
                           "k": args.k, "seed": args.seed},
              "setup_seconds": setup_seconds, "results": results}
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
