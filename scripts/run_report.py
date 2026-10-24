"""Save a synthetic report bundle, or rebuild figures from an existing archive.

Synthetic results verify the pipeline. They are not evidence about MIND users.
Use run_news.py for the real-data comparison after the data pipeline is supplied.
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.experiment import run_grid
from src.sim.news_users import sample_news_users
from src.sim.recommenders import REGISTRY
from src.sim.run_bundle import RunBundle, verify_manifest, write_manifest
from src.sim.world import World, _corr_from_distance


def synthetic_world(rho, N=60, dimensions=16, seed=42, pool_size=100):
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((N, dimensions))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    squared = np.sum((embeddings[:, None] - embeddings[None, :]) ** 2, axis=2)
    distances = np.sqrt(np.maximum(squared, 0))
    unit = np.median(np.where(np.eye(N, dtype=bool), np.inf, distances).min(axis=1))
    correlation = _corr_from_distance(distances, unit, rho)
    world = World(N=N, D=distances / distances.max(), C=correlation, rho=rho,
                  unit=unit / distances.max(), ids=np.array([f"synthetic_{i}" for i in range(N)]),
                  profile_vecs=embeddings, popularity=rng.random(N),
                  cats=np.argmax(embeddings[:, :min(6, dimensions)], axis=1), kind="synthetic")
    profiles = rng.standard_normal((pool_size, dimensions))
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True)
    world.user_pool = {"ids": np.array([f"synthetic_reader_{i}" for i in range(pool_size)]),
                       "profiles": profiles, "seen": np.zeros((pool_size, N), dtype=bool)}
    world.article_metadata = [{"news_id": str(world.ids[i]), "category": int(world.cats[i]),
                               "title": f"Synthetic article {i}"} for i in range(N)]
    return world


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", type=Path, help="Rebuild metrics/figures from a saved run; no simulation")
    ap.add_argument("--out", type=Path, default=Path("results/report_checks"))
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--articles", type=int, default=60)
    ap.add_argument("--dimensions", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n-pop", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rho", type=float, nargs="+", default=[0.5])
    ap.add_argument("--fatigue", type=float, nargs="+", default=[0., 0.5])
    ap.add_argument("--mmr-weights", type=float, nargs="+", default=[0., 0.3, 0.6])
    args = ap.parse_args()
    if args.rebuild is not None:
        from src.sim.reporting import rebuild_report
        verify_manifest(args.rebuild)
        rebuild_report(args.rebuild)
        write_manifest(args.rebuild)
        print("Rebuilt saved outputs without simulation:", args.rebuild)
        return
    if min(args.users, args.dimensions, args.rounds, args.n_pop, args.k, args.workers) < 1 \
            or args.articles < 2:
        ap.error("counts must be positive; articles must be at least two")
    if any(not 0 <= value < 1 for value in args.rho):
        ap.error("rho must be in [0, 1)")
    if any(not 0 <= value <= 1 for value in args.mmr_weights):
        ap.error("MMR weights must be in [0, 1]")
    if any(not np.isfinite(value) or value < 0 for value in args.fatigue):
        ap.error("fatigue must be finite and nonnegative")
    config = {"data_kind": "synthetic", "settings": vars(args).copy(),
              "grid": {"rho": args.rho, "phi": args.fatigue, "info": ["none"]}}
    recs = {name: partial(REGISTRY[name], K=args.k)
            for name in ("none", "random", "popularity", "content")}
    for weight in args.mmr_weights:
        recs[f"mmr_{weight:g}"] = partial(REGISTRY["mmr"], K=args.k, diversity_weight=weight)
    with RunBundle(args.out, config, vars(args), "synthetic") as bundle:
        print("SYNTHETIC ENGINEERING CHECK: not MIND research findings")
        print("Saving complete run to:", bundle.path)
        world_fn = partial(synthetic_world, N=args.articles, dimensions=args.dimensions,
                           seed=args.seed, pool_size=max(args.users, 100))
        df, curves = run_grid(world_fn, config["grid"], base={"T": args.rounds, "gamma": 1.0},
                              U=args.users, n_pop=args.n_pop, seed=args.seed,
                              recommenders=recs, users_fn=partial(sample_news_users, U=args.users),
                              workers=args.workers, keep_curves=True, archive_dir=bundle.path)
        df.to_csv(bundle.path / "original_populations.csv", index=False)
        curves.to_csv(bundle.path / "original_curves.csv", index=False)
        from src.sim.reporting import rebuild_report
        rebuild_report(bundle.path)
        print("Archived population runs:", len(df))
        print("Report preparation can now use saved arrays without running the simulator.")
    print("Saved run:", bundle.path)


if __name__ == "__main__":
    main()
