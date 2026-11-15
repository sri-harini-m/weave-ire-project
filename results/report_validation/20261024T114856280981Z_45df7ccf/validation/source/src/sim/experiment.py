"""Run many populations over a parameter grid and collect results.

Output (one row per population, metrics averaged over its users):
    params..., recommender, pop, local_distance, diversity, welfare,
    welfare_net, fatigue, rounds, [cat_entropy, n_categories], homogeneity
"""
import itertools
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import homogeneity, time_curves, user_metrics
from .params import Params
from .simulate import run_population
from .recommenders import NoRecommender


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]


_WORLDS = {}


def _init(worlds):
    _WORLDS.update(worlds)


def _one(job):
    (rho, pdict, U, seed, rec_factory, users_fn, pop_id, keep_curves,
     archive_dir, job_id, label) = job
    world = _WORLDS[rho]
    p = Params(**pdict)
    rng = np.random.default_rng(seed)
    profiles, seen = users_fn(world, rng) if users_fn else (None, None)
    rec = rec_factory() if rec_factory else NoRecommender()
    archive = None
    if archive_dir is not None:
        from .artifacts import PopulationArchive
        settings = {k: getattr(rec, k) for k in
                    ("K", "prior_weight", "profile_weight", "diversity_weight")
                    if hasattr(rec, k)}
        archive = PopulationArchive(Path(archive_dir) / "populations" / f"job_{job_id:06d}",
                                    metadata={"job_id": job_id, "population": pop_id,
                                              "pop": pop_id, "seed": seed, "rho": rho,
                                              "recommender": label or rec.name,
                                              "algorithm": rec.name, "params": p.as_dict(),
                                              "data_kind": "mind" if world.kind == "news" else world.kind,
                                              "user_ids": getattr(world, "sampled_user_ids", None),
                                              "initial_histories": getattr(world, "sampled_histories", None),
                                              "recommender_settings": settings})
    try:
        run = run_population(world, p, U, rng, rec, profiles, seen, recorder=archive)
    except Exception as error:
        if archive is not None:
            archive.fail(error)
        raise
    um = user_metrics(world, run)
    row = {**pdict, "recommender": label or rec.name, "pop": pop_id,
           "seed": seed, "job_id": job_id,
           **um.drop(columns=["user"]).mean(numeric_only=True).to_dict(),
           "homogeneity": homogeneity(run)}
    curves = None
    if keep_curves:
        curves = time_curves(world, run).assign(**{k: pdict[k] for k in pdict},
                                                recommender=row["recommender"],
                                                pop=pop_id)
    return row, curves


def run_grid(world_fn, grid: dict, base: dict | None = None, n_pop: int = 10,
             U: int = 100, seed: int = 0, recommenders: dict | None = None,
             users_fn=None, workers: int = 1, keep_curves: bool = False,
             archive_dir=None):
    """world_fn(rho) -> World (rebuilt only when rho changes).
    grid: {param_name: [values]}; base: fixed Params overrides.
    recommenders: {name: factory()} or None for the paper's no-slate setup.
    users_fn(world, rng) -> (profiles, seen) for grounded news users, or None.
    With workers > 1, recommender factories and users_fn must be top-level
    functions/classes (or functools.partial of them), not lambdas.
    archive_dir: optional unique run directory for all population artifacts.
    """
    base = base or {}
    recs = recommenders or {None: None}
    combos = expand_grid(grid)
    worlds = {}
    jobs = []
    for ci, combo in enumerate(combos):
        pdict = {**base, **combo}
        rho = pdict.pop("rho", base.get("rho", 0.5))
        if rho not in worlds:
            worlds[rho] = world_fn(rho)
        for rname, rfac in recs.items():
            for k in range(n_pop):
                # Common random numbers: the seed ignores `info` and the
                # recommender, so every regime/recommender faces the SAME users
                # and true values. Differences are then caused by the regime.
                key = repr(sorted((a, b) for a, b in {**pdict, "rho": rho}.items()
                                  if a != "info"))
                s = seed * 1_000_003 + zlib.crc32(key.encode()) % 100_000 * 1000 + k
                jobs.append((rho, dict(pdict), U, s, rfac, users_fn, k, keep_curves,
                             str(archive_dir) if archive_dir is not None else None,
                             len(jobs), rname))

    if archive_dir is not None:
        root = Path(archive_dir)
        root.mkdir(parents=True, exist_ok=True)
        expected = [{"directory": f"job_{j[9]:06d}", "job_id": j[9],
                     "population": j[6], "seed": j[3], "rho": j[0],
                     "params": j[1], "recommender": j[10] or "none"} for j in jobs]
        with (root / "expected_populations.json").open("x") as handle:
            json.dump({"schema_version": 1, "jobs": expected}, handle, indent=2, default=str)
    _init(worlds)
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(worlds,)) as ex:
            out = list(ex.map(_one, jobs, chunksize=8))
    else:
        out = [_one(j) for j in jobs]

    rows, curves = [], []
    for (row, cur), job in zip(out, jobs):
        row["rho"] = job[0]
        rows.append(row)
        if cur is not None:
            curves.append(cur.assign(rho=job[0]))
    df = pd.DataFrame(rows)
    cdf = pd.concat(curves, ignore_index=True) if curves else None
    return df, cdf


def summarise(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Mean and 95% CI (normal approx over populations) per group."""
    g = df.groupby(by)[metrics]
    m, s, n = g.mean(), g.std(), g.count()
    ci = 1.96 * s / np.sqrt(n)
    out = m.join(ci, rsuffix="_ci95")
    return out.reset_index()
