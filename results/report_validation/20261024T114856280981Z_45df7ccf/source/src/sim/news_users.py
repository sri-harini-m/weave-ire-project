"""Real MIND users for the news simulation.

We pre-compute, once, a pool of real users:
    profile  = mean embedding of everything they clicked in MIND (full 384-dim)
    seen     = which catalogue articles they already read (they can't reread)
Each population then samples U users from this pool.

The pool is attached to the World object so it is shared with worker
processes without being copied for every job.
"""
import numpy as np


def attach_user_pool(world, store, max_users: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    uids = store.user_ids()
    if len(uids) > max_users:
        uids = list(rng.choice(uids, size=max_users, replace=False))
    pos = {nid: j for j, nid in enumerate(world.ids)}
    profiles = np.zeros((len(uids), store.embeddings.shape[1]), np.float32)
    seen = np.zeros((len(uids), world.N), bool)
    histories = []
    for k, u in enumerate(uids):
        hist = store.history(u)
        histories.append(list(hist))
        profiles[k] = store.embeddings[[store.index[n] for n in hist]].mean(0)
        for n in hist:
            if n in pos:
                seen[k, pos[n]] = True
    world.user_pool = {"ids": np.array(uids), "profiles": profiles, "seen": seen,
                       "histories": histories}
    return world


def sample_news_users(world, rng, U: int):
    """users_fn for run_grid: returns (profiles [U, dim], seen [U, N])."""
    pool = world.user_pool
    idx = rng.choice(len(pool["ids"]), size=U, replace=False)
    world.sampled_user_ids = pool["ids"][idx].tolist()
    world.sampled_histories = [pool["histories"][i] for i in idx] \
        if "histories" in pool else None
    return pool["profiles"][idx], pool["seen"][idx]
