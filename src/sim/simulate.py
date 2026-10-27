"""The simulation loop.

One call to `run_population` = one population of U users consuming T items.
All users move forward one round at a time (synchronously), so recommenders
that learn from the whole population (popularity, matrix factorisation, ...)
see everyone's feedback from round t before making round t+1 recommendations.
This is the feedback loop studied by Chaney et al.

Each round, for every active user:
  1. score every item by certainty equivalent: mean - gamma/2 * variance
     (minus phi * fatigue, if users anticipate fatigue)
  2. remove items already consumed / already read before the simulation
  3. if there is a recommender, keep only items in the user's slate
  4. pick the best remaining item (or leave, if an outside option is set and
     nothing beats it)
  5. experience it: true value x, minus fatigue penalty
  6. update beliefs (Bayesian, with spillovers) and fatigue state
  7. the recommender observes the feedback
"""
import numpy as np

from .params import Params
from .recommenders import NoRecommender, Recommender, SimState
from .users import Beliefs, Population
from .world import World


def fatigue_kernel(world: World, fatigue_rho: float) -> np.ndarray:
    """Similarity used for fatigue: fatigue_rho ** (distance / neighbour unit).
    Kept separate from C so fatigue works even when rho = 0."""
    key = ("_fk", fatigue_rho)
    if not hasattr(world, "_cache"):
        world._cache = {}
    if key not in world._cache:
        K = np.power(fatigue_rho, world.D / world.unit) if fatigue_rho > 0 \
            else np.eye(world.N)
        np.fill_diagonal(K, 1.0)
        world._cache[key] = K
    return world._cache[key]


def run_population(world: World, p: Params, U: int, rng: np.random.Generator,
                   recommender: Recommender | None = None,
                   profiles: np.ndarray | None = None,
                   seen: np.ndarray | None = None) -> dict:
    pop = Population(world, p, U, rng, profiles=profiles, seen=seen)
    bel = Beliefs(world, pop, p)
    rec = recommender or NoRecommender()
    rec.reset(world, U, rng)

    N, T = world.N, p.T
    Kf = fatigue_kernel(world, p.fatigue_rho) if p.phi > 0 else None
    F = np.zeros((U, N))                     # fatigue state per user per item
    consumed = pop.seen.copy()
    active = np.ones(U, bool)
    ratings = np.full((U, N), np.nan)
    history = [[] for _ in range(U)]

    seq = np.full((U, T), -1)                # item consumed at round t (-1 = none)
    val = np.full((U, T), np.nan)            # true value x of that item
    pen = np.full((U, T), np.nan)            # fatigue penalty at that moment
    left_at = np.full(U, T)                  # round at which the user left

    for t in range(T):
        extra = {"profiles": profiles} if profiles is not None else {}
        state = SimState(world, t, ~consumed, history, ratings, active.copy(), extra=extra)
        slates = rec.recommend(state)

        score = bel.certainty_equivalent(p.gamma)
        if p.phi > 0 and p.fatigue_aware:
            score = score - p.phi * F
        allowed = ~consumed
        if slates is not None:
            mask = np.zeros_like(allowed)
            r, c = np.nonzero(slates >= 0)
            mask[r, slates[r, c]] = True
            allowed &= mask
        score = np.where(allowed, score, -np.inf)

        users, items, fb = [], [], []
        for i in np.flatnonzero(active):
            n = int(np.argmax(score[i]))
            if not np.isfinite(score[i, n]):
                continue                     # empty slate this round
            if p.outside_option is not None and score[i, n] < p.outside_option:
                active[i] = False
                left_at[i] = t
                continue
            penalty = p.phi * F[i, n]
            seq[i, t], val[i, t], pen[i, t] = n, pop.X[i, n], penalty
            consumed[i, n] = True
            history[i].append(n)
            ratings[i, n] = pop.X[i, n] - penalty
            bel.observe(i, n)                # learning spillovers
            users.append(i); items.append(n); fb.append(ratings[i, n])

        if Kf is not None:                   # fatigue decays, then builds up
            F *= p.fatigue_decay
            if users:
                F[users] += Kf[items]
        if users:
            rec.observe(state, np.array(users), np.array(items), np.array(fb))

    return {"seq": seq, "val": val, "pen": pen, "left_at": left_at,
            "X": pop.X, "V": pop.V}
