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
from time import perf_counter

from .params import Params
from .recommenders import IdealRecommender, NoRecommender, Recommender, SimState
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
                   seen: np.ndarray | None = None,
                   recorder=None, query_texts=None, category_profiles=None) -> dict:
    pop = Population(world, p, U, rng, profiles=profiles, seen=seen)
    bel = Beliefs(world, pop, p)
    rec = recommender or NoRecommender()
    rec.reset(world, U, rng)
    # Privileged reference only: ordinary state objects never contain X or V.
    if type(rec) is IdealRecommender:
        rec.set_reference_values(pop.X)

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

    if recorder is not None:
        recorder.start(world, p, rec, rng, profiles if p.reader_history_visible else None, consumed,
                       {"X": pop.X, "V": pop.V, "prior_means": pop.vbar,
                        "latent_profiles": profiles})

    for t in range(T):
        # Readers' real MIND history reaches recommenders only in the
        # "returning user" setting; readers' own beliefs always use it.
        extra = {}
        if p.reader_history_visible:
            if profiles is not None:
                extra["profiles"] = profiles
            if query_texts is not None:
                extra["query_texts"] = query_texts
            if category_profiles is not None:
                extra["category_profiles"] = category_profiles
        state = SimState(world, t, ~consumed, history, ratings, active.copy(),
                         extra=extra)
        if recorder is not None:
            recorder.before_round(state, rec,
                                  {"belief_mean": bel.mean, "belief_variance": bel.var,
                                   "fatigue": F})
        started = perf_counter()
        slates = rec.recommend(state)
        recommendation_seconds = perf_counter() - started

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

        users, items, fb, bored, left_users = [], [], [], [], []
        for i in np.flatnonzero(active):
            n = int(np.argmax(score[i]))
            if not np.isfinite(score[i, n]):
                continue                     # empty slate this round
            if p.outside_option is not None and score[i, n] < p.outside_option:
                active[i] = False
                left_at[i] = t
                left_users.append(i)
                continue
            penalty = p.phi * F[i, n]
            seq[i, t], val[i, t], pen[i, t] = n, pop.X[i, n], penalty
            consumed[i, n] = True
            history[i].append(n)
            ratings[i, n] = pop.X[i, n] - penalty
            bel.observe(i, n)                # learning spillovers
            users.append(i); items.append(n); fb.append(ratings[i, n])
            # Observable coarse self-report; raw latent fatigue is not exposed.
            bored.append(float(penalty > p.boredom_threshold))

        if Kf is not None:                   # fatigue decays, then builds up
            F *= p.fatigue_decay
            if users:
                F[users] += Kf[items]
        user_array, item_array = np.array(users, dtype=int), np.array(items, dtype=int)
        feedback_array = np.array(fb, dtype=float)
        signals = {"boredom": np.array(bored, dtype=float),
                   "left_users": np.array(left_users, dtype=int)}
        if getattr(rec, "accepts_feedback_signals", False):
            rec.observe(state, user_array, item_array, feedback_array, signals=signals)
        elif users:
            rec.observe(state, user_array, item_array, feedback_array)
        if recorder is not None:
            recorder.after_round(t, slates, np.array(users, dtype=int),
                                 np.array(items, dtype=int), np.array(fb, dtype=float), rec,
                                 {"choice_scores": score, "active_after": active,
                                  "fatigue_after": F, "belief_mean_after": bel.mean,
                                  "belief_variance_after": bel.var,
                                  "recommendation_seconds": recommendation_seconds},
                                 signals=signals)

    run = {"seq": seq, "val": val, "pen": pen, "left_at": left_at,
           "X": pop.X, "V": pop.V}
    if recorder is not None:
        recorder.finish(run)
    return run
