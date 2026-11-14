import numpy as np
from time import perf_counter

from .params import Params
from .recommenders import IdealRecommender, NoRecommender, Recommender, SimState
from .users import Beliefs, Population
from .world import World


def fatigue_kernel(world: World, fatigue_rho: float) -> np.ndarray:


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
                   recorder=None, query_texts=None,
                   category_profiles: np.ndarray | None = None,
                   true_profiles: np.ndarray | None = None,
                   history_lens: np.ndarray | None = None) -> dict:
    if true_profiles is None and profiles is not None:
        cand = getattr(world, "sampled_true_profiles", None)
        if cand is not None and len(cand) == U:
            true_profiles = cand
    if history_lens is None and profiles is not None:
        cand = getattr(world, "sampled_history_lens", None)
        if cand is not None and len(cand) == U:
            history_lens = cand
    if category_profiles is None and profiles is not None:
        cand = getattr(world, "sampled_category_profiles", None)
        if cand is not None and len(cand) == U:
            category_profiles = cand

    pop = Population(world, p, U, rng, profiles=profiles, seen=seen,
                     true_profiles=true_profiles, history_lens=history_lens)
    bel = Beliefs(world, pop, p)
    rec = recommender or NoRecommender()
    rec.reset(world, U, rng)

    if type(rec) is IdealRecommender:
        rec.set_reference_values(pop.X)

    gammas = pop.gammas
    phis = pop.phis
    decays = pop.decays

    N, T = world.N, p.T
    Kf = fatigue_kernel(world, p.fatigue_rho) if (phis > 0).any() or p.phi > 0 else None
    F = np.zeros((U, N))
    consumed = pop.seen.copy()
    active = np.ones(U, bool)
    ratings = np.full((U, N), np.nan)
    history = [[] for _ in range(U)]

    seq = np.full((U, T), -1)
    val = np.full((U, T), np.nan)
    pen = np.full((U, T), np.nan)
    left_at = np.full(U, T)

    if recorder is not None:
        recorder.start(world, p, rec, rng, profiles if p.reader_history_visible else None, consumed,
                       {"X": pop.X, "V": pop.V, "prior_means": pop.vbar,
                        "latent_profiles": profiles})

    for t in range(T):


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

        score = bel.certainty_equivalent(gammas)
        if (phis > 0).any() and p.fatigue_aware:
            score = score - phis[:, None] * F
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
                continue
            if p.outside_option is not None and score[i, n] < p.outside_option:
                active[i] = False
                left_at[i] = t
                left_users.append(i)
                continue
            penalty = phis[i] * F[i, n]
            seq[i, t], val[i, t], pen[i, t] = n, pop.X[i, n], penalty
            consumed[i, n] = True
            history[i].append(n)
            ratings[i, n] = pop.X[i, n] - penalty
            bel.observe(i, n)
            users.append(i); items.append(n); fb.append(ratings[i, n])

            bored.append(float(penalty > p.boredom_threshold))

        if Kf is not None:
            F *= decays[:, None]
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
           "X": pop.X, "V": pop.V, "reader_types": pop.reader_types,
           "gammas": gammas, "phis": phis, "decays": decays,
           "history_lens": getattr(pop, "history_lens", None)}
    if recorder is not None:
        recorder.finish(run)
    return run
