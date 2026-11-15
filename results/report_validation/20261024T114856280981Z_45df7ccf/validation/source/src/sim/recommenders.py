"""Plug-in interface for recommendation systems.

The paper's three regimes ("none", "rec", "oracle") are about how much the
user KNOWS; they are set with Params.info. Real recommendation systems are
different: they decide what the user SEES. Each round a recommender returns a
slate of K items per user, and the user picks the best item in the slate
according to their own beliefs.

To add a recommender, subclass Recommender and implement `recommend`
(and `observe` if it learns from feedback). Everything the recommender is
allowed to see is in SimState: it never sees users' true values.
"""
from dataclasses import dataclass, field

import numpy as np

from .world import World


@dataclass
class SimState:
    world: World
    t: int                         # current round
    available: np.ndarray          # [U, N] bool, items the user can still consume
    history: list                  # history[i] = list of items user i consumed, in order
    ratings: np.ndarray            # [U, N] observed feedback (realised utility), NaN if none
    active: np.ndarray             # [U] bool, user is still in the session
    extra: dict = field(default_factory=dict)


class Recommender:
    name = "base"

    def __init__(self, K: int = 10):
        if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 0:
            raise ValueError("K must be a nonnegative integer")
        self.K = K

    def reset(self, world: World, U: int, rng: np.random.Generator):
        """Called once at the start of every population run."""
        self.world, self.U, self.rng = world, U, rng

    def recommend(self, state: SimState) -> np.ndarray | None:
        """Return [U, K] item indices (use -1 for padding), or None for
        'no recommender: the user browses the whole catalogue'."""
        raise NotImplementedError

    def observe(self, state: SimState, users: np.ndarray, items: np.ndarray,
                feedback: np.ndarray):
        """Called after each round with what users consumed and the feedback
        (realised utility). Learning recommenders update themselves here."""
        pass

    # helper for subclasses: top-K available items from a score matrix
    def top_k(self, scores: np.ndarray, available: np.ndarray) -> np.ndarray:
        s = np.where(available & np.isfinite(scores), scores, -np.inf)
        out = np.full((s.shape[0], self.K), -1, dtype=int)
        k = min(self.K, s.shape[1])
        # Stable sorting breaks ties by catalogue index, including at the cutoff.
        idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
        out[:, :k] = np.where(np.isfinite(np.take_along_axis(s, idx, 1)), idx, -1)
        return out

    def eligible(self, state: SimState) -> np.ndarray:
        if state.available.shape != (self.U, self.world.N):
            raise ValueError("available must have shape [U, N]")
        if state.active.shape != (self.U,):
            raise ValueError("active must have shape [U]")
        if state.available.dtype != bool or state.active.dtype != bool:
            raise ValueError("available and active must be boolean arrays")
        return state.available & state.active[:, None]


class NoRecommender(Recommender):
    """Baseline: no recommender at all, users choose from the full catalogue
    using only their own beliefs (the paper's 'no recommendation' world)."""
    name = "none"

    def recommend(self, state):
        return None


class RandomRecommender(Recommender):
    """K random available items. Useful as a sanity check / lower bound."""
    name = "random"

    def recommend(self, state):
        scores = self.rng.random(state.available.shape)
        return self.top_k(scores, self.eligible(state))


class PopularityRecommender(Recommender):
    """Rank by simulated read counts plus an optional MIND popularity prior.

    The prior is normalized to total mass `prior_weight`; each observed read
    adds one count, regardless of the utility reported for that read.
    """
    name = "popularity"

    def __init__(self, K: int = 10, prior_weight: float = 1.0):
        super().__init__(K)
        if not np.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        self.prior_weight = prior_weight

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        prior = np.zeros(world.N) if world.popularity is None \
            else np.array(world.popularity, dtype=float, copy=True)
        if prior.shape != (world.N,) or not np.isfinite(prior).all() or (prior < 0).any():
            raise ValueError("popularity must be finite, nonnegative, with shape [N]")
        # Scale before summing to avoid overflow for large valid counts.
        if prior.size and prior.max() > 0:
            prior /= prior.max()
            prior /= prior.sum()
        self.counts = prior * self.prior_weight

    def recommend(self, state):
        return self.top_k(self.counts, self.eligible(state))

    def observe(self, state, users, items, feedback):
        np.add.at(self.counts, items, 1.0)


def _unit_rows(vectors):
    """Normalize finite vectors; retain zero rows as zero vectors."""
    # Scaling first avoids overflow when squaring large finite components.
    scale = np.max(np.abs(vectors), axis=1, keepdims=True)
    scaled = vectors / np.where(scale > 0, scale, 1)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.where(norm > 0, norm, 1)


class ContentBasedRecommender(PopularityRecommender):
    """Cosine relevance to a visible initial profile plus simulated reads.

    `extra['profiles']` is optional [U, D] public profile data. Each profile
    contributes `profile_weight` normalized vectors; every simulated read
    contributes one normalized article vector. A zero centroid uses popularity.
    Feedback values do not weight this positive-interaction content model.
    """
    name = "content"

    def __init__(self, K: int = 10, prior_weight: float = 1.0,
                 profile_weight: float = 1.0):
        super().__init__(K, prior_weight)
        if not np.isfinite(profile_weight) or profile_weight < 0:
            raise ValueError("profile_weight must be finite and nonnegative")
        self.profile_weight = profile_weight

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        if world.profile_vecs is None:
            raise ValueError("content and MMR require world.profile_vecs [N, D]")
        E = np.asarray(world.profile_vecs, dtype=float)
        if E.ndim != 2 or E.shape[0] != world.N or E.shape[1] == 0 \
                or not np.isfinite(E).all():
            raise ValueError("profile_vecs must be finite with shape [N, D], D > 0")
        self.embeddings = _unit_rows(E)

    def relevance(self, state):
        D = self.embeddings.shape[1]
        centroid = np.zeros((self.U, D))
        profiles = state.extra.get("profiles")
        if profiles is not None:
            profiles = np.asarray(profiles, dtype=float)
            if profiles.shape != (self.U, D) or not np.isfinite(profiles).all():
                raise ValueError("extra['profiles'] must be finite with shape [U, D]")
            centroid += self.profile_weight * _unit_rows(profiles)
        if len(state.history) != self.U:
            raise ValueError("history must contain one list per user")
        for u, history in enumerate(state.history):
            if not len(history):
                continue
            idx = np.asarray(history)
            if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) \
                    or (idx < 0).any() or (idx >= self.world.N).any():
                raise ValueError("history must contain catalogue indices in [0, N)")
            centroid[u] += self.embeddings[idx].sum(axis=0)
        has_profile = np.any(centroid != 0, axis=1)
        scores = np.clip(_unit_rows(centroid) @ self.embeddings.T, -1, 1)
        prior = self.counts / self.counts.max() if self.counts.size \
            and self.counts.max() > 0 else self.counts
        scores[~has_profile] = prior
        return scores

    def recommend(self, state):
        available = self.eligible(state)
        return self.top_k(self.relevance(state), available)


class MMRRecommender(ContentBasedRecommender):
    """Greedy relevance minus diversity_weight times maximum slate similarity.

    Unlike the relevance-weight lambda convention, zero here disables diversity.
    The first selection always uses relevance, with catalogue-order tie breaks.
    """
    name = "mmr"

    def __init__(self, K: int = 10, prior_weight: float = 1.0,
                 profile_weight: float = 1.0, diversity_weight: float = 0.3):
        super().__init__(K, prior_weight, profile_weight)
        if not np.isfinite(diversity_weight) or not 0 <= diversity_weight <= 1:
            raise ValueError("diversity_weight must be finite and between 0 and 1")
        self.diversity_weight = diversity_weight

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        self.similarity = np.clip(self.embeddings @ self.embeddings.T, -1, 1)

    def recommend(self, state):
        available = self.eligible(state).copy()
        relevance = self.relevance(state)
        if self.diversity_weight == 0:
            return self.top_k(relevance, available)
        out = np.full((self.U, self.K), -1, dtype=int)
        redundancy = np.zeros_like(relevance)
        for k in range(min(self.K, self.world.N)):
            scores = relevance if k == 0 else relevance - self.diversity_weight * redundancy
            scores = np.where(available, scores, -np.inf)
            users = np.flatnonzero(available.any(axis=1))
            if not users.size:
                break
            items = scores[users].argmax(axis=1)
            out[users, k] = items
            available[users, items] = False
            similarity = self.similarity[items]
            redundancy[users] = similarity if k == 0 \
                else np.maximum(redundancy[users], similarity)
        return out


REGISTRY = {"none": NoRecommender, "random": RandomRecommender,
            "popularity": PopularityRecommender, "content": ContentBasedRecommender,
            "mmr": MMRRecommender}
