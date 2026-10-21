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
        s = np.where(available, scores, -np.inf)
        k = min(self.K, s.shape[1])
        idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, idx, 1), axis=1)
        idx = np.take_along_axis(idx, order, 1)
        return np.where(np.isfinite(np.take_along_axis(s, idx, 1)), idx, -1)


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
        return self.top_k(scores, state.available)


REGISTRY = {"none": NoRecommender, "random": RandomRecommender}
