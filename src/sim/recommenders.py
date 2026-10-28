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
from collections import Counter
import inspect
from numbers import Real
import re

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

    def get_config(self) -> dict:
        """Return the explicit constructor settings of a registered algorithm."""
        return {name: getattr(self, name) for name in
                inspect.signature(type(self).__init__).parameters if name != "self"}

    def model_state(self) -> dict:
        """Return learned arrays; static catalogue data stays in the world."""
        return {}

    def restore_model_state(self, values: dict):
        if values:
            raise ValueError("This recommender has no learned model arrays")

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

    def model_state(self):
        return {"counts": self.counts.copy()}

    def restore_model_state(self, values):
        if set(values) != {"counts"}:
            raise ValueError("Popularity state requires only counts")
        counts = np.asarray(values["counts"], dtype=float)
        if counts.shape != (self.world.N,) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Invalid learned counts in model state")
        self.counts = counts.copy()


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


class BM25Recommender(PopularityRecommender):
    """Lexical article matching from visible text and simulated reading history.

    The query is the set of unique lowercase alphanumeric tokens in optional
    `extra['query_texts'][u]` and titles/abstracts of simulated reads. BM25 uses
    positive Lucene-style IDF. Empty or out-of-vocabulary queries use popularity.
    No stemming, stopword removal, category expansion, or hidden preferences are
    used. Catalogue text is matched by news_id, not metadata row order.
    """
    name = "bm25"

    def __init__(self, K: int = 10, prior_weight: float = 1.0,
                 k1: float = 1.2, b: float = 0.75):
        super().__init__(K, prior_weight)
        for name, value in (("k1", k1), ("b", b)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
                    or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b must be in [0, 1]")
        self.k1, self.b = k1, b

    @staticmethod
    def tokenize(text):
        return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _text(value):
        if value is None or isinstance(value, (float, np.floating)) and np.isnan(value):
            return ""
        if not isinstance(value, str):
            raise ValueError("Article title and abstract must be strings or missing")
        return value

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        metadata = getattr(world, "article_metadata", None)
        if world.N == 0 and metadata is None:
            metadata = []
        if not isinstance(metadata, (list, tuple)):
            raise ValueError("BM25 requires article_metadata with news_id, title and/or abstract")
        ids = [] if world.N == 0 and world.ids is None else np.asarray(world.ids)
        if np.shape(ids) != (world.N,):
            raise ValueError("BM25 requires world.ids with shape [N]")
        ids = [str(value) for value in ids]
        if len(ids) != len(set(ids)):
            raise ValueError("BM25 catalogue article IDs must be unique")
        by_id = {}
        for record in metadata:
            if not isinstance(record, dict) or "news_id" not in record:
                raise ValueError("Each BM25 article record must contain news_id")
            key = str(record["news_id"])
            if key in by_id:
                raise ValueError("BM25 article_metadata contains duplicate news_id values")
            by_id[key] = record
        if any(key not in by_id for key in ids):
            raise ValueError("BM25 article_metadata is missing catalogue article IDs")
        self.document_tokens = []
        for key in ids:
            record = by_id[key]
            text = self._text(record.get("title")) + " " + self._text(record.get("abstract"))
            self.document_tokens.append(tuple(self.tokenize(text)))
        lengths = np.asarray([len(tokens) for tokens in self.document_tokens], dtype=float)
        if world.N and not lengths.any():
            raise ValueError("BM25 requires at least one nonempty article title or abstract")
        self.average_length = float(lengths.mean()) if world.N else 0.0
        frequencies = [Counter(tokens) for tokens in self.document_tokens]
        df = Counter(term for terms in frequencies for term in terms)
        postings = {}
        for item, terms in enumerate(frequencies):
            norm = self.k1 * (1 - self.b + self.b * lengths[item] / self.average_length)
            for term, tf in terms.items():
                idf = np.log1p((world.N - df[term] + .5) / (df[term] + .5))
                postings.setdefault(term, []).append((item, idf * tf * (self.k1 + 1) / (tf + norm)))
        self.postings = {term: (np.array([item for item, _ in values], dtype=int),
                                np.array([weight for _, weight in values]))
                         for term, values in postings.items()}

    def relevance(self, state):
        if len(state.history) != self.U:
            raise ValueError("history must contain one list per user")
        query_texts = state.extra.get("query_texts")
        if query_texts is None:
            query_texts = [""] * self.U
        if not isinstance(query_texts, (list, tuple, np.ndarray)) \
                or np.ndim(query_texts) != 1 or len(query_texts) != self.U \
                or any(not isinstance(text, str) for text in query_texts):
            raise ValueError("extra['query_texts'] must contain one string per user")
        scores = np.zeros((self.U, self.world.N))
        fallback = self.counts / self.counts.max() if self.counts.size \
            and self.counts.max() > 0 else self.counts
        for u, history in enumerate(state.history):
            query = set(self.tokenize(query_texts[u]))
            if len(history):
                idx = np.asarray(history)
                if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) \
                        or (idx < 0).any() or (idx >= self.world.N).any():
                    raise ValueError("history must contain catalogue indices in [0, N)")
                for item in idx:
                    query.update(self.document_tokens[item])
            matched = False
            # Sorted query terms also make floating-point accumulation repeatable.
            for term in sorted(query):
                if term in self.postings:
                    items, weights = self.postings[term]
                    scores[u, items] += weights
                    matched = True
            if not matched:
                scores[u] = fallback
        return scores

    def recommend(self, state):
        available = self.eligible(state)
        return self.top_k(self.relevance(state), available)


class MatrixFactorizationRecommender(PopularityRecommender):
    """Biased explicit-feedback matrix factorization with regularized ALS.

    Every observed utility, including zero and negative values, is a target.
    Missing entries are never targets. Each feedback batch warm-starts an ALS
    refit over all stored observations. Repeated pairs retain their latest target,
    while popularity counts every read event. Cold users use popularity; items
    without feedback use the fitted global/user baseline instead of latent noise.
    """
    name = "mf"

    def __init__(self, K: int = 10, prior_weight: float = 1.0, factors: int = 8,
                 regularization: float = 1.0, iterations: int = 3):
        super().__init__(K, prior_weight)
        for name, value in (("factors", factors), ("iterations", iterations)):
            if isinstance(value, (bool, np.bool_)) \
                    or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(regularization, (bool, np.bool_)) \
                or not isinstance(regularization, Real) \
                or not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.factors, self.regularization, self.iterations = factors, regularization, iterations

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        self.observed_ratings = np.full((U, world.N), np.nan)
        self.user_factors = rng.normal(0, .1, (U, self.factors))
        self.item_factors = rng.normal(0, .1, (world.N, self.factors))
        self.user_bias = np.zeros(U)
        self.item_bias = np.zeros(world.N)
        self.global_mean = 0.0

    def predictions(self):
        return self.global_mean + self.user_bias[:, None] + self.item_bias[None, :] \
            + self.user_factors @ self.item_factors.T

    def recommend(self, state):
        available = self.eligible(state)
        observed = np.isfinite(self.observed_ratings)
        scores = self.predictions()
        unknown_items = ~observed.any(axis=0)
        scores[:, unknown_items] = self.global_mean + self.user_bias[:, None]
        scores[~observed.any(axis=1)] = self.counts
        return self.top_k(scores, available)

    def observe(self, state, users, items, feedback):
        users, items = np.asarray(users), np.asarray(items)
        try:
            feedback = np.asarray(feedback, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("feedback must contain finite numeric utilities") from error
        if users.ndim != 1 or items.ndim != 1 or feedback.ndim != 1 \
                or users.shape != items.shape or users.shape != feedback.shape:
            raise ValueError("users, items and feedback must be aligned one-dimensional arrays")
        if not np.isfinite(feedback).all():
            raise ValueError("feedback must contain finite numeric utilities")
        if not users.size:
            return
        if not np.issubdtype(users.dtype, np.integer) \
                or not np.issubdtype(items.dtype, np.integer) \
                or (users < 0).any() or (users >= self.U).any() \
                or (items < 0).any() or (items >= self.world.N).any():
            raise ValueError("feedback users/items must be valid integer indices")
        # All input validation precedes changes to either observations or counts.
        for user, item, rating in zip(users, items, feedback):
            self.observed_ratings[user, item] = rating
        np.add.at(self.counts, items, 1.0)
        self._fit()

    def _fit(self):
        observed = np.isfinite(self.observed_ratings)
        values = self.observed_ratings[observed]
        self.global_mean = float(values.mean())
        known_items = observed.any(axis=0)
        # Uniform early feedback can drive every fitted factor to zero. A seeded
        # restart avoids trapping all later ALS updates at that stationary point.
        if not np.any(self.item_factors[known_items]):
            self.item_factors[known_items] = self.rng.normal(0, .1, (known_items.sum(), self.factors))
        ridge = self.regularization * np.eye(self.factors + 1)
        for _ in range(self.iterations):
            for user in np.flatnonzero(observed.any(axis=1)):
                idx = np.flatnonzero(observed[user])
                design = np.column_stack((np.ones(len(idx)), self.item_factors[idx]))
                target = self.observed_ratings[user, idx] - self.global_mean - self.item_bias[idx]
                fitted = np.linalg.solve(design.T @ design + ridge, design.T @ target)
                self.user_bias[user], self.user_factors[user] = fitted[0], fitted[1:]
            for item in np.flatnonzero(known_items):
                idx = np.flatnonzero(observed[:, item])
                design = np.column_stack((np.ones(len(idx)), self.user_factors[idx]))
                target = self.observed_ratings[idx, item] - self.global_mean - self.user_bias[idx]
                fitted = np.linalg.solve(design.T @ design + ridge, design.T @ target)
                self.item_bias[item], self.item_factors[item] = fitted[0], fitted[1:]

    def model_state(self):
        values = super().model_state()
        values.update({name: np.array(getattr(self, name), copy=True) for name in
                       ("observed_ratings", "user_factors", "item_factors",
                        "user_bias", "item_bias", "global_mean")})
        return values

    def restore_model_state(self, values):
        shapes = {"counts": (self.world.N,), "observed_ratings": (self.U, self.world.N),
                  "user_factors": (self.U, self.factors),
                  "item_factors": (self.world.N, self.factors),
                  "user_bias": (self.U,), "item_bias": (self.world.N,), "global_mean": ()}
        if set(values) != set(shapes):
            raise ValueError("MF state must contain every learned model array")
        restored = {}
        for name, shape in shapes.items():
            array = np.asarray(values[name], dtype=float)
            finite = not np.isinf(array).any() if name == "observed_ratings" \
                else np.isfinite(array).all()
            if array.shape != shape or not finite or name == "counts" and (array < 0).any():
                raise ValueError(f"Invalid {name} in MF model state")
            restored[name] = array.copy()
        for name, array in restored.items():
            setattr(self, name, float(array) if name == "global_mean" else array)


REGISTRY = {"none": NoRecommender, "random": RandomRecommender,
            "popularity": PopularityRecommender, "content": ContentBasedRecommender,
            "mmr": MMRRecommender, "bm25": BM25Recommender,
            "mf": MatrixFactorizationRecommender}
