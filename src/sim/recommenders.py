"""Plug-in interface for recommendation systems.

The paper's three regimes ("none", "rec", "oracle") are about how much the
user KNOWS; they are set with Params.info. Real recommendation systems are
different: they decide what the user SEES. Each round a recommender returns a
slate of K items per user, and the user picks the best item in the slate
according to their own beliefs.

To add a recommender, subclass Recommender and implement `recommend`
(and `observe` if it learns from feedback). Ordinary recommenders only use
observable SimState inputs. The ideal reference receives privileged values
through a separate setter, never through the shared state.
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


def _bounded_weight(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
            or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _category_info(world, required=True):
    if world.cats is None:
        if required and world.N:
            raise ValueError("This recommender requires world.cats with shape [N]")
        return np.empty(0, dtype=int), np.full(world.N, -1, dtype=int)
    cats = np.asarray(world.cats)
    if cats.shape != (world.N,) or any(value is None or
            isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in cats):
        raise ValueError("world.cats must contain one nonmissing category per article")
    return np.unique(cats, return_inverse=True)


class IdealRecommender(Recommender):
    """Privileged reference ranking by true gross utility, supplied separately.

    This does not guarantee the highest realized net welfare: readers still
    choose within the returned slate and can experience fatigue.
    """
    name = "ideal"
    reference_only = True

    def __init__(self, K: int = 10, reference_only: bool = True):
        super().__init__(K)
        if reference_only is not True:
            raise ValueError("ideal is a privileged reference_only recommender")
        self.reference_only = reference_only

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        self._reference_values = None

    def set_reference_values(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.U, self.world.N) or not np.isfinite(values).all():
            raise ValueError("Reference values must be finite with shape [U, N]")
        self._reference_values = values.copy()

    def recommend(self, state):
        if self._reference_values is None:
            raise ValueError("ideal requires set_reference_values() after reset")
        return self.top_k(self._reference_values, self.eligible(state))

    def model_state(self):
        return {} if self._reference_values is None \
            else {"reference_values": self._reference_values.copy()}

    def restore_model_state(self, values):
        if not values:
            self._reference_values = None
        elif set(values) == {"reference_values"}:
            self.set_reference_values(values["reference_values"])
        else:
            raise ValueError("ideal state only accepts reference_values")


class CalibratedRecommender(ContentBasedRecommender):
    """Greedy relevance and calibration to an observable category distribution.

    The target combines visible `extra['category_profiles']` raw counts and
    simulated reads. With no counts it uses the catalogue's category mix.
    Candidate score is (1-w)*relevance - w*KL(target || .99*slate_mix+.01*target).
    Within each category, only the most relevant remaining candidate needs to
    be considered because its calibration term is shared by that category.
    """
    name = "calibrated"

    def __init__(self, K: int = 10, prior_weight: float = 1.0,
                 profile_weight: float = 1.0, calibration_weight: float = 0.5):
        super().__init__(K, prior_weight, profile_weight)
        self.calibration_weight = _bounded_weight(calibration_weight, "calibration_weight")

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        self.category_labels, self.article_categories = _category_info(
            world, required=self.calibration_weight > 0)

    def target_distribution(self, state):
        C = len(self.category_labels)
        counts = np.zeros((self.U, C))
        profiles = state.extra.get("category_profiles")
        if profiles is not None:
            profiles = np.asarray(profiles, dtype=float)
            if profiles.shape != (self.U, C) or not np.isfinite(profiles).all() \
                    or (profiles < 0).any():
                raise ValueError("category_profiles must be finite nonnegative counts with shape [U, C]")
            counts += profiles
        if len(state.history) != self.U:
            raise ValueError("history must contain one list per user")
        for user, history in enumerate(state.history):
            if not len(history):
                continue
            idx = np.asarray(history)
            if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) \
                    or (idx < 0).any() or (idx >= self.world.N).any():
                raise ValueError("history must contain catalogue indices in [0, N)")
            if C:
                counts[user] += np.bincount(self.article_categories[idx], minlength=C)
        totals = counts.sum(axis=1, keepdims=True)
        target = counts / np.where(totals > 0, totals, 1)
        if C:
            catalogue = np.bincount(self.article_categories, minlength=C).astype(float)
            target[totals[:, 0] == 0] = catalogue / catalogue.sum()
        return target

    def recommend(self, state):
        if self.calibration_weight == 0:
            return super().recommend(state)
        available = self.eligible(state)
        relevance = self.relevance(state)
        target = self.target_distribution(state)
        out = self.top_k(relevance, available)
        C = len(self.category_labels)
        for user in np.flatnonzero(available.sum(axis=1) > self.K):
            order = np.argsort(-relevance[user], kind="stable") \
                if self.calibration_weight < 1 else np.arange(self.world.N)
            order = order[available[user, order]]
            groups = [order[self.article_categories[order] == category] for category in range(C)]
            pointers = np.zeros(C, dtype=int)
            selected = np.zeros(C)
            positive = target[user] > 0
            for k in range(self.K):
                best_item, best_category, best_score = None, None, -np.inf
                for category, candidates in enumerate(groups):
                    if pointers[category] == len(candidates):
                        continue
                    item = int(candidates[pointers[category]])
                    mix = selected.copy()
                    mix[category] += 1
                    mix /= k + 1
                    smoothed = .99 * mix + .01 * target[user]
                    kl = np.sum(target[user, positive] *
                                np.log(target[user, positive] / smoothed[positive]))
                    score = (1 - self.calibration_weight) * relevance[user, item] \
                        - self.calibration_weight * kl
                    if score > best_score or score == best_score \
                            and (best_item is None or item < best_item):
                        best_item, best_category, best_score = item, category, score
                if best_item is None:
                    break
                out[user, k] = best_item
                selected[best_category] += 1
                pointers[best_category] += 1
        return out


class _ObservableFatigueSignals:
    """Shared validation and storage for opted-in observable feedback only."""
    accepts_feedback_signals = True

    def _reset_signals(self):
        self.departed_users = np.zeros(self.U, dtype=bool)

    def _feedback_inputs(self, users, items, feedback, signals):
        users, items = np.asarray(users), np.asarray(items)
        try:
            if np.iscomplexobj(feedback):
                raise ValueError("Feedback utilities must be real")
            feedback = np.asarray(feedback, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("feedback must contain finite real utilities") from error
        if users.ndim != 1 or items.ndim != 1 or feedback.ndim != 1 \
                or users.shape != items.shape or users.shape != feedback.shape \
                or not np.isfinite(feedback).all():
            raise ValueError("Feedback arrays must be aligned, one-dimensional and finite")
        if users.size and (not np.issubdtype(users.dtype, np.integer)
                          or not np.issubdtype(items.dtype, np.integer)
                          or (users < 0).any() or (users >= self.U).any()
                          or (items < 0).any() or (items >= self.world.N).any()):
            raise ValueError("Feedback users/items must be valid integer indices")
        if signals is None:
            signals = {}
        if not isinstance(signals, dict) or set(signals) - {"boredom", "left_users"}:
            raise ValueError("signals only accepts boredom and left_users")
        boredom = np.asarray(signals.get("boredom", np.zeros(users.size)), dtype=float)
        if boredom.shape != users.shape or not np.isfinite(boredom).all() \
                or not np.isin(boredom, [0, 1]).all():
            raise ValueError("boredom must be aligned binary observations")
        left = np.asarray(signals.get("left_users", []))
        if left.ndim != 1 or left.size and (not np.issubdtype(left.dtype, np.integer)
                or (left < 0).any() or (left >= self.U).any()):
            raise ValueError("left_users must contain valid user indices")
        return users.astype(int), items.astype(int), feedback, boredom, left.astype(int)

    def model_state(self):
        values = super().model_state()
        values["departed_users"] = self.departed_users.copy()
        return values

    def restore_model_state(self, values):
        values = dict(values)
        departed = np.asarray(values.pop("departed_users", None))
        if departed.shape != (self.U,) or departed.dtype != bool:
            raise ValueError("departed_users must be a boolean array with shape [U]")
        super().restore_model_state(values)
        self.departed_users = departed.copy()


class FatiguePopularityRecommender(_ObservableFatigueSignals, PopularityRecommender):
    """Downweight read counts when a consumed article produces reported boredom."""
    name = "fatigue_popularity"

    def __init__(self, K: int = 10, prior_weight: float = 1.0, fatigue_weight: float = 1.0):
        super().__init__(K, prior_weight)
        self.fatigue_weight = _bounded_weight(fatigue_weight, "fatigue_weight")

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        self._reset_signals()

    def observe(self, state, users, items, feedback, signals=None):
        users, items, feedback, boredom, left = self._feedback_inputs(users, items, feedback, signals)
        np.add.at(self.counts, items, 1 - self.fatigue_weight * boredom)
        self.departed_users[left] = True


class FatigueMFRecommender(_ObservableFatigueSignals, MatrixFactorizationRecommender):
    """Fit net utility minus a coarse reported-boredom penalty using observed reads."""
    name = "fatigue_mf"

    def __init__(self, K: int = 10, prior_weight: float = 1.0, factors: int = 8,
                 regularization: float = 1.0, iterations: int = 3,
                 fatigue_weight: float = 1.0):
        super().__init__(K, prior_weight, factors, regularization, iterations)
        self.fatigue_weight = _bounded_weight(fatigue_weight, "fatigue_weight")

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        self._reset_signals()

    def observe(self, state, users, items, feedback, signals=None):
        users, items, feedback, boredom, left = self._feedback_inputs(users, items, feedback, signals)
        adjusted = feedback - self.fatigue_weight * boredom
        for user, item, rating in zip(users, items, adjusted):
            self.observed_ratings[user, item] = rating
        np.add.at(self.counts, items, 1 - self.fatigue_weight * boredom)
        if users.size:
            self._fit()
        self.departed_users[left] = True


class FatigueContentRecommender(_ObservableFatigueSignals, ContentBasedRecommender):
    """Penalize recently consumed categories using only observable reading/boredom.

    Per-category recent counts decay once per elapsed round. A read adds
    1+boredom to its category. Relevance subtracts fatigue_weight times recent
    category load / (1 + total load), so empty-round decay also reduces penalties.
    The ordinary content profile/history calculation is retained.
    """
    name = "fatigue_content"

    def __init__(self, K: int = 10, prior_weight: float = 1.0,
                 profile_weight: float = 1.0, fatigue_weight: float = 1.0,
                 decay: float = 0.8):
        super().__init__(K, prior_weight, profile_weight)
        self.fatigue_weight = _bounded_weight(fatigue_weight, "fatigue_weight")
        self.decay = _bounded_weight(decay, "decay")

    def reset(self, world, U, rng):
        super().reset(world, U, rng)
        self._reset_signals()
        self.category_labels, self.article_categories = _category_info(
            world, required=self.fatigue_weight > 0)
        self.recent_category_counts = np.zeros((U, len(self.category_labels)))
        self.last_round = -1

    def _advance(self, t):
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) \
                or t < 0 or t < self.last_round:
            raise ValueError("state.t must be a nonnegative nondecreasing integer")
        if t > self.last_round:
            self.recent_category_counts *= self.decay ** (t - self.last_round)
            self.last_round = int(t)

    def relevance(self, state):
        base = super().relevance(state)
        self._advance(state.t)
        if self.fatigue_weight == 0 or not len(self.category_labels):
            return base
        normalized = self.recent_category_counts / (1 + self.recent_category_counts.sum(axis=1, keepdims=True))
        return base - self.fatigue_weight * normalized[:, self.article_categories]

    def observe(self, state, users, items, feedback, signals=None):
        users, items, feedback, boredom, left = self._feedback_inputs(users, items, feedback, signals)
        self._advance(state.t)
        np.add.at(self.counts, items, 1.0)
        if len(self.category_labels):
            np.add.at(self.recent_category_counts, (users, self.article_categories[items]), 1 + boredom)
        self.departed_users[left] = True

    def model_state(self):
        values = super().model_state()
        values.update(recent_category_counts=self.recent_category_counts.copy(),
                      last_round=np.asarray(self.last_round))
        return values

    def restore_model_state(self, values):
        values = dict(values)
        recent = np.asarray(values.pop("recent_category_counts", None), dtype=float)
        last = np.asarray(values.pop("last_round", None))
        if recent.shape != (self.U, len(self.category_labels)) or not np.isfinite(recent).all() \
                or (recent < 0).any() or last.shape != () \
                or not np.issubdtype(last.dtype, np.integer) or last < -1:
            raise ValueError("Invalid recent category counts or last_round in fatigue content state")
        super().restore_model_state(values)
        self.recent_category_counts = recent.copy()
        self.last_round = int(last)


REGISTRY = {"none": NoRecommender, "random": RandomRecommender,
            "popularity": PopularityRecommender, "content": ContentBasedRecommender,
            "mmr": MMRRecommender, "bm25": BM25Recommender,
            "mf": MatrixFactorizationRecommender, "ideal": IdealRecommender,
            "calibrated": CalibratedRecommender,
            "fatigue_popularity": FatiguePopularityRecommender,
            "fatigue_mf": FatigueMFRecommender, "fatigue_content": FatigueContentRecommender}
