"""The item space ("world") the simulated users live in.

A World holds everything about the items that does not depend on the user:
    N      number of items
    D      [N, N] normalised distance between items, in [0, 1]
    C      [N, N] correlation between item values (the "spillover" matrix)
    cats   [N] category label per item (None for the circle world)
    ids    [N] item ids (MIND news_ids, or 0..N-1 for the circle)

Two constructors:
    circle_world(N, rho)          exactly the Aridor et al. setup
    news_world(store, N, rho)     the same idea on real MIND articles

In both, items that are close together have correlated values, controlled by
rho in [0, 1): two *neighbouring* items have correlation rho, and correlation
decays exponentially with distance. rho = 0 means no spillover at all.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    N: int
    D: np.ndarray
    C: np.ndarray
    rho: float
    unit: float = 1.0                       # normalised distance of a "neighbour"
    cats: np.ndarray | None = None
    ids: np.ndarray | None = None
    popularity: np.ndarray | None = None   # [N] in [0, 1], news only
    profile_vecs: np.ndarray | None = None  # [N, dim] embeddings, news only
    kind: str = "circle"

    def __post_init__(self):
        # Matrix square root used to draw correlated values: V = L @ z.
        # eigh + clipping is more robust than Cholesky when C is near-singular.
        w, Q = np.linalg.eigh(self.C)
        self.L = Q * np.sqrt(np.clip(w, 0, None))


def _corr_from_distance(dist: np.ndarray, unit: float, rho: float) -> np.ndarray:
    """corr = rho ** (dist / unit). Exponential (Laplacian) kernel: always a
    valid covariance for Euclidean / circular distances."""
    if rho <= 0:
        return np.eye(len(dist))
    C = np.power(rho, dist / unit)
    np.fill_diagonal(C, 1.0)
    return C


def circle_world(N: int = 200, rho: float = 0.5) -> World:
    """Aridor et al.: items 0..N-1 on a circle, d(n,m) = min(|n-m|, N-|n-m|),
    Sigma_nm proportional to rho ** d(n,m)."""
    idx = np.arange(N)
    diff = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(diff, N - diff).astype(float)
    C = _corr_from_distance(d, unit=1.0, rho=rho)
    # Paper's diversity metric divides by N, so normalised distance = d / N.
    return World(N=N, D=d / N, C=C, rho=rho, unit=1.0 / N, ids=idx,
                 kind="circle")


def news_world(store, N: int = 500, rho: float = 0.5, pca_dims: int = 8,
               only_clicked: bool = True, seed: int = 0) -> World:
    """MIND version. Items are N sampled real articles.

    Distance: Euclidean distance between article embeddings after PCA to
    `pca_dims` dimensions. (Full 384-dim embeddings suffer from distance
    concentration: almost every pair is equally far apart, so there would be no
    notion of a "neighbourhood". A low-dimensional projection restores it.)

    Correlation: rho ** (dist / unit), where `unit` = median nearest-neighbour
    distance. This mirrors the circle: an article and its closest neighbour have
    correlation about rho, just like adjacent items on the circle.
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(len(store.articles))
    if only_clicked:  # articles that actually received clicks during the period
        pool = pool[store.popularity > 0]
    rows = np.sort(rng.choice(pool, size=min(N, len(pool)), replace=False))
    N = len(rows)

    E = store.embeddings[rows].astype(np.float64)
    Ec = E - E.mean(axis=0)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    Z = Ec @ Vt[:pca_dims].T                      # [N, pca_dims]

    sq = (Z ** 2).sum(1)
    dist = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))
    np.fill_diagonal(dist, 0.0)
    nn = np.where(np.eye(N, dtype=bool), np.inf, dist).min(axis=1)
    unit = float(np.median(nn))
    C = _corr_from_distance(dist, unit=unit, rho=rho)

    art = store.articles.iloc[rows]
    return World(N=N, D=dist / dist.max(), C=C, rho=rho, unit=unit / dist.max(),
                 cats=art["category"].to_numpy(), ids=art["news_id"].to_numpy(),
                 popularity=store.popularity[rows], profile_vecs=E, kind="news")


def describe(world: World) -> dict:
    """Quick diagnostics: is there a meaningful neighbourhood structure?"""
    off = world.C[~np.eye(world.N, dtype=bool)]
    out = {"N": world.N, "rho": world.rho,
           "corr_mean": float(off.mean()),
           "corr_p50": float(np.median(off)),
           "corr_p99": float(np.quantile(off, 0.99)),
           "min_eig": float(np.linalg.eigvalsh(world.C).min())}
    if world.cats is not None:
        out["n_categories"] = int(len(set(world.cats)))
    return out
