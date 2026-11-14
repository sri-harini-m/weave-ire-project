from dataclasses import dataclass

import numpy as np


@dataclass
class World:
    N: int
    D: np.ndarray
    C: np.ndarray
    rho: float
    unit: float = 1.0
    cats: np.ndarray | None = None
    ids: np.ndarray | None = None
    popularity: np.ndarray | None = None
    profile_vecs: np.ndarray | None = None
    article_metadata: list[dict] | None = None
    kind: str = "circle"

    def __post_init__(self):


        w, Q = np.linalg.eigh(self.C)
        self.L = Q * np.sqrt(np.clip(w, 0, None))


def _corr_from_distance(dist: np.ndarray, unit: float, rho: float) -> np.ndarray:


    if rho <= 0:
        return np.eye(len(dist))
    C = np.power(rho, dist / unit)
    np.fill_diagonal(C, 1.0)
    return C


def circle_world(N: int = 200, rho: float = 0.5) -> World:


    idx = np.arange(N)
    diff = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(diff, N - diff).astype(float)
    C = _corr_from_distance(d, unit=1.0, rho=rho)

    return World(N=N, D=d / N, C=C, rho=rho, unit=1.0 / N, ids=idx,
                 kind="circle")


def news_world(store, N: int = 500, rho: float = 0.5, pca_dims: int = 8,
               only_clicked: bool = True, seed: int = 0) -> World:


    rng = np.random.default_rng(seed)
    pool = np.arange(len(store.articles))
    if only_clicked:
        pool = pool[store.popularity > 0]
    rows = np.sort(rng.choice(pool, size=min(N, len(pool)), replace=False))
    N = len(rows)

    E = store.embeddings[rows].astype(np.float64)
    Ec = E - E.mean(axis=0)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    Z = Ec @ Vt[:pca_dims].T

    sq = (Z ** 2).sum(1)
    dist = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))
    np.fill_diagonal(dist, 0.0)
    nn = np.where(np.eye(N, dtype=bool), np.inf, dist).min(axis=1)
    unit = float(np.median(nn))
    C = _corr_from_distance(dist, unit=unit, rho=rho)

    art = store.articles.iloc[rows]
    text_cols = [c for c in ("news_id", "title", "abstract") if c in art.columns]
    metadata = art[text_cols].to_dict(orient="records")
    return World(N=N, D=dist / dist.max(), C=C, rho=rho, unit=unit / dist.max(),
                 cats=art["category"].to_numpy(), ids=art["news_id"].to_numpy(),
                 popularity=store.popularity[rows], profile_vecs=E,
                 article_metadata=metadata, kind="news")


def describe(world: World) -> dict:

    off = world.C[~np.eye(world.N, dtype=bool)]
    out = {"N": world.N, "rho": world.rho,
           "corr_mean": float(off.mean()),
           "corr_p50": float(np.median(off)),
           "corr_p99": float(np.quantile(off, 0.99)),
           "min_eig": float(np.linalg.eigvalsh(world.C).min())}
    if world.cats is not None:
        out["n_categories"] = int(len(set(world.cats)))
    return out
