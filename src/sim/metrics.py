import numpy as np
import pandas as pd

from .world import World


def user_metrics(world: World, run: dict) -> pd.DataFrame:
    seq, val, pen, D = run["seq"], run["val"], run["pen"], world.D
    rows = []
    n_cats = len(set(world.cats)) if world.cats is not None else None
    for i in range(seq.shape[0]):
        items = seq[i][seq[i] >= 0]
        m = len(items)
        r = {"user": i, "rounds": int(run["left_at"][i]), "n_consumed": m,
             "dropout": int(run["left_at"][i] < seq.shape[1])}
        if "reader_types" in run and run["reader_types"] is not None:
            r["reader_type"] = str(run["reader_types"][i])
        if "gammas" in run and run["gammas"] is not None:
            r["reader_gamma"] = float(run["gammas"][i])
        if "phis" in run and run["phis"] is not None:
            r["reader_phi"] = float(run["phis"][i])
        if "decays" in run and run["decays"] is not None:
            r["reader_fatigue_decay"] = float(run["decays"][i])
        if "history_lens" in run and run["history_lens"] is not None:
            r["history_len"] = int(run["history_lens"][i])
        if m >= 2:
            r["local_distance"] = float(D[items[:-1], items[1:]].mean())
            sub = D[np.ix_(items, items)]
            r["diversity"] = float(sub.sum() / (m * (m - 1)))
        else:
            r["local_distance"] = r["diversity"] = np.nan
        v, p = val[i][seq[i] >= 0], pen[i][seq[i] >= 0]
        r["welfare"] = float(v.mean()) if m else np.nan
        r["welfare_net"] = float((v - p).mean()) if m else np.nan
        r["fatigue"] = float(p.mean()) if m else np.nan
        if world.cats is not None and m:
            _, counts = np.unique(world.cats[items], return_counts=True)
            q = counts / counts.sum()
            h = -(q * np.log(q)).sum()
            r["cat_entropy"] = float(h / np.log(n_cats)) if n_cats > 1 else 0.0
            r["n_categories"] = int(len(counts))
        rows.append(r)
    return pd.DataFrame(rows)


PER_TYPE_METRICS = ("cat_entropy", "diversity", "local_distance", "welfare",
                    "welfare_net", "fatigue", "rounds", "dropout")


def per_type_metrics(um: pd.DataFrame, run: dict) -> dict:


    if "reader_type" not in um.columns or um["reader_type"].nunique() < 2:
        return {}
    out = {}
    for t, grp in um.groupby("reader_type"):
        out[f"n_readers__{t}"] = int(len(grp))
        for m in PER_TYPE_METRICS:
            if m in grp.columns:
                out[f"{m}__{t}"] = float(grp[m].mean())
        idx = grp["user"].to_numpy()
        if len(idx) >= 2:
            out[f"homogeneity__{t}"] = homogeneity({"seq": run["seq"][idx]})
    return out


def homogeneity(run: dict) -> float:

    seq = run["seq"]
    U = seq.shape[0]
    N = max(int(seq.max()) + 1, 1)
    M = np.zeros((U, N), bool)
    r, c = np.nonzero(seq >= 0)
    M[r, seq[r, c]] = True
    M = M.astype(float)
    inter = M @ M.T
    size = M.sum(1)
    union = size[:, None] + size[None, :] - inter
    J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    off = ~np.eye(U, dtype=bool)
    return float(J[off].mean())


def time_curves(world: World, run: dict) -> pd.DataFrame:

    seq, pen = run["seq"], run["pen"]
    T = seq.shape[1]
    out = []
    for t in range(1, T):
        ok = (seq[:, t] >= 0) & (seq[:, t - 1] >= 0)
        d = world.D[seq[ok, t - 1], seq[ok, t]].mean() if ok.any() else np.nan
        out.append({"t": t, "step_distance": float(d),
                    "fatigue": float(np.nanmean(pen[:, t])) if (seq[:, t] >= 0).any() else np.nan,
                    "active_share": float((seq[:, t] >= 0).mean())})
    return pd.DataFrame(out)
