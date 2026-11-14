import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.store import MindStore
from src.sim.users import READER_TYPES

GAMMAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
PHIS = [0.0, 0.2, 0.5, 1.0, 1.5]


def auc(labels, scores):

    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def top1(labels, scores):
    best = np.flatnonzero(scores == scores.max())
    return float(labels[best].mean())


def mrr(labels, scores):
    order = np.argsort(-scores, kind="stable")
    hits = np.flatnonzero(labels[order] == 1)
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


class ReaderModel:


    def __init__(self, store, pca_dims, rho, fatigue_rho, sigma_bar, s2, seed=0):
        rng = np.random.default_rng(seed)
        E = store.embeddings.astype(np.float64)
        self.E = E / np.linalg.norm(E, axis=1, keepdims=True)
        pool = np.flatnonzero(store.popularity > 0)
        fit = rng.choice(pool, size=min(5000, len(pool)), replace=False)
        self.mu = self.E[fit].mean(0)
        _, _, Vt = np.linalg.svd(self.E[fit] - self.mu, full_matrices=False)
        self.W = Vt[:pca_dims].T
        self.Z = (self.E - self.mu) @ self.W

        self.ref = rng.choice(pool, size=min(500, len(pool)), replace=False)
        Zr = self.Z[self.ref]
        d = np.sqrt(((Zr[:, None] - Zr[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        self.unit = float(np.median(d.min(1)))
        self.rho, self.frho = rho, fatigue_rho
        self.sigma_bar, self.s2 = sigma_bar, s2

    def _kernel(self, a, b, base):
        d = np.sqrt(((self.Z[a][:, None] - self.Z[b][None]) ** 2).sum(-1))
        if base <= 0:
            return (d == 0).astype(float)
        return np.power(base, d / self.unit)

    def components(self, hist, cand, decays):

        prof = self.E[hist].mean(0)
        prof /= np.linalg.norm(prof)
        ref_sim = self.E[self.ref] @ prof
        sd = ref_sim.std() or 1.0
        mean = self.sigma_bar * (self.E[cand] @ prof - ref_sim.mean()) / sd

        Css = self._kernel(hist, hist, self.rho) + 1e-6 * np.eye(len(hist))
        Ccs = self._kernel(cand, hist, self.rho)
        A = np.linalg.solve(Css, Ccs.T)
        var = np.clip(self.s2 * (1 - (Ccs * A.T).sum(1)), 0, None)

        Kf = self._kernel(cand, hist, self.frho)
        age = np.arange(len(hist))[::-1]
        fat = {dec: Kf @ (dec ** age) for dec in decays}
        return mean, var, fat


def parse_impressions(path, store, max_impressions, max_history, min_history):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            hist = [store.index[n] for n in parts[3].split() if n in store.index]
            if len(hist) < min_history:
                continue
            cand, lab = [], []
            for item in parts[4].split():
                nid, _, click = item.rpartition("-")
                if nid in store.index:
                    cand.append(store.index[nid])
                    lab.append(int(click))
            lab = np.array(lab)
            if len(lab) < 2 or lab.sum() in (0, len(lab)):
                continue
            yield parts[1], hist[-max_history:], np.array(cand), lab
            max_impressions -= 1
            if max_impressions <= 0:
                return


def summarise(rows):
    out = []
    for name, vals in rows.items():
        v = pd.DataFrame(vals, columns=["auc", "top1", "mrr"])
        r = {"scorer": name, "impressions": len(v)}
        for m in ("auc", "top1", "mrr"):
            mean, se = v[m].mean(), v[m].std(ddof=1) / np.sqrt(len(v))
            r[m] = mean
            r[f"{m}_ci_low"], r[f"{m}_ci_high"] = mean - 1.96 * se, mean + 1.96 * se
        out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-impressions", type=int, default=10000)
    ap.add_argument("--max-history", type=int, default=30,
                    help="most recent history clicks treated as already read")
    ap.add_argument("--min-history", type=int, default=5)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base = cfg["news"]["base"]
    sigma = base.get("sigma", 1.0)
    sigma_i = base.get("sigma_i") or sigma
    s2 = sigma_i ** 2 + base.get("beta", 1.0) ** 2 * sigma ** 2
    store = MindStore(args.config)
    model = ReaderModel(store, cfg["news"]["pca_dims"], args.rho,
                        base.get("fatigue_rho", 0.8), base.get("sigma_bar", 1.0), s2)

    decays = sorted({base.get("fatigue_decay", 0.8)} |
                    {t["fatigue_decay"] for t in READER_TYPES.values()})
    dec0 = base.get("fatigue_decay", 0.8)
    grid = {(g, p): [] for g in GAMMAS for p in PHIS}
    grid_test = {(g, p): [] for g in GAMMAS for p in PHIS}
    rows = {"random": [], "popularity": [], "prior similarity (gamma=0, phi=0)": []}
    rows.update({f"type: {t}": [] for t in READER_TYPES})
    rng = np.random.default_rng(42)
    path = Path(cfg["paths"]["raw_dir"]) / "dev" / "behaviors.tsv"

    n = n_fit = 0
    for user, hist, cand, lab in parse_impressions(path, store, args.max_impressions,
                                                    args.max_history, args.min_history):
        mean, var, fat = model.components(hist, cand, decays)
        def score(vals):
            return (auc(lab, vals), top1(lab, vals), mrr(lab, vals))
        n += 1
        if zlib.crc32(user.encode()) % 2 == 0:
            for (g, p) in grid:
                grid[(g, p)].append(score(mean - 0.5 * g * var - p * fat[dec0]))
            n_fit += 1
            continue
        rows["random"].append(score(rng.random(len(lab))))
        rows["popularity"].append(score(store.popularity[cand]))
        rows["prior similarity (gamma=0, phi=0)"].append(score(mean))
        for (g, p) in grid_test:
            grid_test[(g, p)].append(score(mean - 0.5 * g * var - p * fat[dec0]))
        for t, par in READER_TYPES.items():
            s = mean - 0.5 * par["gamma"] * var - par["phi"] * fat[par["fatigue_decay"]]
            rows[f"type: {t}"].append(score(s))
    n_test = n - n_fit
    if n_fit < 2 or n_test < 2:
        sys.exit(f"Not enough usable impressions in {path} (fit {n_fit}, test {n_test})")

    grid_df = summarise({f"{g}|{p}": v for (g, p), v in grid.items()})
    grid_df[["gamma", "phi"]] = grid_df["scorer"].str.split("|", expand=True).astype(float)
    best = grid_df.sort_values("auc", ascending=False).iloc[0]
    g_b, p_b = best["gamma"], best["phi"]
    best_name = f"best fit on fit half (gamma={g_b:g}, phi={p_b:g})"
    rows[best_name] = grid_test[(g_b, p_b)]
    res = summarise(rows)

    gain = (np.array(grid_test[(g_b, p_b)])[:, 0]
            - np.array(rows["prior similarity (gamma=0, phi=0)"])[:, 0])
    gain_se = gain.std(ddof=1) / np.sqrt(len(gain))
    gain_lo, gain_hi = gain.mean() - 1.96 * gain_se, gain.mean() + 1.96 * gain_se

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "reader_choice_validation.csv", index=False)
    grid_df.drop(columns="scorer").to_csv(out / "reader_choice_validation_grid.csv", index=False)

    def fmt(r, m, pct=False):
        f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3f}")
        return f"{f(r[m])} [{f(r[m + '_ci_low'])}, {f(r[m + '_ci_high'])}]"
    table = "\n".join(f"| {r['scorer']} | {fmt(r, 'auc')} | {fmt(r, 'top1', True)} | {fmt(r, 'mrr')} |"
                      for _, r in res.iterrows())
    pivot = grid_df.pivot(index="gamma", columns="phi", values="auc").round(4)
    try:
        pivot_txt = pivot.to_markdown()
    except ImportError:
        pivot_txt = "```\n" + pivot.to_string() + "\n```"
    md = f"""# Validation of the reader choice rule on real MIND dev impressions

**Setup.** {n:,} real impressions from `{path}` (users with at least {args.min_history}
earlier clicks; the {args.max_history} most recent clicks are treated as already read).
Each shown article is scored with the simulator's choice rule,
mean - gamma/2 x variance - phi x fatigue, using the simulator's kernels
(rho = {args.rho}, fatigue_rho = {base.get('fatigue_rho', 0.8)}, {cfg['news']['pca_dims']}-dim PCA,
prior variance {s2:g}). Past reads are assumed to carry no surprise, so they change
variance and fatigue but not the mean. 95% CIs are normal approximations over impressions.
Users are split in two halves by user id: (gamma, phi) is chosen on the fit half
({n_fit:,} impressions) and every row below is scored on the held-out test half
({n_test:,} impressions).

**Held-out test half**

| Scorer | ROC-AUC | Top-1 hit rate | MRR |
|---|---|---|---|
{table}

Gain of the best fit over plain similarity on the test half (paired, same impressions):
AUC {gain.mean():+.4f} [{gain_lo:+.4f}, {gain_hi:+.4f}].

**AUC over the (gamma, phi) grid, fit half** (rows gamma, columns phi; fatigue decay {dec0}):

{pivot_txt}

Best fit on the fit half: gamma = {g_b:g}, phi = {p_b:g} (fit-half AUC {best['auc']:.4f}).
gamma = phi = 0 is the plain similarity ranking used in the earlier check.
"""
    (out / "reader_choice_validation.md").write_text(md)
    print(res[["scorer", "auc", "top1", "mrr"]].round(4).to_string(index=False))
    print(f"\nBest (gamma, phi) on fit half: ({g_b:g}, {p_b:g}); scored above on test half")
    print(f"Held-out AUC gain over prior similarity: {gain.mean():+.4f} "
          f"[{gain_lo:+.4f}, {gain_hi:+.4f}]")


if __name__ == "__main__":
    main()
