import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {
    "cat_entropy": "Category entropy (topic coverage)",
    "homogeneity": "Jaccard homogeneity (echo chamber)",
    "welfare_net": "Net realized utility",
    "fatigue": "Fatigue penalty",
    "diversity": "Intra-list diversity of reads",
    "rounds": "Rounds stayed",
    "dropout": "Dropout rate",
}
HIGHER_BETTER = {"cat_entropy": True, "homogeneity": False, "welfare_net": True,
                 "fatigue": False, "diversity": True, "rounds": True, "dropout": False}

SETTING_COLS = ["rho", "phi", "gamma", "beta", "sigma_bar", "uncertainty_n0",
                "outside_option", "fatigue_decay", "K", "reader_history_visible",
                "reader_types_enabled"]
REFERENCE_ONLY = {"ideal"}


def setting_columns(df):
    return [c for c in SETTING_COLS if c in df.columns and df[c].nunique(dropna=False) > 1]


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return np.quantile(means, 0.025), np.quantile(means, 0.975)


def summary_table(df, keys, metrics):
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["n_populations"] = len(g)
        for m in metrics:
            lo, hi = ci95(g[m])
            r[m], r[f"{m}_ci_low"], r[f"{m}_ci_high"] = g[m].mean(), lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def paired_table(df, settings, metrics, ref):

    rows = []
    groups = df.groupby(settings, dropna=False) if settings else [((), df)]
    for k, g in groups:
        base = g[g["recommender"] == ref].set_index("seed")
        if base.empty:
            continue
        for rec, h in g.groupby("recommender"):
            if rec == ref:
                continue
            h = h.set_index("seed")
            common = h.index.intersection(base.index)
            if len(common) < 2:
                continue
            r = dict(zip(settings, k if isinstance(k, tuple) else (k,)))
            r.update({"recommender": rec, "reference": ref, "n_pairs": len(common)})
            for m in metrics:
                d = (h.loc[common, m] - base.loc[common, m]).to_numpy()
                lo, hi = boot_ci(d)
                r[f"{m}_diff"], r[f"{m}_diff_ci_low"], r[f"{m}_diff_ci_high"] = d.mean(), lo, hi
                r[f"{m}_significant"] = bool(lo > 0 or hi < 0)
            rows.append(r)
    return pd.DataFrame(rows)


def paired_setting(df, settings, metrics, col, treat, base):

    others = [c for c in settings if c != col]
    rows = []
    for k, g in df.groupby(others + ["recommender"], dropna=False):
        a = g[g[col].astype(str) == str(treat)].set_index("seed")
        b = g[g[col].astype(str) == str(base)].set_index("seed")
        common = a.index.intersection(b.index)
        if len(common) < 2:
            continue
        r = dict(zip(others + ["recommender"], k if isinstance(k, tuple) else (k,)))
        r.update({"comparison": f"{col}={treat} minus {col}={base}", "n_pairs": len(common)})
        for m in metrics:
            d = (a.loc[common, m] - b.loc[common, m]).to_numpy()
            lo, hi = boot_ci(d)
            r[f"{m}_diff"], r[f"{m}_diff_ci_low"], r[f"{m}_diff_ci_high"] = d.mean(), lo, hi
            r[f"{m}_significant"] = bool(lo > 0 or hi < 0)
        rows.append(r)
    return pd.DataFrame(rows)


def per_type_table(df, keys):
    cols = [c for c in df.columns if "__" in c and not c.startswith("n_readers")]
    if not cols:
        return pd.DataFrame()
    long = []
    for c in cols:
        metric, rtype = c.split("__", 1)
        part = df[keys + ["seed", c]].rename(columns={c: "value"})
        part["metric"], part["reader_type"] = metric, rtype
        long.append(part)
    long = pd.concat(long, ignore_index=True)
    rows = []
    for k, g in long.groupby(keys + ["reader_type", "metric"], dropna=False):
        r = dict(zip(keys + ["reader_type", "metric"], k))
        lo, hi = ci95(g["value"])
        r.update({"mean": g["value"].mean(), "ci_low": lo, "ci_high": hi, "n_populations": len(g)})
        rows.append(r)
    return pd.DataFrame(rows)


def kendall_tau(a, b):

    n, s = len(a), 0
    for i, j in itertools.combinations(range(n), 2):
        s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2) if n > 1 else np.nan


def rank_robustness(summary, settings, metrics):
    if not settings:
        return pd.DataFrame()
    rows = []
    s = summary[~summary["recommender"].isin(REFERENCE_ONLY)]
    groups = list(s.groupby(settings, dropna=False))
    for m in metrics:
        if s.groupby(settings, dropna=False)[m].nunique().max() <= 1:
            continue
        taus, winners = [], []
        tables = {}
        for k, g in groups:
            tables[k] = g.set_index("recommender")[m]
            best = g.loc[g[m].idxmax() if HIGHER_BETTER[m] else g[m].idxmin(), "recommender"]
            winners.append(best)
        for (k1, a), (k2, b) in itertools.combinations(tables.items(), 2):
            common = a.index.intersection(b.index)
            if len(common) > 2:
                taus.append(kendall_tau(a[common].to_numpy(), b[common].to_numpy()))
        if taus:
            top = pd.Series(winners).value_counts()
            rows.append({"metric": m, "n_settings": len(groups), "mean_kendall_tau": np.mean(taus),
                         "min_kendall_tau": np.min(taus), "most_frequent_best": top.index[0],
                         "best_in_share_of_settings": top.iloc[0] / len(groups)})
    return pd.DataFrame(rows)


def pick_main(summary, settings, main):
    q = summary.copy()
    for col, val in main.items():
        if col in q.columns:
            try:
                q = q[np.isclose(q[col].astype(float), float(val))]
            except (TypeError, ValueError):
                q = q[q[col].astype(str).str.lower() == str(val).lower()]

    for col in settings:
        if col not in main and q[col].nunique() > 1:
            q = q[q[col] == sorted(q[col].dropna().unique())[0]]
    return q


def scatter(ax, q, x, y):


    labels = q.groupby([q[x].round(6), q[y].round(6)])["recommender"].transform(
        lambda s: " = ".join(s))
    markers = itertools.cycle("o^vD<>ph*X")
    seen = set()
    for idx, r in q.iterrows():
        lab = labels[idx]
        if lab in seen:
            continue
        seen.add(lab)
        ref = r["recommender"] in REFERENCE_ONLY
        ax.errorbar(r[x], r[y], xerr=[[r[x] - r[f"{x}_ci_low"]], [r[f"{x}_ci_high"] - r[x]]],
                    yerr=[[r[y] - r[f"{y}_ci_low"]], [r[f"{y}_ci_high"] - r[y]]],
                    fmt="s" if ref else next(markers), mfc="white" if ref else None,
                    capsize=2, ms=7, label=lab + (" (reference)" if ref else ""))
    ax.set(xlabel=METRICS[x], ylabel=METRICS[y])
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.3)


def figures(df, summary, main_q, settings, figdir):
    figdir.mkdir(parents=True, exist_ok=True)
    made = []
    if {"cat_entropy", "welfare_net"} <= set(main_q.columns):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        scatter(ax, main_q, "cat_entropy", "welfare_net")
        ax.set_title("Topic coverage vs net realized utility")
        fig.tight_layout(); fig.savefig(figdir / "coverage_vs_utility.png", dpi=150); plt.close(fig)
        made.append("coverage_vs_utility.png")
    if {"cat_entropy", "homogeneity"} <= set(main_q.columns):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        scatter(ax, main_q, "cat_entropy", "homogeneity")
        ax.set_title("Filter bubble (left) vs echo chamber (up)")
        fig.tight_layout(); fig.savefig(figdir / "bubble_vs_echo_chamber.png", dpi=150); plt.close(fig)
        made.append("bubble_vs_echo_chamber.png")
    mmr = main_q[main_q["recommender"].astype(str).str.match(r"mmr_[\d.]+$")].copy()
    if len(mmr) > 1:
        mmr["lam"] = mmr["recommender"].str[4:].astype(float)
        mmr = mmr.sort_values("lam")
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
        for a, m in zip(ax, ["cat_entropy", "homogeneity", "welfare_net", "fatigue"]):
            a.errorbar(mmr["lam"], mmr[m], yerr=[mmr[m] - mmr[f"{m}_ci_low"], mmr[f"{m}_ci_high"] - mmr[m]],
                       fmt="o-", capsize=3)
            a.set(xlabel="MMR diversity weight lambda", title=METRICS[m]); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(figdir / "mmr_diversity_curve.png", dpi=150); plt.close(fig)
        made.append("mmr_diversity_curve.png")
    if "K" in settings:
        others = [c for c in settings if c != "K"]
        q = summary
        for c in others:
            q = q[q[c] == main_q[c].iloc[0]] if c in main_q.columns and len(main_q) else q
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for a, m in zip(ax, ["cat_entropy", "homogeneity", "welfare_net"]):
            for rec, g in q.groupby("recommender"):
                g = g.sort_values("K")
                a.plot(g["K"], g[m], "o-", label=rec)
            a.set(xscale="log", xlabel="slate size K", title=METRICS[m]); a.grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figdir / "slate_size.png", dpi=150); plt.close(fig)
        made.append("slate_size.png")
    return made


def type_figure(pt, main, figdir):
    if pt.empty:
        return []
    q = pt
    for col, val in main.items():
        if col in q.columns:
            try:
                q = q[np.isclose(q[col].astype(float), float(val))]
            except (TypeError, ValueError):
                pass
    metrics = [m for m in ["cat_entropy", "homogeneity", "welfare_net", "fatigue", "dropout"]
               if m in set(q["metric"]) and q.loc[q["metric"] == m, "mean"].abs().max() > 0]
    if not metrics:
        return []
    fig, ax = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4))
    ax = np.atleast_1d(ax)
    for a, m in zip(ax, metrics):
        piv = q[q["metric"] == m].pivot_table(index="recommender", columns="reader_type", values="mean")
        piv.plot.bar(ax=a, legend=a is ax[0], width=0.8)
        a.set(title=METRICS.get(m, m), xlabel=""); a.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(figdir / "reader_types.png", dpi=150); plt.close(fig)
    return ["reader_types.png"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder (contains original_populations.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--main", nargs="*", default=["rho=0.5", "phi=0.5", "K=10",
                                                    "reader_history_visible=False"],
                    help="setting used for the headline table and figures")
    ap.add_argument("--refs", nargs="*", default=["none", "random"],
                    help="reference recommenders for paired differences")
    args = ap.parse_args()

    df = pd.read_csv(Path(args.run) / "original_populations.csv")
    metrics = [m for m in METRICS if m in df.columns]
    settings = setting_columns(df)
    keys = settings + ["recommender"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = summary_table(df, keys, metrics)
    summary.to_csv(out / "summary.csv", index=False)
    written = ["summary.csv"]
    for ref in args.refs:
        p = paired_table(df, settings, metrics, ref)
        if not p.empty:
            p.to_csv(out / f"paired_vs_{ref}.csv", index=False)
            written.append(f"paired_vs_{ref}.csv")
    if "reader_history_visible" in settings:
        p = paired_setting(df, settings, metrics, "reader_history_visible", True, False)
        if not p.empty:
            p.to_csv(out / "paired_returning_vs_new.csv", index=False)
            written.append("paired_returning_vs_new.csv")
    pt = per_type_table(df, keys)
    if not pt.empty:
        pt.to_csv(out / "per_type.csv", index=False)
        written.append("per_type.csv")
    rr = rank_robustness(summary, settings, metrics)
    if not rr.empty:
        rr.to_csv(out / "rank_robustness.csv", index=False)
        written.append("rank_robustness.csv")

    main_setting = dict(s.split("=", 1) for s in args.main)
    main_q = pick_main(summary, settings, main_setting)
    figs = figures(df, summary, main_q, settings, out / "figures") + \
        type_figure(pt, main_setting, out / "figures")

    cols = ["recommender"] + [c for m in ["cat_entropy", "homogeneity", "welfare_net", "fatigue",
                                          "rounds", "dropout"] if m in main_q for c in [m]]
    shown = main_q[cols].sort_values("recommender").round(3)
    readme = f"""# Analysis of {Path(args.run).name}

Populations: {len(df)}; settings that vary: {', '.join(settings) or 'none'}.
Main setting used below and in figures: {main_setting}.

## Main-setting results (means; see summary.csv for 95% CIs)

{shown.to_string(index=False)}

## Files

""" + "\n".join(f"- {f}" for f in written + [f"figures/{f}" for f in figs]) + """

Paired differences compare each recommender with the reference on the same simulated
readers (matched by seed); a difference is marked significant when its bootstrap
95% CI excludes zero. Recommenders marked reference-only (ideal) use readers' true
preferences and are excluded from rank robustness.
"""
    (out / "README.md").write_text(readme)
    print(readme)


if __name__ == "__main__":
    main()
