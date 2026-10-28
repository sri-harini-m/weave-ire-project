import argparse
import json
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.experiment import run_grid, summarise
from src.sim.news_users import attach_user_pool, sample_news_users
from src.sim.recommenders import REGISTRY
from src.sim.recommender_config import recommender_options
from src.sim.run_bundle import RunBundle
from src.sim.world import describe, news_world

METRICS = ["local_distance", "diversity", "cat_entropy", "n_categories",
           "welfare", "welfare_net", "fatigue", "rounds", "homogeneity"]


def build_world(store, cfg, rho):
    w = news_world(store, N=cfg["N"], rho=rho, pca_dims=cfg["pca_dims"],
                   only_clicked=cfg["only_clicked"], seed=cfg["seed"])
    w.article_metadata = store.articles.set_index("news_id", drop=False).loc[w.ids].to_dict("records")
    return attach_user_pool(w, store, seed=cfg["seed"])


def plot(s, curves, path, group):
    rhos = sorted(s["rho"].unique())
    rho = 0.5 if 0.5 in rhos else rhos[len(rhos) // 2]
    q = s[s["rho"] == rho]
    panels = [("cat_entropy", "Category entropy (filter bubble: lower = narrower)"),
              ("homogeneity", "Homogeneity (echo chamber: higher = more alike)"),
              ("welfare_net", "Welfare after fatigue"),
              ("fatigue", "Average fatigue penalty")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, (m, title) in zip([ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]], panels):
        for key, grp in q.groupby(group):
            grp = grp.sort_values("phi")
            a.errorbar(grp["phi"], grp[m], yerr=grp[m + "_ci95"], marker="o",
                       capsize=3, label=str(key))
        a.set_title(f"{title}\nrho={rho}", fontsize=10)
        a.set_xlabel("fatigue strength phi")
        a.grid(alpha=0.3)
    ax[0, 0].legend(title=group)
    if curves is not None:
        c = curves[curves["rho"] == rho]
        phi_hi = c["phi"].max()
        c = c[c["phi"] == phi_hi]
        for key, grp in c.groupby(group):
            g = grp.groupby("t")[["step_distance", "fatigue"]].mean()
            ax[0, 2].plot(g.index, g["step_distance"], marker=".", label=str(key))
            ax[1, 2].plot(g.index, g["fatigue"], marker=".", label=str(key))
        ax[0, 2].legend(title=group)
        ax[0, 2].set_title(f"Distance between consecutive reads\nrho={rho}, phi={phi_hi}",
                           fontsize=10)
        ax[1, 2].set_title(f"Fatigue over time\nrho={rho}, phi={phi_hi}", fontsize=10)
        for a in ax[:, 2]:
            a.set_xlabel("round"); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-pop", type=int, default=None)
    ap.add_argument("--recommenders", nargs="*", default=None, choices=list(REGISTRY),
                    help=f"slate recommenders to compare: {list(REGISTRY)}")
    weights = ap.add_mutually_exclusive_group()
    weights.add_argument("--mmr-diversity-weight", type=float, default=None,
                    help="MMR redundancy penalty in [0, 1]; zero equals content-based")
    weights.add_argument("--mmr-diversity-weights", type=float, nargs="+", default=None,
                        help="Save a comparison of several MMR penalty settings")
    ap.add_argument("--out", default="results/news")
    ap.add_argument("--no-archive", action="store_true",
                    help="skip the per-round population archives (saves a lot of disk); "
                         "summary CSVs and figures are still written")
    args = ap.parse_args()

    full = yaml.safe_load(Path(args.config).read_text())
    with RunBundle(args.out, full, vars(args), "mind") as bundle:
        print("Saving complete MIND run to:", bundle.path)
        execute_news(args, full, bundle.path)
    print("Saved run:", bundle.path)


def execute_news(args, full, out):

    from src.data.store import MindStore

    cfg = full["news"]
    store = MindStore(args.config)

    grid = dict(cfg["grid"])
    recs, group = None, "info"
    if args.recommenders:
        settings = full.get("recommenders", {})
        recs = {}
        for r in args.recommenders:
            options = recommender_options(r, settings)
            if r == "mmr":
                options["diversity_weight"] = args.mmr_diversity_weight \
                    if args.mmr_diversity_weight is not None \
                    else options.get("diversity_weight", 0.3)
                if args.mmr_diversity_weights is not None:
                    for weight in args.mmr_diversity_weights:
                        recs[f"mmr_{weight:g}"] = partial(REGISTRY[r], **{**options, "diversity_weight": weight})
                    continue
            recs[r] = partial(REGISTRY[r], **options)
        grid["info"] = ["none"]
        group = "recommender"

    world_fn = partial(build_world, store, cfg)
    w0 = world_fn(grid["rho"][len(grid["rho"]) // 2])
    diag = describe(w0)
    (out / "world.json").write_text(json.dumps(diag, indent=2))
    print("World:", diag)

    df, curves = run_grid(world_fn, grid, base={**cfg["base"], "T": cfg["T"]},
                          n_pop=args.n_pop or cfg["n_pop"], U=cfg["U"],
                          seed=cfg["seed"], recommenders=recs,
                          users_fn=partial(sample_news_users, U=cfg["U"]),
                          workers=args.workers, keep_curves=True,
                          archive_dir=None if getattr(args, "no_archive", False) else out)
    df.to_csv(out / "original_populations.csv", index=False)
    curves.to_csv(out / "original_curves.csv", index=False)
    by = [k for k in grid if k != "info"] + ["info", "recommender"]
    s = summarise(df, by, METRICS)
    s.to_csv(out / "original_summary.csv", index=False)
    plot(s, curves, out / "original_figures.png", group)
    if not getattr(args, "no_archive", False):
        from src.sim.reporting import rebuild_report
        rebuild_report(out)
    print(s.groupby(group)[["cat_entropy", "homogeneity", "welfare_net", "fatigue"]]
          .mean().round(3))
    print("Saved to", out)


if __name__ == "__main__":
    main()
