import argparse
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.experiment import run_grid, summarise
from src.sim.world import circle_world

METRICS = ["local_distance", "diversity", "welfare", "homogeneity"]
LABEL = {"none": "No recommendation", "rec": "Recommendation", "oracle": "Oracle"}
STYLE = {"none": dict(color="#C0504D", marker="o"),
         "rec": dict(color="#4F81BD", marker="s"),
         "oracle": dict(color="#7F7F7F", marker="^", linestyle="--")}


def checks(s: pd.DataFrame) -> list[str]:
    out = []

    def piv(metric, cols, **fix):
        q = s
        for k, v in fix.items():
            q = q[q[k] == v]
        return q.pivot_table(index=cols, columns="info", values=metric)

    g_hi = s["gamma"].max()
    b_mid = sorted(s["beta"].unique())[len(s["beta"].unique()) // 2]

    t = piv("local_distance", "rho", gamma=g_hi, beta=b_mid)
    pos = t.index > 0
    ok = (t.loc[pos, "none"] < t.loc[pos, "rec"]).all() and \
         (t.loc[pos, "none"] < t.loc[pos, "oracle"]).all()
    out.append(("1. Without recommendation, consumption is more local (rho>0)", ok))

    gap = (t["rec"] - t["none"]).loc[pos]
    out.append(("2. That gap grows with rho", gap.iloc[-1] > gap.iloc[0]))

    t0 = piv("local_distance", "gamma", rho=0.0, beta=b_mid)
    rel = ((t0["rec"] - t0["none"]).abs() / t0["none"]).max()
    out.append((f"3. At rho=0 regimes barely differ in locality (max rel diff {rel:.2f})",
                rel < 0.15))

    tg = piv("local_distance", "gamma", rho=s["rho"].max(), beta=b_mid)
    gapg = tg["rec"] - tg["none"]
    out.append(("4. Locality gap grows with risk aversion", gapg.iloc[-1] > gapg.iloc[0]))

    td = piv("diversity", "rho", gamma=g_hi, beta=b_mid)
    strong = td.index >= 0.3
    out.append(("5. Diversity lowest without recommendation (rho>=0.3)",
                (td.loc[strong, "none"] < td.loc[strong, "rec"]).all()))

    th = piv("homogeneity", "beta", rho=0.5 if 0.5 in s["rho"].values else s["rho"].max(),
             gamma=g_hi)
    out.append(("6. Homogeneity highest under recommendation",
                (th["rec"] >= th["none"]).all()))
    out.append(("7. Homogeneity under recommendation rises with beta",
                th["rec"].iloc[-1] > th["rec"].iloc[0]))
    tw = piv("welfare", "rho", gamma=s["gamma"].min(), beta=b_mid)
    out.append(("8. Welfare without recommendation rises with rho (gamma=0)",
                tw["none"].iloc[-1] > tw["none"].iloc[0]))
    gain = tw["rec"] - tw["none"]
    out.append(("9. Welfare gain from recommendation shrinks as rho rises (gamma=0)",
                gain.iloc[-1] < gain.iloc[0]))
    return [f"{'PASS' if ok else 'FAIL'}  {msg}" for msg, ok in out]


def plot(s: pd.DataFrame, path: Path):
    g_hi = s["gamma"].max()
    betas = sorted(s["beta"].unique())
    b_mid = betas[len(betas) // 2]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    panels = [("local_distance", "rho", dict(gamma=g_hi, beta=b_mid),
               "Local consumption (consecutive distance)"),
              ("diversity", "rho", dict(gamma=g_hi, beta=b_mid), "Diversity"),
              ("welfare", "rho", dict(gamma=s["gamma"].min(), beta=b_mid), "Welfare"),
              ("homogeneity", "beta", dict(gamma=g_hi, rho=0.5 if 0.5 in s["rho"].values
                                           else s["rho"].max()),
               "Homogeneity (Jaccard between users)")]
    for a, (m, x, fix, title) in zip(ax.flat, panels):
        q = s
        for k, v in fix.items():
            q = q[q[k] == v]
        for info, grp in q.groupby("info"):
            grp = grp.sort_values(x)
            a.errorbar(grp[x], grp[m], yerr=grp[m + "_ci95"], label=LABEL[info],
                       capsize=3, **STYLE[info])
        a.set_title(title + "\n" + ", ".join(f"{k}={v}" for k, v in fix.items()),
                    fontsize=10)
        a.set_xlabel(x)
        a.grid(alpha=0.3)
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-pop", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/aridor_repro")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))["aridor"]
    grid = cfg["full_grid"] if args.full else cfg["reduced_grid"]
    grid = {**grid, "info": ["none", "rec", "oracle"]}
    n_pop = args.n_pop or cfg["n_pop"]
    base = {"T": cfg["T"], "sigma_bar": cfg["sigma_bar"]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = n_pop * 3
    for v in grid.values():
        n_jobs *= len(v) if v is not grid["info"] else 1
    print(f"Running {n_jobs} populations of {cfg['U']} users...")

    df, _ = run_grid(partial(circle_world, cfg["N"]), grid, base=base,
                     n_pop=n_pop, U=cfg["U"], seed=cfg["seed"],
                     workers=args.workers)
    df.to_csv(out / "populations.csv", index=False)
    by = [k for k in grid if k != "info"] + ["info"]
    s = summarise(df, by, METRICS)
    s.to_csv(out / "summary.csv", index=False)
    plot(s, out / "figures.png")
    res = checks(s)
    (out / "checks.txt").write_text("\n".join(res) + "\n")
    print("\n".join(res))
    print("Saved to", out)


if __name__ == "__main__":
    main()
