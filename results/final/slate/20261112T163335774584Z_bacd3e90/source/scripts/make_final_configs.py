"""Write the configs for the final experiments, derived from config.yaml.

Each experiment changes only the `news` grid / base settings, so every run
shares the same data paths, world and recommender settings.

    python scripts/make_final_configs.py              # writes configs/final/*.yaml
    python scripts/make_final_configs.py --n-pop 2    # quick smoke-test versions

Experiments (see docs/final_experiments.md):
    main        identical readers, fatigue phi in {0, 0.5, 1}          -> RQ1, RQ3
    types       four reader types, history-based uncertainty (n0=20)  -> who is affected
    dropout     readers may leave: outside option x fatigue           -> session length
    n0          history-length uncertainty n0 in {5, 20, 80}          -> sensitivity
    rho         spillover rho in {0, 0.5, 0.9}                        -> sensitivity
    robustness  gamma x beta x sigma_bar                              -> ranking robustness
    pca4/pca16  embedding dimensions for article similarity           -> sensitivity
Slate size and returning users use the main config with command-line flags.
"""
import argparse
import copy
from pathlib import Path

import yaml

EXPERIMENTS = {
    "main": {"grid": {"rho": [0.5], "phi": [0.0, 0.5, 1.0]}},
    "types": {"grid": {"rho": [0.5], "phi": [0.5]},
              "base": {"reader_types_enabled": True, "uncertainty_n0": 20}},
    "dropout": {"grid": {"rho": [0.5], "phi": [0.5, 1.0], "outside_option": [-1.5, -1.0, -0.5]}},
    "n0": {"grid": {"rho": [0.5], "phi": [0.5], "uncertainty_n0": [5, 20, 80]}},
    "rho": {"grid": {"rho": [0.0, 0.5, 0.9], "phi": [0.5]}},
    "robustness": {"grid": {"rho": [0.5], "phi": [0.5], "gamma": [0.5, 1.0, 2.0],
                            "beta": [0.5, 1.0, 2.0], "sigma_bar": [0.5, 1.0, 2.0]},
                   "n_pop": 10},
    "pca4": {"grid": {"rho": [0.5], "phi": [0.5]}, "pca_dims": 4},
    "pca16": {"grid": {"rho": [0.5], "phi": [0.5]}, "pca_dims": 16},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="config.yaml")
    ap.add_argument("--out", default="configs/final")
    ap.add_argument("--n-pop", type=int, default=None, help="override populations (smoke tests)")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in EXPERIMENTS.items():
        cfg = copy.deepcopy(base)
        news = cfg["news"]
        news["grid"] = {**spec["grid"], "info": ["none"]}
        news["base"] = {**news["base"], "reader_history_visible": False, **spec.get("base", {})}
        news["n_pop"] = args.n_pop or spec.get("n_pop", news.get("n_pop", 20))
        if "pca_dims" in spec:
            news["pca_dims"] = spec["pca_dims"]
        header = f"# Final experiment '{name}', generated from {args.base} by scripts/make_final_configs.py\n"
        (out / f"{name}.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False))
        print("wrote", out / f"{name}.yaml")


if __name__ == "__main__":
    main()
