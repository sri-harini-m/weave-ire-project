from functools import partial
import json

import pandas as pd

from .recommender_config import recommender_options
from .recommenders import REGISTRY


MMR_WEIGHTS = [0., .3, .6, .9, 1.]
SLATE_SIZES = [5, 10, 25, 50, 500]


def add_study_arguments(parser, default_k=None):
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--k", "--slate-size", type=int, default=default_k,
                       help="Number of articles in each slate")
    sizes.add_argument("--slate-sizes", type=int, nargs="+",
                       help="Compare several sizes, e.g. 5 10 25 50 500")
    parser.add_argument("--reader-history-visible", type=str.lower,
                        choices=["false", "true", "both"], default=None,
                        help="Expose initial history, or compare both visibility settings")
    parser.add_argument("--snapshot-policy", choices=["all", "final", "none"], default="final",
                        help="Model checkpoints to retain (default: final only)")
    parser.add_argument("--archive-diagnostics", action="store_true",
                        help="Retain hidden simulator diagnostics and full belief arrays")
    parser.add_argument("--outside-option", type=float, default=None,
                        help="Reader departure threshold; omit to keep configured behavior")


def slate_sizes(args, default=10):
    sizes = getattr(args, "slate_sizes", None)
    if sizes is None:
        k = getattr(args, "k", None)
        sizes = [default if k is None else k]
    if not sizes or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in sizes):
        raise ValueError("Slate sizes must be positive integers")
    return list(dict.fromkeys(sizes))


def apply_visibility(args, grid):
    setting = getattr(args, "reader_history_visible", None)
    if setting is not None:
        grid["reader_history_visible"] = [False, True] if setting == "both" else [setting == "true"]
    return grid


def factories(names, settings, mmr_weights=None):
    recs = {}
    for name in names:
        options = recommender_options(name, settings)
        if name == "mmr" and mmr_weights is not None:
            for weight in dict.fromkeys(mmr_weights):
                if not 0 <= weight <= 1:
                    raise ValueError("MMR weights must be in [0, 1]")
                recs[f"mmr_{weight:g}"] = partial(REGISTRY[name], **{**options, "diversity_weight": weight})
        else:
            recs[name] = partial(REGISTRY[name], **options)
    return recs


def write_comparisons(root):

    frame = pd.read_csv(root / "report" / "summary.csv")
    tables = {
        "slate_size_comparison": frame,
        "mmr_sweep": frame[frame["recommender"].astype(str).str.startswith("mmr")],
        "history_visibility_comparison": frame,
        "fatigue_feedback_comparison": frame[frame["recommender"].isin(
            ["popularity", "fatigue_popularity", "content", "fatigue_content", "mf", "fatigue_mf"])],
        "diversity_methods_comparison": frame[frame["recommender"].astype(str).str.startswith("mmr")
                                                | frame["recommender"].isin(["content", "calibrated"])],
    }
    for name, table in tables.items():
        table.to_csv(root / "report" / f"{name}.csv", index=False)
    conditions = pd.read_csv(root / "report" / "conditions.csv")
    (root / "settings_used.json").write_text(json.dumps({
        "schema_version": 1, "conditions": json.loads(conditions.to_json(orient="records")),
        "interpretation": "Exact condition settings; requested_K is the requested slate size."
    }, indent=2) + "\n")
