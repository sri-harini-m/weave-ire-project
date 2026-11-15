"""Rebuild report tables and figures from saved runs, without rerunning readers.

Exposure means an explicitly returned slate. Full-catalogue browsing (`none`)
has no measured rendered exposure. Consumption is measured separately.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import homogeneity, time_curves, user_metrics


METRIC_DEFINITIONS = {
    "cat_entropy": "Mean reader consumption-category entropy / log(catalogue categories).",
    "n_categories": "Mean number of categories consumed per reader.",
    "local_distance": "Mean normalized world distance between consecutive reads per reader.",
    "diversity": "Mean pairwise normalized world distance among each reader's consumed articles.",
    "homogeneity": "Mean pairwise Jaccard overlap of reader article sets; NaN with fewer than two readers.",
    "welfare": "Mean reader mean true utility per consumed article, before fatigue.",
    "welfare_net": "Mean reader mean observed utility per consumed article, after fatigue.",
    "fatigue": "Mean reader mean fatigue penalty per consumed article.",
    "rounds": "Mean recorded exit round; horizon when no explicit exit occurs; not the number of reads.",
    "n_consumed": "Mean actual number of reads per reader.",
    "consumption_share": "Read events / (readers * horizon); not CTR.",
    "dropout_share": "Fraction with an explicit exit before the horizon; exhaustion is not an explicit exit.",
    "catalog_coverage": "Distinct consumed article indices / catalogue size.",
    "consumption_gini": "Gini of read-event counts across all catalogue articles, including zero counts.",
    "consumption_hhi": "Sum of squared article read-event shares; higher means more concentrated consumption.",
    "dominant_category_share": "Mean reader fraction of reads in their most consumed category.",
    "cumulative_welfare": "Mean per-reader sum of pre-fatigue utilities across the run.",
    "cumulative_welfare_net": "Mean per-reader sum of after-fatigue utilities across the run.",
    "total_read_events": "Total article consumptions across readers and rounds.",
    "total_utility_net": "Sum of after-fatigue utilities across all consumed events.",
    "slate_ild": "Mean pairwise normalized world distance within nontrivial explicitly returned slates.",
    "slate_category_coverage": "Mean distinct slate categories / catalogue category count, over nonempty slates.",
    "slate_category_entropy": "Mean normalized category entropy over nonempty explicitly returned slates.",
    "exposure_coverage": "Distinct articles in explicit slates / catalogue size; NaN without explicit exposure events.",
    "exposure_gini": "Gini of explicit slate exposure counts across the catalogue; NaN without exposures.",
    "exposure_hhi": "Sum of squared article exposure shares; NaN without explicit exposures.",
    "exposure_events": "Number of article positions in explicitly returned slates; NaN for none baseline.",
    "step_distance": "Per-round mean world distance between consecutive reads, where both exist.",
    "active_share": "Fraction not explicitly dropped out after each round; exhaustion does not imply dropout.",
    "reported_boredom_rate": "Mean observed coarse boredom flag across reported read events; not latent fatigue or inferred dissatisfaction.",
    "reported_departures": "Total recorded explicit reader-departure signals; missing when no signals were archived.",
    "departures": "Number of explicit departure signals in a round.",
    "recommendation_seconds": "Measured recommendation-call duration for a round, excluding archive writes and reader simulation.",
    "total_recommendation_seconds": "Sum of measured recommendation-call durations across rounds.",
    "ci95": "Normal approximation: 1.96 * sample standard deviation / sqrt(population count), independently per condition; NaN for n < 2.",
    "limitations": "Article overlap is a homogenization proxy, not ideological agreement. No CTR or click model is inferred. Synthetic data are engineering checks, not MIND findings.",
}


def _mean(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _concentration(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if not counts.size or total <= 0:
        return np.nan, np.nan
    sorted_counts = np.sort(counts)
    n = len(counts)
    gini = 2 * np.dot(np.arange(1, n + 1), sorted_counts) / (n * total) - (n + 1) / n
    return float(gini), float(np.square(counts / total).sum())


def _entropy(categories, n_categories):
    if not len(categories):
        return np.nan
    _, counts = np.unique(categories, return_counts=True)
    proportions = counts / counts.sum()
    return float(-(proportions * np.log(proportions)).sum() / np.log(n_categories)) \
        if n_categories > 1 else 0.0


def population_report(world, run, slates=None):
    """Return reader metrics, population metrics and per-round curves.

    `slates` is a list of [U, K] integer arrays, one per round; `None` entries
    represent full-catalogue browsing. Inactive rows must already be padded.
    """
    seq = np.asarray(run["seq"])
    if seq.ndim != 2 or not np.issubdtype(seq.dtype, np.integer):
        raise ValueError("seq must be a [U, T] integer array")
    if (seq < -1).any() or (seq >= world.N).any():
        raise ValueError("seq contains an invalid catalogue index")
    U, T = seq.shape
    val, pen = np.asarray(run["val"]), np.asarray(run["pen"])
    left_at = np.asarray(run["left_at"])
    if val.shape != seq.shape or pen.shape != seq.shape or left_at.shape != (U,):
        raise ValueError("run value, penalty and exit arrays must match seq")
    users = user_metrics(world, run) if U else pd.DataFrame(columns=["user"])
    valid = seq >= 0
    if U:
        users["cumulative_welfare"] = np.where(valid, val, 0).sum(axis=1)
        users["cumulative_welfare_net"] = np.where(valid, val - pen, 0).sum(axis=1)
        users["dropout_share"] = (left_at < T).astype(float)
        users["consumption_share"] = valid.sum(axis=1) / T if T else np.nan
    n_cats = len(np.unique(world.cats)) if world.cats is not None else 0
    dominant = []
    if world.cats is not None:
        for row in seq:
            items = row[row >= 0]
            _, counts = np.unique(world.cats[items], return_counts=True)
            dominant.append(float(counts.max() / counts.sum()) if counts.size else np.nan)
        users["dominant_category_share"] = dominant
    metrics = {c: _mean(users[c]) for c in users.columns if c != "user"}
    counts = np.bincount(seq[valid], minlength=world.N)
    metrics["catalog_coverage"] = float(np.count_nonzero(counts) / world.N) if world.N else np.nan
    metrics["consumption_gini"], metrics["consumption_hhi"] = _concentration(counts)
    metrics["homogeneity"] = homogeneity(run) if U >= 2 and T and valid.any() else np.nan
    metrics["total_read_events"] = int(valid.sum())
    metrics["total_utility_net"] = float(np.where(valid, val - pen, 0).sum())

    # Retain the existing step-distance calculation, then distinguish activity
    # from reading. The original time_curves calls actual reads active_share.
    old_curves = time_curves(world, run) if T > 1 else pd.DataFrame()
    steps = dict(zip(old_curves.get("t", []), old_curves.get("step_distance", [])))
    curve_rows = []
    for t in range(T):
        selected = valid[:, t]
        prefix = seq[:, :t + 1]
        cat_entropies = [_entropy(world.cats[r[r >= 0]], n_cats) for r in prefix] \
            if world.cats is not None else []
        curve_rows.append({"t": t, "step_distance": steps.get(t, np.nan),
                           "fatigue": _mean(pen[selected, t]),
                           "welfare_net": _mean((val - pen)[selected, t]),
                           "active_share": float((left_at > t).mean()) if U else np.nan,
                           "consumption_share": float(selected.mean()) if U else np.nan,
                           "cat_entropy": _mean(cat_entropies)})
    curves = pd.DataFrame(curve_rows, columns=["t", "step_distance", "fatigue", "welfare_net",
                                               "active_share", "consumption_share", "cat_entropy"])

    exposure_counts = np.zeros(world.N, dtype=int)
    ild, category_coverage, category_entropy = [], [], []
    explicit_slates = False
    if slates is not None:
        if len(slates) != T:
            raise ValueError("slates must have one entry per simulation round")
        for slate in slates:
            if slate is None:
                continue
            explicit_slates = True
            slate = np.asarray(slate)
            if slate.ndim != 2 or slate.shape[0] != U or not np.issubdtype(slate.dtype, np.integer):
                raise ValueError("each slate must be an integer [U, K] array")
            if (slate < -1).any() or (slate >= world.N).any():
                raise ValueError("slate contains an invalid catalogue index")
            for row in slate:
                items = row[row >= 0]
                if len(items) != len(np.unique(items)):
                    raise ValueError("slate contains duplicate articles")
                np.add.at(exposure_counts, items, 1)
                if len(items) >= 2:
                    d = world.D[np.ix_(items, items)]
                    ild.append(float(d.sum() / (len(items) * (len(items) - 1))))
                if len(items) and world.cats is not None:
                    category_coverage.append(len(np.unique(world.cats[items])) / n_cats)
                    category_entropy.append(_entropy(world.cats[items], n_cats))
    n_exposures = int(exposure_counts.sum())
    metrics.update(slate_ild=_mean(ild), slate_category_coverage=_mean(category_coverage),
                   slate_category_entropy=_mean(category_entropy),
                   exposure_events=n_exposures if explicit_slates else np.nan,
                   exposure_coverage=float(np.count_nonzero(exposure_counts) / world.N)
                   if n_exposures and world.N else np.nan)
    metrics["exposure_gini"], metrics["exposure_hhi"] = _concentration(exposure_counts)
    return users, metrics, curves


def _metadata(directory):
    meta = json.loads((directory / "metadata.json").read_text())
    params_file = directory / "params.json"
    params = meta.get("params", {})
    if not params and params_file.exists():
        params = json.loads(params_file.read_text())
    settings = meta.get("recommender_settings", {})
    row = {k: v for k, v in params.items() if v is None or isinstance(v, (str, bool, int, float))}
    row.update({k: v for k, v in settings.items() if v is None or isinstance(v, (str, bool, int, float))})
    for key in ("rho", "phi", "info", "diversity_weight", "data_kind", "requested_K", "reference_only"):
        if key in meta:
            row[key] = meta[key]
    row["recommender"] = meta.get("recommender", meta.get("algorithm", "unknown"))
    row["algorithm"] = meta.get("algorithm", row["recommender"])
    row["population"] = meta.get("population", meta.get("pop", directory.name))
    row["seed"] = meta.get("seed")
    row["archive"] = str(directory)
    return row


def _read_slates(directory, T):
    outputs = sorted((directory / "rounds").glob("round_*_output.npz"))
    if not outputs:
        return None
    if len(outputs) != T:
        raise ValueError(f"Incomplete round outputs in {directory}")
    if [p.name for p in outputs] != [f"round_{t:04d}_output.npz" for t in range(T)]:
        raise ValueError(f"Round indices are not contiguous in {directory}")
    slates = []
    for path in outputs:
        with np.load(path, allow_pickle=False) as output:
            slate = None if bool(output["slate_is_none"]) else output["slates"].copy()
        if slate is not None:
            input_path = path.with_name(path.name.replace("_output.npz", "_input.npz"))
            with np.load(input_path, allow_pickle=False) as inputs:
                active = inputs["active"]
                available = inputs["available"]
            if slate.ndim != 2 or slate.shape[0] != len(active) \
                    or not np.issubdtype(slate.dtype, np.integer) \
                    or (slate < -1).any() or (slate >= available.shape[1]).any():
                raise ValueError(f"Saved slate has invalid shape or indices: {path}")
            slate[~active] = -1
            for u, row in enumerate(slate):
                items = row[row >= 0]
                if len(items) and not available[u, items].all():
                    raise ValueError(f"Saved slate contains unavailable articles: {path}")
        slates.append(slate)
    return slates


def _observed_metrics(directory, curves, metrics):
    """Add only recorded feedback signals and measured call times."""
    flags, departures, times = [], [], []
    for t in curves["t"]:
        path = directory / "rounds" / f"round_{t:04d}_output.npz"
        if not path.exists():
            flags.append(None); departures.append(np.nan); times.append(np.nan)
            continue
        with np.load(path, allow_pickle=False) as saved:
            boredom = saved["feedback_boredom"].copy() if "feedback_boredom" in saved else None
            if boredom is not None and (boredom.ndim != 1 or not np.isfinite(boredom).all()
                                         or (boredom < 0).any() or (boredom > 1).any()):
                raise ValueError("Saved boredom signals must be finite values in [0, 1]")
            flags.append(boredom)
            departures.append(len(saved["feedback_left_users"]) if "feedback_left_users" in saved else np.nan)
            times.append(float(saved["diagnostic_recommendation_seconds"])
                         if "diagnostic_recommendation_seconds" in saved else np.nan)
    curves["reported_boredom_rate"] = [_mean(f) if f is not None else np.nan for f in flags]
    curves["departures"] = departures
    curves["recommendation_seconds"] = times
    reported = [f for f in flags if f is not None]
    metrics["reported_boredom_rate"] = _mean(np.concatenate(reported)) if reported else np.nan
    metrics["reported_departures"] = float(np.nansum(departures)) if np.isfinite(departures).any() else np.nan
    metrics["total_recommendation_seconds"] = float(np.nansum(times)) if np.isfinite(times).any() else np.nan


def _constructor_columns():
    from inspect import signature
    from .recommenders import REGISTRY
    return {name for rec in REGISTRY.values() for name in signature(rec.__init__).parameters if name != "self"}


def paired_comparisons(frame, condition_columns, metric_columns, model_columns=None):
    """Return exact matched-population differences, preserving model controls.

    Only the named contrast changes. Missing population IDs/seeds are never
    matched. A nonunique baseline is rejected instead of being averaged.
    """
    model_columns = set(model_columns or _constructor_columns())
    identity = {"recommender", "algorithm", "reference_only"}
    conditions = [c for c in condition_columns if c in frame]
    measurements = [m for m in metric_columns if m in frame]
    required = ["seed", "population"]
    if not all(c in frame for c in required):
        usable = frame.iloc[:0]
    else:
        usable = frame[frame[required].notna().all(axis=1)]
    algorithm = usable.get("algorithm", usable.get("recommender", pd.Series(dtype=str)))
    outputs = {}

    def match(left, right, keys, comparison):
        keys = list(dict.fromkeys([c for c in keys + required if c in frame]))
        schema = list(dict.fromkeys(["comparison", "left_recommender", "right_recommender"]
                                   + conditions + required + ["left_archive", "right_archive"]
                                   + [m + suffix for m in measurements for suffix in ("_left", "_right", "_difference")]))
        if left.empty or right.empty:
            result = pd.DataFrame(columns=schema)
        else:
            if right.duplicated(keys).any():
                raise ValueError(f"Ambiguous matched baseline for {comparison}; model settings or population controls differ")
            merged = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="many_to_one")
            result = pd.DataFrame(index=merged.index)
            result["comparison"] = comparison
            result["left_recommender"] = merged["recommender"] if "recommender" in keys else merged["recommender_left"]
            result["right_recommender"] = merged["recommender"] if "recommender" in keys else merged["recommender_right"]
            for c in conditions + required:
                result[c] = merged[c] if c in keys else merged[c + "_left"]
            for side in ("left", "right"):
                result[side + "_archive"] = merged.get("archive_" + side, pd.Series(index=merged.index, dtype=str))
            for metric in measurements:
                result[metric + "_left"] = merged[metric + "_left"]
                result[metric + "_right"] = merged[metric + "_right"]
                result[metric + "_difference"] = pd.to_numeric(merged[metric + "_left"], errors="coerce") \
                    - pd.to_numeric(merged[metric + "_right"], errors="coerce")
            result = result.reindex(columns=schema)
        result.attrs["candidate_left_rows"] = len(left)
        result.attrs["matched_rows"] = len(result)
        result.attrs["unmatched_left_rows"] = len(left) - len(result)
        return result

    if "reader_history_visible" in usable:
        history_left = usable[usable["reader_history_visible"].eq(True)]
        history_right = usable[usable["reader_history_visible"].eq(False)]
    else:
        history_left = history_right = usable.iloc[:0]
    outputs["paired_history_visibility"] = match(history_left, history_right,
        [c for c in conditions if c != "reader_history_visible"], "returning_minus_new")
    aware_outputs = []
    coverage = {"candidate_left_rows": 0, "matched_rows": 0, "unmatched_left_rows": 0}
    for aware, blind in (("fatigue_popularity", "popularity"), ("fatigue_content", "content"), ("fatigue_mf", "mf")):
        left, right = usable[algorithm.eq(aware)], usable[algorithm.eq(blind)]
        shared = [c for c in model_columns if c in frame and left[c].notna().any() and right[c].notna().any()]
        controls = [c for c in conditions if c not in identity | model_columns] + shared
        paired = match(left, right, controls, "fatigue_aware_minus_blind")
        for key in coverage:
            coverage[key] += paired.attrs[key]
        aware_outputs.append(paired)
    populated = [table for table in aware_outputs if len(table)]
    outputs["paired_fatigue_feedback"] = pd.concat(populated, ignore_index=True) \
        if populated else aware_outputs[0].copy()
    outputs["paired_fatigue_feedback"].attrs.update(coverage)
    baseline = usable[algorithm.eq("none")]
    methods = usable[~algorithm.eq("none")]
    # Baseline has no personalized model, so only common constructor controls
    # (normally K) can be matched; every left model configuration remains a row.
    shared = [c for c in model_columns if c in frame and methods[c].notna().any() and baseline[c].notna().any()
              and c != "reference_only"]
    controls = [c for c in conditions if c not in identity | model_columns] + shared
    outputs["paired_vs_none"] = match(methods, baseline, controls, "method_minus_none")
    return outputs


def _save_paired_tables(frame, condition_columns, metric_columns, output, model_columns):
    paired = paired_comparisons(frame, condition_columns, metric_columns, model_columns)
    coverage = []
    for name, table in paired.items():
        table.to_csv(output / f"{name}.csv", index=False)
        coverage.append({"comparison": name, **table.attrs})
        differences = [c for c in table if c.endswith("_difference")]
        controls = [c for c in ["comparison", "left_recommender", "right_recommender"] + condition_columns if c in table]
        if len(table):
            summary = _summarize(table, list(dict.fromkeys(controls)), differences)
            summary = summary.rename(columns={"n_populations": "n_pairs"})
        else:
            summary = pd.DataFrame(columns=list(dict.fromkeys(controls)) + ["n_pairs"] + differences)
        summary.to_csv(output / f"{name}_summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "paired_comparison_coverage.csv", index=False)
    (output / "paired_comparison_definitions.json").write_text(json.dumps({
        "direction": "Each difference is left minus right for the same saved seed and population.",
        "history": "Returning-reader visibility minus new-reader visibility; all other simulator/model controls match.",
        "fatigue_feedback": "Fatigue-aware algorithm minus its blind counterpart; shared model settings match and aware-only knobs remain labeled.",
        "vs_none": "Each exact method/configuration minus full-catalogue browsing at matching simulator controls and slate-size request.",
        "limits": "Descriptive matched simulation comparisons. Unmatched rows are counted in paired_comparison_coverage.csv; reference_only rows remain explicitly flagged."
    }, indent=2) + "\n")


def _summarize(frame, group_columns, value_columns):
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_populations"] = len(group)
        for value in value_columns:
            series = pd.to_numeric(group[value], errors="coerce").dropna()
            row[value] = float(series.mean()) if len(series) else np.nan
            row[value + "_ci95"] = float(1.96 * series.std(ddof=1) / np.sqrt(len(series))) \
                if len(series) >= 2 else np.nan
            row[value + "_n"] = len(series)
        rows.append(row)
    return pd.DataFrame(rows)


def _figures(summary, curve_summary, output, condition_columns=None, model_columns=None):
    if len(summary) > 24:
        from .study_figures import study_figures
        if condition_columns is None:
            condition_columns = [c for c in summary if c not in METRIC_DEFINITIONS
                                 and c not in ("condition_id", "n_populations")
                                 and not c.endswith(("_ci95", "_n"))]
        return study_figures(summary, output, condition_columns,
                             _constructor_columns() if model_columns is None else model_columns)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    synthetic = "data_kind" in summary and summary["data_kind"].eq("synthetic").all()
    prefix = "SYNTHETIC engineering check: " if synthetic else ""
    def label(row):
        identity = f"{row['condition_id']} | {row['recommender']}"
        pieces = []
        for key in ("rho", "phi", "diversity_weight"):
            if key in row and pd.notna(row[key]):
                pieces.append(f"{key}={row[key]:g}")
        return identity + ("\n" + " | ".join(pieces) if pieces else "")
    labels = {row["condition_id"]: label(row) for _, row in summary.iterrows()}
    targets = ["cat_entropy", "diversity", "homogeneity", "fatigue", "welfare_net",
               "catalog_coverage", "consumption_gini", "dropout_share", "slate_ild", "exposure_coverage"]
    for metric in targets:
        if metric not in summary or not summary[metric].notna().any():
            continue
        fig, ax = plt.subplots(figsize=(max(10, len(summary) * .6), 7))
        valid = summary[metric].notna()
        selected = summary[valid]
        x = np.arange(len(selected))
        ci = selected[metric + "_ci95"].fillna(0)
        ax.errorbar(x, selected[metric], yerr=ci, fmt="o", capsize=3)
        ax.set_xticks(x, [labels[c] for c in selected["condition_id"]],
                      rotation=45, ha="right", fontsize=8)
        ax.set(xlabel="Condition (all remaining parameters in conditions.csv)", ylabel=metric,
               title=f"{prefix}{metric}\nPopulation means and approximate 95% CIs")
        ax.grid(alpha=.25)
        fig.subplots_adjust(left=.12, right=.97, top=.85, bottom=.43)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{metric}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    for metric in ("cat_entropy", "step_distance", "fatigue", "welfare_net", "active_share", "consumption_share"):
        if metric not in curve_summary or not curve_summary[metric].notna().any():
            continue
        fig, ax = plt.subplots(figsize=(11, max(5.5, len(summary) * .34)))
        for condition, group in curve_summary.groupby("condition_id", sort=False):
            group = group.sort_values("t")
            ax.plot(group["t"], group[metric], label=labels[condition])
            lo = group[metric] - group[metric + "_ci95"]
            hi = group[metric] + group[metric + "_ci95"]
            ax.fill_between(group["t"], lo, hi, alpha=.12)
        ax.set(xlabel="Round (zero based)", ylabel=metric,
               title=f"{prefix}{metric} by round\nConditions kept separate")
        ax.legend(title="Condition", fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1),
                  borderaxespad=0, ncol=max(1, len(summary) // 20))
        ax.grid(alpha=.25)
        fig.subplots_adjust(left=.1, right=.62, top=.84, bottom=.14)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"curve_{metric}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    if {"diversity", "welfare_net"}.issubset(summary.columns):
        tradeoff = summary[summary["diversity"].notna() & summary["welfare_net"].notna()]
        if len(tradeoff):
            fig, ax = plt.subplots(figsize=(11, max(5.5, len(tradeoff) * .34)))
            markers = ("o", "s", "^", "D", "v", "P", "X")
            for i, (_, row) in enumerate(tradeoff.iterrows()):
                xerr = row["diversity_ci95"] if pd.notna(row["diversity_ci95"]) else 0
                yerr = row["welfare_net_ci95"] if pd.notna(row["welfare_net_ci95"]) else 0
                ax.errorbar(row["diversity"], row["welfare_net"], xerr=xerr, yerr=yerr,
                            fmt=markers[i % len(markers)], capsize=3,
                            label=labels[row["condition_id"]])
            ax.set(xlabel="Consumption diversity (normalized world distance)",
                   ylabel="Net welfare (utility after fatigue)",
                   title=f"{prefix}Diversity–welfare comparison\nPopulation means and approximate 95% CIs")
            ax.legend(title="Condition", fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1),
                      borderaxespad=0, ncol=max(1, len(tradeoff) // 20))
            ax.grid(alpha=.25)
            fig.subplots_adjust(left=.1, right=.62, top=.84, bottom=.14)
            for suffix in ("png", "svg"):
                fig.savefig(output / f"tradeoff_diversity_welfare.{suffix}", dpi=180, bbox_inches="tight")
            plt.close(fig)


def rebuild_report(run_dir):
    """Recompute tables and plots using only population archives on disk.

    Outputs are in `run_dir/report/`. Existing report outputs can be regenerated;
    population inputs, raw results and checkpoints are never overwritten here.
    """
    from .artifacts import load_world

    root = Path(run_dir)
    run_status_path = root / "run.json"
    if run_status_path.exists() and json.loads(run_status_path.read_text()).get("status") == "failed":
        raise ValueError("Report refused: the experiment run is marked failed; inspect its saved logs.")
    if (root / "manifest.json").exists():
        from .run_bundle import verify_manifest
        verify_manifest(root, ignore_report=True)
    population_root = root / "populations"
    plan_path = root / "expected_populations.json"
    if plan_path.exists():
        jobs = json.loads(plan_path.read_text())["jobs"]
        planned_names = [job["directory"] for job in jobs]
        if len(planned_names) != len(set(planned_names)):
            raise ValueError("Report refused: expected population directories are duplicated")
        missing = []
        for name in planned_names:
            directory = population_root / name
            if not directory.resolve().is_relative_to(population_root.resolve()):
                raise ValueError("Report refused: expected population directory escapes the run")
            status_path = directory / "status.json"
            status = json.loads(status_path.read_text()).get("status") if status_path.exists() else None
            if status != "complete" or not (directory / "run.npz").is_file():
                missing.append(name)
        if missing:
            raise ValueError("Report refused: planned populations are incomplete or missing: " + ", ".join(missing))
    paths = sorted(population_root.glob("**/run.npz"))
    if plan_path.exists():
        unplanned = [str(path.parent.relative_to(population_root)) for path in paths
                     if str(path.parent.relative_to(population_root)) not in planned_names]
        if unplanned:
            raise ValueError("Report refused: unplanned population archives: " + ", ".join(unplanned))
    candidate_dirs = {p.parent for pattern in ("**/metadata.json", "**/status.json", "**/run.npz")
                      for p in population_root.glob(pattern)}
    incomplete = []
    for directory in sorted(candidate_dirs):
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()).get("status") if status_path.exists() else None
        if status not in (None, "complete") or not (directory / "run.npz").exists():
            incomplete.append(f"{directory.name} ({status or 'raw results missing'})")
    if incomplete:
        raise ValueError("Report refused: incomplete population archives: " + ", ".join(incomplete)
                         + ". Completed population arrays remain saved; inspect failed jobs before reporting.")
    if not paths:
        raise ValueError(f"No saved population runs under {root / 'populations'}")
    populations, users, curves, skipped = [], [], [], []
    metadata_columns = set()
    model_columns = set()
    for run_path in paths:
        directory = run_path.parent
        meta = _metadata(directory)
        metadata_columns.update(meta)
        document = json.loads((directory / "metadata.json").read_text())
        model_columns.update(k for k, v in document.get("recommender_settings", {}).items()
                             if v is None or isinstance(v, (str, bool, int, float)))
        world = load_world(directory)
        with np.load(run_path, allow_pickle=False) as stored:
            run = {k: stored[k].copy() for k in stored.files}
        slate_data = _read_slates(directory, run["seq"].shape[1])
        user, metrics, curve = population_report(world, run, slate_data)
        _observed_metrics(directory, curve, metrics)
        populations.append({**meta, **metrics})
        for key, value in meta.items():
            user[key] = value
            curve[key] = value
        users.append(user)
        curves.append(curve)
    if not populations:
        raise ValueError("No complete saved populations are available for reporting")
    population_frame = pd.DataFrame(populations)
    group_columns = sorted(metadata_columns - {"population", "seed", "archive"})
    metric_columns = [c for c in population_frame if c not in metadata_columns]
    summary = _summarize(population_frame, group_columns, metric_columns)
    summary.insert(0, "condition_id", [f"C{i:03d}" for i in range(len(summary))])
    user_frame = pd.concat(users, ignore_index=True)
    curve_frame = pd.concat(curves, ignore_index=True)
    curve_metrics = [c for c in curve_frame if c not in metadata_columns and c != "t"]
    curve_summary = _summarize(curve_frame, group_columns + ["t"], curve_metrics) \
        if len(curve_frame) else pd.DataFrame()
    if len(curve_summary):
        curve_summary = curve_summary.merge(summary[["condition_id"] + group_columns],
                                            on=group_columns, how="left", validate="many_to_one")
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("populations", population_frame), ("users", user_frame),
                        ("curves", curve_frame), ("summary", summary),
                        ("curves_summary", curve_summary),
                        ("conditions", summary[["condition_id"] + group_columns])):
        frame.to_csv(output / f"{name}.csv", index=False)
    _save_paired_tables(population_frame, group_columns, metric_columns, output, model_columns)
    (output / "metrics_definitions.json").write_text(json.dumps(METRIC_DEFINITIONS, indent=2) + "\n")
    (output / "sources.json").write_text(json.dumps({"population_archives": population_frame["archive"].tolist(),
        "skipped_incomplete_archives": skipped, "simulator_rerun": False}, indent=2) + "\n")
    columns = [c for c in ("condition_id", "recommender", "requested_K", "K", "reader_history_visible",
                            "reference_only", "rho", "phi", "diversity_weight",
                            "data_kind", "n_populations", "cat_entropy", "diversity", "homogeneity",
                            "fatigue", "welfare_net", "catalog_coverage", "slate_ild") if c in summary]
    lines = ["# Saved experiment comparison", "", "Rebuilt from saved arrays. No simulator was rerun.", "",
             "Condition IDs map to every saved parameter in conditions.csv.",
             "Rows with n_populations=1 have no estimated confidence interval.",
             "Synthetic rows are engineering checks, not MIND research findings.",
             "Exposure metrics describe explicit returned slates. Full-catalogue browsing has no measured slate exposure.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in summary.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                cell = "NA"
            elif col in metric_columns:
                ci = row[col + "_ci95"]
                cell = f"{value:.4g}" + (f" ± {ci:.3g}" if pd.notna(ci) else " (CI unavailable)")
            else:
                cell = str(value)
            cells.append(cell.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Full numeric results, effective sample counts, and CI widths are in summary.csv.",
              "Plots use summary.csv and curves_summary.csv. Regenerate these outputs from the same archives when presentation needs change.", ""]
    (output / "summary.md").write_text("\n".join(lines))
    _figures(summary, curve_summary, output / "figures", group_columns, model_columns)
    return {"output": str(output), "populations": len(population_frame),
            "conditions": len(summary), "summary": summary}
