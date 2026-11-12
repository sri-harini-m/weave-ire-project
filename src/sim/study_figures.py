"""Bounded study figures: exact condition lines, small canvases, no pooling."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("cat_entropy", "homogeneity", "welfare_net", "fatigue")
CANVAS = (15, 10)
LINES_PER_PAGE = 16


def _groups(frame, columns):
    if not columns:
        yield (), frame
    else:
        for key, group in frame.groupby(columns, dropna=False, sort=False):
            yield key if isinstance(key, tuple) else (key,), group


def _value(value):
    return None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value


def _line_label(row, line_id):
    label = f"{line_id} {row.get('recommender', row.get('algorithm', 'method'))}"
    if pd.notna(row.get("diversity_weight")):
        label += f" (MMR weight={row['diversity_weight']:g})"
    if pd.notna(row.get("fatigue_weight")):
        label += f" (fatigue weight={row['fatigue_weight']:g})"
    reference = row.get("reference_only", False)
    if pd.notna(reference) and bool(reference):
        label += " [privileged reference]"
    return label


def _save_figure(fig, base):
    for suffix in ("png", "svg"):
        fig.savefig(base.with_suffix("." + suffix), dpi=160, bbox_inches="tight")


def study_figures(summary, output, condition_columns, model_columns):
    """Save compact pages and their exact line/facet/source tables.

    Every simulator parameter is a facet control. Every constructor setting is
    an exact line control. Only slate size changes along a K line. A duplicate
    x value within one exact line is rejected rather than averaged.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in condition_columns if c in summary]
    model = set(model_columns) | {"recommender", "algorithm", "reference_only"}
    x_column = "requested_K" if "requested_K" in summary and summary["requested_K"].notna().any() else "K"
    if x_column not in summary:
        # Large non-study reports remain bounded and retain each condition.
        frame = summary.copy()
        frame["_condition_position"] = np.arange(len(frame))
        x_column = "_condition_position"
    else:
        frame = summary.copy()
    facet_columns = [c for c in conditions if c not in model | {"K", "requested_K"}]
    line_columns = [c for c in conditions if c not in facet_columns + ["K", "requested_K"]]
    line_rows, facet_rows, source_rows = [], [], []
    identities = {}
    rendered = []
    synthetic = "data_kind" in summary and summary["data_kind"].eq("synthetic").all()
    prefix = "SYNTHETIC engineering check — " if synthetic else ""
    colors = plt.get_cmap("tab20")

    for facet_number, (facet_key, facet) in enumerate(_groups(frame, facet_columns)):
        facet_id = f"F{facet_number:03d}"
        controls = dict(zip(facet_columns, facet_key))
        facet_rows.append({"facet_id": facet_id, **controls})
        lines = []
        for line_key, group in _groups(facet, line_columns):
            if group[x_column].duplicated().any():
                raise ValueError("Study plot contains duplicate x values within an exact configuration; refusing to pool controls")
            configuration = dict(zip(line_columns, line_key))
            identity = json.dumps({k: _value(v) for k, v in configuration.items()}, sort_keys=True)
            if identity not in identities:
                line_id = f"L{len(identities):03d}"
                identities[identity] = line_id
                line_rows.append({"line_id": line_id, **configuration})
            line_id = identities[identity]
            lines.append((line_id, group.sort_values(x_column)))
        for start in range(0, len(lines), LINES_PER_PAGE):
            page = start // LINES_PER_PAGE
            base = output / f"study_slate_size_{facet_id}_page_{page:02d}"
            fig, axes = plt.subplots(2, 2, figsize=CANVAS, squeeze=False)
            handles, labels = [], []
            for local, (line_id, group) in enumerate(lines[start:start + LINES_PER_PAGE]):
                label = _line_label(group.iloc[0], line_id)
                color = colors(int(line_id[1:]) % 20)
                reference = group.iloc[0].get("reference_only", False)
                reference = pd.notna(reference) and bool(reference)
                first_handle = None
                for ax, metric in zip(axes.flat, METRICS):
                    if metric not in group:
                        continue
                    finite = group[metric].notna() & group[x_column].notna()
                    points = group[finite]
                    if not len(points):
                        continue
                    ci = points.get(metric + "_ci95", pd.Series(np.nan, index=points.index)).fillna(0)
                    handle = ax.errorbar(points[x_column], points[metric], yerr=ci,
                                         color=color, marker="o", markersize=3, capsize=2,
                                         linestyle="--" if reference else "-")
                    if first_handle is None:
                        first_handle = handle
                if first_handle is not None:
                    handles.append(first_handle)
                    labels.append(label)
                for _, row in group.iterrows():
                    source_rows.append({"figure": base.name, "facet_id": facet_id, "line_id": line_id,
                                        "condition_id": row["condition_id"], "x_column": x_column,
                                        "x": row[x_column]})
            for ax, metric in zip(axes.flat, METRICS):
                ax.set(xlabel="Requested slate size K" if x_column != "_condition_position" else "Condition position",
                       ylabel=metric, title=metric)
                ax.grid(alpha=.25)
            shown = [f"{c}={_value(controls[c])}" for c in ("rho", "phi", "reader_history_visible") if c in controls]
            fig.suptitle(f"{prefix}Slate-size comparison | {facet_id}\n" + ", ".join(shown)
                         + "\nExact remaining controls: study_figure_facets.csv; lines: study_figure_lines.csv", fontsize=11)
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7,
                           bbox_to_anchor=(.5, .02), title="Exact algorithm/settings lines")
            fig.subplots_adjust(left=.08, right=.97, top=.82, bottom=.30, hspace=.42, wspace=.25)
            _save_figure(fig, base)
            plt.close(fig)
            rendered.append(base.name)

    # Fixed-K MMR sweeps use the actual penalty weight as x, rather than treating
    # the weight as a relevance coefficient or averaging differently named runs.
    if "diversity_weight" in frame and frame["diversity_weight"].notna().any():
        algorithm = frame.get("algorithm", frame["recommender"])
        mmr = frame[algorithm.eq("mmr") & frame["diversity_weight"].notna()]
        mmr_facets = facet_columns + [c for c in ("requested_K", "K") if c in conditions]
        mmr_lines = [c for c in line_columns if c not in ("recommender", "diversity_weight")]
        for facet_number, (facet_key, facet) in enumerate(_groups(mmr, mmr_facets)):
            facet_id = f"M{facet_number:03d}"
            controls = dict(zip(mmr_facets, facet_key))
            facet_rows.append({"facet_id": facet_id, **controls})
            lines = list(_groups(facet, mmr_lines))
            for start in range(0, len(lines), LINES_PER_PAGE):
                base = output / f"study_mmr_weight_{facet_id}_page_{start // LINES_PER_PAGE:02d}"
                fig, axes = plt.subplots(2, 2, figsize=CANVAS, squeeze=False)
                handles, labels = [], []
                for local, (line_key, group) in enumerate(lines[start:start + LINES_PER_PAGE]):
                    if group["diversity_weight"].duplicated().any():
                        raise ValueError("MMR sweep contains duplicate weights within exact controls; refusing to pool")
                    group = group.sort_values("diversity_weight")
                    configuration = dict(zip(mmr_lines, line_key))
                    line_id = f"{facet_id}L{local + start:03d}"
                    line_rows.append({"line_id": line_id, **configuration})
                    first_handle = None
                    for ax, metric in zip(axes.flat, METRICS):
                        if metric not in group:
                            continue
                        points = group[group[metric].notna()]
                        if not len(points):
                            continue
                        ci = points.get(metric + "_ci95", pd.Series(np.nan, index=points.index)).fillna(0)
                        handle = ax.errorbar(points["diversity_weight"], points[metric], yerr=ci,
                                             marker="o", capsize=2, color=colors(local % 20))
                        first_handle = first_handle or handle
                    if first_handle is not None:
                        handles.append(first_handle)
                        labels.append(f"{line_id} MMR; exact settings in study_figure_lines.csv")
                    for _, row in group.iterrows():
                        source_rows.append({"figure": base.name, "facet_id": facet_id, "line_id": line_id,
                                            "condition_id": row["condition_id"], "x_column": "diversity_weight",
                                            "x": row["diversity_weight"]})
                for ax, metric in zip(axes.flat, METRICS):
                    ax.set(xlabel="MMR diversity penalty weight (0 = content ranking)", ylabel=metric, title=metric)
                    ax.grid(alpha=.25)
                shown = [f"{c}={_value(v)}" for c, v in controls.items()
                         if c in ("rho", "phi", "reader_history_visible", "requested_K", "K")]
                fig.suptitle(f"{prefix}MMR penalty sweep | {facet_id}\n" + ", ".join(shown), fontsize=11)
                if handles:
                    fig.legend(handles, labels, loc="lower center", fontsize=7,
                               bbox_to_anchor=(.5, .04), ncol=2)
                fig.subplots_adjust(left=.08, right=.97, top=.83, bottom=.23, hspace=.4, wspace=.25)
                _save_figure(fig, base)
                plt.close(fig)
                rendered.append(base.name)
    for filename, records in (("study_figure_lines.csv", line_rows),
                              ("study_figure_facets.csv", facet_rows),
                              ("study_figure_sources.csv", source_rows)):
        pd.DataFrame(records).to_csv(output / filename, index=False)
    (output / "study_figure_policy.json").write_text(json.dumps({
        "canvas_inches": list(CANVAS), "max_lines_per_page": LINES_PER_PAGE,
        "metrics": list(METRICS), "files": rendered,
        "notes": "No conditions pooled. Time-curve values remain in curves_summary.csv; large-study figures focus on K and MMR-weight comparisons."
    }, indent=2) + "\n")
    return rendered
