"""Exact matched comparisons and bounded study figures, without simulation."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.reporting import (_figures, _metadata, _observed_metrics,
                               paired_comparisons, population_report)
from src.sim.study_figures import study_figures
from src.sim.world import World


def test_study_metadata_keeps_controls_and_ignores_storage_policy(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "params": {"reader_history_visible": True, "boredom_threshold": .25, "gamma": 2.},
        "recommender_settings": {"K": 500, "reference_only": True},
        "requested_K": 500, "reference_only": True, "recommender": "ideal", "algorithm": "ideal",
        "archive": {"snapshot_policy": "none", "diagnostics": False, "shared_world_reference": "worlds/x.npz"}}))
    row = _metadata(tmp_path)
    assert row["requested_K"] == 500 and row["reference_only"] is True
    assert row["reader_history_visible"] is True and row["boredom_threshold"] == .25
    assert row["gamma"] == 2. and row["algorithm"] == "ideal"
    assert row["archive"] == str(tmp_path)
    assert not {"snapshot_policy", "diagnostics", "shared_world_reference"} & row.keys()


def test_light_result_arrays_need_no_hidden_preferences():
    world = World(N=2, D=np.array([[0., 1.], [1., 0.]]), C=np.eye(2), rho=0.,
                  cats=np.array(["a", "b"]))
    light = {"seq": np.array([[0, 1]]), "val": np.array([[3., 4.]]),
             "pen": np.array([[1., 1.]]), "left_at": np.array([2])}
    users, metrics, curves = population_report(world, light)
    assert "X" not in light and "V" not in light
    assert metrics["welfare_net"] == 2.5
    assert metrics["catalog_coverage"] == 1.
    assert len(users) == 1 and len(curves) == 2


def row(algorithm, value, history=False, **overrides):
    values = {"recommender": algorithm, "algorithm": algorithm, "rho": .5, "phi": 1.,
              "gamma": 1., "reader_history_visible": history, "boredom_threshold": .25,
              "requested_K": 10, "K": 10, "reference_only": False,
              "seed": 42, "population": 0, "data_kind": "synthetic", "welfare_net": value}
    values.update(overrides)
    return values


def comparisons(rows):
    frame = pd.DataFrame(rows)
    conditions = [c for c in frame if c not in ("welfare_net", "seed", "population", "archive")]
    model = {"K", "prior_weight", "profile_weight", "fatigue_weight", "decay", "factors",
             "regularization", "iterations", "reference_only"}
    return paired_comparisons(frame, conditions, ["welfare_net"], model)


def test_paired_differences_match_seeds_population_and_model_controls():
    tables = comparisons([
        row("none", 1.), row("none", 1.5, history=True),
        row("content", 2., profile_weight=1.), row("content", 4., history=True, profile_weight=1.),
        row("content", 99., history=True, rho=.9, profile_weight=1.),
        row("fatigue_content", 5., profile_weight=1., fatigue_weight=1., decay=.8),
        row("mf", 2., factors=4, regularization=1., iterations=3),
        row("fatigue_mf", 3., factors=4, regularization=1., iterations=3, fatigue_weight=1.),
        row("fatigue_mf", 7., factors=8, regularization=1., iterations=3, fatigue_weight=1.),
        row("ideal", 8., reference_only=True),
    ])
    history = tables["paired_history_visibility"]
    content = history[history["left_recommender"].eq("content")]
    assert content["welfare_net_difference"].tolist() == [2.]
    assert content["reader_history_visible"].tolist() == [True]
    fatigue = tables["paired_fatigue_feedback"]
    assert set(fatigue["left_recommender"]) == {"fatigue_content", "fatigue_mf"}
    assert sorted(fatigue["welfare_net_difference"].tolist()) == [1., 3.]
    assert fatigue.loc[fatigue["left_recommender"].eq("fatigue_mf"), "factors"].tolist() == [4.]
    assert fatigue.attrs["unmatched_left_rows"] == 1
    baseline = tables["paired_vs_none"]
    assert baseline.loc[baseline["left_recommender"].eq("content"), "welfare_net_difference"].tolist() == [1., 2.5]
    ideal = baseline[baseline["left_recommender"].eq("ideal")]
    assert ideal["reference_only"].tolist() == [True]
    assert ideal["welfare_net_difference"].tolist() == [7.]


def test_paired_comparisons_do_not_match_different_seeds_or_gamma():
    tables = comparisons([row("none", 1.), row("content", 4., seed=43), row("content", 5., gamma=2.)])
    assert tables["paired_vs_none"].empty
    assert tables["paired_vs_none"].attrs["unmatched_left_rows"] == 2


def test_paired_comparisons_reject_ambiguous_baselines():
    with pytest.raises(ValueError, match="Ambiguous matched baseline"):
        comparisons([row("none", 1.), row("none", 2.), row("content", 4.)])


def test_observable_signal_metrics_use_recorded_flags_and_times(tmp_path):
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    np.savez_compressed(rounds / "round_0000_output.npz", feedback_boredom=np.array([1., 0.]),
                        feedback_left_users=np.array([], dtype=int), diagnostic_recommendation_seconds=.2)
    np.savez_compressed(rounds / "round_0001_output.npz", feedback_boredom=np.array([1.]),
                        feedback_left_users=np.array([1]), diagnostic_recommendation_seconds=.3)
    curves, metrics = pd.DataFrame({"t": [0, 1]}), {}
    _observed_metrics(tmp_path, curves, metrics)
    assert metrics["reported_boredom_rate"] == pytest.approx(2 / 3)
    assert metrics["reported_departures"] == 1.
    assert metrics["total_recommendation_seconds"] == .5
    assert curves["reported_boredom_rate"].tolist() == [.5, 1.]
    assert curves["departures"].tolist() == [0, 1]


def large_summary():
    names = ["none", "random", "popularity", "content", "mf", "fatigue_popularity", "fatigue_content",
             "fatigue_mf", "calibrated", "bm25", "ideal"] + [f"mmr_{w:g}" for w in (0., .3, .6, .9, 1.)]
    rows = []
    for k in (5, 10, 25, 50, 500):
        for history in (False, True):
            for phi in (0., 1.):
                for name in names:
                    algorithm = "mmr" if name.startswith("mmr_") else name
                    record = {"condition_id": f"C{len(rows):03d}", "recommender": name, "algorithm": algorithm,
                              "requested_K": k, "K": k, "rho": .5, "phi": phi, "reader_history_visible": history,
                              "gamma": 1., "boredom_threshold": .25, "data_kind": "synthetic",
                              "reference_only": algorithm == "ideal",
                              "diversity_weight": float(name[4:]) if algorithm == "mmr" else np.nan}
                    for metric in ("cat_entropy", "homogeneity", "welfare_net", "fatigue"):
                        record[metric] = .2 + k / 1000
                        record[metric + "_ci95"] = .01
                    rows.append(record)
    return pd.DataFrame(rows)


def test_large_study_dispatch_has_bounded_canvases_and_exact_controls(tmp_path, monkeypatch):
    from src.sim import study_figures as module
    summary = large_summary()
    captured = []
    monkeypatch.setattr(module, "_save_figure", lambda fig, base: captured.append(
        (tuple(fig.get_size_inches()), fig._suptitle.get_text())))
    condition_columns = [c for c in summary if c not in ("condition_id", "cat_entropy", "homogeneity", "welfare_net", "fatigue")
                         and not c.endswith("_ci95")]
    _figures(summary, pd.DataFrame(), tmp_path, condition_columns, {"K", "diversity_weight"})
    assert len(summary) == 320
    assert captured and all(width <= 15 and height <= 10 for (width, height), _ in captured)
    assert all("SYNTHETIC" in title for _, title in captured)
    facets = pd.read_csv(tmp_path / "study_figure_facets.csv")
    primary = facets[facets["facet_id"].str.startswith("F")]
    assert len(primary) == 4
    assert set(zip(primary["phi"], primary["reader_history_visible"])) == {(0., False), (0., True), (1., False), (1., True)}
    lines = pd.read_csv(tmp_path / "study_figure_lines.csv")
    mmr = lines[lines["recommender"].fillna("").str.startswith("mmr_")]
    assert set(mmr["diversity_weight"]) == {0., .3, .6, .9, 1.}
    sources = pd.read_csv(tmp_path / "study_figure_sources.csv")
    main_sources = sources[sources["figure"].str.startswith("study_slate_size")]
    assert len(main_sources) == 320
    assert main_sources["condition_id"].nunique() == 320


def test_study_figures_do_not_pool_extra_simulator_controls(tmp_path, monkeypatch):
    from src.sim import study_figures as module
    monkeypatch.setattr(module, "_save_figure", lambda fig, base: None)
    summary = large_summary().query("recommender == 'content' and reader_history_visible == False and phi == 0").copy()
    extra = summary.copy()
    extra["gamma"] = 2.
    extra["condition_id"] = "extra_" + extra["condition_id"]
    summary = pd.concat([summary, extra], ignore_index=True)
    conditions = ["recommender", "algorithm", "K", "requested_K", "rho", "phi", "gamma",
                  "reader_history_visible", "boredom_threshold", "reference_only", "data_kind"]
    study_figures(summary, tmp_path, conditions, {"K"})
    facets = pd.read_csv(tmp_path / "study_figure_facets.csv")
    assert sorted(facets["gamma"].tolist()) == [1., 2.]


def test_study_plot_rejects_duplicate_x_values_in_exact_line(tmp_path, monkeypatch):
    from src.sim import study_figures as module
    monkeypatch.setattr(module, "_save_figure", lambda fig, base: None)
    source = large_summary().iloc[[0]]
    duplicate = pd.concat([source, source], ignore_index=True)
    with pytest.raises(ValueError, match="refusing to pool"):
        study_figures(duplicate, tmp_path,
                      ["recommender", "K", "requested_K", "rho", "phi", "reader_history_visible"], {"K"})
