"""Known-answer metrics and offline report outputs, without simulation runs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sim.reporting import _summarize, population_report, rebuild_report
from src.sim.world import World


def world():
    return World(N=4, D=np.ones((4, 4)) - np.eye(4), C=np.eye(4), rho=0.,
                 cats=np.array(["sports", "sports", "politics", "politics"]),
                 ids=np.array(["a", "b", "c", "d"]), profile_vecs=np.eye(4), kind="news")


def run():
    return {"seq": np.array([[0, 1], [0, -1]]),
            "val": np.array([[4., 6.], [2., np.nan]]),
            "pen": np.array([[1., 2.], [.5, np.nan]]),
            "left_at": np.array([2, 1]), "X": np.zeros((2, 4)), "V": np.zeros(4)}


def test_known_consumption_metrics_and_actual_activity():
    users, metrics, curves = population_report(world(), run())
    assert metrics["catalog_coverage"] == .5
    assert metrics["consumption_gini"] == pytest.approx(7 / 12)
    assert metrics["consumption_hhi"] == pytest.approx(5 / 9)
    assert metrics["homogeneity"] == .5
    assert metrics["dropout_share"] == .5
    assert metrics["n_consumed"] == 1.5
    assert metrics["consumption_share"] == .75
    assert metrics["total_read_events"] == 3
    assert metrics["total_utility_net"] == 8.5
    assert metrics["cumulative_welfare_net"] == 4.25
    assert metrics["cat_entropy"] == 0
    assert metrics["dominant_category_share"] == 1
    assert curves["active_share"].tolist() == [1., .5]
    assert curves["consumption_share"].tolist() == [1., .5]
    assert np.isnan(metrics["exposure_events"])


def test_exposure_and_consumption_are_separate():
    slates = [np.array([[0, 2], [0, 3]]), np.array([[1, 2], [-1, -1]])]
    _, metrics, _ = population_report(world(), run(), slates)
    assert metrics["catalog_coverage"] == .5
    assert metrics["exposure_coverage"] == 1
    assert metrics["exposure_events"] == 6
    assert metrics["slate_ild"] == 1
    assert metrics["slate_category_coverage"] == 1
    assert metrics["slate_category_entropy"] == 1


def test_empty_reads_and_missing_categories_have_missing_averages():
    w = world()
    w.cats = None
    r = {"seq": np.full((1, 1), -1), "val": np.full((1, 1), np.nan),
         "pen": np.full((1, 1), np.nan), "left_at": np.array([1])}
    _, metrics, curves = population_report(w, r, [None])
    assert np.isnan(metrics["welfare_net"])
    assert np.isnan(metrics["consumption_gini"])
    assert np.isnan(metrics["homogeneity"])
    assert np.isnan(metrics["exposure_coverage"])
    assert metrics["total_read_events"] == 0
    assert curves["active_share"].tolist() == [1.]
    assert curves["consumption_share"].tolist() == [0.]


def test_confidence_intervals_keep_all_conditions_separate():
    frame = pd.DataFrame({"recommender": ["mmr"] * 3, "phi": [0., 0., 1.],
                          "diversity_weight": [.3] * 3, "welfare": [1., 3., 8.]})
    summary = _summarize(frame, ["recommender", "phi", "diversity_weight"], ["welfare"])
    first, second = summary.iloc[0], summary.iloc[1]
    assert first["welfare"] == 2.
    assert first["welfare_ci95"] == pytest.approx(1.96)
    assert first["welfare_n"] == 2
    assert second["welfare"] == 8.
    assert np.isnan(second["welfare_ci95"])


def test_reporting_rejects_wrong_slate_shapes_and_indices():
    with pytest.raises(ValueError):
        population_report(world(), run(), [np.array([[0], [4]]), None])
    with pytest.raises(ValueError):
        population_report(world(), run(), [np.array([[0, 0], [1, 2]]), None])


def test_report_refuses_incomplete_population_archives(tmp_path):
    directory = tmp_path / "populations" / "job_00001"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text("{}")
    (directory / "status.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="incomplete population archives"):
        rebuild_report(tmp_path)


def test_report_refuses_missing_planned_populations(tmp_path):
    completed = tmp_path / "populations" / "job_000000"
    completed.mkdir(parents=True)
    (completed / "status.json").write_text(json.dumps({"status": "complete"}))
    np.savez_compressed(completed / "run.npz", **run())
    (tmp_path / "expected_populations.json").write_text(json.dumps({"schema_version": 1,
        "jobs": [{"directory": "job_000000", "job_id": 0},
                 {"directory": "job_000001", "job_id": 1}]}))
    with pytest.raises(ValueError, match="planned populations are incomplete or missing: job_000001"):
        rebuild_report(tmp_path)


def test_report_refuses_failed_root_run(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="experiment run is marked failed"):
        rebuild_report(tmp_path)


def test_offline_rebuild_saves_tables_and_vector_figures(tmp_path, monkeypatch):
    # No simulator is invoked. Only known saved arrays are loaded.
    from src.sim import artifacts
    monkeypatch.setattr(artifacts, "load_world", lambda directory: world())
    for pop in range(2):
        directory = tmp_path / "populations" / f"job_{pop:05d}"
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(json.dumps({"params": {"phi": .5, "info": "none"},
            "rho": 0., "recommender": "content", "population": pop, "seed": pop,
            "data_kind": "synthetic", "recommender_settings": {"K": 2}}))
        np.savez_compressed(directory / "run.npz", **run())
    result = rebuild_report(tmp_path)
    output = tmp_path / "report"
    assert result["populations"] == 2 and result["conditions"] == 1
    for name in ("summary.csv", "populations.csv", "users.csv", "curves.csv",
                 "curves_summary.csv", "conditions.csv", "summary.md", "metrics_definitions.json",
                 "figures/welfare_net.png", "figures/welfare_net.svg",
                 "figures/tradeoff_diversity_welfare.png", "figures/tradeoff_diversity_welfare.svg"):
        assert (output / name).is_file()
    summary = pd.read_csv(output / "summary.csv")
    assert summary.loc[0, "n_populations"] == 2
    assert summary.loc[0, "welfare_net_ci95"] == 0
    assert "Synthetic rows" in (output / "summary.md").read_text()
    # Rebuilding is intentionally idempotent and never changes archived arrays.
    before = (tmp_path / "populations/job_00000/run.npz").read_bytes()
    rebuild_report(tmp_path)
    assert (tmp_path / "populations/job_00000/run.npz").read_bytes() == before
