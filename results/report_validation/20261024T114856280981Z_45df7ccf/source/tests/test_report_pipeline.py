"""End-to-end archive, replay, provenance, and offline report checks."""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_report import synthetic_world
from src.sim.artifacts import PopulationArchive, load_checkpoint, load_state, load_world
from src.sim.experiment import run_grid
from src.sim.news_users import sample_news_users
from src.sim.params import Params
from src.sim.recommenders import REGISTRY
from src.sim.reporting import rebuild_report
from src.sim.run_bundle import RunBundle, file_digest, write_manifest
from src.sim.simulate import run_population


@pytest.mark.parametrize("name", list(REGISTRY))
def test_recording_preserves_trajectory_and_replays_every_round(tmp_path, name):
    world = synthetic_world(.5, N=10, dimensions=4, pool_size=4)
    profiles = world.user_pool["profiles"][:3]
    params = Params(T=3, phi=.5)
    expected = run_population(world, params, 3, np.random.default_rng(9),
                              REGISTRY[name](K=3), profiles=profiles)
    archive = PopulationArchive(tmp_path / name)
    actual = run_population(world, params, 3, np.random.default_rng(9),
                            REGISTRY[name](K=3), profiles=profiles, recorder=archive)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    reloaded_world = load_world(archive.path)
    for t in range(params.T):
        state = load_state(archive.path, t, reloaded_world)
        rec = load_checkpoint(archive.path / "models" / f"before_{t:04d}.npz", reloaded_world)
        prediction = rec.recommend(state)
        with np.load(archive.path / "rounds" / f"round_{t:04d}_output.npz", allow_pickle=False) as output:
            if prediction is None:
                assert output["slate_is_none"]
            else:
                np.testing.assert_array_equal(prediction, output["slates"])


def test_rebuild_uses_saved_data_without_simulator_and_preserves_raw_files(tmp_path, monkeypatch):
    recs = {"content": partial(REGISTRY["content"], K=3),
            "mmr_0": partial(REGISTRY["mmr"], K=3, diversity_weight=0.)}
    grid = {"rho": [.5], "phi": [0., .5], "info": ["none"]}
    run_grid(partial(synthetic_world, N=10, dimensions=4, pool_size=5), grid,
             base={"T": 3}, U=3, n_pop=2, seed=8, recommenders=recs,
             users_fn=partial(sample_news_users, U=3), archive_dir=tmp_path)
    raw = {str(p): file_digest(p) for p in (tmp_path / "populations").rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Report generation must not rerun a simulation")

    monkeypatch.setattr("src.sim.experiment.run_population", forbidden)
    result = rebuild_report(tmp_path)
    assert result["populations"] == 8
    assert result["conditions"] == 4
    summary = pd.read_csv(tmp_path / "report" / "summary.csv")
    assert (summary["n_populations"] == 2).all()
    assert set(summary["data_kind"]) == {"synthetic"}
    first = summary.copy()
    rebuild_report(tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "report" / "summary.csv"), first)
    assert raw == {str(p): file_digest(p) for p in (tmp_path / "populations").rglob("*") if p.is_file()}
    assert list((tmp_path / "report" / "figures").glob("*.png"))
    assert list((tmp_path / "report" / "figures").glob("*.svg"))
    assert (tmp_path / "report" / "metrics_definitions.json").exists()


def test_run_bundle_saves_logs_provenance_failure_and_manifest(tmp_path, capsys):
    bundle = RunBundle(tmp_path, {"test": True}, {"seed": 4}, "synthetic")
    with pytest.raises(RuntimeError, match="intentional failure"):
        with bundle:
            print("saved stdout")
            print("saved stderr", file=sys.stderr)
            raise RuntimeError("intentional failure")
    assert json.loads((bundle.path / "run.json").read_text())["status"] == "failed"
    log = (bundle.path / "run.log").read_text()
    assert "saved stdout" in log and "saved stderr" in log
    assert "RuntimeError: intentional failure" in log
    assert (bundle.path / "source" / "src" / "sim" / "recommenders.py").exists()
    manifest = json.loads((bundle.path / "manifest.json").read_text())
    for record in manifest["files"]:
        assert file_digest(bundle.path / record["path"]) == record["sha256"]
    other = RunBundle(tmp_path, {}, {}, "synthetic")
    assert other.path != bundle.path


def test_grid_saves_failed_population_and_keeps_initial_checkpoint(tmp_path, monkeypatch):
    def broken(self, state):
        raise RuntimeError("recommendation failed")

    monkeypatch.setattr(REGISTRY["none"], "recommend", broken)
    with pytest.raises(RuntimeError, match="recommendation failed"):
        run_grid(partial(synthetic_world, N=6, dimensions=3),
                 {"rho": [.5]}, base={"T": 2}, U=2, n_pop=1, archive_dir=tmp_path)
    directory = tmp_path / "populations" / "job_000000"
    assert json.loads((directory / "status.json").read_text())["status"] == "failed"
    assert (directory / "models" / "initial.npz").exists()
    assert (directory / "rounds" / "round_0000_input.npz").exists()


def test_parallel_population_archives_have_distinct_metadata(tmp_path):
    df, _ = run_grid(partial(synthetic_world, N=8, dimensions=3),
                     {"rho": [.5]}, base={"T": 2}, U=2, n_pop=2,
                     recommenders={"popularity": partial(REGISTRY["popularity"], K=2)},
                     archive_dir=tmp_path, workers=2)
    assert len(df) == 2
    directories = sorted((tmp_path / "populations").iterdir())
    assert len(directories) == 2
    metadata = [json.loads((p / "metadata.json").read_text()) for p in directories]
    assert {m["population"] for m in metadata} == {0, 1}
    assert len({m["seed"] for m in metadata}) == 2
    assert all((p / "run.npz").exists() for p in directories)
