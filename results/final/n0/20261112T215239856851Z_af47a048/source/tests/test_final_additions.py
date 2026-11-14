"""Tests for the final-submission additions: per-type results, setting columns
not overwritten by per-reader values, and the validation / fatigue helpers."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sim.experiment import run_grid  # noqa: E402
from src.sim.metrics import per_type_metrics, user_metrics  # noqa: E402
from src.sim.params import Params  # noqa: E402
from src.sim.simulate import run_population  # noqa: E402
from src.sim.world import circle_world  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_per_type_metrics_split_by_reader_type():
    w = circle_world(100, 0.5)
    p = Params(T=8, reader_types_enabled=True)
    run = run_population(w, p, 40, np.random.default_rng(0))
    um = user_metrics(w, run)
    out = per_type_metrics(um, run)
    types = set(um["reader_type"])
    assert len(types) > 1
    assert sum(out[f"n_readers__{t}"] for t in types) == 40
    for t in types:
        expected = um.loc[um["reader_type"] == t, "welfare_net"].mean()
        assert np.isclose(out[f"welfare_net__{t}"], expected)
        assert 0 <= out[f"homogeneity__{t}"] <= 1


def test_per_type_metrics_empty_for_homogeneous_readers():
    w = circle_world(100, 0.5)
    run = run_population(w, Params(T=5), 10, np.random.default_rng(0))
    assert per_type_metrics(user_metrics(w, run), run) == {}


def test_grid_settings_not_overwritten_by_reader_values():
    """With reader types, each reader has its own phi/gamma; the population row
    must still report the grid's phi/gamma, not the readers' average."""
    df, _ = run_grid(lambda rho: circle_world(60, rho), {"rho": [0.5], "phi": [0.0, 0.5]},
                     base={"T": 5, "reader_types_enabled": True}, n_pop=2, U=20)
    assert set(df["phi"]) == {0.0, 0.5}
    assert "reader_phi" in df.columns          # readers' average kept under its own name
    assert any(c.startswith("welfare_net__") for c in df.columns)


def test_validation_scoring_helpers():
    v = load_script("validate_reader_choices")
    labels = np.array([0, 1, 0, 0])
    assert v.auc(labels, np.array([0.1, 0.9, 0.2, 0.3])) == 1.0
    assert v.auc(labels, np.array([0.9, 0.1, 0.8, 0.7])) == 0.0
    assert v.top1(labels, np.array([0.1, 0.9, 0.2, 0.3])) == 1.0
    assert v.top1(labels, np.ones(4)) == 0.25          # ties -> expected hit rate
    assert v.mrr(labels, np.array([0.9, 0.5, 0.1, 0.0])) == 0.5


def test_fatigue_streaks_and_choice_model_find_no_effect_in_random_order():
    f = load_script("estimate_fatigue")
    rng = np.random.default_rng(0)
    topics = list("abcdef")
    seqs = [list(rng.choice(topics, size=40, p=[.4, .2, .15, .1, .1, .05])) for _ in range(400)]
    st = f.streak_table(seqs, max_k=4, n_shuffles=3)
    assert np.allclose(st["ratio_observed_to_shuffled"].iloc[0], 1, atol=0.1)
    shuffled = [list(rng.permutation(s)) for s in seqs]
    w, se = f.fit_logit(*f.choice_events(seqs, 0.8, seed=0))
    ws, ses = f.fit_logit(*f.choice_events(shuffled, 0.8, seed=0))
    assert abs((w[2] - ws[2]) / np.hypot(se[2], ses[2])) < 3   # no order effect


def test_fatigue_choice_model_detects_satiation():
    f = load_script("estimate_fatigue")
    rng = np.random.default_rng(1)
    topics = list("abcdef")
    seqs = []
    for _ in range(400):
        s, last = [], None
        for _ in range(40):                                # never repeat the last topic
            s.append(last := rng.choice([t for t in topics if t != last]))
        seqs.append(s)
    shuffled = [list(rng.permutation(s)) for s in seqs]
    w, se = f.fit_logit(*f.choice_events(seqs, 0.5, seed=0))
    ws, ses = f.fit_logit(*f.choice_events(shuffled, 0.5, seed=0))
    assert (w[2] - ws[2]) / np.hypot(se[2], ses[2]) < -3      # satiation detected


def test_analysis_pairs_by_seed(tmp_path):
    a = load_script("analyse_results")
    rows = []
    for seed in range(5):
        for rec, shift in [("none", 0.0), ("content", -0.2)]:
            rows.append({"recommender": rec, "seed": seed, "rho": 0.5,
                         "cat_entropy": 0.5 + 0.01 * seed + shift})
    df = pd.DataFrame(rows)
    p = a.paired_table(df, [], ["cat_entropy"], "none")
    assert np.isclose(p.loc[0, "cat_entropy_diff"], -0.2)
    assert p.loc[0, "cat_entropy_significant"]
