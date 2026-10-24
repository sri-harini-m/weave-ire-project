"""Lossless local population archives and safe classical-model checkpoints.

Arrays use NPZ without pickle. JSON stores schemas and metadata. Hidden simulator
values are saved separately from observable recommender inputs.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from .recommenders import REGISTRY, SimState
from .world import World


SCHEMA_VERSION = 1
WORLD_FIELDS = ("N", "D", "C", "rho", "unit", "cats", "ids", "popularity",
                "profile_vecs", "kind")
BIT_GENERATORS = {name: getattr(np.random, name) for name in
                  ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64")}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Cannot archive JSON value of type {type(value).__name__}")


def _json_text(value):
    return json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(_json_text(value) + "\n")


def _write_npz(path, values, metadata=None):
    """Store numeric/string arrays, preserving object arrays through JSON."""
    arrays, objects = {}, {}
    for key, value in values.items():
        if value is None:
            continue
        array = np.asarray(value)
        if array.dtype.hasobject:
            objects[key] = {"shape": list(array.shape), "values": array.tolist()}
        else:
            arrays[key] = array
    arrays["__metadata__"] = np.asarray(_json_text({
        "schema_version": SCHEMA_VERSION, "objects": objects,
        "metadata": metadata or {},
    }))
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        header = json.loads(str(stored["__metadata__"]))
        if header["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported archive schema")
        arrays = {key: stored[key].copy() for key in stored.files
                  if key != "__metadata__"}
    for key, value in header.get("objects", {}).items():
        arrays[key] = np.asarray(value["values"], dtype=object).reshape(value["shape"])
    return arrays, header.get("metadata", {})


def world_digest(world):
    """Identify all catalogue fields, including article order and embeddings."""
    digest = hashlib.sha256()
    for name in WORLD_FIELDS:
        value = getattr(world, name)
        digest.update(name.encode())
        if value is None:
            digest.update(b"null")
            continue
        array = np.asarray(value)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        if array.dtype.hasobject:
            digest.update(_json_text(array.tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def save_world(world, path):
    path = Path(path)
    fields = {name: getattr(world, name) for name in WORLD_FIELDS}
    fields["L"] = world.L
    _write_npz(path, fields, {"absent": [k for k, v in fields.items() if v is None],
                              "digest": world_digest(world)})


def load_world(path):
    path = Path(path)
    if path.is_dir():
        path = path / "world.npz"
    values, metadata = _read_npz(path)
    fields = {name: values.get(name) for name in WORLD_FIELDS}
    for name in ("N", "rho", "unit", "kind"):
        fields[name] = fields[name].item()
    world = World(**fields)
    # Retain the exact original matrix square root, including eigenvector signs.
    world.L = values["L"]
    if world_digest(world) != metadata["digest"]:
        raise ValueError("World archive digest mismatch")
    article_path = path.parent / "article_metadata.json"
    if article_path.exists():
        world.article_metadata = json.loads(article_path.read_text(encoding="utf-8"))
    return world


def _model_settings(rec):
    if rec.name not in REGISTRY or type(rec) is not REGISTRY[rec.name]:
        raise ValueError("Checkpointing supports only registered recommender classes")
    keys = ("K", "prior_weight", "profile_weight", "diversity_weight")
    return {key: getattr(rec, key) for key in keys if hasattr(rec, key)}


def save_checkpoint(rec, path):
    """Save learned state and RNG; static embedding data remains in the world."""
    generator = type(rec.rng.bit_generator).__name__
    if generator not in BIT_GENERATORS:
        raise ValueError("Unsupported random bit generator")
    metadata = {"algorithm": rec.name, "settings": _model_settings(rec),
                "U": rec.U, "world_digest": world_digest(rec.world),
                "rng": rec.rng.bit_generator.state}
    values = {"counts": rec.counts} if hasattr(rec, "counts") else {}
    _write_npz(path, values, metadata)


def load_checkpoint(path, world):
    """Restore one known algorithm without importing arbitrary classes."""
    values, metadata = _read_npz(path)
    if metadata["world_digest"] != world_digest(world):
        raise ValueError("Checkpoint world does not match the supplied catalogue")
    algorithm = metadata["algorithm"]
    if algorithm not in REGISTRY:
        raise ValueError("Unknown checkpoint recommender")
    state = metadata["rng"]
    generator = state["bit_generator"]
    if generator not in BIT_GENERATORS:
        raise ValueError("Unsupported random bit generator")
    rng = np.random.Generator(BIT_GENERATORS[generator]())
    rec = REGISTRY[algorithm](**metadata["settings"])
    rec.reset(world, metadata["U"], rng)
    if "counts" in values:
        counts = values["counts"]
        if counts.shape != (world.N,) or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Invalid learned counts in checkpoint")
        rec.counts = counts
    # reset currently draws no values; setting after reset also supports future resets.
    rng.bit_generator.state = state
    return rec


def load_state(population_path, t, world=None):
    """Reload the exact observable input saved before round t."""
    root = Path(population_path)
    world = load_world(root) if world is None else world
    prefix = root / "rounds" / f"round_{t:04d}_input"
    arrays, _ = _read_npz(prefix.with_suffix(".npz"))
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    extra = metadata["extra"]
    if "profiles" in arrays:
        extra["profiles"] = arrays["profiles"]
    return SimState(world, metadata["t"], arrays["available"], metadata["history"],
                    arrays["ratings"], arrays["active"], extra)


class PopulationArchive:
    """Flush one population's inputs, model snapshots, and responses per round.

    This stores enough data to recompute reports and reproduce recommendations.
    It is not a simulator-resume implementation.
    """

    def __init__(self, path, metadata=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "models").mkdir()
        (self.path / "rounds").mkdir()
        self.metadata = metadata or {}
        self.started = False
        self.finished = False
        self._pending = None
        self._rec = None
        self._rounds = 0
        self._clock = perf_counter()
        self._created = datetime.now(timezone.utc).isoformat()
        _write_json(self.path / "metadata.json", self.metadata)
        (self.path / "responses.jsonl").touch(exist_ok=False)
        self._status("created")

    def _status(self, state, **extra):
        document = {"schema_version": SCHEMA_VERSION, "status": state,
                    "created_utc": self._created,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "rounds_saved": self._rounds,
                    "elapsed_seconds": perf_counter() - self._clock, **extra}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path,
                                         prefix=".status-", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(_json_text(document) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, self.path / "status.json")
        self._log(state)

    def _log(self, text):
        with (self.path / "population.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now(timezone.utc).isoformat()} {text}\n")
            stream.flush()

    def start(self, world, params, rec, rng, profiles, seen, initial):
        if self.started or self.finished:
            raise ValueError("Population archive already started or closed")
        self.world, self._rec = world, rec
        save_world(world, self.path / "world.npz")
        _write_json(self.path / "world.json", {"N": world.N, "rho": world.rho,
                    "unit": world.unit, "kind": world.kind,
                    "digest": world_digest(world)})
        if hasattr(world, "article_metadata"):
            articles = world.article_metadata
            if hasattr(articles, "to_dict"):
                articles = articles.to_dict(orient="records")
            _write_json(self.path / "article_metadata.json", articles)
        _write_json(self.path / "params.json", params.as_dict())
        observable = {"profiles": profiles, "seen": seen}
        _write_npz(self.path / "initial_observable.npz", observable,
                   {"profiles_present": profiles is not None, "seen_present": seen is not None,
                    "U": rec.U, "rng": rng.bit_generator.state})
        _write_npz(self.path / "initial_hidden.npz", initial,
                   {"purpose": "Simulator diagnostics only; never recommender inputs"})
        save_checkpoint(rec, self.path / "models" / "initial.npz")
        self.started = True
        self._status("running")

    def before_round(self, state, rec, diagnostics=None):
        if not self.started or self.finished or self._pending is not None:
            raise ValueError("Archive is not ready for another round")
        if state.t != self._rounds:
            raise ValueError("Archive rounds must be sequential and start at zero")
        prefix = self.path / "rounds" / f"round_{state.t:04d}_input"
        arrays = {"available": state.available, "active": state.active,
                  "ratings": state.ratings, "profiles": state.extra.get("profiles")}
        arrays.update({f"diagnostic_{key}": value for key, value in (diagnostics or {}).items()})
        _write_npz(prefix.with_suffix(".npz"), arrays)
        _write_json(prefix.with_suffix(".json"), {"t": state.t, "history": state.history,
                    "extra": {k: v for k, v in state.extra.items() if k != "profiles"}})
        save_checkpoint(rec, self.path / "models" / f"before_{state.t:04d}.npz")
        self._pending = state.t
        self._rec = rec

    def after_round(self, t, slates, users, items, feedback, rec, diagnostics=None):
        if self.finished or self._pending != t:
            raise ValueError("Output must match the pending archived round")
        arrays = {"slates": np.empty((rec.U, 0), dtype=int) if slates is None else slates,
                  "slate_is_none": slates is None, "users": users, "items": items,
                  "feedback": feedback}
        arrays.update({f"diagnostic_{key}": value for key, value in (diagnostics or {}).items()})
        _write_npz(self.path / "rounds" / f"round_{t:04d}_output.npz", arrays)
        save_checkpoint(rec, self.path / "models" / f"after_{t:04d}.npz")
        response = {"t": t, "slate_is_none": slates is None, "slates": slates,
                    "users": users, "items": items, "feedback": feedback}
        with (self.path / "responses.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(response), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._pending = None
        self._rounds += 1
        self._rec = rec
        self._status("running")

    def finish(self, run):
        if not self.started or self.finished or self._pending is not None:
            raise ValueError("Cannot finish an unstarted, incomplete, or closed archive")
        _write_npz(self.path / "run.npz", run)
        save_checkpoint(self._rec, self.path / "models" / "final.npz")
        from .metrics import time_curves, user_metrics
        user_metrics(self.world, run).to_csv(self.path / "user_metrics.csv", index=False)
        curves = time_curves(self.world, run).reindex(
            columns=["t", "step_distance", "fatigue", "active_share"])
        curves.to_csv(self.path / "curves.csv", index=False)
        self.finished = True
        self._status("complete")

    def fail(self, error):
        if self.finished:
            raise ValueError("Cannot mark a complete archive as failed")
        self.finished = True
        self._status("failed", error_type=type(error).__name__, error=str(error))
