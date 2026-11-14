import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, indent=2, default=str) + "\n"


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root):
    root = Path(root)
    files = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
              "sha256": file_digest(p)}
             for p in sorted(root.rglob("*"))
             if p.is_file() and p != root / "manifest.json"]
    (root / "manifest.json").write_text(_json({"schema_version": 1,
                                              "updated_utc": _now(), "files": files}))


def verify_manifest(root, ignore_report=True):

    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported file manifest schema")
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid path in file manifest")
        if ignore_report and relative.parts and relative.parts[0] == "report":
            continue
        path = root / relative
        if not path.is_file() or path.stat().st_size != record["bytes"] \
                or file_digest(path) != record["sha256"]:
            raise ValueError(f"Saved artifact integrity check failed: {relative}")
    return True


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.log.flush()


class RunBundle:


    def __init__(self, output_base, config, arguments, data_kind):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = Path(output_base) / f"{stamp}_{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {"schema_version": 1, "started_utc": _now(),
                         "status": "running", "data_kind": data_kind,
                         "arguments": arguments, "config": config}

    def __enter__(self):
        self.log = (self.path / "run.log").open("w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(self.stdout, self.log), _Tee(self.stderr, self.log)
        try:
            self._capture_provenance()
            self._save_metadata()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _save_metadata(self):
        (self.path / "run.json").write_text(_json(self.metadata))

    def _capture_provenance(self):
        repo = Path(__file__).resolve().parents[2]
        environment = {"python": sys.version, "executable": sys.executable,
                       "platform": platform.platform(),
                       "packages": sorted({f"{d.metadata['Name']}=={d.version}"
                                           for d in importlib.metadata.distributions()})}
        (self.path / "environment.json").write_text(_json(environment))
        (self.path / "config.json").write_text(_json(self.metadata["config"]))
        for name, args in {"git_revision.txt": ["rev-parse", "HEAD"],
                           "git_status.txt": ["status", "--short", "--branch"],
                           "working_tree.patch": ["diff", "--binary", "HEAD"]}.items():
            result = subprocess.run(["git", *args], cwd=repo, text=True,
                                    capture_output=True, check=False)
            (self.path / name).write_text(result.stdout + result.stderr)
        source = self.path / "source"
        source.mkdir()
        for folder in ("src", "scripts", "tests", "docs", "configs"):
            for original in sorted((repo / folder).rglob("*")):
                if original.is_file() and "__pycache__" not in original.parts \
                        and original.suffix in (".py", ".md", ".yaml", ".json"):
                    target = source / original.relative_to(repo)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, target)
        for name in ("README.md", "requirements.txt", "config.yaml", "pytest.ini"):
            if (repo / name).is_file():
                shutil.copy2(repo / name, source / name)
        processed = self.metadata["config"].get("paths", {}).get("processed_dir")
        inputs = []
        if processed and self.metadata["data_kind"] == "mind":
            directory = Path(processed)
            for p in sorted(directory.rglob("*")):
                if p.is_file():
                    inputs.append({"path": str(p.resolve()), "bytes": p.stat().st_size,
                                   "sha256": file_digest(p)})
        (self.path / "input_files.json").write_text(_json(inputs))

    def __exit__(self, error_type, error, tb):
        if error is not None:
            traceback.print_exception(error_type, error, tb, file=sys.stderr)
        self.metadata.update(status="failed" if error is not None else "complete",
                             finished_utc=_now())
        if error is not None:
            self.metadata["error"] = {"type": error_type.__name__, "message": str(error)}
        try:
            self._save_metadata()
        finally:
            sys.stdout, sys.stderr = self.stdout, self.stderr
            self.log.close()
        write_manifest(self.path)
        return False
