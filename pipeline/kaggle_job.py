"""Run a batch of GPU jobs on a Kaggle notebook and bring the results back.

Flow (shared by the shots and music stages):
  1. write inputs + jobs.json into build/kaggle/<name>/data and publish it as a
     private Kaggle dataset (new version each time),
  2. push the kernel in kaggle/<name>/ with that dataset attached and GPU enabled,
  3. poll `kaggle kernels status` until complete / error / timeout,
  4. download /kaggle/working into build/kaggle/<name>/out and read results.json.

Each job carries the cache key it was submitted with; results are accepted only
when the key matches, so a stale dataset version can never be mistaken for new
output. State is kept in outputs/.cache/kaggle_<name>.json so you can submit,
close the laptop, and `--collect` later.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .common import CACHE_DIR, ROOT, say
from .providers import kaggle_username

BUILD = ROOT / "build" / "kaggle"


def _kaggle(*args: str, check: bool = True) -> str:
    if not shutil.which("kaggle"):
        raise SystemExit("kaggle CLI not found; `uv sync` installs it, run via `uv run` (see README)")
    res = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout + res.stderr


def state_path(name: str) -> Path:
    return CACHE_DIR / f"kaggle_{name}.json"


def load_state(name: str) -> dict:
    p = state_path(name)
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(name: str, state: dict) -> None:
    p = state_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2) + "\n")


def submit(name: str, jobs: list[dict], files: dict[str, Path], settings: dict) -> dict:
    """files: name-in-dataset -> local path. Returns the saved state."""
    user = kaggle_username()
    if not user:
        raise SystemExit("Kaggle username unknown; set KAGGLE_USERNAME or ~/.kaggle/kaggle.json")
    slug = f"tizita-{name.replace('_', '-')}"
    data_dir = BUILD / name / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    for dst, src in files.items():
        shutil.copy2(src, data_dir / dst)
    submitted_at = time.time()
    (data_dir / "jobs.json").write_text(json.dumps({"submitted_at": submitted_at, "settings": settings,
                                                    "jobs": jobs}, ensure_ascii=False, indent=2))
    (data_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}))
    exists = "ready" in _kaggle("datasets", "status", f"{user}/{slug}", check=False).lower()
    if exists:
        _kaggle("datasets", "version", "-p", str(data_dir), "-m", f"jobs {int(submitted_at)}", "-r", "zip")
    else:
        _kaggle("datasets", "create", "-p", str(data_dir), "-r", "zip")
    # Wait for the new dataset version to be processed before the kernel mounts it.
    for _ in range(60):
        if "ready" in _kaggle("datasets", "status", f"{user}/{slug}", check=False).lower():
            break
        time.sleep(10)

    kdir = BUILD / name / "kernel"
    if kdir.exists():
        shutil.rmtree(kdir)
    shutil.copytree(ROOT / "kaggle" / name, kdir)
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    meta["id"] = f"{user}/{meta['id'].split('/')[-1]}"
    meta["dataset_sources"] = [f"{user}/{slug}"]
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    out = _kaggle("kernels", "push", "-p", str(kdir))
    say(f"kaggle:{name}", out.strip().splitlines()[-1] if out.strip() else "pushed")
    state = {"kernel": meta["id"], "dataset": f"{user}/{slug}", "submitted_at": submitted_at,
             "jobs": {j["id"]: j["key"] for j in jobs}, "status": "running"}
    save_state(name, state)
    return state


def status(name: str) -> str:
    st = load_state(name)
    if not st:
        return "none"
    out = _kaggle("kernels", "status", st["kernel"], check=False).lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in out:
            return s
    return out.strip()[:80]


def wait(name: str, timeout_s: int, poll_s: int = 60) -> str:
    t0 = time.time()
    while True:
        s = status(name)
        if s in ("complete", "error", "cancel"):
            return s
        if time.time() - t0 > timeout_s:
            return "timeout"
        say(f"kaggle:{name}", f"status={s}, waited {int(time.time() - t0)}s")
        time.sleep(poll_s)


def collect(name: str) -> tuple[Path, list[dict]]:
    """Download kernel output; return (output dir, results list)."""
    st = load_state(name)
    out = BUILD / name / "out"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    _kaggle("kernels", "output", st["kernel"], "-p", str(out))
    res_file = next(out.rglob("results.json"), None)
    results = json.loads(res_file.read_text())["results"] if res_file else []
    # Keep the kernel log for FAILURES.md evidence.
    for log in out.rglob("*.log"):
        dst = ROOT / "outputs" / "logs" / f"kaggle_{name}_{int(st['submitted_at'])}.log"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, dst)
    st["status"] = "collected"
    save_state(name, st)
    return out, results
