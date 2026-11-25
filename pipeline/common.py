"""Shared plumbing for every stage: storyboard contract, run log, cache, retries.

Every stage reads storyboard.json through `load_storyboard()` (which validates it
against schema/storyboard.schema.json), logs every model call through `log_call()`
into runs.jsonl, and decides whether to regenerate an item through `Cache`.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import jsonschema

ROOT = Path(__file__).resolve().parent.parent
STORYBOARD = ROOT / "storyboard.json"
SCHEMA = ROOT / "schema" / "storyboard.schema.json"
OUTPUTS = ROOT / "outputs"
RUNS = ROOT / "runs.jsonl"
CACHE_DIR = OUTPUTS / ".cache"

try:  # .env is optional: a key-less clone can still run OFFLINE=1
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

OFFLINE = os.environ.get("OFFLINE", "0") not in ("0", "", "false")
RUN_ID = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
os.environ["RUN_ID"] = RUN_ID  # share one id across stage subprocesses


# --------------------------------------------------------------------------- contract
class StoryboardError(Exception):
    pass


def load_storyboard(path: Path = STORYBOARD) -> dict:
    """Load and validate the storyboard. Raises StoryboardError with every problem."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_storyboard(data)
    return data


def validate_storyboard(data: dict) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    problems = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]
    # Semantic checks JSON Schema cannot express.
    ids = [s["id"] for s in data.get("shots", []) if isinstance(s, dict) and "id" in s]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"shots: duplicate ids {sorted(dupes)}")
    char_ids = {c.get("id") for c in data.get("characters", []) if isinstance(c, dict)}
    for s in data.get("shots", []):
        if not isinstance(s, dict):
            continue
        for c in s.get("characters", []):
            if c not in char_ids:
                problems.append(f"shots/{s.get('id')}: unknown character '{c}'")
    total = sum(s.get("duration_s", 0) for s in data.get("shots", []) if isinstance(s, dict))
    total += data.get("end_card", {}).get("duration_s", 0)
    lo, hi = 60, 90
    if not (lo <= total <= hi):
        problems.append(f"planned runtime {total:.1f}s is outside {lo}-{hi}s")
    if problems:
        raise StoryboardError("storyboard.json is invalid:\n  - " + "\n  - ".join(problems))


def save_storyboard(data: dict, path: Path = STORYBOARD) -> None:
    validate_storyboard(data)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shots_filter(storyboard: dict, only: str | None) -> list[dict]:
    """`only` is a comma-separated list of shot ids (from SHOT=... in the Makefile)."""
    shots = storyboard["shots"]
    if not only:
        return shots
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    unknown = wanted - {s["id"] for s in shots}
    if unknown:
        raise SystemExit(f"unknown shot id(s): {sorted(unknown)}")
    return [s for s in shots if s["id"] in wanted]


# --------------------------------------------------------------------------- hashing
def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, Path):
            h.update(file_hash(p).encode() if p.exists() else b"<missing>")
        else:
            h.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- cache
class Cache:
    """Per-stage manifest: item id -> {key, path}. An item is fresh when the key of
    its current inputs matches the stored key and the output file still exists."""

    def __init__(self, stage: str):
        self.stage = stage
        self.path = CACHE_DIR / f"{stage}.json"
        self.data: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def fresh(self, item: str, key: str) -> Path | None:
        entry = self.data.get(item)
        if entry and entry["key"] == key:
            p = ROOT / entry["path"]
            if p.exists():
                return p
        return None

    def put(self, item: str, key: str, out: Path, **extra: Any) -> None:
        self.data[item] = {"key": key, "path": str(out.relative_to(ROOT)), **extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def get(self, item: str) -> dict | None:
        return self.data.get(item)


# --------------------------------------------------------------------------- run log
# List-price equivalents in USD. These are ESTIMATES for COSTS.md; every value was
# entered by hand from the provider's public pricing page and must be re-checked.
# Keys are matched by prefix against the model id.
LIST_PRICE = {
    # Gemini paid tier, per call, at our typical size (~300 in / ~400 out tokens).
    "gemini-text": 0.0005,
    # Gemini TTS paid tier: audio out ~25 tokens/s; we bill ~$0.0003 per second of speech.
    "gemini-tts-per-audio-s": 0.0003,
    # Cloudflare Workers AI beyond the free 10k neurons/day ($0.011 / 1k neurons).
    "@cf/black-forest-labs/flux-1-schnell": 0.0011,
    "@cf/black-forest-labs/flux-2-klein": 0.0030,
    # Hosted T4 on a public cloud, per GPU-second (~$0.35/h).
    "gpu-t4-per-s": 0.35 / 3600,
}


def estimate_cost(model: str, *, audio_s: float = 0.0, gpu_s: float = 0.0) -> float:
    if gpu_s:
        return round(gpu_s * LIST_PRICE["gpu-t4-per-s"], 5)
    if "tts" in model:
        return round(audio_s * LIST_PRICE["gemini-tts-per-audio-s"], 5)
    if model.startswith("gemini"):
        return LIST_PRICE["gemini-text"]
    for prefix, price in LIST_PRICE.items():
        if model.startswith(prefix):
            return price
    return 0.0


def log_call(
    *,
    stage: str,
    provider: str,
    model: str,
    item: str | None = None,
    params: dict | None = None,
    seed: int | None = None,
    latency_ms: float | None = None,
    status: str = "ok",
    cache_hit: bool = False,
    gpu_s: float | None = None,
    cost_usd: float | None = None,
    attempt: int = 1,
    error: str | None = None,
    output: str | None = None,
    **extra: Any,
) -> dict:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "stage": stage,
        "item": item,
        "provider": provider,
        "model": model,
        "params": params or {},
        "seed": seed,
        "latency_ms": round(latency_ms) if latency_ms is not None else None,
        "gpu_s": gpu_s,
        "status": status,
        "cache_hit": cache_hit,
        "attempt": attempt,
        "cost_usd_list_price": cost_usd,
        "free_tier": True,
        "output": output,
        "error": error,
        **extra,
    }
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def say(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


# --------------------------------------------------------------------------- retries
class TransientError(Exception):
    """Worth retrying: 429, 5xx, timeouts."""


class QuotaExhausted(Exception):
    """Daily / weekly quota gone. Do not retry; fall back."""


@dataclass
class Attempt:
    n: int = 0
    errors: list[str] = field(default_factory=list)


def with_retries(fn: Callable[[Attempt], Any], *, attempts: int = 3, base_s: float = 2.0,
                 stage: str = "", item: str = "") -> Any:
    """Call fn(attempt) up to `attempts` times on TransientError, with jittered
    exponential backoff. QuotaExhausted and other errors propagate immediately."""
    state = Attempt()
    while True:
        state.n += 1
        try:
            return fn(state)
        except TransientError as e:
            state.errors.append(str(e))
            if state.n >= attempts:
                raise
            wait = base_s * (2 ** (state.n - 1)) + random.uniform(0, 1)
            say(stage, f"{item}: transient error ({e}); retry {state.n}/{attempts - 1} in {wait:.1f}s")
            time.sleep(wait)


def classify_http(resp) -> None:
    """Raise the right exception class for a non-2xx `requests` response."""
    if resp.ok:
        return
    body = resp.text[:2000]
    if resp.status_code == 429:
        low = body.lower()
        # Per-minute limits recover on their own; per-day limits do not.
        if "perday" in low.replace(" ", "").replace("_", "") or "per day" in low or "daily" in low:
            raise QuotaExhausted(f"HTTP 429 daily quota: {body}")
        raise TransientError(f"HTTP 429: {body}")
    if resp.status_code >= 500:
        raise TransientError(f"HTTP {resp.status_code}: {body}")
    raise RuntimeError(f"HTTP {resp.status_code}: {body}")


@contextmanager
def timer() -> Iterator[dict]:
    t = {"start": time.perf_counter(), "ms": 0.0}
    try:
        yield t
    finally:
        t["ms"] = (time.perf_counter() - t["start"]) * 1000


# --------------------------------------------------------------------------- env/tools
def require_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"missing environment variable(s) {missing}; copy .env.example to .env")
    return {n: os.environ[n] for n in names}


def have_env(*names: str) -> bool:
    return all(os.environ.get(n) for n in names)


def ffmpeg(*args: str, quiet: bool = True) -> subprocess.CompletedProcess:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found; run `brew install ffmpeg` (see README)")
    cmd = ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error" if quiet else "info", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{res.stderr[-2000:]}")
    return res


def probe_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(res.stdout.strip())


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def main_guard(fn: Callable[[], None]) -> None:
    try:
        fn()
    except StoryboardError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
