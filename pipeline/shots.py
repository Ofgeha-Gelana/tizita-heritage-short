"""Stage: moving shots (Wan 2.2 TI2V-5B image-to-video on a Kaggle T4).

For every shot with video_mode == "wan" the cache key is
    hash(keyframe file, motion prompt, seed, model, size, frames, steps).
Only shots whose key changed are sent to Kaggle, as one batch. Shots with
video_mode == "kenburns", or whose Wan job failed 3 times, are animated from the
keyframe in the assembly stage instead, so the film always finishes.

    python -m pipeline.shots [--quality draft|final] [--shots s01,s04]
    python -m pipeline.shots --submit-only     # launch and return
    python -m pipeline.shots --collect         # fetch a finished batch
"""
from __future__ import annotations

import argparse
import shutil

from . import kaggle_job as K
from .common import (OFFLINE, OUTPUTS, Cache, estimate_cost, have_env, load_storyboard, log_call,
                     main_guard, rel, say, shots_filter, stable_hash)
from .keyframes import KF_DIR

STAGE = "shots"
NAME = "wan22_i2v"
CLIP_DIR = OUTPUTS / "clips"
MAX_FAILS = 3


def job_spec(sb: dict, shot: dict, quality: str) -> dict:
    w, h = sb["format"]["video_size_final" if quality == "final" else "video_size_draft"]
    frames = 121  # 5 s at 24 fps; Wan quality drops on longer clips, assembly stretches/holds
    steps = 40 if quality == "final" else 25
    kf = KF_DIR / f"{shot['id']}.png"
    key = stable_hash(kf, shot["motion_prompt"], shot["seed"], sb["models"]["video"], [w, h], frames, steps)
    return {"id": shot["id"], "key": key, "image": f"{shot['id']}.png", "prompt": shot["motion_prompt"],
            "negative_prompt": sb.get("negative_prompt", ""), "seed": shot["seed"], "width": w, "height": h,
            "num_frames": frames, "steps": steps, "guidance_scale": 5.0, "quality": quality}


def pending_jobs(sb: dict, only: str | None, quality: str) -> list[dict]:
    cache = Cache(STAGE)
    jobs = []
    for shot in shots_filter(sb, only):
        if shot["video_mode"] != "wan":
            continue
        if not (KF_DIR / f"{shot['id']}.png").exists():
            say(STAGE, f"{shot['id']}: no keyframe yet; skipped (run keyframes first)")
            continue
        spec = job_spec(sb, shot, quality)
        if cache.fresh(shot["id"], spec["key"]):
            say(STAGE, f"{shot['id']}: cached")
            continue
        # A final-quality clip also satisfies a draft request.
        if quality == "draft" and cache.fresh(shot["id"], job_spec(sb, shot, "final")["key"]):
            say(STAGE, f"{shot['id']}: cached (final)")
            continue
        fails = (cache.get(f"fail/{shot['id']}") or {}).get("count", 0)
        if fails >= MAX_FAILS:
            say(STAGE, f"{shot['id']}: failed {fails}x; using Ken Burns fallback")
            continue
        jobs.append(spec)
    return jobs


def ingest(sb: dict) -> int:
    """Copy finished clips into outputs/clips and log one runs.jsonl line per job."""
    st = K.load_state(NAME)
    out, results = K.collect(NAME)
    cache = Cache(STAGE)
    ok = 0
    for r in results:
        sid, key = r["id"], r["key"]
        if st.get("jobs", {}).get(sid) != key:
            say(STAGE, f"{sid}: result key {key} is not from the last submission; ignored")
            continue
        gpu_s = r.get("gpu_s")
        common = dict(stage=STAGE, provider="kaggle", model=sb["models"]["video"], item=sid, seed=r.get("seed"),
                      latency_ms=(r.get("wall_s") or 0) * 1000, gpu_s=gpu_s,
                      params={k: r.get(k) for k in ("width", "height", "num_frames", "steps", "guidance_scale",
                                                    "quality", "backend", "gpu_name", "peak_vram_gb")},
                      cost_usd=estimate_cost("gpu", gpu_s=gpu_s or 0))
        if r.get("status") == "ok":
            src = next(out.rglob(r["file"]), None)
            if src:
                dst = CLIP_DIR / f"{sid}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                cache.put(sid, key, dst, quality=r.get("quality"), gpu_s=gpu_s)
                log_call(**common, output=rel(dst))
                ok += 1
                continue
            r["error"] = f"clip file {r['file']} missing from kernel output"
        prev = (cache.get(f"fail/{sid}") or {}).get("count", 0)
        cache.data[f"fail/{sid}"] = {"key": key, "path": "", "count": prev + 1, "last_error": r.get("error")}
        cache.path.write_text(__import__("json").dumps(cache.data, indent=2, sort_keys=True) + "\n")
        log_call(**common, status="error", error=str(r.get("error"))[:500])
    say(STAGE, f"ingested {ok}/{len(results)} clips")
    return ok


def run(only: str | None = None, quality: str = "draft", submit_only: bool = False,
        collect_only: bool = False, timeout_s: int = 4 * 3600) -> None:
    sb = load_storyboard()
    if collect_only:
        ingest(sb)
        return
    jobs = pending_jobs(sb, only, quality)
    if not jobs:
        say(STAGE, "nothing to generate")
        return
    if OFFLINE or not (have_env("KAGGLE_USERNAME") or K.kaggle_username()):
        say(STAGE, f"{len(jobs)} shot(s) need Wan but Kaggle is unavailable/OFFLINE; Ken Burns will be used")
        return
    files = {j["image"]: KF_DIR / j["image"] for j in jobs}
    K.submit(NAME, jobs, files, {"model": sb["models"]["video"], "fps": sb["format"]["fps"]})
    say(STAGE, f"submitted {len(jobs)} job(s): {[j['id'] for j in jobs]}")
    if submit_only:
        say(STAGE, "run `make shots-collect` when the Kaggle kernel finishes")
        return
    s = K.wait(NAME, timeout_s)
    say(STAGE, f"kernel finished with status={s}")
    if s in ("complete", "error"):
        ingest(sb)  # an errored kernel may still have finished some clips
    else:
        log_call(stage=STAGE, provider="kaggle", model=sb["models"]["video"], status="timeout",
                 error=f"kernel not done after {timeout_s}s", params={"jobs": [j["id"] for j in jobs]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots")
    ap.add_argument("--quality", choices=["draft", "final"], default="draft")
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    a = ap.parse_args()
    run(a.shots, a.quality, a.submit_only, a.collect, a.timeout)


if __name__ == "__main__":
    main_guard(main)
