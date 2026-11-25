"""Stage: music (ACE-Step 1.5 on Kaggle) plus a placeholder drone for the radio cut.

Candidates: one per seed in music.candidate_seeds, cache key =
hash(caption, bpm, key, duration, seed, model). You listen under the narration,
set music.chosen (a seed) and music.choice_reason in storyboard.json, and the
assembly stage uses that file. With music.chosen == null the placeholder is used.

    python -m pipeline.music                 # candidates (submit, wait, collect)
    python -m pipeline.music --submit-only | --collect
    python -m pipeline.music --placeholder   # only the synthetic temp track
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import kaggle_job as K
from .common import (OFFLINE, OUTPUTS, Cache, estimate_cost, ffmpeg, load_storyboard, log_call,
                     main_guard, rel, say, stable_hash)

STAGE = "music"
NAME = "acestep"
MUSIC_DIR = OUTPUTS / "music"


def candidate_path(seed: int) -> Path:
    return MUSIC_DIR / f"candidate_seed{seed}.wav"


def placeholder(duration_s: float) -> Path:
    """Deterministic temp track: a soft E-minor drone with slow tremolo. Not AI,
    not logged as a model call; it exists only so the radio cut has a bed."""
    out = MUSIC_DIR / "placeholder_drone.wav"
    cache = Cache(STAGE)
    key = stable_hash("drone-v1", round(duration_s, 1))
    if cache.fresh("placeholder", key):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    expr = ("0.10*sin(2*PI*82.41*t)+0.07*sin(2*PI*123.47*t)+0.05*sin(2*PI*164.81*t)"
            "+0.03*sin(2*PI*196.00*t)*(0.5+0.5*sin(2*PI*0.07*t))")
    ffmpeg("-f", "lavfi", "-i", f"aevalsrc={expr}:s=48000:d={duration_s:.2f}",
           "-af", f"tremolo=f=0.15:d=0.3,lowpass=f=1200,afade=t=in:d=3,afade=t=out:st={max(duration_s - 4, 0):.2f}:d=4",
           "-ac", "2", str(out))
    cache.put("placeholder", key, out)
    return out


def chosen_track(sb: dict) -> Path | None:
    seed = sb["music"]["chosen"]
    if seed is None:
        return None
    p = candidate_path(seed)
    return p if p.exists() else None


def job_for(sb: dict, seed: int) -> dict:
    m = sb["music"]
    key = stable_hash(m["caption"], m["bpm"], m["key"], m["duration_s"], seed, sb["models"]["music"])
    return {"id": f"seed{seed}", "key": key, "seed": seed, "caption": m["caption"], "bpm": m["bpm"],
            "key_scale": m["key"], "duration_s": m["duration_s"]}


def ingest(sb: dict) -> None:
    st = K.load_state(NAME)
    out, results = K.collect(NAME)
    cache = Cache(STAGE)
    for r in results:
        common = dict(stage=STAGE, provider="kaggle", model=sb["models"]["music"], item=r["id"], seed=r["seed"],
                      latency_ms=(r.get("wall_s") or 0) * 1000, gpu_s=r.get("gpu_s"),
                      params={"caption": sb["music"]["caption"], "bpm": sb["music"]["bpm"], "key": sb["music"]["key"],
                              "duration_s": sb["music"]["duration_s"], "planner": r.get("planner"),
                              "gpu_name": r.get("gpu_name"), "peak_vram_gb": r.get("peak_vram_gb")},
                      cost_usd=estimate_cost("gpu", gpu_s=r.get("gpu_s") or 0))
        if st.get("jobs", {}).get(r["id"]) != r["key"]:
            continue
        src = next(out.rglob(r["file"]), None) if r.get("status") == "ok" else None
        if src:
            dst = candidate_path(r["seed"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            cache.put(r["id"], r["key"], dst)
            log_call(**common, output=rel(dst))
            say(STAGE, f"{r['id']}: {rel(dst)}")
        else:
            log_call(**common, status="error", error=str(r.get("error"))[:500])
    if sb["music"]["chosen"] is None:
        say(STAGE, "listen to outputs/music/candidate_*.wav and set music.chosen + music.choice_reason")


def run(submit_only: bool = False, collect_only: bool = False, timeout_s: int = 2 * 3600) -> None:
    sb = load_storyboard()
    placeholder(sum(s["duration_s"] for s in sb["shots"]) + sb["end_card"]["duration_s"] + 2)
    if collect_only:
        ingest(sb)
        return
    cache = Cache(STAGE)
    jobs = [j for j in (job_for(sb, s) for s in sb["music"]["candidate_seeds"]) if not cache.fresh(j["id"], j["key"])]
    if not jobs:
        say(STAGE, "all candidates cached")
        return
    if OFFLINE or not K.kaggle_username():
        say(STAGE, f"{len(jobs)} candidate(s) missing but Kaggle unavailable/OFFLINE; placeholder will be used")
        return
    K.submit(NAME, jobs, {}, {"model": sb["models"]["music"], "use_lm": True})
    if submit_only:
        return
    s = K.wait(NAME, timeout_s)
    if s in ("complete", "error"):
        ingest(sb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--placeholder", action="store_true")
    a = ap.parse_args()
    if a.placeholder:
        sb = load_storyboard()
        say(STAGE, rel(placeholder(sum(s["duration_s"] for s in sb["shots"]) + sb["end_card"]["duration_s"] + 2)))
        return
    run(a.submit_only, a.collect)


if __name__ == "__main__":
    main_guard(main)
