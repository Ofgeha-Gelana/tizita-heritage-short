"""One command for the whole film:  make film  ==  python -m pipeline.run

validate -> narration -> keyframes (refs + shots) -> shots (Wan on Kaggle) ->
music (ACE-Step on Kaggle) -> assemble (both languages, both aspect ratios).

Every stage skips items whose inputs have not changed, so a second run with an
unchanged storyboard makes zero model calls. With OFFLINE=1 nothing is called and
the film is assembled from whatever is already in outputs/ (fallbacks fill gaps).
"""
from __future__ import annotations

import argparse
import time

from . import assemble, keyframes, music, narration, shots
from .common import OFFLINE, RUN_ID, RUNS, load_storyboard, main_guard, say


def calls_since(offset: int) -> int:
    if not RUNS.exists():
        return 0
    with open(RUNS, "rb") as f:
        f.seek(offset)
        return sum(1 for line in f if b'"provider": "local"' not in line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", choices=["draft", "final"], default="draft")
    ap.add_argument("--shots", help="limit generation stages to these shot ids")
    ap.add_argument("--skip-gpu", action="store_true", help="do not submit Kaggle jobs this run")
    a = ap.parse_args()

    t0 = time.time()
    offset = RUNS.stat().st_size if RUNS.exists() else 0
    sb = load_storyboard()
    say("run", f"run_id={RUN_ID} storyboard valid: {len(sb['shots'])} shots, OFFLINE={OFFLINE}")

    narration.run(only=a.shots)
    keyframes.run_refs(sb)
    keyframes.run_shots(load_storyboard(), a.shots)
    if not a.skip_gpu:
        shots.run(a.shots, a.quality)
        music.run()
    assemble.run()
    say("run", f"done in {time.time() - t0:.0f}s; {calls_since(offset)} model call(s) logged to runs.jsonl")


if __name__ == "__main__":
    main_guard(main)
