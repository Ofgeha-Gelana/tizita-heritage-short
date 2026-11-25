"""Stage: keyframes (Cloudflare Workers AI, FLUX.2 klein; FLUX.1 schnell fallback).

1. Reference sheets: for every character, one image per `reference.candidate_seeds`.
   You look at them and set `reference.chosen_seed` in storyboard.json.
2. Keyframes: one image per shot = style block + character descriptions + image
   prompt, with the chosen reference sheet(s) passed as reference images, so the
   character stays the same person across shots.

Every image is cropped/resized to format.keyframe_size. Cache key includes the
prompt, seed, model, size and the hash of the reference image file, so changing a
reference regenerates only the shots that character appears in.

    python -m pipeline.keyframes refs            # reference sheet candidates
    python -m pipeline.keyframes shots [--shots s03]
    python -m pipeline.keyframes explore --shots s03 --seeds 1,2,3,4
    python -m pipeline.keyframes promote --shots s03 --seeds 3   # keep that exact image

FLUX.2 klein on Cloudflare is not deterministic per seed (EXPERIMENTS.md E15): the same
prompt and seed gave a visibly different image. So `promote` copies the explored file
into place and writes the cache entry; later runs reuse that file instead of redrawing.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image, ImageOps

from . import providers as P
from .common import (OFFLINE, OUTPUTS, Cache, QuotaExhausted, estimate_cost, load_storyboard, log_call,
                     main_guard, rel, say, shots_filter, stable_hash, timer, with_retries)

STAGE = "keyframes"
KF_DIR = OUTPUTS / "keyframes"
REF_DIR = OUTPUTS / "references"


def ref_path(char_id: str, seed: int) -> Path:
    return REF_DIR / f"{char_id}_seed{seed}.png"


def chosen_ref(sb: dict, char_id: str) -> Path | None:
    c = next(c for c in sb["characters"] if c["id"] == char_id)
    seed = c["reference"]["chosen_seed"]
    if seed is None:
        return None
    p = ref_path(char_id, seed)
    if not p.exists():
        return None
    return front_view(p)


def front_view(sheet: Path) -> Path:
    """Crop the left (front) view out of a three-view turnaround sheet.

    Passing the whole sheet made FLUX.2 copy its layout: the same person appeared two
    or three times in one keyframe (EXPERIMENTS.md E14). A single figure fixes that."""
    out = sheet.with_name(sheet.stem + "_front.png")
    if out.exists() and out.stat().st_mtime >= sheet.stat().st_mtime:
        return out
    img = Image.open(sheet).convert("RGB")
    w, h = img.size
    img.crop((0, 0, w // 3, h)).save(out)
    return out


def shot_prompt(sb: dict, shot: dict) -> str:
    chars = {c["id"]: c for c in sb["characters"]}
    who = "; ".join(f"the {chars[c]['name'].split('(')[0].strip().lower()} is {chars[c]['description']}"
                    for c in shot["characters"])
    parts = [shot["image_prompt"], who, sb["style_block"]]
    return ". ".join(p for p in parts if p)


def fit(img_bytes: bytes, size: tuple[int, int]) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return ImageOps.fit(img, size, Image.LANCZOS, centering=(0.5, 0.45))


def generate(sb: dict, *, item: str, prompt: str, seed: int, size: tuple[int, int], refs: list[Path],
             out: Path) -> str:
    """Try the primary model, fall back to schnell on quota or repeated failure.
    Returns the model id that produced the image."""
    models = [sb["models"]["image"], sb["models"]["image_fallback"]]
    ref_bytes = [r.read_bytes() for r in refs]
    last_err = None
    for model in models:
        def call(attempt, model=model):
            with timer() as t:
                try:
                    data = P.cf_image(model, prompt, seed=seed, width=size[0], height=size[1],
                                      reference_images=ref_bytes if "flux-2" in model else None)
                except Exception as e:
                    log_call(stage=STAGE, provider="cloudflare", model=model, item=item, seed=seed,
                             attempt=attempt.n, status="error", error=str(e)[:2000],
                             params={"size": size, "refs": [rel(r) for r in refs]})
                    raise
            return data, t["ms"], attempt.n
        try:
            data, ms, n = with_retries(call, stage=STAGE, item=item)
        except (QuotaExhausted, Exception) as e:  # noqa: BLE001
            last_err = e
            say(STAGE, f"{item}: {model} failed ({str(e)[:120]}); trying fallback")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        fit(data, size).save(out)
        log_call(stage=STAGE, provider="cloudflare", model=model, item=item, seed=seed, attempt=n,
                 latency_ms=ms, output=rel(out), cost_usd=estimate_cost(model),
                 params={"prompt": prompt, "size": size, "refs": [rel(r) for r in refs] if "flux-2" in model else []},
                 fallback=model != models[0])
        say(STAGE, f"{item}: {model} seed {seed} in {ms/1000:.1f}s -> {rel(out)}")
        return model
    raise SystemExit(f"{item}: all image models failed: {last_err}")


def run_refs(sb: dict) -> None:
    cache = Cache(STAGE)
    size = (768, 768)
    for c in sb["characters"]:
        if c["real_person"]:
            say(STAGE, f"{c['id']}: real person — make sure ETHICS.md#{c.get('consent_ref')} holds consent")
        prompt = f"{c['reference']['prompt']}. The character: {c['description']}. {sb['style_block']}"
        for seed in c["reference"]["candidate_seeds"]:
            item = f"ref/{c['id']}/{seed}"
            key = stable_hash(prompt, seed, sb["models"]["image"], size)
            out = ref_path(c["id"], seed)
            if cache.fresh(item, key):
                say(STAGE, f"{item}: cached")
                continue
            if OFFLINE:
                continue
            model = generate(sb, item=item, prompt=prompt, seed=seed, size=size, refs=[], out=out)
            cache.put(item, key, out, model=model)
    unchosen = [c["id"] for c in sb["characters"] if c["reference"]["chosen_seed"] is None]
    if unchosen:
        say(STAGE, f"look at {rel(REF_DIR)} and set reference.chosen_seed for {unchosen} in storyboard.json")


def run_shots(sb: dict, only: str | None = None, seeds_override: list[int] | None = None) -> None:
    cache = Cache(STAGE)
    size = tuple(sb["format"]["keyframe_size"])
    for shot in shots_filter(sb, only):
        refs = [r for r in (chosen_ref(sb, c) for c in shot["characters"]) if r]
        missing = [c for c in shot["characters"] if chosen_ref(sb, c) is None]
        if missing:
            say(STAGE, f"{shot['id']}: no chosen reference for {missing}; generating from text only")
        prompt = shot_prompt(sb, shot)
        for seed in seeds_override or [shot["seed"]]:
            explore = seeds_override is not None
            item = f"{'explore/' if explore else ''}{shot['id']}" + (f"/{seed}" if explore else "")
            out = (KF_DIR / "explore" / f"{shot['id']}_seed{seed}.png") if explore else KF_DIR / f"{shot['id']}.png"
            key = stable_hash(prompt, seed, sb["models"]["image"], size, *refs)
            if cache.fresh(item, key):
                say(STAGE, f"{item}: cached")
                continue
            if OFFLINE:
                continue
            model = generate(sb, item=item, prompt=prompt, seed=seed, size=size, refs=refs, out=out)
            cache.put(item, key, out, model=model, seed=seed)


def promote(sb: dict, shot_id: str, seed: int) -> None:
    import shutil
    shot = next(s for s in shots_filter(sb, shot_id))
    if shot["seed"] != seed:
        raise SystemExit(f"set shots[{shot_id}].seed = {seed} in storyboard.json first, so the cache key matches")
    src = KF_DIR / "explore" / f"{shot_id}_seed{seed}.png"
    if not src.exists():
        raise SystemExit(f"{rel(src)} not found; run explore first")
    refs = [r for r in (chosen_ref(sb, c) for c in shot["characters"]) if r]
    size = tuple(sb["format"]["keyframe_size"])
    key = stable_hash(shot_prompt(sb, shot), seed, sb["models"]["image"], size, *refs)
    out = KF_DIR / f"{shot_id}.png"
    shutil.copyfile(src, out)
    Cache(STAGE).put(shot_id, key, out, model=sb["models"]["image"], seed=seed, promoted_from=rel(src))
    log_call(stage=STAGE, provider="local", model="promote", item=shot_id, seed=seed, cache_hit=True,
             output=rel(out), params={"from": rel(src)})
    say(STAGE, f"{shot_id}: promoted {rel(src)} -> {rel(out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["refs", "shots", "explore", "promote", "all"], nargs="?", default="all")
    ap.add_argument("--shots")
    ap.add_argument("--seeds", help="comma-separated seeds for explore mode")
    a = ap.parse_args()
    sb = load_storyboard()
    if a.what in ("refs", "all"):
        run_refs(sb)
    if a.what in ("shots", "all"):
        run_shots(sb, a.shots)
    if a.what == "explore":
        if not (a.shots and a.seeds):
            raise SystemExit("explore needs --shots and --seeds")
        run_shots(sb, a.shots, [int(s) for s in a.seeds.split(",")])
    if a.what == "promote":
        if not (a.shots and a.seeds):
            raise SystemExit("promote needs --shots and --seeds (one seed)")
        promote(sb, a.shots, int(a.seeds))


if __name__ == "__main__":
    main_guard(main)
