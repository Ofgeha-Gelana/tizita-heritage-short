"""Smoke test: call each provider once, log latency to runs.jsonl, and check that
the model ids in storyboard.json still exist (the challenge brief warns they move).

    make smoke
"""
from __future__ import annotations

import shutil
import subprocess

from . import providers as P
from .common import have_env, load_storyboard, log_call, main_guard, say, timer

STAGE = "smoke"


def check(name: str, fn):
    try:
        with timer() as t:
            out = fn()
        return True, t["ms"], out
    except Exception as e:  # noqa: BLE001 - smoke test reports everything
        return False, t["ms"], str(e)[:300]


def main() -> None:
    sb = load_storyboard()
    m = sb["models"]
    results = []

    if have_env("GEMINI_API_KEY"):
        ok, ms, models = check("gemini-list", P.gemini_list_models)
        if ok:
            wanted = [("text", m["text"]), ("text_lite", m["text_lite"]), ("tts", m["tts"])]
            wanted += [("tts_fallbacks", x) for x in m.get("tts_fallbacks", [])]
            for key, mid in wanted:
                if mid not in models:
                    near = [x for x in models if ("tts" in x) == ("tts" in mid)][:8]
                    say(STAGE, f"WARNING models.{key}='{mid}' not offered; similar: {near}")
        ok, ms, out = check("gemini-text", lambda: P.gemini_text(m["text"], "Reply with the single word: ok", temperature=0))
        log_call(stage=STAGE, provider="gemini", model=m["text"], latency_ms=ms, status="ok" if ok else "error",
                 error=None if ok else out, params={"prompt": "ping"})
        results.append(("gemini text", ok, ms, out if not ok else out[0].strip()))
        ok, ms, out = check("gemini-tts", lambda: P.gemini_tts(m["tts"], "ሰላም", voice=sb["voice"]["name"], style=""))
        log_call(stage=STAGE, provider="gemini", model=m["tts"], latency_ms=ms, status="ok" if ok else "error",
                 error=None if ok else out, params={"text": "ሰላም", "voice": sb["voice"]["name"]})
        results.append(("gemini tts (am)", ok, ms, f"{len(out[0])} bytes wav" if ok else out))
    else:
        results.append(("gemini", False, 0, "GEMINI_API_KEY not set"))

    if have_env("CF_ACCOUNT_ID", "CF_API_TOKEN"):
        ok, ms, flux = check("cf-list", P.cf_list_flux_models)
        if ok:
            say(STAGE, f"Cloudflare FLUX models: {flux}")
        for key in ("image", "image_fallback"):
            ok, ms, out = check(key, lambda k=key: P.cf_image(m[k], "a clay coffee pot on green grass, gouache", seed=1, width=512, height=512))
            log_call(stage=STAGE, provider="cloudflare", model=m[key], latency_ms=ms, seed=1,
                     status="ok" if ok else "error", error=None if ok else out, params={"w": 512, "h": 512})
            results.append((f"cloudflare {key}", ok, ms, f"{len(out)} bytes" if ok else out))
    else:
        results.append(("cloudflare", False, 0, "CF_ACCOUNT_ID / CF_API_TOKEN not set"))

    if shutil.which("kaggle") or P.kaggle_username():
        ok, ms, out = check("kaggle", lambda: subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "--page-size", "1"], capture_output=True, text=True, check=True).stdout)
        log_call(stage=STAGE, provider="kaggle", model="kaggle-api", latency_ms=ms, status="ok" if ok else "error",
                 error=None if ok else out)
        results.append(("kaggle api", ok, ms, "reachable" if ok else out))
    else:
        results.append(("kaggle", False, 0, "kaggle CLI / ~/.kaggle/kaggle.json not configured"))

    results.append(("ffmpeg", bool(shutil.which("ffmpeg")), 0, shutil.which("ffmpeg") or "not installed"))

    print("\nprovider                  ok    latency  detail")
    for name, ok, ms, detail in results:
        print(f"{name:25} {'yes' if ok else 'NO ':4} {ms:7.0f}ms  {str(detail)[:90]}")


if __name__ == "__main__":
    main_guard(main)
