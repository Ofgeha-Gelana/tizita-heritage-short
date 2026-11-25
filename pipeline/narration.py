"""Stage: narration. One wav per shot per language with Gemini TTS.

Free-tier reality (FAILURES.md F4/F5): every TTS model allows only 10 requests per
day and 3 per minute, and a style instruction in the prompt is often spoken aloud.
So this stage:

* sends the narration text ONLY (no style prompt; voice choice carries the tone),
* in BATCH mode (>= 3 stale lines in a language) speaks the whole script in ONE
  request and splits it into per-shot files at the N-1 longest pauses, which also
  keeps one consistent narrator. If the split does not look right it falls back to
* PER-LINE mode, one request per line. A single edited line always takes this
  path, so changing one line costs exactly one model call.
* walks a chain of TTS models (models.tts then models.tts_fallbacks) and remembers
  for the rest of the day which ones are out of daily quota.

Each line is cached by (language, text, voice, model chain); trimmed to 0.1 s of
silence at each end so the assembly stage controls the timing.

    python -m pipeline.narration [--lang am] [--shots s01,s03] [--per-line]
"""
from __future__ import annotations

import argparse
import array
import io
import json
import time
import wave
from datetime import datetime, timedelta, timezone

from . import providers as P
from .common import (CACHE_DIR, OFFLINE, OUTPUTS, Cache, QuotaExhausted, TransientError, estimate_cost,
                     load_storyboard, log_call, main_guard, rel, say, shots_filter, stable_hash, timer,
                     with_retries)

STAGE = "narration"
MIN_INTERVAL_S = 21.0   # 3 requests/minute per model on the free tier
BATCH_MIN_LINES = 3
EXHAUSTED = CACHE_DIR / "tts_quota_exhausted.json"
_last_call = [0.0]


# --------------------------------------------------------------------------- helpers
def throttle() -> None:
    wait = _last_call[0] + MIN_INTERVAL_S - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()


def quota_day() -> str:
    """Gemini daily quotas reset at midnight Pacific time."""
    return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


def exhausted_today() -> set[str]:
    if not EXHAUSTED.exists():
        return set()
    data = json.loads(EXHAUSTED.read_text())
    return {m for m, day in data.items() if day == quota_day()}


def mark_exhausted(model: str) -> None:
    data = json.loads(EXHAUSTED.read_text()) if EXHAUSTED.exists() else {}
    data[model] = quota_day()
    EXHAUSTED.parent.mkdir(parents=True, exist_ok=True)
    EXHAUSTED.write_text(json.dumps(data, indent=2) + "\n")


def probe_duration(path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def read_pcm(wav_bytes: bytes) -> tuple[array.array, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return array.array("h", w.readframes(w.getnframes())), w.getframerate()


def write_pcm(path, samples: array.array, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def frame_rms(samples: array.array, rate: int, frame_s: float = 0.02) -> list[float]:
    n = max(int(rate * frame_s), 1)
    out = []
    for i in range(0, len(samples), n):
        chunk = samples[i:i + n]
        out.append((sum(x * x for x in chunk) / max(len(chunk), 1)) ** 0.5)
    return out


def silence_threshold(rms: list[float]) -> float:
    loud = sorted(rms)[int(len(rms) * 0.95)] if rms else 0
    return max(loud * 0.06, 60.0)


def trim(samples: array.array, rate: int, keep_s: float = 0.1) -> array.array:
    rms = frame_rms(samples, rate)
    thr = silence_threshold(rms)
    voiced = [i for i, r in enumerate(rms) if r > thr]
    if not voiced:
        return samples
    n = int(rate * 0.02)
    start = max(voiced[0] * n - int(rate * keep_s), 0)
    end = min((voiced[-1] + 1) * n + int(rate * keep_s), len(samples))
    return samples[start:end]


def split_on_pauses(samples: array.array, rate: int, texts: list[str]) -> list[array.array] | None:
    """Cut one recording of N lines into N parts at the N-1 longest internal pauses.
    Returns None when the result does not look like the script (wrong proportions)."""
    n_parts = len(texts)
    rms = frame_rms(samples, rate)
    thr = silence_threshold(rms)
    voiced = [i for i, r in enumerate(rms) if r > thr]
    if not voiced:
        return None
    first, last = voiced[0], voiced[-1]
    runs, start = [], None
    for i in range(first, last + 1):
        if rms[i] <= thr and start is None:
            start = i
        elif rms[i] > thr and start is not None:
            runs.append((i - start, start, i))
            start = None
    if len(runs) < n_parts - 1:
        return None
    # Pauses between lines are only slightly longer than pauses at commas inside a
    # line (EXPERIMENTS.md E13), so "the N-1 longest" is not enough. Predict where
    # each boundary should fall from the cumulative text length, then take the
    # longest pause near each prediction (score = length / (1 + distance)).
    total_c = sum(len(t) for t in texts)
    span = last - first
    cand = [r for r in runs if r[0] >= 10]  # >= 0.2 s
    cuts, used = [], set()
    acc = 0
    for t in texts[:-1]:
        acc += len(t)
        target = first + span * acc / total_c
        window = span / n_parts * 0.6
        options = [r for r in cand if r[1] not in used and abs((r[1] + r[2]) / 2 - target) <= window]
        if not options:
            return None
        best = max(options, key=lambda r: r[0] / (1 + abs((r[1] + r[2]) / 2 - target) / 25))
        used.add(best[1])
        cuts.append(best)
    cuts.sort(key=lambda r: r[1])
    if len({c[1] for c in cuts}) != n_parts - 1:
        return None
    n = int(rate * 0.02)
    bounds = [first * n] + [((a + b) // 2) * n for _, a, b in cuts] + [(last + 1) * n]
    parts = [trim(samples[bounds[i]:bounds[i + 1]], rate) for i in range(n_parts)]
    # Sanity: each part's length should be roughly proportional to its text length.
    total_s = sum(len(p) for p in parts) / rate
    total_c = sum(len(t) for t in texts)
    for p, t in zip(parts, texts):
        ratio = (len(p) / rate) / (total_s * len(t) / total_c)
        if not 0.45 <= ratio <= 2.2:
            return None
    return parts


def voice_for(sb: dict, lang: str) -> str:
    return sb["voice"].get("per_language", {}).get(lang, {}).get("name", sb["voice"]["name"])


def model_chain(sb: dict) -> list[str]:
    m = sb["models"]
    return [m["tts"], *m.get("tts_fallbacks", [])]


def narration_path(lang: str, shot_id: str):
    return OUTPUTS / "narration" / lang / f"{shot_id}.wav"


def tts_call(sb: dict, text: str, voice: str, item: str, lang: str) -> tuple[bytes, str, float]:
    """Try each model in the chain; returns (wav_bytes, model, latency_ms)."""
    skip = exhausted_today()
    for model in model_chain(sb):
        if model in skip:
            continue

        def call(attempt, model=model):
            throttle()
            with timer() as t:
                try:
                    data, usage = P.gemini_tts(model, text, voice=voice, style="")
                except Exception as e:
                    log_call(stage=STAGE, provider="gemini", model=model, item=item, attempt=attempt.n,
                             status="error", error=str(e)[:2000],
                             params={"voice": voice, "lang": lang, "chars": len(text)})
                    raise
            return data, usage, t["ms"], attempt.n
        try:
            data, usage, ms, n = with_retries(call, stage=STAGE, item=item)
            return data, model, ms
        except QuotaExhausted:
            mark_exhausted(model)
            say(STAGE, f"{item}: {model} is out of daily quota; next model")
        except (TransientError, RuntimeError) as e:
            say(STAGE, f"{item}: {model} failed ({str(e)[:120]}); next model")
    raise SystemExit(f"{item}: no TTS model left today (all out of quota or failing); "
                     f"quota resets at midnight Pacific. See runs.jsonl.")


def save_line(cache: Cache, item: str, key: str, out, samples, rate, *, model, ms, text, voice, lang, mode,
              batch_item=None) -> None:
    write_pcm(out, samples, rate)
    dur = len(samples) / rate
    log_call(stage=STAGE, provider="gemini" if mode == "per-line" else "local",
             model=model if mode == "per-line" else "split", item=item, latency_ms=ms if mode == "per-line" else 0,
             params={"voice": voice, "lang": lang, "chars": len(text), "mode": mode, "batch": batch_item},
             output=rel(out), audio_s=round(dur, 2),
             cost_usd=estimate_cost(model, audio_s=dur) if mode == "per-line" else 0.0)
    cache.put(item, key, out, model=model, duration_s=round(dur, 3), mode=mode)
    expected = len(text.split()) / 1.2 + 1.5
    flag = "  WARNING: longer than expected — listen for extra speech" if dur > expected else ""
    say(STAGE, f"{item}: {dur:.2f}s ({mode}, {model}){flag}")


# --------------------------------------------------------------------------- stage
def run(lang: str | None = None, only: str | None = None, per_line: bool = False) -> None:
    sb = load_storyboard()
    langs = [lang] if lang else [sb["languages"]["primary"], *sb["languages"]["extra"]]
    cache = Cache(STAGE)
    chain = model_chain(sb)
    for lg in langs:
        voice = voice_for(sb, lg)
        stale = []
        for shot in shots_filter(sb, only):
            text = shot["narration"].get(lg, "").strip()
            item = f"{lg}/{shot['id']}"
            if not text:
                if lg != sb["languages"]["primary"]:
                    say(STAGE, f"{item}: no '{lg}' text yet (run `make translate`); skipped")
                continue
            key = stable_hash("tts-v2", lg, text, voice, chain)
            if cache.fresh(item, key):
                say(STAGE, f"{item}: cached")
                continue
            stale.append((shot, text, item, key))
        if not stale or OFFLINE:
            if stale:
                say(STAGE, f"{lg}: {len(stale)} stale line(s) but OFFLINE=1")
            continue

        if len(stale) >= BATCH_MIN_LINES and not per_line:
            texts = [t for _, t, _, _ in stale]
            batch_item = f"{lg}/batch:{','.join(s['id'] for s, *_ in stale)}"
            script = "\n\n".join(texts)
            wav, model, ms = tts_call(sb, script, voice, batch_item, lg)
            samples, rate = read_pcm(wav)
            raw = OUTPUTS / "narration" / lg / "_batch_last.wav"
            write_pcm(raw, samples, rate)
            log_call(stage=STAGE, provider="gemini", model=model, item=batch_item, latency_ms=ms,
                     params={"voice": voice, "lang": lg, "chars": len(script), "lines": len(texts), "mode": "batch"},
                     output=rel(raw), audio_s=round(len(samples) / rate, 2),
                     cost_usd=estimate_cost(model, audio_s=len(samples) / rate))
            parts = split_on_pauses(samples, rate, texts)
            if parts:
                say(STAGE, f"{lg}: 1 request for {len(texts)} lines ({len(samples) / rate:.1f}s), split on pauses")
                for (shot, text, item, key), part in zip(stale, parts):
                    save_line(cache, item, key, narration_path(lg, shot["id"]), part, rate, model=model, ms=ms,
                              text=text, voice=voice, lang=lg, mode="batch-split", batch_item=batch_item)
                continue
            say(STAGE, f"{lg}: batch recording did not split cleanly into {len(texts)} lines; per-line instead")

        for shot, text, item, key in stale:
            wav, model, ms = tts_call(sb, text, voice, item, lg)
            samples, rate = read_pcm(wav)
            save_line(cache, item, key, narration_path(lg, shot["id"]), trim(samples, rate), rate,
                      model=model, ms=ms, text=text, voice=voice, lang=lg, mode="per-line")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang")
    ap.add_argument("--shots")
    ap.add_argument("--per-line", action="store_true", help="never batch")
    a = ap.parse_args()
    run(a.lang, a.shots, a.per_line)


if __name__ == "__main__":
    main_guard(main)
