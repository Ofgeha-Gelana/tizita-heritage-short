"""Thin REST clients for the hosted providers. Each function makes exactly one HTTP
call, raises TransientError / QuotaExhausted / RuntimeError, and returns raw data.
Logging and retries happen in the caller so every attempt is visible in runs.jsonl.
"""
from __future__ import annotations

import base64
import io
import os
import wave

import requests

from .common import QuotaExhausted, TransientError, classify_http, require_env

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = 180


# --------------------------------------------------------------------------- Gemini

def _http(method, url, **kw):
    """requests call whose timeouts and dropped connections are retryable (TransientError)."""
    try:
        return requests.request(method, url, **kw)
    except (requests.Timeout, requests.ConnectionError) as e:
        raise TransientError(f"network: {e!r}"[:400]) from e

def _gemini_headers() -> dict:
    key = require_env("GEMINI_API_KEY")["GEMINI_API_KEY"]
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def gemini_list_models() -> list[str]:
    names, token = [], None
    while True:
        params = {"pageSize": 200, **({"pageToken": token} if token else {})}
        r = _http("GET", f"{GEMINI_BASE}/models", headers=_gemini_headers(), params=params, timeout=60)
        classify_http(r)
        body = r.json()
        names += [m["name"].removeprefix("models/") for m in body.get("models", [])]
        token = body.get("nextPageToken")
        if not token:
            return names


def gemini_text(model: str, prompt: str, *, temperature: float = 0.7, seed: int | None = None,
                json_mode: bool = False) -> tuple[str, dict]:
    cfg: dict = {"temperature": temperature}
    if seed is not None:
        cfg["seed"] = seed
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
    r = _http("POST", f"{GEMINI_BASE}/models/{model}:generateContent", headers=_gemini_headers(),
                      json=payload, timeout=TIMEOUT)
    classify_http(r)
    body = r.json()
    try:
        text = "".join(p.get("text", "") for p in body["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected Gemini response: {str(body)[:400]}") from e
    return text, body.get("usageMetadata", {})


def gemini_tts(model: str, text: str, *, voice: str, style: str) -> tuple[bytes, dict]:
    """Returns (wav_bytes, usage). Gemini TTS returns raw 16-bit PCM mono at 24 kHz."""
    # "Say <how>: <text>" is the documented form. A free-standing instruction paragraph
    # gets read aloud (FAILURES.md F4), so the style must end in a colon before the text.
    prompt = f"{style.rstrip(' .:')}: {text}" if style else text
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    r = _http("POST", f"{GEMINI_BASE}/models/{model}:generateContent", headers=_gemini_headers(),
                      json=payload, timeout=TIMEOUT)
    classify_http(r)
    body = r.json()
    try:
        part = body["candidates"][0]["content"]["parts"][0]["inlineData"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"no audio in TTS response: {str(body)[:400]}") from e
    pcm = base64.b64decode(part["data"])
    rate = 24000
    mime = part.get("mimeType", "")
    if "rate=" in mime:
        rate = int(mime.split("rate=")[1].split(";")[0])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue(), body.get("usageMetadata", {})


# --------------------------------------------------------------------------- Cloudflare
def _cf() -> tuple[str, dict]:
    env = require_env("CF_ACCOUNT_ID", "CF_API_TOKEN")
    url = f"https://api.cloudflare.com/client/v4/accounts/{env['CF_ACCOUNT_ID']}/ai/run/"
    return url, {"Authorization": f"Bearer {env['CF_API_TOKEN']}"}


def cf_image(model: str, prompt: str, *, seed: int, width: int, height: int, steps: int = 4,
             reference_images: list[bytes] | None = None) -> bytes:
    """One image from Cloudflare Workers AI. FLUX.2 models take multipart form data
    (and up to four reference images); FLUX.1 schnell takes JSON and returns 1024px."""
    base, headers = _cf()
    if "flux-2" in model:
        form = {
            "prompt": (None, prompt),
            "seed": (None, str(seed)),
            "width": (None, str(width)),
            "height": (None, str(height)),
            "steps": (None, str(steps)),
        }
        for i, img in enumerate((reference_images or [])[:4]):
            form[f"input_image_{i}"] = (f"ref{i}.png", img, "image/png")
        r = _http("POST", base + model, headers=headers, files=form, timeout=TIMEOUT)
    else:
        r = _http("POST", base + model, headers=headers, timeout=TIMEOUT,
                          json={"prompt": prompt, "steps": min(steps, 8)})  # schnell rejects seed (FAILURES.md F6)
    if r.status_code == 429 or (r.status_code == 400 and "neuron" in r.text.lower()):
        # Workers AI reports the daily free allowance running out this way.
        raise QuotaExhausted(f"Cloudflare quota: {r.text[:300]}")
    classify_http(r)
    if r.headers.get("content-type", "").startswith("image/"):
        return r.content
    body = r.json()
    result = body.get("result", body)
    b64 = result.get("image") if isinstance(result, dict) else None
    if not b64:
        raise RuntimeError(f"no image in Cloudflare response: {str(body)[:400]}")
    return base64.b64decode(b64)


def cf_list_flux_models() -> list[str]:
    env = require_env("CF_ACCOUNT_ID", "CF_API_TOKEN")
    r = _http("GET", 
        f"https://api.cloudflare.com/client/v4/accounts/{env['CF_ACCOUNT_ID']}/ai/models/search",
        headers={"Authorization": f"Bearer {env['CF_API_TOKEN']}"},
        params={"search": "flux", "per_page": 50}, timeout=60,
    )
    classify_http(r)
    return [m["name"] for m in r.json().get("result", [])]


def kaggle_username() -> str | None:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    from pathlib import Path
    import json
    p = Path.home() / ".kaggle" / "kaggle.json"
    if p.exists():
        return json.loads(p.read_text()).get("username")
    return None
