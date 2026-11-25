# Failures

Each entry records what broke, the evidence, what was tried, and what fixed it or how it was worked around. Newest entries go at the bottom.

### F1: No ffmpeg, and system Python 3.9 on the dev machine (2026-11-25)
- **Broke:** The fresh macOS machine had `/usr/bin/python3` 3.9.6 and no `ffmpeg`. The pipeline uses 3.10+ syntax, and assembly is entirely FFmpeg.
- **Tried:** `brew install ffmpeg` failed three times on our network: Homebrew's Portable Ruby upgrade failed, and the bottle download from ghcr.io stalled. The evermeet.cx static build API returned 410, and its redirect went to an unknown mirror, so we did not use it.
- **Fixed:** We added the `imageio-ffmpeg` dependency, whose PyPI wheel bundles a static ffmpeg 9.0 with libx264, libass, loudnorm and sidechaincompress. `common.ffmpeg_exe()` uses the system ffmpeg when one is on PATH and this bundled one otherwise. The wheel ships no `ffprobe`, so `probe_duration` falls back to parsing `ffmpeg -i`. The first radio cut then built: 73.8 s, −14.15 LUFS, −1.0 dBTP.
- Python is pinned through `uv` (`requires-python >=3.10`, `uv sync` downloads a managed interpreter), so a clean clone does not depend on the system Python. `make setup` checks for ffmpeg and stops with an install hint.

### F2: The model docs do not show the call we need (2026-11-25)
- **Broke (before running):** The Wan2.2-TI2V-5B-Diffusers model card shows **text-to-video only** in diffusers. Image-to-video appears only through the official `generate.py`. The card also says the offloaded CLI needs about **24 GB VRAM**, while a Kaggle T4 has 16 GB. Cloudflare's FLUX.2 klein page documents only a `multipart` wrapper with no field names.
- **Workaround in code:** The Wan kernel tries `WanImageToVideoPipeline` (fp16, CPU offload, VAE tiling) first and then falls back to `generate.py --offload_model True --convert_model_dtype --t5_cpu`, recording which backend ran for each clip. Klein is called with the field names used by Cloudflare's FLUX.2 models (`prompt`, `width`, `height`, `steps`, `seed`, `input_image_N`). The smoke test checks it, and FLUX.1 schnell is the fallback.
- **Result:** _to fill in after the first Kaggle run (outcome, backend, peak VRAM, s/clip)._

### F6: FLUX.1 schnell on Cloudflare rejects `seed` (2026-11-25)
- **Broke:** The smoke test call to `@cf/black-forest-labs/flux-1-schnell` failed with `HTTP 400 AiError: Bad input: Additional or unevaluated properties '/seed' at '/' not allowed`. `num_steps` is rejected too. Only `prompt` and `steps` are accepted.
- **Fixed:** The schnell request now sends `prompt` and `steps` only. FLUX.2 klein, the primary model, still takes a seed.
- **Cost:** Fallback images cannot be reproduced from a seed. runs.jsonl marks them `fallback: true`, and the cache keeps the file, so a rerun does not redraw them.

<!-- Template:
### Fn: title (date)
- **Broke:** what happened, with the exact error and the runs.jsonl item / kernel log
- **Tried:** each attempt and its result
- **Fixed / worked around:** what worked, and what it cost (time, quality, quota)
-->

### F3: Model ids in the brief are not all real, and Gemini text returns 503 (2026-11-25)
- **Broke:** The first `make smoke` call logged `gemini-3.8-flash` → **HTTP 503 "model is currently experiencing high demand"**. The brief's "Flash-Lite for bulk calls" assumed a `gemini-3.8-flash-lite`, but that id **is not offered** by the API (`models.list`).
- **Found:** `gemini-3.8-flash-tts` works for Amharic on the first try (3.8 s for a one-word line). The Lite **text** models that exist are `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`. A `gemini-3.8-flash-lite-tts` also exists.
- **Fixed:** `models.text_lite` → `gemini-3.5-flash-lite`, and `models.tts_fallback` → `gemini-3.8-flash-lite-tts` (a separate model, so likely a separate rate-limit bucket). The 503 is handled by `with_retries` (3 attempts, backoff). The smoke test now warns whenever a storyboard model id disappears.
- **Evidence:** runs.jsonl `stage=smoke`.

### F4: Gemini TTS read the style instruction aloud, and the free tier allows 3 TTS requests per minute (2026-11-25)
- **Broke:** The first full Amharic narration pass produced 10 lines of **14.8–18.9 s each** for lines of 5–11 words (planned 5–8 s). The duration hardly depended on the word count, which pointed to a fixed amount of extra audio on every line. A Gemini transcription of `s05` (5 words, 16.1 s) came back as *"Read this slowly and warmly, like someone remembering something precious. With gentle pauses between sentences. Do not rush."* followed by the Amharic: the model had **spoken the English style paragraph**. The same pass also hit **HTTP 429, `generate_content_free_tier_requests`, limit 3/min, model `gemini-3.8-flash-tts`**. The retries got every line through, but only on the third attempt.
- **Tried:** (1) Changing the prompt to the documented `Say <how>: <text>` form. A re-transcription of `s05` was only the Amharic line (*እጣኑ ይጨሳል፤ ቤቱ በሰላም ይመላል።*), but it was still 11.3 s for 5 words, because "say slowly" made the delivery drag. (2) Changing the style to "warmly and gently, at a calm natural pace".
- **Fixed:** The prompt is built as `f"{style}: {text}"` in `providers.gemini_tts`. `narration.py` paces calls 21 s apart, which keeps them under 3/min instead of relying on 429-and-retry. It also warns when a line is longer than `words / 1.2 + 2.5` s, so a model that speaks extra text gets flagged automatically. `classify_http` now treats only *per-day* 429s as quota exhaustion. Per-minute 429s are retried, and errors are logged at full length so the quota metric is visible.
- **Side effect to watch:** One retake landed on the fallback `gemini-3.8-flash-lite-tts`. Lines from two different models may not sound like the same narrator. The narration cache records which `model` produced each line.

### F5: Gemini TTS free tier is 10 requests per model per day (2026-11-25)
- **Broke:** After about 13 TTS calls (smoke test plus two narration passes), every call to `gemini-3.8-flash-tts` returned **429 `generate_content_free_tier_requests`, limit 10, "requests per day"**. `gemini-3.8-flash-lite-tts` hit the same limit. The brief describes Gemini TTS as "free on the API"; the real allowance is 10 requests per model per day, which is less than a single per-line pass for two languages plus retakes.
- **Tried:** (1) Retrying, which is useless against a daily limit and wasted calls. (2) Batching all 10 lines into one request and splitting on pauses (E13): 1 request instead of 10, but the split was rejected (see E13). (3) A longer model chain.
- **Fixed / worked around:** `models.tts_fallbacks` is now a chain (`3.8-flash-lite-tts` → `3.1-flash-tts-preview` → `2.5-flash-preview-tts`). Per-day 429s mark the model exhausted in `outputs/.cache/tts_quota_exhausted.json` until midnight Pacific, so later runs skip it without spending a request. The Amharic narration was completed on `gemini-3.1-flash-tts-preview` (s01–s09) and `gemini-2.5-flash-preview-tts` (s10): **3.6–6.8 s per line, 49.6 s total**.
- **Still open:** s10 comes from a different model than s01–s09, so the narrator may not sound like the same person. Plan: after the quota resets (11:00 EAT), regenerate it on the same model as the rest, which is a one-call `make shot SHOT=s10`-style edit.
