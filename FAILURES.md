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

### F7: Kaggle `kernels push` returned 409 Conflict (2026-11-25)
- **Broke:** `make shots-submit` uploaded the inputs dataset `tizita-wan22-i2v`, then `kaggle kernels push` failed with `409 Client Error: Conflict ... SaveKernel`. The error says nothing about the cause.
- **Tried:** (1) Renaming the dataset to `…-inputs` still gave 409, because the old dataset kept the slug. (2) Bisected with throwaway kernels: CPU, GPU and T4 with internet all pushed fine, and a trivial kernel named `tizita-wan22-i2v` still gave 409. So the slug was taken: kernels and datasets appear to share one namespace per user. (3) The probes then used up the batch tier's limit of 2 concurrent GPU sessions (`Maximum batch GPU session count of 2 reached`), so we deleted them once they had finished.
- **Fixed:** The kernel is now `tizita-wan22-shots` and its datasets are `tizita-<name>-inputs`. The Wan batch (10 draft jobs) and the ACE-Step batch then pushed. The limit of 2 sessions means Wan and ACE-Step can run together, but nothing else can run alongside them.

### F8: Wan 2.2 ran out of memory on a Kaggle T4, and the fallback could not install (2026-11-26)
- **Broke:** Kernel v1: `diffusers-i2v-float16` loaded, then s01 failed after 15 s with `CUDA out of memory ... GPU 0 has a total capacity of 14.56 GiB of which 98.81 MiB is free`. The kernel then dropped diffusers for *every* remaining shot and used the official repo, whose `requirements.txt` failed to build `flash_attn`, and `generate.py` died on `ModuleNotFoundError: No module named 'decord'`. Result: 0/10 clips, about 9 minutes of GPU time. Log: `outputs/logs/kaggle_wan22_i2v_1795640071.log`.
- **Diagnosis:** 121 frames at 480×832 is about 48k video tokens. A T4 (sm75) has no flash attention, so PyTorch SDPA may fall back to the "math" kernel, which builds the full attention matrix. The fp16 5B transformer alone is about 10 GB of the 15 GB.
- **Fixed (kernel v2):** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The math and flash SDPA backends are disabled, so only memory-efficient attention runs. On OOM the kernel walks a ladder (121 → 81 → 49 frames, then sequential CPU offload) and keeps the rung that worked for later shots. Each result records `requested_frames`, `num_frames` and `ladder_rung`. A shorter clip is acceptable because assembly slows clips by up to 1.6× and then holds the last frame. The official fallback now installs only `decord easydict einops dashscope opencv-python-headless`.
- **Run 2 (v2) also failed:** The ladder stepped down instantly. Every rung hit OOM with the same 14.39 GiB still allocated, because the live exception traceback kept the failed run's activations alive. The kernel then crashed with no log line while reloading for sequential offload, probably out of host RAM.
- **Fixed (v3):** The UMT5 text encoder runs once, in fp32, split across both T4s (`device_map="auto"`), saves the embeddings, and exits. Then one worker process per T4 loads the pipeline *without* the text encoder, fully on the GPU, and takes half the shots. Memory is released outside the `except` block before each retry. Frames are capped at 81, with a 4.5 h budget.
- **Result:** **10/10 clips on the first rung** (81 frames, 480×832, 25 steps, fp16, no offload). Each took 768–792 s, two in parallel, so about 66 min of wall time. Peak VRAM was 14.9–15.3 GB on a 15 GB card. Total GPU time across runs 1–3 was about 2.3 h.

### F9: ACE-Step 1.5 does not install on Kaggle's Python 3.13 (2026-11-26)
- **Broke:** Kernel v1 failed in 25 s: `ERROR: Package 'ace-step' requires a different Python: 3.13.16 not in '<3.13,>=3.11'`. The `requirements.txt` fallback then failed to build `flash-attn`.
- **Diagnosis:** The Kaggle image moved to Python 3.13. ACE-Step's `nano-vllm` dependency ships a prebuilt flash-attn wheel only for CPython 3.11 on Linux x86_64.
- **Fixed (kernel v2):** We follow the README instead of pip: `pip install uv`, then `uv sync --python 3.11` (uv downloads its own 3.11), then `os.execve` the same script inside `.venv` with `ACE_INNER=1`. On a T4 (sm75, which flash-attn does not support) the LM planner uses the `pt` backend directly instead of trying `vllm` first.
- **Result:** v2 installed and generated 3 × 80 s candidates in 6–9 s each, after about 60 s of model loading. The LM planner ran on the `pt` backend. A follow-up: `kaggle kernels output` then tried to list the repo, venv and weights in `/kaggle/working`, hit `429 Too Many Requests`, and hung for over 10 minutes (F10).

### F10: Collecting kernel output was rate-limited (2026-11-26)
- **Broke:** `make music-collect` downloaded the whole `/kaggle/working`, including the cloned ACE-Step repo, its `.venv` and the model checkpoints. One attempt ran for over 10 minutes, the next was refused with `429 Too Many Requests ... You have been rate limited`, and another hit a connect timeout.
- **Fixed:** (1) The ACE-Step kernel keeps its repo and temp files in `/tmp`, so only results, logs and audio become output. (2) `kaggle_job.collect` passes `--file-pattern` so it downloads only `results.json`, `*.log`, `clips/*.mp4` and `music/*.wav`. (3) Gemini/Cloudflare network timeouts are now raised as `TransientError` and retried.

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
