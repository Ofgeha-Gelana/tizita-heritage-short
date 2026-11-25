# Architecture

## Shape

Six stages surround one contract file. Each stage is a module that reads `storyboard.json`, validates it, does its work, and writes files into `outputs/`. Stages never call each other's internals. They meet only through the storyboard and through files at known paths.

| Stage | Module | Reads from storyboard | Writes | Model / tool | Runs on |
|---|---|---|---|---|---|
| script | `pipeline/script.py` | narration, subtitle_en | narration.<lang> (only with `--write`) | Gemini Flash / Flash-Lite | Gemini API |
| narration | `pipeline/narration.py` | narration.<lang>, voice | `outputs/narration/<lang>/<shot>.wav` | Gemini TTS (fallback TTS model) | Gemini API |
| keyframes | `pipeline/keyframes.py` | characters, style_block, image_prompt, seed | `outputs/references/*.png`, `outputs/keyframes/<shot>.png` | FLUX.2 klein 4B (fallback FLUX.1 schnell) | Cloudflare Workers AI |
| shots | `pipeline/shots.py` + `kaggle/wan22_i2v/run.py` | motion_prompt, seed, video_mode, format | `outputs/clips/<shot>.mp4` | Wan 2.2 TI2V-5B | Kaggle T4 |
| music | `pipeline/music.py` + `kaggle/acestep/run.py` | music.* | `outputs/music/candidate_seed<N>.wav` | ACE-Step 1.5 turbo | Kaggle T4 |
| assemble | `pipeline/assemble.py` | everything | `outputs/final/tizita_<lang>_<export>.mp4`, `.srt`, `.report.json` | FFmpeg (no AI) | local |

`pipeline/run.py` runs these in order (`make film`). Every stage can also be run on its own through the Makefile.

## The contract

`schema/storyboard.schema.json` (JSON Schema 2020-12, `additionalProperties: false` throughout) defines:

- **Film-level fields**: audience, intent, languages, format (master aspect, keyframe and video sizes, export list), `style_block` (appended to every image prompt), and the model ids. Because the model ids live here rather than in code, swapping a model is a one-line data change that also invalidates the right caches.
- **Characters**: one fixed `description` that is attached verbatim to every prompt the character appears in, plus the reference-sheet prompt, candidate seeds and `chosen_seed`. A `real_person: true` character must carry a `consent_ref` into `ETHICS.md`, and the schema enforces this.
- **Shots** (8–12): `id`, `beat`, `reason` (why the shot exists), `shot_size`, `duration_s`, `characters`, `image_prompt`, `motion_prompt` (motion and camera only, because the keyframe already holds the subject), `seed`, `video_mode` (`wan` | `kenburns`), `narration` keyed by language, and `subtitle_en`.
- **Music**: caption, bpm, key, duration, candidate seeds, `chosen`, `choice_reason`, and the spotting `cues`.
- **End card**: lines, one of which must contain the word "AI".

`common.validate_storyboard()` adds the checks a schema cannot express: unique shot ids, characters that exist, and a planned runtime of 60–90 s. Every stage calls `load_storyboard()`, so an invalid storyboard stops the run before any model is called.

## Caching: regenerate only what changed

`common.Cache(stage)` keeps a manifest in `outputs/.cache/<stage>.json` that maps an item id to `{key, path}`. An item is fresh when the key computed from its current inputs equals the stored key and the file still exists. Keys are `sha256` over the exact inputs that stage uses:

| Item | Key inputs |
|---|---|
| narration `am/s03` | language, text, voice, style instruction, TTS model |
| reference `emama/202` | sheet prompt + description + style block, seed, model, size |
| keyframe `s03` | full prompt (shot + characters + style), seed, model, size, **bytes of the chosen reference images** |
| clip `s03` | **bytes of the keyframe**, motion prompt, seed, model, size, frames, steps |
| music `seed22` | caption, bpm, key, duration, seed, model |
| segment / mix (assembly) | source file bytes, timing, export size |

Because the file hashes of upstream outputs are part of each key, a change propagates exactly as far as it should. Editing one narration line regenerates one wav, and the timeline and mix are then rebuilt locally. Editing one `image_prompt` regenerates that keyframe, and then that clip, because its keyframe bytes changed. Choosing a different reference sheet regenerates only the shots that character appears in. A second `make film` with no edits logs zero model calls (`make costs` lists calls per run).

## Logging

`common.log_call()` appends one JSON line per model call attempt to `runs.jsonl`. Each line records: `ts`, `run_id`, `stage`, `item`, `provider`, `model`, `params` (prompt, size, voice and so on), `seed`, `latency_ms`, `gpu_s`, `status`, `attempt`, `error`, `output`, and `cost_usd_list_price`. Failed attempts are logged too, which keeps the failure history visible. Kaggle jobs report their own wall time, GPU name and peak VRAM from inside the kernel. Those values are logged when results are collected. Assembly writes one `provider: "local"` line per export, recording duration, LUFS and true peak.

## Failure handling

| Failure | Handling |
|---|---|
| HTTP 429 (rate) / 5xx / timeout | `with_retries`: 3 attempts, jittered exponential backoff (2 s, 4 s …), each attempt logged |
| Daily quota exhausted (`QuotaExhausted`) | No retry; the stage switches to the fallback model (`tts_fallback`, `image_fallback`) and logs `fallback: true` |
| TTS fails on every model | The stage stops with a message, because a film without its narration is not worth assembling |
| Wan job fails in the kernel | The kernel catches it per job, tries the official `generate.py` backend, and records the error in `results.json`. Results are saved after every job, so a kernel timeout keeps finished clips |
| Wan fails 3× for a shot | `shots.py` stops resubmitting it, and assembly animates the keyframe with a Ken Burns pan/zoom |
| Kaggle unavailable / `OFFLINE=1` | GPU stages are skipped. Assembly uses clip → Ken Burns keyframe → black, per shot |
| Stale Kaggle dataset version | Each job carries its cache key, and results are accepted only when the key matches the last submission |
| No chosen score | The placeholder drone (deterministic, synthesised in FFmpeg) is used and the run says so |
| Missing Ethiopic font on another machine | The end card drops the Ge'ez characters rather than drawing empty boxes |

## Assembly details

- **Timing follows the voice.** Each shot lasts `max(duration_s, 0.4 s + narration + 0.6 s)`. Each language therefore gets its own timeline: Afaan Oromo runs longer than Amharic.
- **Picture.** A Wan clip has its first 0.25 s trimmed, is slowed by at most 1.6×, and then holds its last frame. A Ken Burns segment is a 2× pre-scaled `zoompan`, with the direction cycling through in / right / out / left / up unless the shot sets one. The master is 9:16. The 9:16 export is crop-filled, and the 16:9 export puts the vertical picture over a blurred, darkened copy of itself. This trade-off is discussed in `EXPERIMENTS.md`.
- **Sound.** The narration is placed with `adelay` and normalised to −16 LUFS. The music is normalised to −16 LUFS, then lowered by `music_gain_db` and faded. It is ducked with `sidechaincompress=threshold=0.02:ratio=6:attack=300:release=1500`, keyed by the voice, then mixed. A **two-pass `loudnorm`** brings the final mix to −14 LUFS integrated, −1 dBTP.
- **Subtitles.** These are ASS cards of at most 2 lines × 40 characters, timed proportionally across the narration. In 9:16 they sit 22% above the bottom edge, clear of the feed's on-screen buttons. They are burned in, and an `.srt` file is also written.
- **End card.** A Pillow-rendered still is shown for 3 s with fades. The MP4 `comment` metadata also states that the film is AI-made.
- Each export writes `<name>.report.json` with its measured duration, LUFS, true peak and per-shot source (clip / kenburns / black). This lets a reader verify the deliverables without opening the video.

## Why this shape

- **One JSON contract** is what makes partial regeneration and the "change one line live" demo possible. It also keeps prompts and seeds out of the code.
- **Kaggle as a batch queue, not an interactive notebook.** GPU time is the binding constraint, so the pipeline sends only changed shots, all in one kernel run, and can disconnect while it runs (`shots-submit` / `shots-collect`).
- **Fallbacks at every GPU stage** mean `make film` always produces a watchable film, and every fallback is logged where it happened.
