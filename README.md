# Tizita (ትዝታ): a heritage short made by a pipeline

A 74-second short film about a grandmother's coffee ceremony, narrated in **Amharic**, with English subtitles, an original score and an AI end card. It is exported as vertical 9:16 and horizontal 16:9. An Afaan Oromo version is wired into the pipeline but not shipped; see [Status](#status).

The film is not edited by hand. `storyboard.json` is the single contract. Each stage in `pipeline/` reads it and writes its outputs. Every model call is logged to `runs.jsonl`, and a re-run regenerates only what changed.

```
storyboard.json ──► narration (Gemini TTS) ───────────────────────────────┐
      │        ──► keyframes (FLUX.2 klein, Cloudflare) ──► shots (Wan 2.2, Kaggle T4) ──► assemble (FFmpeg) ──► outputs/final/*.mp4
      │        ──► music (ACE-Step 1.5, Kaggle) ──────────────────────────┘
      └─ validated against schema/storyboard.schema.json before every stage
```

## The film

| File | Format |
|---|---|
| [`outputs/final/tizita_am_vertical.mp4`](outputs/final/tizita_am_vertical.mp4) | 1080×1920, 24 fps, H.264/AAC, 73.8 s |
| [`outputs/final/tizita_am_horizontal.mp4`](outputs/final/tizita_am_horizontal.mp4) | 1920×1080 (the 9:16 master over a blurred fill), same audio |
| [`outputs/final/tizita_am.en.srt`](outputs/final/tizita_am.en.srt) | English subtitles as a sidecar (also burned in) |
| [`outputs/radio_cut_am.mp4`](outputs/radio_cut_am.mp4) | Audio-first prototype: narration, subtitles and score on black, used for the first audience test |

Loudness is −14.2 LUFS integrated, −1.4 dBTP true peak (`outputs/final/*.report.json`). The score is ducked under the voice with a sidechain compressor. 9 of 10 shots are Wan 2.2 image-to-video clips; s04 is a Ken Burns move on its keyframe because its clip morphed (EXPERIMENTS.md E17).

| Stage | Model | Where it runs |
|---|---|---|
| Narration | Gemini TTS (`gemini-3.1-flash-tts-preview`, voice *Sulafat*) | Gemini API, free tier |
| Keyframes and character sheets | FLUX.2 klein 4B | Cloudflare Workers AI, free tier |
| Motion | Wan 2.2 TI2V-5B (diffusers) | Kaggle, 2× T4 |
| Score | ACE-Step 1.5 | Kaggle, T4 |
| Edit, mix, subtitles, end card | FFmpeg + Pillow | locally |

## Run it

```bash
git clone <this repo> && cd tizita-heritage-short
brew install uv                 # or: pipx install uv / curl -LsSf https://astral.sh/uv/install.sh | sh
make setup                      # uv-managed Python + deps, creates .env from .env.example
make film                       # the whole pipeline, end to end
```

FFmpeg comes from the `imageio-ffmpeg` wheel (a static ffmpeg 9.0 with libx264 and libass), so no system install is needed. A system `ffmpeg` on `PATH` is used instead if present (FAILURES.md F1).

`make film OFFLINE=1` makes no model calls. It assembles the film from what is already in `outputs/`, so it works without any keys. If a clip or keyframe is missing, a fallback fills the gap.

### Keys (all free, no payment card)

| Variable | Where | Used for |
|---|---|---|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) | narration (TTS), translation |
| `CF_ACCOUNT_ID`, `CF_API_TOKEN` | Cloudflare dashboard → Workers AI → API token with *Workers AI: Read/Edit* | keyframes |
| `KAGGLE_API_TOKEN` (`KGAT_…`) + `KAGGLE_USERNAME` | kaggle.com → Settings → API → *Create New Token* (Kaggle CLI ≥ 2 reads the token from the environment; a legacy `~/.kaggle/kaggle.json` also works). The account must be phone-verified for GPUs. At most 2 GPU jobs run at once | Wan 2.2 video, ACE-Step music |

`.env` is git-ignored. Run `make smoke` once to call each provider and log its latency.

### Everyday commands

| Command | What it does |
|---|---|
| `make validate` | Checks the storyboard against the schema, plus semantic checks (unique ids, known characters, 60–90 s runtime, an end card that mentions AI, consent for real people) |
| `make radio-cut` | Narration over the chosen score (or a placeholder drone before one is chosen) on black, with subtitles, written to `outputs/radio_cut_am.mp4`. This is the first prototype to test with the audience |
| `make refs` | Character reference-sheet candidates. Pick one and set `reference.chosen_seed` |
| `make keyframes [SHOT=s03]` | One keyframe per shot, conditioned on the chosen reference sheet |
| `make explore SHOT=s03 SEEDS=1,2,3,4` | Seed exploration for one shot |
| `make promote SHOT=s03 SEED=2` | Keeps an explored image exactly. FLUX.2 klein on Cloudflare is not deterministic per seed, so the cache, not the seed, is what makes it reproducible (E15) |
| `make shots [QUALITY=final]` | Wan 2.2 image-to-video on Kaggle for changed shots only. `shots-submit` / `shots-collect` let you launch a run and fetch it later |
| `make music` | Three ACE-Step candidates. Set `music.chosen` and `music.choice_reason` |
| `make translate TO=om` | Translates Amharic narration to Afaan Oromo with Gemini and writes it into the storyboard. Check the output with a native speaker first; `pipeline.script backtranslate` helps |
| `make assemble` | Edit, duck, subtitles, loudness, end card, both aspect ratios, every language |
| `make shot SHOT=s03` | Regenerates one shot and re-assembles. This is what the walkthrough shows |
| `make costs` | Per-stage latency, GPU minutes and list-price table from `runs.jsonl` |
| `make report` | Renders `REPORT.md` to `report.pdf` with headless Chrome (no pandoc or LaTeX needed) |
| `make test` | Unit tests |

### Changing one shot

Edit `shots[i].narration.am` or `shots[i].image_prompt` in `storyboard.json`, then run `make film`. Only that item's cache key changes, so `runs.jsonl` gains one new model call and the other shots report `cached`.

## What it costs

$0 spent: everything runs on free tiers. At list price one film costs about **$0.83**, roughly 90% of it the Wan 2.2 clips (~13 T4-minutes each). A re-run with no storyboard change makes 0 model calls. `COSTS.md` has the per-stage latency, GPU minutes and the free-tier limits we actually hit, all from `runs.jsonl` (`make costs`).

## Status

| | |
|---|---|
| Done | Amharic film in both aspect ratios; radio cut; 10 keyframes with a consistent character; 10 Wan clips; chosen ACE-Step score; 10 FAILURES entries; 17 EXPERIMENTS rows, 7 with results |
| Not shipped | **Afaan Oromo version.** The translation came from the fallback text model while the main one was overloaded, and it has visible errors (a stray Arabic letter, mistranslated ceremony terms); see `outputs/translations/`. It is not voiced until a native speaker corrects it. The pipeline handles `om` end to end once the text is in `storyboard.json` |
| Known limits | Wan clips are 81 frames (3.4 s) at 480×832, a T4 memory limit (F8). Assembly slows each by up to 1.6× and holds the last frame to fill the shot. The narration was not reviewed for pronunciation by a native speaker before shipping; that review belongs in DESIGN.md |

## Repository map

| Path | Contents |
|---|---|
| `storyboard.json` | Shots, durations, prompts, seeds, narration, music cues, and the stated reason for each shot |
| `schema/` | JSON Schema for the storyboard |
| `pipeline/` | `common.py` (contract, cache, retries, log), `providers.py`, and one module per stage |
| `kaggle/` | The GPU kernels pushed to Kaggle by `pipeline/kaggle_job.py` |
| `outputs/` | Generated media, cache manifests (`outputs/.cache`) and per-export reports |
| `runs.jsonl` | One line per model call |
| `DESIGN.md` · `ARCHITECTURE.md` · `EXPERIMENTS.md` · `FAILURES.md` · `COSTS.md` · `ETHICS.md` · `REPORT.md` | Process documentation |

The film is labelled as AI-made on the end card, in the MP4 `comment` metadata, and through the platform's AI-content label when posted. The characters are fictional composites; no real person's likeness or voice is used (ETHICS.md).
