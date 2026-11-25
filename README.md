# Tizita (ትዝታ): a heritage short made by a pipeline

A 60–90 second short film about a grandmother's coffee ceremony, narrated in **Amharic** (plus an **Afaan Oromo** version), with English subtitles, an original score and an AI end card. It is exported as vertical 9:16 and horizontal 16:9.

The film is not edited by hand. `storyboard.json` is the single contract. Each stage in `pipeline/` reads it and writes its outputs. Every model call is logged to `runs.jsonl`, and a re-run regenerates only what changed.

```
storyboard.json ──► narration (Gemini TTS) ───────────────────────────────┐
      │        ──► keyframes (FLUX.2 klein, Cloudflare) ──► shots (Wan 2.2, Kaggle T4) ──► assemble (FFmpeg) ──► outputs/final/*.mp4
      │        ──► music (ACE-Step 1.5, Kaggle) ──────────────────────────┘
      └─ validated against schema/storyboard.schema.json before every stage
```

## Run it

```bash
git clone <this repo> && cd tizita-heritage-short
brew install ffmpeg uv          # or: apt install ffmpeg && pipx install uv
make setup                      # Python 3.11 + deps via uv, creates .env from .env.example
make film                       # the whole pipeline, end to end
```

`make film OFFLINE=1` makes no model calls. It assembles the film from what is already in `outputs/`, so it works without any keys. If a clip or keyframe is missing, a fallback fills the gap.

### Keys (all free, no payment card)

| Variable | Where | Used for |
|---|---|---|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) | narration (TTS), translation |
| `CF_ACCOUNT_ID`, `CF_API_TOKEN` | Cloudflare dashboard → Workers AI → API token with *Workers AI: Read/Edit* | keyframes |
| `KAGGLE_API_TOKEN` (`KGAT_…`) + `KAGGLE_USERNAME` | kaggle.com → Settings → API → *Create New Token* (Kaggle CLI ≥ 2 reads the token from the environment; a legacy `~/.kaggle/kaggle.json` also works). The account must be phone-verified for GPUs | Wan 2.2 video, ACE-Step music |

`.env` is git-ignored. Run `make smoke` once to call each provider and log its latency.

### Everyday commands

| Command | What it does |
|---|---|
| `make validate` | Checks the storyboard against the schema, plus semantic checks (unique ids, known characters, 60–90 s runtime, an end card that mentions AI, consent for real people) |
| `make radio-cut` | Narration over a placeholder drone on black, with subtitles, written to `outputs/radio_cut_am.mp4`. This is the first prototype to test with the audience |
| `make refs` | Character reference-sheet candidates. Pick one and set `reference.chosen_seed` |
| `make keyframes [SHOT=s03]` | One keyframe per shot, conditioned on the chosen reference sheet |
| `make explore SHOT=s03 SEEDS=1,2,3,4` | Seed exploration for one shot |
| `make shots [QUALITY=final]` | Wan 2.2 image-to-video on Kaggle for changed shots only. `shots-submit` / `shots-collect` let you launch a run and fetch it later |
| `make music` | Three ACE-Step candidates. Set `music.chosen` and `music.choice_reason` |
| `make translate TO=om` | Translates Amharic narration to Afaan Oromo with Gemini and writes it into the storyboard |
| `make assemble` | Edit, duck, subtitles, loudness, end card, both aspect ratios, every language |
| `make shot SHOT=s03` | Regenerates one shot and re-assembles. This is what the walkthrough shows |
| `make costs` | Per-stage latency, GPU minutes and list-price table from `runs.jsonl` |
| `make test` | Unit tests |

### Changing one shot

Edit `shots[i].narration.am` or `shots[i].image_prompt` in `storyboard.json`, then run `make film`. Only that item's cache key changes, so `runs.jsonl` gains one new model call and the other shots report `cached`.

## What it costs

Everything runs on free tiers. `COSTS.md` holds the measured latency and GPU minutes per stage, plus a list-price equivalent computed from `runs.jsonl` (`make costs`).

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

The film is labelled as AI-made on the end card, in the file metadata, and through the platform's AI-content label when posted.
