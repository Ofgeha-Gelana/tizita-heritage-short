# Experiments

Each entry follows **hypothesis → change → result → decision**. Only one variable changes per entry, and the seed is fixed unless the seed is the variable. Evidence is the `runs.jsonl` lines (filter by `run_id` or `item`) and the files named.

The first entries below are planned. Their results are filled in only from real runs. An entry without a result is still open.

| # | Hypothesis | Change (one variable) | Evidence | Result | Decision |
|---|---|---|---|---|---|
| E1 | Passing the chosen reference sheet into FLUX.2 klein keeps Emama recognisably the same across shots, better than text description alone | `s02,s04,s08` with refs vs with `chosen_seed: null`, same seeds | `outputs/keyframes/explore/`, runs.jsonl `stage=keyframes` | | |
| E2 | The same reference prompt with 4 seeds gives at least one usable sheet | `candidate_seeds` 101/202/303/404 | `outputs/references/` | | |
| E3a | A free-standing style paragraph steers Gemini TTS without being spoken | style paragraph + blank line + text (first pass) vs `Say <how>: <text>` | runs.jsonl `stage=narration`; transcription of s05 | **Refuted.** The paragraph was spoken aloud: 14.8–18.9 s per line, and the transcript starts with the English instruction. With `Say …:` the transcript is only the Amharic | Use `Say <how>: <text>`; keep the duration check in code (F4) |
| E3 | A style instruction improves Amharic TTS more than switching voices | voice `Sulafat` + style vs `Sulafat` without style vs another voice + style, shot s08 | runs.jsonl `stage=narration`, durations | | |
| E4 | Amharic narration runs slower than the 2 words/s rule of thumb | measured wav duration ÷ word count (`make wordcount` vs report.json) | narration cache `duration_s` | | |
| E5 | Drafting Wan at 480×832 / 25 steps costs under a third of the GPU time of 704×1280 / 40 steps | one shot, both settings, same seed | runs.jsonl `stage=shots`, `gpu_s` | | |
| E6 | fp16 on a T4 produces clean Wan frames (no black / NaN frames) | the dtype the kernel picks | kernel log `outputs/logs/kaggle_wan22_i2v_*.log` | | |
| E7 | A motion-only prompt is more stable than re-describing the subject | s04 motion prompt with vs without subject text | clips + notes | | |
| E8 | The ACE-Step LM planner ("thinking") improves fit with the caption | 3 candidates with planner on vs off | `outputs/music/`, runs.jsonl `params.planner` | | |
| E9 | Ducking at ratio 6 keeps the narration intelligible at music gain −6 dB on a phone speaker | `music_gain_db` −6 vs −10, listened to on a phone | `report.json` LUFS + listening notes | | |
| E10 | A blur-fill 16:9 from the 9:16 master is acceptable compared with re-generating horizontal keyframes | blind A/B with ≥5 people on one shot | votes | | |
| E11 | Gemini Flash translates Amharic narration into natural spoken Afaan Oromo | `make translate TO=om`, then back-translate and a native-speaker check | `pipeline.script backtranslate --lang om` output | | |
| E12 | Gemini TTS pronounces Afaan Oromo (Qubee) names and long vowels correctly | om narration rated by a native speaker | narration files + notes | | |
| E13 | Speaking all 10 lines in ONE TTS request and splitting at pauses gives usable per-shot files for 1/10 of the quota | batch mode vs per-line, Amharic, same voice | `outputs/narration/am/_batch_last.wav`; runs.jsonl `item=am/batch:…` | **Not usable yet.** The batch was 58.6 s of audio (vs 49.6 s for per-line takes). Gaps between lines were 0.84–1.04 s, and pauses inside lines at `፣`/`፤` were 0.52–0.72 s, so 'N-1 longest pauses' mis-cut. Predicting boundaries from text length got close, but some boundaries were 2–3 s off because the pace varies by line; the proportion check rejected the split | Keep per-line as the default. Batch stays as a quota saver with an automatic fallback. Next idea: an explicit separator token between lines |
