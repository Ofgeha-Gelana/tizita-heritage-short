# Report: Tizita (ትዝታ)

A 74-second heritage short in Amharic with English subtitles, regenerated from one file (`storyboard.json`) by one command (`make film`). Every claim below points to a file, a `runs.jsonl` item, an EXPERIMENTS/FAILURES entry or a commit.

**Deliverables:** `outputs/final/tizita_am_vertical.mp4` (1080×1920) · `outputs/final/tizita_am_horizontal.mp4` (1920×1080) · `outputs/radio_cut_am.mp4` · repository with `runs.jsonl` (127 lines), FAILURES F1–F10, EXPERIMENTS E1–E17.

## 1. What I set out to make, and for whom

**Audience.** Ethiopian diaspora grandchildren aged 18–35, watching on a phone in a vertical feed, often with the sound off at first and often in their second language. They grew up with *buna* at home and now live far from their grandparents (DESIGN.md, written before any media was generated).

**Intent.** At the end the viewer should feel *tizita*, the warm ache of nostalgic memory, and the small, concrete urge to call a grandparent today. The last line says it directly: *"Don't leave before the third round. Call them today."*

**Why this story.** I considered three ideas (DESIGN.md, *Ideation*). The coffee ceremony won on three counts. It needs one character and one place, which matters when every image is generated and consistency is the main risk. The audience recognises every step. And its three rounds (*abol, tona, bereka*) give the film a built-in structure. Each of the 10 shots has a one-sentence `reason` in the storyboard.

**How it is built.** Narration with Gemini TTS, character sheets and keyframes with FLUX.2 klein (Cloudflare), motion with Wan 2.2 TI2V-5B on Kaggle T4s, the score with ACE-Step 1.5 on Kaggle, and the edit, ducking, loudness (−14 LUFS / −1 dBTP), subtitles and end card in FFmpeg. Every stage reads only its fields from the storyboard, caches by a content hash of those fields plus upstream files, and logs each model call. With nothing changed, a re-run makes **0 model calls** (COSTS.md, run `20261126T095003Z`).

## 2. What I learned from my audience

I tested twice with the same three viewers, all Ethiopian peers in the audience's age range. They watched in person on my laptop while I watched their reactions (DESIGN.md).

**Test 1: radio cut** (`outputs/radio_cut_am.mp4`: narration, subtitles and score on black). The question was whether the story lands on sound and words alone, before any GPU time is spent on pictures. All three were amazed by the audio quality. No problem with the script, voice or pacing came up, so the narration went into the full cut unchanged. That is a result, not a non-event: the most expensive stage (video) was built on a script that had already held an audience.

**Test 2: full cut** (`outputs/final/tizita_am_vertical.mp4`). All three liked it ("amazing", "good", "the rests are very good"). Each one asked for one thing, and each ask was about *identity* or *craft*, not story:

| Viewer said | What I heard | What I changed |
|---|---|---|
| "It is better if the music behind is tizita or 'hageregna'." | The score sounds generic ambient, not Ethiopian | Rewrote the score caption for traditional instruments (krar, masenqo, washint, soft kebero) and generated three new candidates. After comparing, **I kept the original (seed 22)**. The request was tested, not just accepted (`a91b3d1`) |
| "Good if it is with good photo, specially with habesha face." | The characters did not read as *our* people | Named Habesha features and *shuruba* braids in the character descriptions, chose new reference seeds, and regenerated every shot with a person (`a91b3d1`, `5def2e2`) |
| "Better if with a best video transformation… to be smooth." | Hard cuts, and clips freezing on their last frame | Crossfades between shots, motion interpolation instead of a freeze, and a slow push-in over every clip, with no change to narration timing (`pipeline/assemble.py`) |

**The lesson:** for a heritage film, "is this us?" matters as much as "is this good?". The face and the sound are what make the audience recognise themselves, and an AI model's default "Ethiopian" was not specific enough until the prompt named the features.

## 3. The hardest problem and how I solved it

**Getting Wan 2.2 to produce clips on a free 15 GB T4.** It took three Kaggle runs (FAILURES.md F8, commits `1dd5bb1`, `045ab50`).

1. **Run 1:** 0 of 10 clips. Out of memory on the first shot after 15 s, with 14.4 GiB allocated. The kernel then gave up on diffusers for *all* shots and fell back to the official repo, whose install failed on `flash_attn` and whose script died on a missing `decord` (log in `outputs/logs/`).
2. **Run 2:** 0 of 10. I added memory-efficient attention and a ladder that steps down the frame count on OOM (121 → 81 → 49). Every rung failed instantly with the *same* 14.39 GiB allocated, because the live exception traceback kept the failed attempt's activations alive. The kernel then crashed while reloading.
3. **Run 3:** **10 of 10 on the first rung.** I separated the model rather than shrinking the job. The 5B-parameter T5 text encoder runs once in fp32, split across both T4s, writes the prompt embeddings and exits. Then one worker per T4 loads the video model *without* the text encoder, fully on the GPU, and takes half the shots. Memory is released outside the `except` block. Each clip took ~13 min, and the batch about 66 min wall time, at a peak of 14.9–15.3 GB.

What made this survivable was a design choice made on day one. Every shot has a fallback (`video_mode: kenburns`), so a GPU failure costs motion, never the film. The fallback was used once: s04's clip turned the roasting pan into a camera, and switching that shot was a one-line storyboard change (`ff3ad17`, E17).

**Second hardest: keeping Emama the same person.** Conditioning FLUX.2 on her three-view reference sheet made the model copy the *layout*, so she appeared two or three times per frame. Cropping the sheet to the front view fixed all 10 shots (E14, `0d98167`).

## 4. Suggested practices, tested honestly

**A practice that beat my instinct: a reference sheet before keyframes. It works, but only as a single figure.** My instinct was to describe Emama in text in every prompt. A reference image does keep her face, shawl and dress consistent across the 5 shots she appears in (s02, s03, s04, s07, s08 in `outputs/keyframes/`), but not as the "character sheet" usually recommended. The full turnaround sheet produced duplicated characters in 4 of 10 shots. Its front-view crop produced duplicates in 0 of 10 (E14, archive in `outputs/keyframes/e14_turnaround_ref/`).

**A practice that did not work: putting delivery style in the TTS prompt.** The common advice is to steer Gemini TTS with a style line such as "warm, gentle, calm…". The model **read the instruction aloud**. Lines came out 14.8–18.9 s long instead of about 5 s, and a transcription began with the English instruction (E3a, F4). Text-only prompts give consistent 3.6–6.8 s lines. Delivery is now set by the voice choice (*Sulafat*), and the style text is kept only as a note for humans (`voice.style` in the schema says so).

**Also tested and refuted:**
- **"Fix the seed for reproducibility"** does not hold on Cloudflare's FLUX.2 klein. The same prompt and seed gave a different composition minutes later. Reproducibility now comes from the cache: `make promote` keeps the exact chosen file (E15).
- **Batching all narration into one TTS call to save quota** (10× fewer requests) failed. Pauses inside lines were as long as pauses between lines, so the split was unreliable (E13). Per-line stays the default.

## 5. Beyond the floor

- **A second language, attempted and honestly not shipped.** The pipeline handles Afaan Oromo end to end: schema, translation, per-language TTS and per-language exports. The translation step ran while `gemini-3.8-flash` was overloaded (503 on both days), so it fell back to the lite model. That draft has visible errors: a stray Arabic letter in s04, "roasting" rendered as "prepared for grain" in s03, and "first round" rendered as "the front house" in s06 (`outputs/translations/om_draft_flash_lite.txt`, E11). I chose not to voice an uncorrected text in a language whose speakers are the audience.
- **Two-GPU parallel video generation** on Kaggle's free tier: one worker per T4 with a shared embedding pre-pass, which halved the wall time.
- **Audio done to broadcast-style spec:** two-pass loudness normalisation to −14 LUFS with a −1.5 dBTP target, because the −1 target measured −0.91 after AAC encoding. The score is ducked with a sidechain compressor, not a static volume cut.
- **Failures as data:** 10 FAILURES entries, each with the exact error, what was tried and what it cost. Seven experiments have measured results, including three that refuted common advice.

## 6. What I would build next

1. **Final-quality clips.** The 10 shipped clips are drafts: 480×832, 25 steps, 81 frames. With more time I would run 704×1280 / 40 steps for the shots that earned it, using the same kernel and `QUALITY=final`.
2. **A corrected Afaan Oromo version.** Send the draft to a native speaker, write the corrected lines into the storyboard, and run `make film`. Only the 10 `om` narration calls and the `om` exports would be new.
3. **Automatic artefact checks.** The s04 morph was caught by eye. A cheap CLIP similarity check between each clip's first and last frame and its keyframe would flag morphs before assembly.
4. **Pronunciation review in the loop.** Add a field in the storyboard for a native speaker's sign-off per line, and refuse to export a language until every line has one.
