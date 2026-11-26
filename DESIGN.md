# Design

## Audience and intent

*Written before any image, clip or score was generated, on 2026-11-25.*

**Who.** Ethiopian diaspora grandchildren aged 18–35. They grew up with buna at home, now live far from their grandparents, and often speak Amharic or Afaan Oromo as a second language.

**Where.** On a phone, in a vertical feed (TikTok, Instagram Reels, YouTube Shorts), often without sound at first. This is why the subtitles are burned in and kept inside the safe area. Family members also watch, forwarded on WhatsApp or Telegram.

**What they should feel at the end.** The warm ache of *tizita*, the nostalgic memory the film is named after, and the small, concrete urge to call a grandparent today.

**Define.** How might we help a grandchild abroad feel the pull of the morning coffee ceremony strongly enough to pick up the phone?

**Ideation (options considered before choosing).**
1. A grandmother's coffee ceremony as the thread of memory (chosen). It has one character and one place, which keeps consistency manageable. The ritual is universally recognised by the audience, and the three rounds give it a built-in structure.
2. A walk to the market, which needs a crowd and several locations and so carries a higher consistency risk.
3. A proverb told as a fable, which is more abstract and less personal.

**Constraints.** 60–90 s, eight to twelve shots, free-tier models only, three days. The characters are fictional composites, so no real likeness or voice is used (see `ETHICS.md`).

**Why each shot exists.** Each shot's `reason` field in `storyboard.json` states it.

## User test 1: radio cut

*Prototype: `outputs/radio_cut_am.mp4`: Amharic narration, English subtitles and the ACE-Step seed 22 score on a black screen (commit `2e5e43a`), shown on 2026-11-26 before the full cut.*

| # | Who (role, not name) | How | What they said / did | Insight |
|---|---|---|---|---|
| 1–3 | Viewers A, B, C (the same three as test 2) | In person, on the author's laptop, author watching | Author's note: all three were amazed by the audio quality | On sound alone, the narration, voice and mix held their attention. The audio carried the story without pictures, so the script and narration stayed unchanged |

**What changes in the storyboard because of this.** Nothing. The test was meant to find problems with the script, voice or pacing before spending GPU time on pictures, and none came up. The narration lines and their timing went into the full cut unchanged.

## User test 2: near-final cut

*Prototype: `outputs/final/tizita_am_vertical.mp4` (commit `e288566`), shown on 2026-11-26.*

| # | Who (role, not name) | How | What they said / did | Insight |
|---|---|---|---|---|
| 1 | Viewer A, woman, peer in the cohort's age range (Ethiopian) | In person, on the author's laptop, author watching | "It is amazing, it is better if the music behind is tizita or 'hageregna'. The rests are very good." | The score reads as generic ambient, not as Ethiopian music. She wants the traditional *tizita* mode or *hageregna* (folk/home-country) sound the film is named after |
| 2 | Viewer B, man, peer (Ethiopian) | In person, on the author's laptop, author watching | "It is good, but good if it is with good photo, specially with habesha face." | The characters do not read clearly as Habesha (Ethiopian/Eritrean highland) faces. For this audience, recognising "our people" is part of the pull |
| 3 | Viewer C, man, peer (Ethiopian) | In person, on the author's laptop, author watching | "It is amazing, but better if with a best video transformation… to be smooth." | The hard cuts between shots, and the hold at the end of each slowed clip, feel abrupt. Smoother transitions are wanted |

Viewers are listed by role; their names are kept off the public repo. The author watched their reactions while they watched, not only what they said afterwards.

All three liked the film overall ("amazing", "good", "the rests are very good"). Nobody commented on the narration, story or subtitles, so those stay. The three asks are about **sound identity**, **face identity** and **smoothness**.

**What changed.**

- **Habesha faces:** Emama's and the grandchild's descriptions now name Habesha features (an oval face with high cheekbones, a slim straight nose, almond-shaped eyes) and *shuruba* braids. The s07 neighbours are described as Habesha. New reference seeds were chosen (707, 444) and the 6 keyframes with people regenerated. The 5 Wan clips are being regenerated (s04 stays a Ken Burns move) (storyboard commit `a91b3d1`, keyframes `5def2e2`).
- **Smooth transitions:** 0.5 s crossfades between shots. Slowed clips use motion interpolation up to 2.4×, not frame duplication up to 1.6× followed by a freeze, and a slow push-in keeps every clip segment moving. Narration and subtitle timing are unchanged (`pipeline/assemble.py`).
- **Traditional music:** the score caption was rewritten for tizita / hageregna instruments (krar, masenqo, washint, soft kebero) and three new candidates generated (seeds 44, 55, 66). After listening, the director **kept seed 22**: the request was heard and tested, and the original stayed.
