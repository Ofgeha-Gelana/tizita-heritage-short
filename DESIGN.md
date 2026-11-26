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

*Prototype: `outputs/radio_cut_am.mp4`, Amharic narration over a placeholder drone on black, with English subtitles.*

> **To fill in after showing it to at least three people from the audience.** Record who (a role, not a name: for example "cousin, 24, Toronto, understands Amharic but reads it slowly"), how (in person / WhatsApp / Slack), what they said in their own words, and where they reacted. Do not summarise. Quotes are the evidence.

| # | Who (role, not name) | How | What they said / did | Insight |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**What changes in the storyboard because of this.** List each change with the shot id. The git diff of `storyboard.json` after this commit is the proof.

-

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

- _(filled in as each change is made, with its commit hash)_
