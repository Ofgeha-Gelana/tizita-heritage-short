# Ethics: consent, labelling, licences, data

## Consent

| Person | Appears as | Likeness used? | Voice used? | Consent |
|---|---|---|---|---|
| Emama (grandmother) | Main character | **No.** A fictional composite, described in text only. No photo of a real person was given to any model | No. The narration uses a stock Gemini TTS voice | Not required: not a real person |
| Grandchild (narrator) | Seen from behind, present day | No, fictional | No, stock TTS voice | Not required |

No real person's face, likeness or voice is generated, so the consent gate does not apply. The schema enforces this going forward: a character with `real_person: true` must have a `consent_ref` pointing to a record below, or `make validate` fails.

### Template, if a real relative is later included

> **Consent record: <anchor id>**
> Name / relation · Date · Recorded how (audio or video file kept privately, not in this repo) · What they agreed to: likeness ☐ voice ☐ their story ☐ · The exact narration lines they approved (paste them) · Where the film will be posted · Their right to withdraw.

## Labelling plan

- **In the film**: a 3-second end card states "Made with AI: the images, motion, music and voice were generated." The schema requires an end card line containing "AI".
- **In the file**: the MP4 `comment` metadata names the models used. Provenance metadata from the providers, such as SynthID in Gemini audio, is not stripped.
- **On the platform**: the AI-generated content disclosure is switched on when posting (TikTok "AI-generated content", YouTube "Altered or synthetic content", Meta "AI info").

## Licences

| Component | Licence / terms | Notes |
|---|---|---|
| Wan 2.2 TI2V-5B | Apache 2.0 | Outputs may be used commercially |
| ACE-Step 1.5 | MIT; trained on licensed, royalty-free and synthetic data | Outputs may be used commercially |
| FLUX.2 klein 4B | Apache 2.0 | The 9B variant is non-commercial and is **not** used |
| FLUX.1 schnell (fallback) | Apache 2.0 | |
| Gemini Flash / TTS (free tier) | Google Gemini API Additional Terms | On the free tier, Google may use inputs and outputs to improve its products (see Data below) |
| Cloudflare Workers AI | Cloudflare terms; model licences above | |
| FFmpeg | LGPL/GPL (Homebrew build) | A tool only; not distributed |
| Kefa font (end card, macOS system font) | Apple system font licence | Used for rendering; not redistributed |
| This repository's code | MIT (add `LICENSE` if publishing) | |

## Style and identity

The prompts describe qualities (gouache, grain, golden light, earthy palette). They never name a living artist.

## Private data

The free Gemini tier may use prompts for training. No private family details go into any prompt: no real names, addresses, or photos. The story is a composite of a common ritual.
