"""Stage: script (Gemini Flash). Text work on the storyboard itself.

    python -m pipeline.script translate --to om [--write]
        Translate the primary-language narration of every shot into another
        language, using the English subtitle as a meaning anchor. Prints a diff;
        with --write, saves into storyboard.json (after validation).
    python -m pipeline.script backtranslate --lang om
        Translate a language back to English so a non-speaker can spot drift.
    python -m pipeline.script wordcount
        Words per shot and estimated speaking time, before any TTS call.

Shot ids are passed in and must come back unchanged; the response is JSON-checked.
"""
from __future__ import annotations

import argparse
import json

from . import providers as P
from .common import (estimate_cost, load_storyboard, log_call, main_guard, save_storyboard, say,
                     timer, with_retries)

STAGE = "script"
LANG_NAMES = {"am": "Amharic (Ge'ez script)", "om": "Afaan Oromo (Qubee Latin script)",
              "ti": "Tigrinya (Ge'ez script)", "so": "Somali", "en": "English"}


def ask_json(model: str, prompt: str, item: str, seed: int = 7) -> dict:
    def call(attempt):
        with timer() as t:
            text, usage = P.gemini_text(model, prompt, temperature=0.3, seed=seed, json_mode=True)
        log_call(stage=STAGE, provider="gemini", model=model, item=item, seed=seed, attempt=attempt.n,
                 latency_ms=t["ms"], params={"prompt_chars": len(prompt), "usage": usage},
                 cost_usd=estimate_cost(model))
        return json.loads(text)
    return with_retries(call, stage=STAGE, item=item)


def translate(target: str, write: bool) -> None:
    sb = load_storyboard()
    src = sb["languages"]["primary"]
    lines = [{"id": s["id"], "source": s["narration"][src], "english_meaning": s["subtitle_en"]} for s in sb["shots"]]
    prompt = f"""You are translating narration for a short family heritage film about an Ethiopian coffee ceremony.
Translate each "source" line from {LANG_NAMES[src]} into natural, spoken {LANG_NAMES[target]}.
Write for the ear: short sentences, warm and simple, as a grandchild would speak to family.
Keep cultural words as they are said in {LANG_NAMES[target]} (for example the names of the three coffee rounds).
Use "english_meaning" only to check meaning. Return JSON: {{"lines": [{{"id": ..., "text": ...}}]}} with the same ids.

{json.dumps(lines, ensure_ascii=False, indent=1)}"""
    out = ask_json(sb["models"]["text"], prompt, f"translate/{src}->{target}")
    got = {r["id"]: r["text"].strip() for r in out.get("lines", [])}
    ids = [s["id"] for s in sb["shots"]]
    if set(got) != set(ids):
        raise SystemExit(f"translation returned ids {sorted(got)} but expected {ids}; not saved")
    for s in sb["shots"]:
        old = s["narration"].get(target, "")
        mark = "=" if old == got[s["id"]] else "*"
        print(f"{mark} {s['id']}: {got[s['id']]}")
        s["narration"][target] = got[s["id"]]
    if write:
        save_storyboard(sb)
        say(STAGE, f"saved {target} narration to storyboard.json — have a native speaker check it")
    else:
        say(STAGE, "dry run; pass --write to save")


def backtranslate(lang: str) -> None:
    sb = load_storyboard()
    lines = [{"id": s["id"], "text": s["narration"].get(lang, "")} for s in sb["shots"]]
    prompt = (f"Translate each line from {LANG_NAMES[lang]} into plain English, literally. "
              f'Return JSON {{"lines": [{{"id": ..., "english": ...}}]}}.\n\n{json.dumps(lines, ensure_ascii=False)}')
    out = ask_json(sb["models"]["text_lite"], prompt, f"backtranslate/{lang}")
    back = {r["id"]: r["english"] for r in out.get("lines", [])}
    for s in sb["shots"]:
        print(f"{s['id']}\n  intended: {s['subtitle_en']}\n  back:     {back.get(s['id'])}")


def wordcount() -> None:
    sb = load_storyboard()
    total_w = 0
    for s in sb["shots"]:
        for lg, text in s["narration"].items():
            w = len(text.split())
            if lg == sb["languages"]["primary"]:
                total_w += w
            print(f"{s['id']} {lg}: {w:3d} words, ~{w / 2.0:.1f}s at 2 w/s, planned {s['duration_s']}s")
    print(f"primary total: {total_w} words (~{total_w / 2.0:.0f}s at 2 words/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("translate")
    t.add_argument("--to", required=True, choices=list(LANG_NAMES))
    t.add_argument("--write", action="store_true")
    b = sub.add_parser("backtranslate")
    b.add_argument("--lang", required=True, choices=list(LANG_NAMES))
    sub.add_parser("wordcount")
    a = ap.parse_args()
    if a.cmd == "translate":
        translate(a.to, a.write)
    elif a.cmd == "backtranslate":
        backtranslate(a.lang)
    else:
        wordcount()


if __name__ == "__main__":
    main_guard(main)
