# One command for the film:   make film
# Everything runs through uv so a clean clone gets the same Python and packages.
PY      := uv run python -m
SHOT    ?=
LANG_   ?=
QUALITY ?= draft
export OFFLINE

.PHONY: help setup smoke validate test wordcount translate radio-cut refs keyframes explore promote \
        shots shots-submit shots-collect music music-submit music-collect assemble film \
        final shot costs report clean-cache

help:
	@grep -E '^## ' Makefile | sed 's/## //'

## setup          install Python 3.11 + deps with uv, check ffmpeg
setup:
	uv sync
	@command -v ffmpeg >/dev/null || (echo "ffmpeg missing: brew install ffmpeg (macOS) / apt install ffmpeg" && exit 1)
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example — fill in your keys")

## smoke          call every provider once, log latency to runs.jsonl
smoke:
	$(PY) pipeline.smoke

## validate       check storyboard.json against the schema
validate:
	$(PY) pipeline.validate

## test           unit tests (schema, cache, subtitles)
test:
	uv run pytest -q

## wordcount      words per shot vs planned duration
wordcount: validate
	$(PY) pipeline.script wordcount

## translate      Amharic -> Afaan Oromo narration (TO=om), writes storyboard.json
translate: validate
	$(PY) pipeline.script translate --to $(or $(TO),om) --write

## radio-cut      narration over placeholder music on black (checkpoint 1)
radio-cut: validate
	$(PY) pipeline.narration $(if $(LANG_),--lang $(LANG_))
	$(PY) pipeline.music --placeholder
	$(PY) pipeline.assemble --radio $(if $(LANG_),--lang $(LANG_))

## refs           character reference-sheet candidates
refs: validate
	$(PY) pipeline.keyframes refs

## keyframes      keyframe per shot (SHOT=s03 for one)
keyframes: validate
	$(PY) pipeline.keyframes shots $(if $(SHOT),--shots $(SHOT))

## explore        several seeds for one shot: make explore SHOT=s03 SEEDS=1,2,3,4
explore: validate
	$(PY) pipeline.keyframes explore --shots $(SHOT) --seeds $(SEEDS)

promote: validate  ## keep an explored keyframe: make promote SHOT=s06 SEED=3006
	$(PY) pipeline.keyframes promote --shots $(SHOT) --seeds $(SEED)

## shots          Wan 2.2 on Kaggle, wait for results (QUALITY=draft|final)
shots: validate
	$(PY) pipeline.shots --quality $(QUALITY) $(if $(SHOT),--shots $(SHOT))
shots-submit: validate
	$(PY) pipeline.shots --quality $(QUALITY) --submit-only $(if $(SHOT),--shots $(SHOT))
shots-collect:
	$(PY) pipeline.shots --collect

## music          ACE-Step score candidates on Kaggle
music: validate
	$(PY) pipeline.music
music-submit: validate
	$(PY) pipeline.music --submit-only
music-collect:
	$(PY) pipeline.music --collect

## assemble       edit, mix, subtitles, end card, both aspect ratios, all languages
assemble: validate
	$(PY) pipeline.assemble $(if $(LANG_),--lang $(LANG_))

## film           the whole pipeline end to end (OFFLINE=1: assemble from outputs/ only)
film:
	$(PY) pipeline.run --quality $(QUALITY)

## final          same, with final-quality (720p) Wan clips
final:
	$(PY) pipeline.run --quality final

## shot           regenerate one shot only, then re-assemble: make shot SHOT=s03
shot: validate
	$(PY) pipeline.run --quality $(QUALITY) --shots $(SHOT)

## costs          per-stage latency / GPU minutes / list price from runs.jsonl
costs:
	$(PY) pipeline.costs

report:  ## REPORT.md -> report.pdf (headless Chrome)
	$(PY) pipeline.report_pdf

## clean-cache    forget cache manifests (outputs are kept)
clean-cache:
	rm -rf outputs/.cache outputs/segments outputs/tmp
