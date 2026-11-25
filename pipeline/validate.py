"""Validate storyboard.json against schema/storyboard.schema.json (+ semantic checks).

    make validate
"""
import sys

from .common import STORYBOARD, StoryboardError, load_storyboard


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else STORYBOARD
    try:
        sb = load_storyboard(path)
    except StoryboardError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    total = sum(s["duration_s"] for s in sb["shots"]) + sb["end_card"]["duration_s"]
    print(f"{path}: valid — {len(sb['shots'])} shots, planned {total:.0f}s incl. end card")


if __name__ == "__main__":
    main()
