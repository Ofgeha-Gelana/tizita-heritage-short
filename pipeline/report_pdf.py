"""REPORT.md -> report.pdf (Markdown -> styled HTML -> headless Chrome print).

    make report

No LaTeX or pandoc needed: markdown-it-py renders the HTML, and Chrome/Chromium prints
it. Set CHROME=/path/to/chrome if it is not in the usual place.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

from .common import ROOT, main_guard, say

STAGE = "report"
CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, "Kefa", "Noto Sans Ethiopic", sans-serif;
       font-size: 10.5pt; line-height: 1.5; color: #1d1b19; }
h1 { font-size: 20pt; margin: 0 0 6pt; color: #3b2414; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #d8cbbd;
     color: #3b2414; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 18pt; }
li { margin: 2pt 0; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.8pt; background: #f4efe9;
       padding: 0 2pt; border-radius: 2pt; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; border-left: 3px solid #b5651d; background: #fbf6f0; }
strong { color: #2a1a0e; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; }
th, td { border: 1px solid #d8cbbd; padding: 3pt 6pt; text-align: left; vertical-align: top; }
"""


def find_chrome() -> str:
    for c in CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    raise SystemExit("Chrome/Chromium not found; set CHROME=/path/to/chrome")


def main() -> None:
    src = ROOT / "REPORT.md"
    out = ROOT / "report.pdf"
    body = MarkdownIt("commonmark").enable("table").render(src.read_text())
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Tizita report</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", page.as_uri()],
                       check=True, capture_output=True, timeout=120)
    say(STAGE, f"{out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main_guard(main)
