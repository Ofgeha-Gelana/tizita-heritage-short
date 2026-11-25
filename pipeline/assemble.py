"""Stage: assembly (FFmpeg, deterministic, no model calls).

Picture is cut to the voice: each shot lasts max(storyboard duration, narration
length + padding), so the narration in each language sets the timeline.

Per shot and per export (vertical 9:16, horizontal 16:9) a segment is rendered from
  1. the Wan clip in outputs/clips/<id>.mp4 (weak first 0.25 s trimmed, gently
     slowed or held on the last frame to fill the shot), else
  2. a Ken Burns pan/zoom over the keyframe, else
  3. black (radio cut / nothing generated yet).
Segments are cached by a hash of their inputs.

Sound: narration placed per shot -> voice stem; score (chosen ACE-Step candidate,
or the placeholder drone) -> music stem; both loudness-matched, music ducked under
the voice with sidechaincompress, mixed, then two-pass loudnorm to -14 LUFS / -1 dBTP.

Then English subtitles (ASS, kept inside the vertical safe area) are burned in,
the AI end card appended, and outputs/final/tizita_<lang>_<export>.mp4 written.

    python -m pipeline.assemble [--lang am] [--radio]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import (OUTPUTS, ffmpeg, load_storyboard, log_call, main_guard, probe_duration, rel,
                     say, stable_hash, timer)
from .keyframes import KF_DIR
from .music import chosen_track, placeholder
from .narration import narration_path
from .shots import CLIP_DIR

STAGE = "assemble"
SEG_DIR = OUTPUTS / "segments"
FINAL = OUTPUTS / "final"
TMP = OUTPUTS / "tmp"
FPS = 24
NAR_LEAD = 0.4     # seconds of picture before the narrator speaks
NAR_TAIL = 0.6     # seconds after the line ends before the cut
X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS)]
KB_CYCLE = ["in", "right", "out", "left", "up"]
ETHIOPIC = re.compile(r"[ሀ-᎟ⶀ-⷟꬀-꬯]")
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/KefaIII.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansEthiopic-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


# --------------------------------------------------------------------------- timeline
def timeline(sb: dict, lang: str) -> list[dict]:
    t, out = 0.0, []
    for s in sb["shots"]:
        wav = narration_path(lang, s["id"])
        has_text = bool(s["narration"].get(lang, "").strip())
        nar = probe_duration(wav) if (has_text and wav.exists()) else 0.0
        length = max(s["duration_s"], (NAR_LEAD + nar + NAR_TAIL) if nar else 0)
        out.append({"id": s["id"], "start": t, "len": round(length, 3), "nar": wav if nar else None,
                    "nar_start": t + NAR_LEAD, "nar_len": nar, "subtitle": s["subtitle_en"], "shot": s})
        t += length
    return out


# --------------------------------------------------------------------------- picture
def fit_filter(src_w: int, src_h: int, W: int, H: int) -> str:
    """Crop-fill when aspect ratios are close; otherwise centre the picture over a
    blurred, darkened copy of itself (vertical master shown in a 16:9 frame)."""
    if abs((src_w / src_h) / (W / H) - 1) < 0.15:
        return f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    return (f"split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"boxblur=30:3,eq=brightness=-0.12:saturation=0.8[bg];"
            f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def kenburns_filter(shot: dict, idx: int, n_frames: int, src: tuple[int, int]) -> str:
    kb = shot.get("kenburns", {})
    direction = kb.get("direction", KB_CYCLE[idx % len(KB_CYCLE)])
    a = kb.get("amount", 0.12)
    w, h = src
    # Pre-upscale 2x so zoompan's integer pixel steps don't jitter.
    pre = f"scale={w * 2}:{h * 2}:flags=lanczos"
    p = f"(on/{max(n_frames - 1, 1)})"
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    z, x, y = {
        "in": (f"1+{a}*{p}", cx, cy),
        "out": (f"1+{a}-{a}*{p}", cx, cy),
        "right": (f"1+{a}", f"(iw-iw/zoom)*{p}", cy),
        "left": (f"1+{a}", f"(iw-iw/zoom)*(1-{p})", cy),
        "down": (f"1+{a}", cx, f"(ih-ih/zoom)*{p}"),
        "up": (f"1+{a}", cx, f"(ih-ih/zoom)*(1-{p})"),
    }[direction]
    return f"{pre},zoompan=z='{z}':x='{x}':y='{y}':d={n_frames}:s={w}x{h}:fps={FPS}"


def render_segment(sb: dict, entry: dict, idx: int, W: int, H: int, export: str, radio: bool) -> Path:
    shot, length = entry["shot"], entry["len"]
    clip, kf = CLIP_DIR / f"{shot['id']}.mp4", KF_DIR / f"{shot['id']}.png"
    mode = "black" if radio else ("clip" if (clip.exists() and shot["video_mode"] == "wan") else
                                  ("kenburns" if kf.exists() else "black"))
    src = clip if mode == "clip" else kf if mode == "kenburns" else None
    key = stable_hash("seg-v2", mode, src if src else "", length, W, H, shot.get("kenburns", {}), idx % len(KB_CYCLE))
    out = SEG_DIR / export / f"{shot['id']}_{key}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    n = round(length * FPS)
    if mode == "clip":
        cd = max(probe_duration(clip) - 0.25, 0.5)
        factor = min(length / cd, 1.6)  # slow gently; beyond 1.6x hold the last frame
        hold = max(length - cd * factor, 0) + 0.1
        with Image.open(kf) if kf.exists() else Image.new("RGB", (9, 16)) as im:
            sw, sh = im.size
        vf = (f"trim=start=0.25,setpts=(PTS-STARTPTS)*{factor:.4f},fps={FPS},"
              f"tpad=stop_mode=clone:stop_duration={hold:.2f},{fit_filter(sw, sh, W, H)},trim=end_frame={n}")
        ffmpeg("-i", str(clip), "-filter_complex", vf, "-an", *X264, "-frames:v", str(n), str(out))
    elif mode == "kenburns":
        with Image.open(kf) as im:
            sw, sh = im.size
        vf = f"{kenburns_filter(shot, idx, n, (sw, sh))},{fit_filter(sw, sh, W, H)}"
        ffmpeg("-i", str(kf), "-filter_complex", vf, "-an", *X264, "-frames:v", str(n), str(out))
    else:
        ffmpeg("-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}:d={length}", "-an", *X264,
               "-frames:v", str(n), str(out))
    return out


def find_font(text: str) -> str | None:
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            if ETHIOPIC.search(text) and "Kefa" not in f and "Ethiopic" not in f:
                continue
            return f
    return None


def end_card(sb: dict, W: int, H: int, export: str) -> Path:
    ec = sb["end_card"]
    key = stable_hash("endcard-v1", ec, W, H)
    out = SEG_DIR / export / f"endcard_{key}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), (28, 20, 14))
    d = ImageDraw.Draw(img)
    base = min(W, H)
    y = H * 0.38
    for i, line in enumerate(ec["lines"]):
        size = int(base * (0.075 if i == 0 else 0.034))
        fp = find_font(line)
        if fp is None:  # no Ethiopic font on this machine: keep the Latin part
            line = ETHIOPIC.sub("", line).strip(" ·")
            fp = find_font(line)
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
        for row in textwrap.wrap(line, width=int(W / (size * 0.52))):
            tw = d.textlength(row, font=font)
            d.text(((W - tw) / 2, y), row, font=font, fill=(240, 226, 200) if i == 0 else (214, 196, 170))
            y += size * 1.35
        y += size * 0.6
    png = out.with_suffix(".png")
    img.save(png)
    dur = ec["duration_s"]
    ffmpeg("-loop", "1", "-i", str(png), "-vf", f"fade=t=in:d=0.6,fade=t=out:st={dur - 0.5}:d=0.5",
           "-t", str(dur), "-an", *X264, str(out))
    return out


# --------------------------------------------------------------------------- subtitles
def _ts(t: float, ass: bool) -> str:
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    if ass:
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def subtitle_events(tl: list[dict]) -> list[tuple[float, float, str]]:
    """Split each line into cards of at most 2 rows x 40 chars, timed by length."""
    events = []
    for e in tl:
        text = e["subtitle"].strip()
        if not text:
            continue
        start = e["nar_start"] if e["nar_len"] else e["start"] + 0.3
        end = start + (e["nar_len"] + 0.3 if e["nar_len"] else e["len"] - 0.6)
        rows = textwrap.wrap(text, 40)
        cards = [" \\N".join(rows[i:i + 2]).replace(" \\N", "\\N") for i in range(0, len(rows), 2)]
        total = sum(len(c) for c in cards)
        t = start
        for c in cards:
            d = (end - start) * len(c) / total
            events.append((t, t + d - 0.05, c))
            t += d
    return events


def write_subs(tl: list[dict], W: int, H: int, path: Path) -> Path:
    vertical = H > W
    size = round(H * (0.028 if vertical else 0.045))
    margin_v = round(H * (0.22 if vertical else 0.07))  # keep clear of feed UI in 9:16
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Helvetica,{size},&H00F0F4F8,&H000000FF,&H00101010,&H64000000,0,0,0,0,100,100,0,0,1,{max(2, size // 18)},1,2,{round(W * 0.08)},{round(W * 0.08)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [f"Dialogue: 0,{_ts(a, True)},{_ts(b, True)},Default,,0,0,0,,{t}" for a, b, t in subtitle_events(tl)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_srt(tl: list[dict], path: Path) -> Path:
    rows = []
    for i, (a, b, t) in enumerate(subtitle_events(tl), 1):
        rows.append(f"{i}\n{_ts(a, False)} --> {_ts(b, False)}\n{t.replace(chr(92) + 'N', chr(10))}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- sound
def measure_loudness(path: Path, extra_filters: str = "") -> dict:
    import subprocess
    af = (extra_filters + "," if extra_filters else "") + "loudnorm=I=-14:TP=-1:LRA=11:print_format=json"
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", af, "-f", "null", "-"],
                         capture_output=True, text=True)
    blob = res.stderr[res.stderr.rfind("{"):res.stderr.rfind("}") + 1]
    return json.loads(blob)


def mix_audio(sb: dict, tl: list[dict], total: float, lang: str, music: Path, radio: bool) -> Path:
    nar_inputs = [e for e in tl if e["nar"]]
    gain = sb["music"].get("music_gain_db", -6)
    key = stable_hash("mix-v1", lang, music, gain, total,
                      [(e["id"], round(e["nar_start"], 3), e["nar"]) for e in nar_inputs])
    out = TMP / f"mix_{lang}_{'radio' if radio else 'film'}_{key}.wav"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    args, chains = [], []
    for i, e in enumerate(nar_inputs):
        args += ["-i", str(e["nar"])]
        ms = int(e["nar_start"] * 1000)
        chains.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,adelay={ms}|{ms}[n{i}]")
    m = len(nar_inputs)
    args += ["-stream_loop", "-1", "-i", str(music)]
    fade_out = max(total - 3.5, 0)
    graph = ";".join(chains) + ";" + "".join(f"[n{i}]" for i in range(m)) + \
        f"amix=inputs={m}:normalize=0:dropout_transition=0,apad,atrim=0:{total:.3f}," \
        f"loudnorm=I=-16:TP=-2:LRA=11[voice];" \
        f"[{m}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{total:.3f},asetpts=PTS-STARTPTS," \
        f"loudnorm=I=-16:TP=-2:LRA=11,volume={gain}dB,afade=t=in:d=2,afade=t=out:st={fade_out:.2f}:d=3.5[music];" \
        f"[voice]asplit=2[v1][vsc];" \
        f"[music][vsc]sidechaincompress=threshold=0.02:ratio=6:attack=300:release=1500[ducked];" \
        f"[v1][ducked]amix=inputs=2:normalize=0:duration=first[mix]"
    pre = TMP / f"premix_{key}.wav"
    ffmpeg(*args, "-filter_complex", graph, "-map", "[mix]", "-t", f"{total:.3f}", "-ar", "48000", str(pre))
    # Two-pass loudness normalisation to the platform target.
    m1 = measure_loudness(pre)
    ln = (f"loudnorm=I=-14:TP=-1:LRA=11:measured_I={m1['input_i']}:measured_TP={m1['input_tp']}:"
          f"measured_LRA={m1['input_lra']}:measured_thresh={m1['input_thresh']}:offset={m1['target_offset']}:linear=true")
    ffmpeg("-i", str(pre), "-af", ln, "-ar", "48000", str(out))
    pre.unlink(missing_ok=True)
    return out


# --------------------------------------------------------------------------- build
def build(sb: dict, lang: str, radio: bool) -> list[Path]:
    if not any(s["narration"].get(lang, "").strip() for s in sb["shots"]):
        say(STAGE, f"{lang}: no narration text; skipped")
        return []
    missing = [s["id"] for s in sb["shots"] if s["narration"].get(lang, "").strip()
               and not narration_path(lang, s["id"]).exists()]
    if missing:
        say(STAGE, f"{lang}: narration missing for {missing}; run narration first")
        return []
    tl = timeline(sb, lang)
    body = sum(e["len"] for e in tl)
    ec_dur = sb["end_card"]["duration_s"]
    total = body + ec_dur
    music = placeholder(total + 2) if radio else (chosen_track(sb) or placeholder(total + 2))
    if not radio and music.name.startswith("placeholder"):
        say(STAGE, "no chosen score yet; using the placeholder drone")
    audio = mix_audio(sb, tl, total, lang, music, radio)
    exports = [{"name": "radio", "width": 1280, "height": 720}] if radio else sb["format"]["exports"]
    outs = []
    for ex in exports:
        W, H, name = ex["width"], ex["height"], ex["name"]
        with timer() as t:
            segs = [render_segment(sb, e, i, W, H, name, radio) for i, e in enumerate(tl)]
            if not radio:
                segs.append(end_card(sb, W, H, name))
            lst = TMP / f"concat_{lang}_{name}.txt"
            lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
            picture = TMP / f"picture_{lang}_{name}.mp4"
            ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(picture))
            subs = write_subs(tl, W, H, TMP / f"subs_{lang}_{name}.ass")
            final = (OUTPUTS / f"radio_cut_{lang}.mp4") if radio else FINAL / f"tizita_{lang}_{name}.mp4"
            final.parent.mkdir(parents=True, exist_ok=True)
            sub_path = str(subs).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            ffmpeg("-i", str(picture), "-i", str(audio), "-vf", f"subtitles='{sub_path}'",
                   *X264, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
                   "-map", "0:v", "-map", "1:a", "-t", f"{total:.3f}",
                   "-metadata", "comment=Made with AI (FLUX.2 klein, Wan 2.2, ACE-Step 1.5, Gemini TTS)", str(final))
        dur = probe_duration(final)
        loud = measure_loudness(final)
        report = {"file": rel(final), "lang": lang, "export": name, "duration_s": round(dur, 2),
                  "integrated_lufs": float(loud["input_i"]), "true_peak_dbtp": float(loud["input_tp"]),
                  "shots": [{"id": e["id"], "start": round(e["start"], 2), "len": e["len"],
                             "source": "black" if radio else ("clip" if (CLIP_DIR / f"{e['id']}.mp4").exists() and e["shot"]["video_mode"] == "wan" else "kenburns" if (KF_DIR / f"{e['id']}.png").exists() else "black")}
                            for e in tl],
                  "music": rel(music)}
        log_call(stage=STAGE, provider="local", model="ffmpeg", item=f"{lang}/{name}", latency_ms=t["ms"],
                 params={"W": W, "H": H, "radio": radio}, output=rel(final), cost_usd=0.0,
                 duration_s=report["duration_s"], lufs=report["integrated_lufs"], tp=report["true_peak_dbtp"])
        (final.parent / f"{final.stem}.report.json").write_text(json.dumps(report, indent=2) + "\n")
        say(STAGE, f"{rel(final)}: {dur:.1f}s, {report['integrated_lufs']} LUFS, TP {report['true_peak_dbtp']} dBTP")
        outs.append(final)
    write_srt(tl, (OUTPUTS if radio else FINAL) / f"tizita_{lang}{'_radio' if radio else ''}.en.srt")
    return outs


def run(lang: str | None = None, radio: bool = False) -> None:
    sb = load_storyboard()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found; `brew install ffmpeg`")
    langs = [lang] if lang else [sb["languages"]["primary"], *sb["languages"]["extra"]]
    for lg in langs:
        build(sb, lg, radio)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang")
    ap.add_argument("--radio", action="store_true", help="narration + placeholder music on black")
    a = ap.parse_args()
    run(a.lang, a.radio)


if __name__ == "__main__":
    main_guard(main)
