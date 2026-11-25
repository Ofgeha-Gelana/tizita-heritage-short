import copy
import json

import pytest

from pipeline import common
from pipeline.assemble import subtitle_events
from pipeline.common import Cache, StoryboardError, stable_hash, validate_storyboard


@pytest.fixture
def sb():
    return json.loads(common.STORYBOARD.read_text(encoding="utf-8"))


def test_storyboard_is_valid(sb):
    validate_storyboard(sb)


@pytest.mark.parametrize("mutate, msg", [
    (lambda s: s["shots"][0].pop("reason"), "reason"),
    (lambda s: s["shots"].__setitem__(1, {**s["shots"][1], "id": s["shots"][0]["id"]}), "duplicate"),
    (lambda s: s["shots"][2]["characters"].append("ghost"), "unknown character"),
    (lambda s: s["end_card"].__setitem__("lines", ["The end"]), "AI"),
    (lambda s: [sh.__setitem__("duration_s", 3) for sh in s["shots"]], "runtime"),
    (lambda s: s["characters"][0].__setitem__("real_person", True), "consent_ref"),
])
def test_bad_storyboards_rejected(sb, mutate, msg):
    bad = copy.deepcopy(sb)
    mutate(bad)
    with pytest.raises(StoryboardError) as e:
        validate_storyboard(bad)
    assert msg.lower() in str(e.value).lower()


def test_hash_changes_only_with_inputs():
    assert stable_hash("a", 1) == stable_hash("a", 1)
    assert stable_hash("a", 1) != stable_hash("a", 2)


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path / ".cache")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    out = tmp_path / "x.wav"
    out.write_bytes(b"x")
    c = Cache("t")
    assert c.fresh("s01", "k1") is None
    c.put("s01", "k1", out)
    assert Cache("t").fresh("s01", "k1") == out
    assert Cache("t").fresh("s01", "k2") is None
    out.unlink()
    assert Cache("t").fresh("s01", "k1") is None


def test_subtitles_wrap_and_time():
    tl = [{"subtitle": "Emama always said: coffee is only an excuse. What matters is being together, every single day.",
           "nar_start": 1.0, "nar_len": 6.0, "start": 0.6, "len": 8.0}]
    ev = subtitle_events(tl)
    assert len(ev) >= 2
    for a, b, text in ev:
        assert 1.0 <= a < b <= 7.3
        assert all(len(row) <= 40 for row in text.split("\\N"))
        assert text.count("\\N") <= 1


def _synthetic_recording(lengths_s, rate=24000, inner_gap=0.18, line_gap=0.55):
    import array
    import math
    out = array.array("h", [0] * int(rate * 0.3))
    for n, L in enumerate(lengths_s):
        words = max(int(L / 0.45), 1)
        for w in range(words):
            seg = int(rate * (L / words - inner_gap))
            out.extend(int(6000 * math.sin(2 * math.pi * (180 + 10 * n) * i / rate)) for i in range(seg))
            out.extend([0] * int(rate * inner_gap))
        out.extend([0] * int(rate * line_gap))
    return out, rate


def test_batch_split_recovers_lines():
    from pipeline.narration import split_on_pauses
    lengths = [3.0, 3.4, 3.8, 3.2, 2.2, 3.0, 2.6, 3.9, 3.9, 4.6]
    samples, rate = _synthetic_recording(lengths)
    texts = ["x" * int(L * 10) for L in lengths]
    parts = split_on_pauses(samples, rate, texts)
    assert parts is not None and len(parts) == len(lengths)
    for p, L in zip(parts, lengths):
        assert abs(len(p) / rate - L) < 0.5


def test_batch_split_rejects_wrong_proportions():
    from pipeline.narration import split_on_pauses
    samples, rate = _synthetic_recording([3.0, 3.0, 3.0])
    assert split_on_pauses(samples, rate, ["a" * 5, "b" * 5, "c" * 200]) is None
