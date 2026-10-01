"""Regression tests for two grader fixes: the สระไอ (ไ/ใ) rhyme fallback and the
ครุ/ลหุ overrides for open สระออ syllables and บ/บ่.

Run:  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/test_grader_fixes.py
(from the repository root), or via pytest. Self-contained, no pytest required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thaiwrite.prosody.grader_meter import (_is_sara_ai_syllable, rhymes,  # noqa: E402
                                            score_meter, weight)
from thaiwrite.prosody.data.audit_poems import KARU_LAHU_PROBES, POEMS      # noqa: E402


# --- rhyme: ไ and ใ spell the same open /aj/ ---------------------------------

def test_sara_ai_pairs_rhyme():
    # ไ/ใ + cluster ล / silent ห / cluster ร: khavee reads ล as a final and the
    # romanizer adds a final to ใ-clusters, so these were rejected.
    for a, b in [("ใจ", "ใกล้"), ("ไกล", "ใหญ่"), ("ไพร", "ใกล้"), ("ไกล", "ใคร"),
                 ("ใคร", "ไกล"), ("ใหม่", "ใกล้"), ("ใด", "ใกล้"), ("ไทย", "ใคร"),
                 ("ใคร", "ใกล้"), ("ไกล", "ใส")]:
        assert rhymes(a, b), (a, b)
        assert rhymes(b, a), (b, a)


def test_sara_ai_rhymes_with_ai_after_short_a():
    # -ัย is the same rime; is_sumpus remaps it, the fallback keeps that.
    assert rhymes("ใกล้", "ชัย")
    assert rhymes("ไกล", "ภัย")


def test_sara_ai_unchanged_passes():
    for a, b in [("ใจ", "ใส"), ("ใจ", "ไป"), ("ใจ", "ใหญ่"), ("ใหล", "ใด"), ("ไฟล์", "ใจ")]:
        assert rhymes(a, b), (a, b)


def test_sara_ai_different_rime_still_fails():
    # different vowel (length / quality) or a real final must not rhyme
    for a, b in [("ใกล้", "กาย"),    # /aj/ vs /aːj/
                 ("ใกล้", "เกิน"), ("ไกล", "กัน"), ("ไกล", "กล"), ("ใจ", "แจ"),
                 ("ใจ", "เจ้า"), ("ใจ", "เอา"), ("ใคร", "ครู"), ("ใหญ่", "หญิง")]:
        assert not rhymes(a, b), (a, b)
        assert not rhymes(b, a), (b, a)


def test_sara_ai_syllable_detector():
    for w in ("ใจ", "ไกล", "ใกล้", "ใคร", "ใหญ่", "ไทย", "ไหล", "ไขว่", "ไฟล์", "ไอ"):
        assert _is_sara_ai_syllable(w), w
    # polysyllabic tokens, other vowels, and non-ไ/ใ words are excluded
    for w in ("ใจดี", "ไปมา", "ใจคน", "ชัย", "กาย", "เกิน", "ไ", ""):
        assert not _is_sara_ai_syllable(w), w


# --- weight: open สระออ is ครุ; บ/บ่ is ลหุ -----------------------------------

def test_open_sara_or_is_karu():
    for s in ("ทอ", "พอ", "ขอ", "รอ", "งอ", "หน่อ", "พ่อ", "หนอ", "คลอ", "ต่อ", "ออ"):
        assert weight(s) == "karu", s


def test_bo_negation_is_lahu():
    assert weight("บ") == "lahu"
    assert weight("บ่") == "lahu"


def test_weight_non_regression():
    for k, sylls in KARU_LAHU_PROBES.items():
        for s in sylls:
            assert weight(s) == k, (s, k)
    for s in ("ก็", "ณ", "ธ", "จะ", "และ", "พระ", "ทะ"):
        assert weight(s) == "lahu", s
    for s in ("เธอ", "คือ", "มือ", "กร", "ขน", "บอ", "บน", "บาป", "กบ"):
        assert weight(s) == "karu", s


# --- layer integration -------------------------------------------------------

def _chan():
    p = next(p for p in POEMS if p.meter_id == "intharawichian_chan_11")
    return [list(l) for l in p.lines]


def test_chan_weight_layer_uses_overrides():
    lines = _chan()
    lines[0][0] = "ทอ"      # ครุ slot: open สระออ now passes
    lines[1][0] = "บ"       # ลหุ slot: บ now passes
    r = score_meter([tuple(l) for l in lines], "intharawichian_chan_11", eval_rhyme=False)
    assert r["layer_rates"]["kru_lahu"] == 1.0, r["layer_rates"]
    lines = _chan()
    lines[0][2] = "ทอ"      # ลหุ slot: a long syllable must now FAIL (was a false pass)
    r = score_meter([tuple(l) for l in lines], "intharawichian_chan_11", eval_rhyme=False)
    assert r["layer_rates"]["kru_lahu"] < 1.0, r["layer_rates"]


def test_kap_rhyme_layer_sara_ai():
    # กาพย์ยานี วรรค๒ท้าย → วรรค๓ท้าย with ไกล → ใคร (previously a rhyme failure)
    p = next(p for p in POEMS if p.meter_id == "kap_yani_11")
    lines = [list(l) for l in p.lines]
    lines[1][5], lines[2][4] = "ไกล", "ใคร"
    r = score_meter([tuple(l) for l in lines], "kap_yani_11")
    assert r["layer_rates"]["rhyme"] == 1.0, r["layer_rates"]


# --- runner ------------------------------------------------------------------

def _main() -> int:
    tests = [t for name, t in sorted(globals().items()) if name.startswith("test_")]
    print(f"Running {len(tests)} tests...")
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(_main())
