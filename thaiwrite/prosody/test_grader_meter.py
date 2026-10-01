"""Tests for grader_meter — canonical audit poems score 1.0; layer-broken
variants score < 0.7.

Run:  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/test_grader_meter.py
(from the repository root), or via pytest. Self-contained, no pytest required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thaiwrite.prosody.grader_meter import score_meter
from thaiwrite.prosody.data.audit_poems import POEMS, Poem

BROKEN_MAX = 0.7


def _poem(substr: str) -> Poem:
    return next(p for p in POEMS if substr in p.source)


def _lines(p: Poem) -> list[tuple[str, ...]]:
    return [tuple(l) for l in p.lines]


def _set(lines, line_index, syll_index, value):
    """Return a copy of `lines` with one position replaced."""
    out = [list(l) for l in lines]
    n = len(out[line_index])
    i = syll_index if syll_index >= 0 else n + syll_index
    out[line_index][i] = value
    return [tuple(l) for l in out]


def _fill_all(lines, value):
    """Replace every syllable (preserving line lengths)."""
    return [tuple(value for _ in l) for l in lines]


# --- canonical → 1.0 -------------------------------------------------------

def test_canonical_scores():
    cases = [
        (_poem("ลิลิตพระลอ"), "khlong_si_suphap", True),
        (_poem("นิราศภูเขาทอง"), "klon_paet", True),
        (_poem("พระอภัยมณี"), "klon_paet", True),
        (_poem("ขึ้นกกตกทุกข์ยาก"), "kap_yani_11", True),
        (_poem("รำลึกพระคุณแม่"), "intharawichian_chan_11", False),  # synthetic วรรค: no rhyme
    ]
    for p, meter, eval_rhyme in cases:
        r = score_meter(_lines(p), meter, eval_rhyme=eval_rhyme)
        assert r["score"] == 1.0, f"{p.source[:20]}: {r['score']} {r['layer_rates']}"


# --- broken layers → < 0.7 -------------------------------------------------

def test_broken_tone_khlong():
    p = _poem("ลิลิตพระลอ")
    spec_lines = _lines(p)
    # blank every tone-bearing position with a plain (no-mark) syllable
    from thaiwrite.prosody import meter_specs as M
    broken = spec_lines
    for tc in M.KHLONG_SI_SUPHAP.tone_constraints:
        broken = _set(broken, tc.position.line_index, tc.position.syllable_index, "มา")
    r = score_meter(broken, "khlong_si_suphap")
    assert r["score"] < BROKEN_MAX, r


def test_broken_rhyme_klon():
    p = _poem("นิราศภูเขาทอง")
    lines = _lines(p)
    lines = _set(lines, 0, -1, "ปู")   # สดับ last
    lines = _set(lines, 1, -1, "เกะ")  # รับ last
    lines = _set(lines, 2, -1, "กก")   # รอง last
    r = score_meter(lines, "klon_paet")
    assert r["layer_rates"]["rhyme"] == 0.0, r
    assert r["score"] < BROKEN_MAX, r


def test_broken_kru_lahu_chan():
    p = _poem("รำลึกพระคุณแม่")
    broken = _fill_all(_lines(p), "จะ")   # ลหุ everywhere → ครุ positions fail
    r = score_meter(broken, "intharawichian_chan_11", eval_rhyme=False)
    assert r["score"] < BROKEN_MAX, r


def test_broken_count():
    p = _poem("ขึ้นกกตกทุกข์ยาก")
    lines = _lines(p)
    # pad each วรรค far beyond tolerance
    lines = [tuple(list(l) + ["มา"] * 4) for l in lines]
    r = score_meter(lines, "kap_yani_11")
    assert r["layer_rates"]["count"] == 0.0, r
    assert r["score"] < BROKEN_MAX, r


# --- runner ----------------------------------------------------------------

def _main() -> int:
    print("CANONICAL (expect 1.0):")
    for p in POEMS:
        er = p.eval_rhyme
        r = score_meter(_lines(p), p.meter_id, eval_rhyme=er)
        rhyme_note = "" if er else "  (rhyme N/A)"
        print(f"  {r['score']:.3f}  {r['meter']:<22} {p.source[:30]}{rhyme_note}  {r['layer_rates']}")

    tests = [t for name, t in sorted(globals().items()) if name.startswith("test_")]
    print(f"\nRunning {len(tests)} tests...")
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
