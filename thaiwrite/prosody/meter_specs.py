"""Formal encodings of four classical Thai poetic meters (ฉันทลักษณ์) for
programmatic prosody grading.

Each meter is a `MeterSpec`: a flat, line-indexed structure that the downstream
grader (`grader_meter.py`) iterates generically while still expressing
meter-specific constraints: per-line syllable counts (with tolerance), tone-mark
requirements at fixed positions, ครุ/ลหุ (heavy/light) patterns, and rhyme
(สัมผัส) linkages.

Canonical addressing
--------------------
A poem is normalized to a flat list of *lines*, one line per วรรค (the smallest
sequential syllable segment). บาท / บท roles are retained as metadata on each
`LineSpec` but are NOT used for addressing. Any syllable is addressed by
`Position(line_index, syllable_index)`, both 0-based; a NEGATIVE
`syllable_index` counts from the end of the line (-1 = last syllable), so rules
survive within-tolerance count variation (critical for กลอนแปด's 7–9 window).

Rhyme targets are a tuple of `PositionRange`s; a single receiving position is a
length-1 range, and disjoint allowed positions (e.g. กลอนแปด "คำ ๓ หรือ ๕") are
separate ranges. `external_rhyme_links` reference the NEXT stanza's line block
(the grader offsets by `len(lines)`), so one stanza stays the spec unit.

Every rule carries a `citation` (see `docs/meter-specs.md` §Sources for full source
records; web sources accessed May 2026). Rules are verified against
ราชบัณฑิตยสภา-aligned references and, for โคลงสี่สุภาพ, against the ลิลิตพระลอ
exemplar word-by-word.

NOTE: the canonical อินทรวิเชียรฉันท์ ๑๑ weight pattern is `ค ค ล ค ค | ล ล ค ล ค ค`
(5+6 split), encoded below; the sometimes-cited `ค-ล-ค-ค-ล-ล-ค | ค-ล-ค-ค` (7+4
split) is non-canonical.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


# --- enums -----------------------------------------------------------------

class ToneMark(str, Enum):
    """Required tone-mark class at a fixed position.

    Maps to `pythainlp.khavee.KhaveeVerifier.check_aek_too`, which classifies a
    syllable as carrying ไม้เอก / ไม้โท.
    """
    AEK = "aek"   # ไม้เอก (อ่)
    TOO = "too"   # ไม้โท  (อ้)


class Weight(str, Enum):
    """ครุ / ลหุ — syllabic weight, for ฉันท์ meters. Resolved via
    `pythainlp.util.sound_syllable` / `syllable_length`."""
    KARU = "karu"   # ครุ — heavy
    LAHU = "lahu"   # ลหุ — light


class VerseRole(str, Enum):
    """Human-readable role of a line within its native grouping. Metadata only,
    never used for addressing."""
    WAK_NA = "wak_na"      # วรรคหน้า (lead segment of a บาท)
    WAK_LANG = "wak_lang"  # วรรคหลัง (tail segment of a บาท)
    SADAB = "sadab"        # วรรคสดับ (กลอน)
    RAP = "rap"            # วรรครับ  (กลอน)
    RONG = "rong"          # วรรครอง  (กลอน)
    SONG = "song"          # วรรคส่ง   (กลอน)
    SOI = "soi"            # คำสร้อย (optional add-on segment)


# --- coordinates -----------------------------------------------------------

@dataclass(frozen=True)
class Position:
    """A single syllable coordinate. `syllable_index` may be negative
    (from end). `line_index` indexes the flattened `MeterSpec.lines` list."""
    line_index: int
    syllable_index: int


@dataclass(frozen=True)
class PositionRange:
    """An inclusive contiguous span of syllable positions on ONE line. A
    single-position target has `start == end`. Indices may be negative
    (resolved against the line's actual syllable count at grade time)."""
    line_index: int
    start: int
    end: int  # inclusive

    @property
    def is_single(self) -> bool:
        return self.start == self.end


# --- per-line structure ----------------------------------------------------

@dataclass(frozen=True)
class LineSpec:
    """One วรรค: a syllable sequence with a count window and a role.

    `canonical_count` is the textbook count; `min_count`/`max_count` are the
    accepted window (equal to canonical for exact meters). บาท splits (5+6,
    5+2, 5+4) are encoded as separate consecutive lines.
    """
    role: VerseRole
    canonical_count: int
    min_count: int
    max_count: int
    optional: bool = False           # คำสร้อย / variant segments
    bat_index: Optional[int] = None  # which บาท this line belongs to (0-based)
    bot_index: Optional[int] = None  # which บท this line belongs to (0-based)
    note: str = ""


# --- constraints -----------------------------------------------------------

@dataclass(frozen=True)
class ToneConstraint:
    """A required tone mark (เอก/โท) at a fixed position."""
    position: Position
    mark: ToneMark
    citation: str
    optional: bool = False


@dataclass(frozen=True)
class WeightPattern:
    """ครุ/ลหุ pattern for one line, one `Weight` per syllable index.
    `len(pattern)` must equal that line's `canonical_count`."""
    line_index: int
    pattern: tuple[Weight, ...]
    citation: str


@dataclass(frozen=True)
class RhymeLink:
    """A สัมผัส linkage: the `source` syllable must rhyme with AT LEAST ONE
    candidate syllable across `targets`. Each target is a `PositionRange`; a
    single receiving position is a length-1 range, and disjoint allowed
    positions (e.g. กลอนแปด "คำ ๓ หรือ ๕") are separate ranges in the tuple.
    Graded with `KhaveeVerifier.is_sumpus(source_syll, candidate_syll)` over
    every candidate in every target range.
    """
    source: Position
    targets: tuple[PositionRange, ...]
    citation: str
    kind: str = "internal"   # "internal" | "external" (cross-stanza)
    optional: bool = False
    note: str = ""


# --- top-level spec --------------------------------------------------------

@dataclass(frozen=True)
class MeterSpec:
    """Complete formal spec for one meter, sufficient for generic grading.

    `lines` defines ONE stanza (บท unit). `external_rhyme_links` reference the
    next stanza's line block (line indices offset by `len(lines)`); they are
    `optional` because they apply only when a following stanza exists.
    `stanza_repeats == 0` means the stanza count is set by the prompt, not fixed.
    """
    meter_id: str            # ascii slug, e.g. "kap_yani_11"
    name_th: str             # "กาพย์ยานี ๑๑"
    name_romanized: str
    lines: tuple[LineSpec, ...]
    rhyme_links: tuple[RhymeLink, ...] = ()
    external_rhyme_links: tuple[RhymeLink, ...] = ()
    tone_constraints: tuple[ToneConstraint, ...] = ()
    weight_patterns: tuple[WeightPattern, ...] = ()
    stanza_repeats: int = 0
    citation: str = ""
    description: str = ""

    @property
    def lines_per_stanza(self) -> int:
        return len(self.lines)

    @property
    def mandatory_lines(self) -> tuple[LineSpec, ...]:
        return tuple(ln for ln in self.lines if not ln.optional)

    def to_dict(self) -> dict:
        """Language-neutral JSON-export path. The str-Enum mixin makes enums
        serialize to their string values, so
        `json.dumps(spec.to_dict(), ensure_ascii=False)` is publishable as-is."""
        return asdict(self)


# ===========================================================================
# Meter 1 — โคลงสี่สุภาพ
# ===========================================================================
# 4 บาท; วรรคหน้า 5 + วรรคหลัง 2 for บาท 1–3, วรรคหลัง 4 for บาท 4 (30 คำ + optional
# คำสร้อย). เอก 7 / โท 4 at fixed positions; 3 สัมผัสบังคับ. Diagram verified
# against ลิลิตพระลอ exemplar. Source: th.wikipedia.org/wiki/โคลงสี่สุภาพ (raw
# diagram template) + digitalschool.club thai2_12 (สัมผัส).
_KS = "th.wikipedia.org/wiki/โคลงสี่สุภาพ (diagram, verified vs ลิลิตพระลอ), 2026-05-23"
_KS_RHYME = "digitalschool.club/.../thai2_12 + th.wikipedia โคลงสี่สุภาพ, 2026-05-23"

KHLONG_SI_SUPHAP = MeterSpec(
    meter_id="khlong_si_suphap",
    name_th="โคลงสี่สุภาพ",
    name_romanized="Khlong Si Suphap",
    description="4 บาท; วรรคหน้า ๕ + วรรคหลัง ๒ (บาท ๑–๓), วรรคหลัง ๔ (บาท ๔). "
                "บังคับเอก ๗ โท ๔ + คำสร้อยท้ายบาท ๑ และ ๓ (ไม่บังคับ).",
    citation=_KS,
    lines=(
        LineSpec(VerseRole.WAK_NA,   5, 5, 5, bat_index=0, bot_index=0, note="บาท ๑ วรรคหน้า"),
        LineSpec(VerseRole.WAK_LANG, 2, 2, 2, bat_index=0, bot_index=0, note="บาท ๑ วรรคหลัง"),
        LineSpec(VerseRole.WAK_NA,   5, 5, 5, bat_index=1, bot_index=0, note="บาท ๒ วรรคหน้า"),
        LineSpec(VerseRole.WAK_LANG, 2, 2, 2, bat_index=1, bot_index=0, note="บาท ๒ วรรคหลัง"),
        LineSpec(VerseRole.WAK_NA,   5, 5, 5, bat_index=2, bot_index=0, note="บาท ๓ วรรคหน้า"),
        LineSpec(VerseRole.WAK_LANG, 2, 2, 2, bat_index=2, bot_index=0, note="บาท ๓ วรรคหลัง"),
        LineSpec(VerseRole.WAK_NA,   5, 5, 5, bat_index=3, bot_index=0, note="บาท ๔ วรรคหน้า"),
        LineSpec(VerseRole.WAK_LANG, 4, 4, 4, bat_index=3, bot_index=0, note="บาท ๔ วรรคหลัง"),
        LineSpec(VerseRole.SOI, 2, 2, 2, optional=True, bat_index=0, bot_index=0,
                 note="คำสร้อยท้ายบาท ๑ (ไม่บังคับ)"),
        LineSpec(VerseRole.SOI, 2, 2, 2, optional=True, bat_index=2, bot_index=0,
                 note="คำสร้อยท้ายบาท ๓ (ไม่บังคับ)"),
    ),
    tone_constraints=(
        # เอก ๗ แห่ง
        ToneConstraint(Position(0, 3), ToneMark.AEK, _KS),  # บาท๑ คำ๔
        ToneConstraint(Position(2, 1), ToneMark.AEK, _KS),  # บาท๒ คำ๒
        ToneConstraint(Position(3, 0), ToneMark.AEK, _KS),  # บาท๒ คำ๖
        ToneConstraint(Position(4, 2), ToneMark.AEK, _KS),  # บาท๓ คำ๓
        ToneConstraint(Position(5, 1), ToneMark.AEK, _KS),  # บาท๓ คำ๗
        ToneConstraint(Position(6, 1), ToneMark.AEK, _KS),  # บาท๔ คำ๒
        ToneConstraint(Position(7, 0), ToneMark.AEK, _KS),  # บาท๔ คำ๖
        # โท ๔ แห่ง
        ToneConstraint(Position(0, 4), ToneMark.TOO, _KS),  # บาท๑ คำ๕
        ToneConstraint(Position(3, 1), ToneMark.TOO, _KS),  # บาท๒ คำ๗
        ToneConstraint(Position(6, 4), ToneMark.TOO, _KS),  # บาท๔ คำ๕
        ToneConstraint(Position(7, 1), ToneMark.TOO, _KS),  # บาท๔ คำ๗
    ),
    rhyme_links=(
        # คำสุดท้ายบาท ๑ (คำ๗) → คำที่ ๕ ของบาท ๒
        RhymeLink(Position(1, 1), (PositionRange(2, 4, 4),), _KS_RHYME,
                  note="ส่งสัมผัสบาท๑→บาท๒ คำ๕"),
        # คำสุดท้ายบาท ๑ (คำ๗) → คำที่ ๕ ของบาท ๓
        RhymeLink(Position(1, 1), (PositionRange(4, 4, 4),), _KS_RHYME,
                  note="ส่งสัมผัสบาท๑→บาท๓ คำ๕"),
        # คำสุดท้ายบาท ๒ (คำ๗) → คำที่ ๕ ของบาท ๔
        RhymeLink(Position(3, 1), (PositionRange(6, 4, 4),), _KS_RHYME,
                  note="ส่งสัมผัสบาท๒→บาท๔ คำ๕"),
    ),
)


# ===========================================================================
# Meter 2 — กลอนแปด (กลอนสุภาพ)
# ===========================================================================
# 1 บท = 4 วรรค (สดับ/รับ/รอง/ส่ง), ~8 พยางค์/วรรค (รับได้ ๗–๙). สัมผัสภายในบท +
# สัมผัสระหว่างบท. เสียงวรรณยุกต์ท้ายวรรคเป็นข้อแนะนำ (ไม่บังคับใน v1).
# Source: dhanitarampaigul.wordpress (ฉันทลักษณ์) + khuntong52.blogspot (สัมผัส).
_KP = "klon-paet: dhanitarampaigul.wordpress.com (ฉันทลักษณ์) + khuntong52.blogspot.com (สัมผัส), 2026-05-23"

KLON_PAET = MeterSpec(
    meter_id="klon_paet",
    name_th="กลอนแปด",
    name_romanized="Klon Paet",
    description="1 บท = ๔ วรรค (สดับ/รับ/รอง/ส่ง); ๘ พยางค์/วรรค (รับได้ ๗–๙). "
                "ไม่มีบังคับรูปวรรณยุกต์เอก/โท; บังคับสัมผัสนอกตามแผน.",
    citation=_KP,
    lines=(
        LineSpec(VerseRole.SADAB, 8, 7, 9, bat_index=0, bot_index=0, note="วรรคสดับ"),
        LineSpec(VerseRole.RAP,   8, 7, 9, bat_index=0, bot_index=0, note="วรรครับ"),
        LineSpec(VerseRole.RONG,  8, 7, 9, bat_index=1, bot_index=0, note="วรรครอง"),
        LineSpec(VerseRole.SONG,  8, 7, 9, bat_index=1, bot_index=0, note="วรรคส่ง"),
    ),
    rhyme_links=(
        # คำสุดท้ายวรรคสดับ → คำที่ ๓ หรือ ๕ ของวรรครับ (encoded as range คำ๓–๕)
        RhymeLink(Position(0, -1), (PositionRange(1, 2, 2), PositionRange(1, 4, 4)), _KP,
                  note="สดับ(ท้าย) → รับ คำ๓ หรือ คำ๕ (สารานุกรมไทยฯ)"),
        # คำสุดท้ายวรรครับ → คำสุดท้ายวรรครอง
        RhymeLink(Position(1, -1), (PositionRange(2, -1, -1),), _KP,
                  note="รับ(ท้าย) → รอง(ท้าย)"),
        # คำสุดท้ายวรรครอง → คำที่ ๓ หรือ ๕ ของวรรคส่ง
        RhymeLink(Position(2, -1), (PositionRange(3, 2, 2), PositionRange(3, 4, 4)), _KP,
                  note="รอง(ท้าย) → ส่ง คำ๓ หรือ คำ๕ (สารานุกรมไทยฯ)"),
    ),
    external_rhyme_links=(
        # คำสุดท้ายวรรคส่ง → คำสุดท้ายวรรครับของบทถัดไป
        RhymeLink(Position(3, -1), (PositionRange(1, -1, -1),), _KP,
                  kind="external", optional=True,
                  note="สัมผัสระหว่างบท: ส่ง(ท้าย) → รับ(ท้าย) บทถัดไป"),
    ),
)


# ===========================================================================
# Meter 3 — กาพย์ยานี ๑๑
# ===========================================================================
# 1 บท = 2 บาท; แต่ละบาทแบ่ง 5+6 (รวม ๑๑) → ๔ วรรค/บท. ไม่มี ครุ/ลหุ.
# Source: th.wikipedia.org/wiki/กาพย์ยานี + trueplookpanya/learning/detail/32162.
_KY = "th.wikipedia.org/wiki/กาพย์ยานี + trueplookpanya.com/learning/detail/32162, 2026-05-23"

KAP_YANI_11 = MeterSpec(
    meter_id="kap_yani_11",
    name_th="กาพย์ยานี ๑๑",
    name_romanized="Kap Yani Sip-et",
    description="2 บาท/บท; แต่ละบาท ๕+๖ พยางค์ (รวม ๑๑). บังคับสัมผัสนอก ๒ ตำแหน่ง/บท "
                "+ สัมผัสระหว่างบท. ไม่มีบังคับรูปวรรณยุกต์/ครุลหุ.",
    citation=_KY,
    lines=(
        LineSpec(VerseRole.WAK_NA,   5, 5, 5, bat_index=0, bot_index=0, note="บาทเอก วรรคหน้า ๕"),
        LineSpec(VerseRole.WAK_LANG, 6, 6, 6, bat_index=0, bot_index=0, note="บาทเอก วรรคหลัง ๖"),
        LineSpec(VerseRole.WAK_NA,   5, 5, 5, bat_index=1, bot_index=0, note="บาทโท วรรคหน้า ๕"),
        LineSpec(VerseRole.WAK_LANG, 6, 6, 6, bat_index=1, bot_index=0, note="บาทโท วรรคหลัง ๖"),
    ),
    rhyme_links=(
        # คำสุดท้ายวรรค ๑ (คำ๕) → คำที่ ๓ ของวรรค ๒
        RhymeLink(Position(0, 4), (PositionRange(1, 2, 2),), _KY,
                  note="วรรค๑(ท้าย) → วรรค๒ คำ๓"),
        # คำสุดท้ายวรรค ๒ (คำ๖) → คำสุดท้ายวรรค ๓ (คำ๕)
        RhymeLink(Position(1, 5), (PositionRange(2, 4, 4),), _KY,
                  note="วรรค๒(ท้าย) → วรรค๓(ท้าย)"),
    ),
    external_rhyme_links=(
        # คำสุดท้ายบท (วรรค ๔ ท้าย) → คำสุดท้ายวรรค ๒ ของบทถัดไป
        RhymeLink(Position(3, -1), (PositionRange(1, -1, -1),), _KY,
                  kind="external", optional=True,
                  note="สัมผัสระหว่างบท: วรรค๔(ท้าย) → วรรค๒(ท้าย) บทถัดไป"),
    ),
)


# ===========================================================================
# Meter 4 — อินทรวิเชียรฉันท์ ๑๑
# ===========================================================================
# Line structure + rhyme = กาพย์ยานี ๑๑, PLUS a fixed ครุ/ลหุ pattern:
#   วรรคหน้า (5): ค ค ล ค ค  |  วรรคหลัง (6): ล ล ค ล ค ค
# Source: phkruthai.in.th (4.3 อินทรวิเชียรฉันท์ ๑๑) + krupiyarerk.wordpress.com.
_IW = "intharawichian: phkruthai.in.th/.../4-3-อินทรวเชยรฉนท-11 + krupiyarerk.wordpress.com, 2026-05-23"
_K, _L = Weight.KARU, Weight.LAHU

INTHARAWICHIAN_CHAN_11 = MeterSpec(
    meter_id="intharawichian_chan_11",
    name_th="อินทรวิเชียรฉันท์ ๑๑",
    name_romanized="Intharawichian Chan Sip-et",
    description="โครงสร้าง/สัมผัสเหมือนกาพย์ยานี ๑๑ แต่บังคับ ครุ/ลหุ: "
                "วรรคหน้า ค-ค-ล-ค-ค, วรรคหลัง ล-ล-ค-ล-ค-ค.",
    citation=_IW,
    lines=(
        LineSpec(VerseRole.WAK_NA,   5, 5, 5, bat_index=0, bot_index=0, note="บาทเอก วรรคหน้า ๕"),
        LineSpec(VerseRole.WAK_LANG, 6, 6, 6, bat_index=0, bot_index=0, note="บาทเอก วรรคหลัง ๖"),
        LineSpec(VerseRole.WAK_NA,   5, 5, 5, bat_index=1, bot_index=0, note="บาทโท วรรคหน้า ๕"),
        LineSpec(VerseRole.WAK_LANG, 6, 6, 6, bat_index=1, bot_index=0, note="บาทโท วรรคหลัง ๖"),
    ),
    weight_patterns=(
        WeightPattern(0, (_K, _K, _L, _K, _K), _IW),           # วรรคหน้า: ค ค ล ค ค
        WeightPattern(1, (_L, _L, _K, _L, _K, _K), _IW),       # วรรคหลัง: ล ล ค ล ค ค
        WeightPattern(2, (_K, _K, _L, _K, _K), _IW),
        WeightPattern(3, (_L, _L, _K, _L, _K, _K), _IW),
    ),
    rhyme_links=(
        RhymeLink(Position(0, 4), (PositionRange(1, 2, 2),), _IW,
                  note="วรรค๑(ท้าย) → วรรค๒ คำ๓"),
        RhymeLink(Position(1, 5), (PositionRange(2, 4, 4),), _IW,
                  note="วรรค๒(ท้าย) → วรรค๓(ท้าย)"),
    ),
    external_rhyme_links=(
        RhymeLink(Position(3, -1), (PositionRange(1, -1, -1),), _IW,
                  kind="external", optional=True,
                  note="สัมผัสระหว่างบท: วรรค๔(ท้าย) → วรรค๒(ท้าย) บทถัดไป"),
    ),
)


# --- registry --------------------------------------------------------------

_ALL_METERS = (KHLONG_SI_SUPHAP, KLON_PAET, KAP_YANI_11, INTHARAWICHIAN_CHAN_11)

#: Grader entry point — `METER_SPECS[name_th]` → `MeterSpec`.
METER_SPECS: dict[str, MeterSpec] = {m.name_th: m for m in _ALL_METERS}

#: Same registry keyed by ascii `meter_id`.
METER_SPECS_BY_ID: dict[str, MeterSpec] = {m.meter_id: m for m in _ALL_METERS}


# --- self-consistency check -------------------------------------------------

def _syll_in_line(idx: int, line: LineSpec) -> bool:
    """True if syllable index `idx` (possibly negative = from end) can resolve
    within a line that may hold up to `max_count` syllables."""
    return idx < line.max_count if idx >= 0 else -idx <= line.max_count


def check_specs() -> list[str]:
    """Validate internal consistency of every registered meter. Returns a list
    of human-readable problems (empty == all good). Raises nothing."""
    problems: list[str] = []

    seen_ids: set[str] = set()
    for name, spec in METER_SPECS.items():
        if spec.meter_id in seen_ids:
            problems.append(f"{name}: duplicate meter_id {spec.meter_id!r}")
        seen_ids.add(spec.meter_id)

        nlines = len(spec.lines)
        for i, ln in enumerate(spec.lines):
            if not (ln.min_count <= ln.canonical_count <= ln.max_count):
                problems.append(f"{name} line{i}: count window "
                                f"{ln.min_count}/{ln.canonical_count}/{ln.max_count} invalid")

        def _check_pos(pos: Position, label: str) -> None:
            if not (0 <= pos.line_index < nlines):
                problems.append(f"{name} {label}: line_index {pos.line_index} out of range")
            elif not _syll_in_line(pos.syllable_index, spec.lines[pos.line_index]):
                problems.append(f"{name} {label}: syllable_index {pos.syllable_index} "
                                f"exceeds line{pos.line_index} (max {spec.lines[pos.line_index].max_count})")

        for tc in spec.tone_constraints:
            _check_pos(tc.position, f"tone {tc.mark.value}")

        for wp in spec.weight_patterns:
            if not (0 <= wp.line_index < nlines):
                problems.append(f"{name} weight: line_index {wp.line_index} out of range")
            elif len(wp.pattern) != spec.lines[wp.line_index].canonical_count:
                problems.append(f"{name} weight line{wp.line_index}: pattern len "
                                f"{len(wp.pattern)} != canonical {spec.lines[wp.line_index].canonical_count}")

        def _check_targets(rl: RhymeLink, label: str) -> None:
            # external targets are interpreted against the next stanza's
            # identical block, so they are bound by nlines too.
            for tr in rl.targets:
                if not (0 <= tr.line_index < nlines):
                    problems.append(f"{name} {label} tgt: line_index {tr.line_index} out of range")
                    continue
                tline = spec.lines[tr.line_index]
                if not (_syll_in_line(tr.start, tline) and _syll_in_line(tr.end, tline)):
                    problems.append(f"{name} {label} tgt: range [{tr.start},{tr.end}] exceeds line{tr.line_index}")

        for rl in spec.rhyme_links:
            _check_pos(rl.source, "rhyme src")
            _check_targets(rl, "rhyme")

        for rl in spec.external_rhyme_links:
            _check_pos(rl.source, "ext-rhyme src")
            _check_targets(rl, "ext-rhyme")

        # JSON round-trip
        try:
            json.dumps(spec.to_dict(), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            problems.append(f"{name}: to_dict() not JSON-serializable ({exc})")

    return problems


def _summary() -> None:
    print(f"{'meter':<26}{'lines':>6}{'syll':>6}{'tone':>6}{'rhyme':>7}{'wt':>5}")
    for name, spec in METER_SPECS.items():
        total_syll = sum(ln.canonical_count for ln in spec.mandatory_lines)
        nrhyme = len(spec.rhyme_links) + len(spec.external_rhyme_links)
        print(f"{name:<26}{spec.lines_per_stanza:>6}{total_syll:>6}"
              f"{len(spec.tone_constraints):>6}{nrhyme:>7}{len(spec.weight_patterns):>5}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    _summary()
    issues = check_specs()
    if issues:
        print("\nPROBLEMS:")
        for p in issues:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\ncheck_specs: OK, all 4 meters internally consistent.")
