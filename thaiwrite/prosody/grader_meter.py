"""Rule-based meter grader for the Thai prosody benchmark.

Scores a poem's *meter adherence* against `meter_specs`, generically across the
four meters. Layered, weighted scoring in [0, 1]; per-rule breakdown returned.

Design follows the pythainlp.khavee audit (`docs/khavee-audit.md`):
  * Tone (เอก/โท): delegate to `khavee.check_aek_too` (audit: 11/11 reliable).
  * ครุ/ลหุ: delegate to `khavee.check_karu_lahu` (audit: 46/46 reliable),
    with two overrides: open สระออ (ทอ, พอ, ขอ) is ครุ; บ/บ่ is ลหุ (see `weight`).
  * Rhyme (สัมผัส): `khavee.is_sumpus` fast path + a phonetic-rime fallback that
    closes its อักษรนำ / การันต์ gaps (audit: is_sumpus alone 8/11, with fallback
    14/14), and a สระไอ fallback for ไ/ใ syllables (ใจ/ใกล้, ไกล/ใคร; see
    `_is_sara_ai_syllable`).
  * Syllable count: `syllable_tokenize(engine="dict")` by default, a soft signal
    (penalty, not gate). The paper's tables recompute the count layer from the
    count-of-record syllable counts (`count_api.py`; see
    `reproduce/make_paper_tables.py`).

Input convention: a poem is **newline-delimited วรรค**,
one วรรค per line, ordered to match `meter_specs[…].lines`. `parse_poem` turns
raw text into the `lines` structure; `score_meter` grades a pre-split `lines`.
"""

from __future__ import annotations

import re

from pythainlp.khavee import KhaveeVerifier
from pythainlp.tokenize import syllable_tokenize
from pythainlp.transliterate import romanize

from thaiwrite.prosody import meter_specs as M

_KV = KhaveeVerifier()

Lines = list[tuple[str, ...]]


# --- tokenization ----------------------------------------------------------

def split_syllables(text: str) -> list[str]:
    """Best-available syllable split (dict engine). NOTE: ~76% line-level on
    classical verse (LLM-gold audit), so callers treat counts as approximate.
    Whitespace-only tokens are dropped: the dict engine returns a ' ' token for
    each internal space (e.g. โคลง วรรคหน้า/หลัง on one line), which would
    otherwise be miscounted as a syllable."""
    text = text.strip()
    if not text:
        return []
    return [t for t in syllable_tokenize(text, engine="dict") if t.strip()]


def parse_poem(text: str) -> Lines:
    """Raw newline-delimited-วรรค text → `lines` (one tuple of syllables per
    non-empty line), ready for `score_meter`."""
    return [tuple(split_syllables(ln)) for ln in text.splitlines() if ln.strip()]


# --- position resolution ---------------------------------------------------

def _norm(idx: int, n: int) -> int:
    return idx if idx >= 0 else n + idx


def _syll_at(lines: Lines, line_index: int, syllable_index: int) -> str | None:
    if not (0 <= line_index < len(lines)):
        return None
    line = lines[line_index]
    i = _norm(syllable_index, len(line))
    return line[i] if 0 <= i < len(line) else None


def _sylls_in_range(lines: Lines, rng: M.PositionRange) -> list[str]:
    if not (0 <= rng.line_index < len(lines)):
        return []
    line = lines[rng.line_index]
    n = len(line)
    s, e = _norm(rng.start, n), _norm(rng.end, n)
    return [line[i] for i in range(s, e + 1) if 0 <= i < n]


# --- rhyme: is_sumpus + phonetic-rime fallback -----------------------------

_LEAD_V = "เแโใไ"
_LOW_AKSORN = "งญณนมยรลฬวฮ"  # low-class consonants that take silent ห นำ


def _strip_silent_h(w: str) -> str:
    """Drop a silent leading ห นำ (หนด→นด, ใหล→ใล) so is_sumpus can read the rime."""
    if len(w) >= 2 and w[0] == "ห" and w[1] in _LOW_AKSORN:
        return w[1:]
    if len(w) >= 3 and w[0] in _LEAD_V and w[1] == "ห" and w[2] in _LOW_AKSORN:
        return w[0] + w[2:]
    return w


def _roman_rime(w: str) -> str:
    """Romanized rime = from the first vowel onward (strips the onset cluster).
    Catches การันต์ finals khavee mishandles, e.g. ศักดิ์→'sak'→'ak'. pythainlp's
    royin engine can raise on malformed/odd tokens → degrade to '' (no rime)."""
    try:
        r = romanize(w)
    except Exception:  # noqa: BLE001
        return ""
    i = 0
    while i < len(r) and r[i] not in "aeiou":
        i += 1
    return r[i:]


_TONE_MARKS = "่้๊๋"
_KARAN = re.compile(r"[ก-ฮ]์")      # a letter silenced by การันต์ (ไฟล์ → ไฟ)


def _is_sara_ai_syllable(w: str) -> bool:
    """True if `w` is ONE syllable spelled with ไ or ใ (ใจ, ไกล, ใกล้, ใคร, ใหญ่, ไทย,
    ไหล, ไฟล์). ไ and ใ write the same short vowel /aj/, and a ไ/ใ syllable has no
    spoken final: a trailing ล/ร/ว belongs to the onset cluster (ไกล, ใคร, ไขว่), ห is
    a silent leader (ใหญ่, ไหล), and ย in ไทย or a การันต์ letter is silent. khavee's
    check_marttra reads the cluster ล / silent ย as a final (ไกล, ใกล้, ไทย → แม่เกย) and
    the royin romanizer adds a spurious final to ใ-clusters (ใกล้ → 'kain'), so both
    rhyme paths reject ใจ–ใกล้, ไกล–ใคร. Onset ≤ 2 consonants and no other vowel sign,
    so polysyllabic tokens (ใจดี, ไปมา, ใจคน) are excluded."""
    if len(w) < 2 or w[0] not in "ไใ":
        return False
    body = "".join(c for c in _KARAN.sub("", w[1:]) if c not in _TONE_MARKS)
    return 1 <= len(body) <= 2 and all("ก" <= c <= "ฮ" for c in body)


def _ai_rime(w: str) -> tuple:
    """(สระ, มาตรา) as is_sumpus reads it, except that an open ไ/ใ syllable is read
    as (ไอ, กา), the corrected rime. -ัย (ชัย) keeps is_sumpus's own remap."""
    if _is_sara_ai_syllable(w):
        return ("ไอ", "กา")
    s, m = _safe(_KV.check_sara, w), _safe(_KV.check_marttra, w)
    if (s, m) == ("อะ", "เกย"):
        return ("ไอ", "กา")
    return (s, m)


def rhymes(a: str, b: str) -> bool:
    """True if a and b rhyme (สัมผัสสระ). is_sumpus fast-path, then fallbacks for
    the อักษรนำ / การันต์ words is_sumpus misses (audit §3a) and for open ไ/ใ
    syllables (สระไอ). Crash-safe: any pythainlp failure on a pathological token
    degrades to 'no rhyme' (not an error)."""
    try:
        if _KV.is_sumpus(a, b):
            return True
        ha, hb = _strip_silent_h(a), _strip_silent_h(b)
        if (ha, hb) != (a, b) and _KV.is_sumpus(ha, hb):
            return True
    except Exception:  # noqa: BLE001
        pass
    # สระไอ fallback, only when at least one side is an open ไ/ใ syllable: compare
    # the corrected rimes (ใจ–ใกล้, ไกล–ใคร, ใกล้–ชัย rhyme; ใกล้–กาย does not).
    if (_is_sara_ai_syllable(a) or _is_sara_ai_syllable(b)) and _ai_rime(a) == _ai_rime(b):
        return True
    ra = _roman_rime(a)
    return bool(ra) and ra == _roman_rime(b)


# --- per-layer checks ------------------------------------------------------

def _layer_count(lines: Lines, spec: M.MeterSpec) -> dict:
    details, passed, total = [], 0, 0
    for i, ls in enumerate(spec.lines):
        if ls.optional and i >= len(lines):
            continue  # omitted optional (คำสร้อย) — not penalized
        total += 1
        n = len(lines[i]) if i < len(lines) else 0
        ok = ls.min_count <= n <= ls.max_count
        passed += ok
        details.append((i, n, f"{ls.min_count}-{ls.max_count}", ok))
    return {"passed": passed, "total": total, "details": details}


def _safe(fn, *a, **k):
    """khavee helpers IndexError on pathological tokens (e.g. consonant-less วรรค);
    degrade to None so one bad syllable doesn't crash the whole grade."""
    try:
        return fn(*a, **k)
    except Exception:  # noqa: BLE001
        return None


#: มาตราตัวสะกด with a stop final = คำตาย (dead). Short-vowel แม่ ก กา is also
#: dead but khavee's check_aek_too already credits those as เอก.
_DEAD_MATRA = {"กก", "กด", "กบ"}


def _is_dead_syllable(syll: str) -> bool:
    """True if `syll` is a คำตาย with a stop final (แม่กก/กด/กบ). Robust to การันต์
    and silent-ร clusters (สมุทร, มนุษย์, ทรัพย์) that check_aek_too mishandles."""
    return _safe(_KV.check_marttra, syll) in _DEAD_MATRA


def _layer_tone(lines: Lines, spec: M.MeterSpec) -> dict:
    details, passed = [], 0
    for tc in spec.tone_constraints:
        syll = _syll_at(lines, tc.position.line_index, tc.position.syllable_index)
        got = _safe(_KV.check_aek_too, syll, dead_syllable_as_aek=True) if syll else None
        ok = got == tc.mark.value
        # เอก-ตาย rule: a คำตาย satisfies a บังคับเอก position. khavee credits simple
        # dead syllables but misses complex ones (polysyllabic / การันต์ / silent-ร),
        # so fall back to มาตรา. Applies to เอก only (โท has no ตาย substitution).
        if not ok and syll and tc.mark is M.ToneMark.AEK and _is_dead_syllable(syll):
            ok = True
            got = f"{got}|dead→aek"
        passed += ok
        details.append((tc.position.line_index, tc.position.syllable_index,
                        syll, tc.mark.value, str(got), ok))
    return {"passed": passed, "total": len(spec.tone_constraints), "details": details}


#: Open syllable with long สระออ and no final: onset (≤ 2 consonants: ขอ, หนอ, คลอ)
#: + optional tone mark + อ. khavee's check_sara drops that อ and falls into its
#: implicit-โอะ branch for 2-letter (ทอ, พอ, ขอ, รอ) and 4-letter (หน่อ) spellings,
#: so check_karu_lahu calls them ลหุ; a long vowel is ครุ.
_OPEN_SARA_OR = re.compile(r"[ก-ฮ]{1,2}[่้๊๋]?อ")
#: บ / บ่ (short negation) is ลหุ. khavee special-cases บ่ but reads bare บ as แม่กบ → ครุ.
_LAHU_WORDS = {"บ", "บ่"}


def weight(syll: str) -> str | None:
    """ครุ/ลหุ of one syllable: khavee.check_karu_lahu plus the two overrides above."""
    if syll in _LAHU_WORDS:
        return "lahu"
    if _OPEN_SARA_OR.fullmatch(syll):
        return "karu"
    return _safe(_KV.check_karu_lahu, syll)


def _layer_kru_lahu(lines: Lines, spec: M.MeterSpec) -> dict:
    details, passed, total = [], 0, 0
    for wp in spec.weight_patterns:
        for i, w in enumerate(wp.pattern):
            syll = _syll_at(lines, wp.line_index, i)
            got = weight(syll) if syll else None
            ok = got == w.value
            total += 1
            passed += ok
            details.append((wp.line_index, i, syll, w.value, str(got), ok))
    return {"passed": passed, "total": total, "details": details}


def _layer_rhyme(lines: Lines, spec: M.MeterSpec) -> dict:
    # Internal links only; external (cross-บท) links need a following stanza
    # and are out of scope for single-บท grading in v1.
    details, passed = [], 0
    for rl in spec.rhyme_links:
        src = _syll_at(lines, rl.source.line_index, rl.source.syllable_index)
        cands: list[str] = []
        for rng in rl.targets:
            cands += _sylls_in_range(lines, rng)
        matched = [c for c in cands if src and rhymes(src, c)]
        ok = bool(matched)
        passed += ok
        details.append((src, cands, matched, ok))
    return {"passed": passed, "total": len(spec.rhyme_links), "details": details}


# --- scoring ---------------------------------------------------------------

#: Per-meter layer weights (weighted-layer scoring). Only layers
#: with ≥1 rule for the meter contribute; weights are renormalized over those.
LAYER_WEIGHTS: dict[str, dict[str, float]] = {
    "khlong_si_suphap":       {"count": 0.25, "tone": 0.40, "rhyme": 0.35},
    "klon_paet":              {"count": 0.40, "rhyme": 0.60},
    "kap_yani_11":            {"count": 0.40, "rhyme": 0.60},
    "intharawichian_chan_11": {"count": 0.20, "rhyme": 0.30, "kru_lahu": 0.50},
}


def _resolve(meter: str | M.MeterSpec) -> M.MeterSpec:
    if isinstance(meter, M.MeterSpec):
        return meter
    return M.METER_SPECS.get(meter) or M.METER_SPECS_BY_ID[meter]


def score_meter(lines: Lines, meter: str | M.MeterSpec, eval_rhyme: bool = True) -> dict:
    """Score meter adherence in [0, 1] for a pre-split poem.

    `lines` — list of syllable-tuples, one per วรรค, indexed to match the spec.
    `eval_rhyme=False` excludes the rhyme layer (e.g. grading isolated วรรค that
    are not one rhyming บท). Returns the score plus a per-layer breakdown.
    """
    spec = _resolve(meter)
    layers = {
        "count": _layer_count(lines, spec),
        "tone": _layer_tone(lines, spec),
        "rhyme": _layer_rhyme(lines, spec),
        "kru_lahu": _layer_kru_lahu(lines, spec),
    }
    weights = LAYER_WEIGHTS[spec.meter_id]

    num = den = 0.0
    rates: dict[str, float | None] = {}
    for name, w in weights.items():
        if name == "rhyme" and not eval_rhyme:
            rates[name] = None
            continue
        L = layers[name]
        if L["total"] == 0:           # layer not applicable to this meter
            rates[name] = None
            continue
        rate = L["passed"] / L["total"]
        rates[name] = rate
        num += w * rate
        den += w

    score = num / den if den else 0.0
    return {
        "meter_id": spec.meter_id,
        "meter": spec.name_th,
        "score": round(score, 4),
        "layer_rates": {k: (round(v, 4) if v is not None else None) for k, v in rates.items()},
        "layers": layers,
    }


def grade_text(text: str, meter: str | M.MeterSpec, eval_rhyme: bool = True) -> dict:
    """Convenience: grade raw newline-delimited-วรรค text."""
    return score_meter(parse_poem(text), meter, eval_rhyme=eval_rhyme)
