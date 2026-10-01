"""Meter-compliant SOURCE poems that feed the cloze + edit generators.

`make_prompts.py` reads `SOURCE_POEMS`. Priority order:
  1. `_COMPOSED`     LLM-composed โคลง + ฉันท์ (composed_poems.py), contamination-free.
  2. `_SEED`         audit corpus (real, hand-segmented, verified; audit_poems.py).
  3. `_REAL`         real public-domain stanzas, grader-verified (real_source_poems.py).
  4. `_PLACEHOLDER`  auto-harvested gpt-5.5 poems (placeholder_poems.py).
  5. `STUDENT_POEMS` additional hand-verified poems (empty by default; append below).
The pool is **capped per meter at TARGET**, so earlier sources are used first and
generated placeholders only fill the remaining gap. Every poem must score 1.0 in
`grader_meter.score_meter` (the entry bar); see `source_poems_guide.md`.
"""

from __future__ import annotations

from collections import Counter

from thaiwrite.prosody.data.audit_poems import POEMS as _SEED, Poem

try:
    from thaiwrite.prosody.data.real_source_poems import REAL_POEMS as _REAL
except ImportError:
    _REAL = ()
try:
    from thaiwrite.prosody.data.placeholder_poems import PLACEHOLDER_POEMS as _PLACEHOLDER
except ImportError:
    _PLACEHOLDER = ()
try:
    # Opus-composed โคลง + ฉันท์ (contamination-free, rule-verified 1.0). Top priority
    # for the rigid/edit meters: used ahead of placeholders AND public-domain classics.
    from thaiwrite.prosody.data.composed_poems import COMPOSED_POEMS as _COMPOSED
except ImportError:
    _COMPOSED = ()

#: Additional hand-authored, grader-verified poems. Append `Poem(...)` entries here
#: (shape in audit_poems.py); they enter the pool after the sources above.
STUDENT_POEMS: tuple[Poem, ...] = (
    # --- add verified source poems below ---
)

#: per-meter source-poem target. โคลง/ฉันท์ use 12 so that all 12 composed,
#: contamination-free poems per meter are included.
TARGET = {"khlong_si_suphap": 12, "intharawichian_chan_11": 12,
          "klon_paet": 14, "kap_yani_11": 20}


def _capped_pool() -> tuple[Poem, ...]:
    """Composed (Opus, contamination-free) first for rigid meters, then real seed/classics,
    then placeholders, then additional hand-authored poems, capped per meter."""
    seen: Counter[str] = Counter()
    pool: list[Poem] = []
    for p in (*_COMPOSED, *_SEED, *_REAL, *_PLACEHOLDER, *STUDENT_POEMS):
        if seen[p.meter_id] < TARGET.get(p.meter_id, 9_999):
            pool.append(p)
            seen[p.meter_id] += 1
    return tuple(pool)


#: Consumed by make_prompts.py.
SOURCE_POEMS: tuple[Poem, ...] = _capped_pool()
