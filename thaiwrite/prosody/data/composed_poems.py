"""LLM-composed source poems for the two rigid/edit meters (โคลง + ฉันท์).

Recent real โคลง/ฉันท์ could not be obtained as plain text on the open web
(recent verse is mostly published as images or behind paywalls), so the cloze and
edit source poems for these meters were composed fresh by Claude Opus 4.7. They
are contamination-free by construction (newly written, never published) and each
is rule-verified to score exactly 1.0 (the composer supplies its own gold syllable
segmentation, so the count layer is exact and bypasses the tokenizer). 12 โคลง +
12 ฉันท์, curated (near-duplicate templates and filler-heavy poems regenerated).

Loaded from `composed/all_composed.json` (the verified output of
composed/collect_composed.py). `source_poems.py` uses these first for these meters,
ahead of the generated placeholders AND the public-domain classics in
`real_source_poems.py` (which carry memorization risk for cloze/edit). กลอน and
กาพย์ยานี are unaffected (this module holds no poems for them).
"""
from __future__ import annotations

import json
from pathlib import Path

from thaiwrite.prosody.data.audit_poems import Poem

_JSON = Path(__file__).resolve().parent.parent / "composed" / "all_composed.json"


def _load() -> tuple[Poem, ...]:
    if not _JSON.exists():
        return ()
    data = json.loads(_JSON.read_text(encoding="utf-8"))
    # `source` is the THEME: make_prompts uses it as the cloze/edit topic for the
    # coherence judge, so keep it a clean theme. Provenance (Opus-composed vs
    # real-2025) is tracked in the JSON's `provenance` field, not here.
    return tuple(
        Poem(meter_id=p["meter_id"], source=p.get("source", ""),
             lines=tuple(tuple(v) for v in p["lines"]))
        for p in data
    )


COMPOSED_POEMS: tuple[Poem, ...] = _load()
