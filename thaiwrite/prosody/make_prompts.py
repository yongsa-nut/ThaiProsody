"""Prompt generator for the prosody benchmark.

Produces the three task modes into `data/prompts_{mode}_{public,private}.jsonl`:

  - compose: fully generated here (4 meters x 40 topics = 160). Needs no source
    poem, just {meter, n_stanzas, topic}.
  - cloze: blanks the *hard* positions (rhyme / tone / ครุ-ลหุ, from
    `meter_specs`) of a meter-compliant SOURCE poem.
  - edit: injects a tone-mark violation (โคลง) or ครุ/ลหุ violation (ฉันท์)
    on a specific word of a source poem; gold = rule satisfaction after the edit
    (multi-reference, blind: `broken_rules` is grading-side only).
    Edit mode covers **โคลง + ฉันท์ only**: กลอนแปด and กาพย์ยานี bind no
    tone/weight, so "tone-mark errors" have no gradeable target.

cloze and edit draw on the pool of meter-compliant source poems assembled in
`data/source_poems.py` (composed, audit-corpus, public-domain and generated
poems, each verified to score 1.0 under the rule grader). Run from the repo root:
  python thaiwrite/prosody/make_prompts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # repo root

from thaiwrite.prosody import meter_specs as M
from thaiwrite.prosody.data.source_poems import SOURCE_POEMS
from thaiwrite.prosody.data.topics import TOPICS, PUBLIC_CUTOFF
from thaiwrite.prosody.grader_meter import grade_text

DATA = Path(__file__).with_name("data")
SLUG = {"khlong_si_suphap": "khlong", "klon_paet": "klon8",
        "kap_yani_11": "yani", "intharawichian_chan_11": "chan"}
N_STANZAS = {"khlong_si_suphap": 1, "klon_paet": 2, "kap_yani_11": 2,
             "intharawichian_chan_11": 2}
AEK_MARK, TOO_MARK = "่", "้"   # ไม้เอก ่ , ไม้โท ้


def _resolve(idx: int, n: int) -> int:
    return idx if idx >= 0 else n + idx


def _render(lines) -> str:
    """One วรรค per line (the benchmark input convention)."""
    return "\n".join("".join(l) for l in lines)


# --- compose ---------------------------------------------------------------

def gen_compose() -> dict[str, list[dict]]:
    out = {"public": [], "private": []}
    for meter_id, slug in SLUG.items():
        n = 0
        for ti, (_cat, topic) in enumerate(TOPICS):
            n += 1
            split = "public" if ti < PUBLIC_CUTOFF else "private"
            out[split].append({
                "id": f"prosody-{slug}-compose-{n:03d}",
                "task_mode": "compose",
                "meter": M.METER_SPECS_BY_ID[meter_id].name_th,
                "meter_id": meter_id,
                "n_stanzas": N_STANZAS[meter_id],
                "topic": topic,
            })
    return out


# --- hard positions (for cloze) -------------------------------------------

def hard_positions(spec: M.MeterSpec) -> list[tuple[int, int, str]]:
    """(line_index, syllable_index, constraint_label) for meter-exercising slots:
    rhyme endpoints, tone-mark positions, ครุ/ลหุ positions."""
    pos: list[tuple[int, int, str]] = []
    for rl in spec.rhyme_links:
        pos.append((rl.source.line_index, rl.source.syllable_index, "rhyme"))
    for tc in spec.tone_constraints:
        pos.append((tc.position.line_index, tc.position.syllable_index, f"tone_{tc.mark.value}"))
    for wp in spec.weight_patterns:
        for i, w in enumerate(wp.pattern):
            pos.append((wp.line_index, i, w.value))
    return pos


# --- cloze -----------------------------------------------------------------

def _cloze_cands(poem) -> list[tuple[int, int, str]]:
    """Resolved, deduped, in-range hard positions for this poem."""
    spec = M.METER_SPECS_BY_ID[poem.meter_id]
    seen, cands = set(), []
    for li, si, label in hard_positions(spec):
        if not (0 <= li < len(poem.lines)):
            continue
        idx = _resolve(si, len(poem.lines[li]))
        if 0 <= idx < len(poem.lines[li]) and (li, idx) not in seen:
            seen.add((li, idx))
            cands.append((li, idx, label))
    return cands


def gen_cloze(variants_per_poem: int = 4, blanks_per_item: int = 2) -> list[dict]:
    """Up to `variants_per_poem` cloze items per source poem, each blanking a
    DIFFERENT set of hard positions (partitioned across the poem's positions, so
    variants are distinct). Capped at the number of available positions: a meter
    that binds few positions (กาพย์ยานี: 2 rhyme links) yields fewer variants."""
    prompts = []
    ctr: dict[str, int] = {}
    for poem in SOURCE_POEMS:
        spec = M.METER_SPECS_BY_ID[poem.meter_id]
        cands = _cloze_cands(poem)
        if not cands:
            continue
        n_var = min(variants_per_poem, len(cands))
        for v in range(n_var):
            # partition positions across variants so each variant blanks different slots
            picks = [cands[i] for i in range(len(cands)) if i % n_var == v][:blanks_per_item]
            if not picks:
                continue
            lines = [list(l) for l in poem.lines]
            blanks = []
            for li, idx, label in picks:
                lines[li][idx] = "____"
                blanks.append({"line": li, "syllable": idx, "constraint": label})
            slug = SLUG[poem.meter_id]
            ctr[slug] = ctr.get(slug, 0) + 1
            prompts.append({
                "id": f"prosody-{slug}-cloze-{ctr[slug]:03d}",
                "task_mode": "cloze",
                "meter": spec.name_th,
                "meter_id": poem.meter_id,
                "topic": poem.source.split("—")[-1].strip()[:40] or "บทร้อยกรอง",
                "poem_with_blanks": _render(lines),
                "n_blanks": len(blanks),
                "blank_positions": blanks,
                "_source": poem.source,
            })
    return prompts


# --- edit ------------------------------------------------------------------

def _strip_mark(syll: str, mark: str) -> str:
    return syll.replace(mark, "")


def _edit_slots(poem) -> list[tuple[int, int, str, object]]:
    """Breakable single positions: tone slots (โคลง) or ครุ/ลหุ lahu slots (ฉันท์).
    Returns (line, syllable, kind, mark) tuples; empty for กลอน/กาพย์."""
    spec = M.METER_SPECS_BY_ID[poem.meter_id]
    slots: list[tuple[int, int, str, object]] = []
    if spec.tone_constraints:
        for tc in spec.tone_constraints:
            slots.append((tc.position.line_index, tc.position.syllable_index, "tone", tc.mark))
    elif spec.weight_patterns:
        for wp in spec.weight_patterns:
            for i, w in enumerate(wp.pattern):
                if w is M.Weight.LAHU:           # break a ลหุ slot with a ครุ syllable
                    slots.append((wp.line_index, i, "weight", None))
    return slots


def gen_edit(variants_per_poem: int = 4) -> tuple[list[dict], list[str]]:
    """Up to `variants_per_poem` edit items per source poem: each injects ONE
    violation at a DIFFERENT position (single-error items are cleanly gradeable).
    โคลง + ฉันท์ only (the tone/weight-bearing meters)."""
    prompts, skipped = [], []
    ctr: dict[str, int] = {}
    for poem in SOURCE_POEMS:
        spec = M.METER_SPECS_BY_ID[poem.meter_id]
        slots = _edit_slots(poem)
        if not slots:                             # กลอน / กาพย์ยานี: no tone/weight rule
            skipped.append(poem.meter_id)
            continue
        made = 0
        for li, si, kind, mark in slots:
            if made >= variants_per_poem:
                break
            lines = [list(l) for l in poem.lines]
            idx = _resolve(si, len(lines[li]))
            if kind == "tone":
                m = AEK_MARK if mark is M.ToneMark.AEK else TOO_MARK
                if m not in lines[li][idx]:        # nothing to strip → skip this slot
                    continue
                lines[li][idx] = _strip_mark(lines[li][idx], m)
                broken = [f"tone_violation_line_{li}_syllable_{idx}"]
            else:
                lines[li][idx] = "มา"              # ครุ in a ลหุ slot
                broken = [f"weight_violation_line_{li}_syllable_{idx}"]
            input_poem = _render(lines)
            # validity gate: the injection must actually break the meter, else the
            # "fix it" task has nothing to repair (no-op strips, misaligned slots).
            if grade_text(input_poem, poem.meter_id, eval_rhyme=True)["score"] >= 1.0:
                continue
            slug = SLUG[poem.meter_id]
            ctr[slug] = ctr.get(slug, 0) + 1
            made += 1
            prompts.append({
                "id": f"prosody-{slug}-edit-{ctr[slug]:03d}",
                "task_mode": "edit",
                "meter": spec.name_th,
                "meter_id": poem.meter_id,
                "topic": poem.source.split("—")[-1].strip()[:40] or "บทร้อยกรอง",
                "input_poem": input_poem,
                "broken_rules": broken,   # grading-side ground truth; never shown to model
                "_source": poem.source,
            })
    return prompts, skipped


# --- write -----------------------------------------------------------------

def _write(name: str, rows: list[dict]) -> None:
    path = DATA / name
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):3d}  {name}")


def main() -> None:
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser(description="Generate prosody prompts (compose/cloze/edit).")
    ap.add_argument("--cloze-variants", type=int, default=4, help="cloze items per source poem")
    ap.add_argument("--cloze-blanks", type=int, default=2, help="blanks per cloze item")
    ap.add_argument("--edit-variants", type=int, default=4, help="edit items per source poem")
    args = ap.parse_args()

    comp = gen_compose()
    _write("prompts_compose_public.jsonl", comp["public"])
    _write("prompts_compose_private.jsonl", comp["private"])

    cloze = gen_cloze(args.cloze_variants, args.cloze_blanks)
    _write("prompts_cloze_public.jsonl", cloze)
    edit, skipped = gen_edit(args.edit_variants)
    _write("prompts_edit_public.jsonl", edit)           # โคลง + ฉันท์ only

    def _by_meter(rows):
        return dict(Counter(r["meter_id"] for r in rows))
    print(f"\ncompose: {len(comp['public'])} public / {len(comp['private'])} private  (FULL)")
    print(f"cloze:   {len(cloze)} from {len(SOURCE_POEMS)} source poems "
          f"(≤{args.cloze_variants}/poem) — by meter: {_by_meter(cloze)}")
    print(f"edit:    {len(edit)} (โคลง + ฉันท์ only, ≤{args.edit_variants}/poem) — "
          f"by meter: {_by_meter(edit)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
