"""Generate cloze + edit items from the contamination-free pool (data/pool.json).

Same construction as `make_prompts.py` (cloze: blank the meter-bearing "hard"
positions (rhyme sources, tone-mark slots, ครุ/ลหุ slots), partitioned across
variants; edit: inject ONE violation (strip the required tone mark in โคลง, or put a
heavy syllable มา in a light slot in ฉันท์), self-validated to actually break the
meter), with three changes:
  * the poem is shown in its ORTHOGRAPHIC form (the student's own spelling), not as
    joined gold syllables; a blank covers the written unit holding the target
    syllable (e.g. the whole word สราญ when สะ is targeted);
  * a โคลง คำสร้อย is shown in its traditional place (after วรรค 6);
  * every item keeps its source-stanza id + provenance, and the reference segmentation
    needed for gold-anchored grading.
Edit mode covers โคลง + ฉันท์ only (กลอน/กาพย์ bind no tone/weight rule).

  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/make_items.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from thaiwrite.prosody import meter_specs as M                      # noqa: E402
from thaiwrite.prosody.grader_meter import score_meter              # noqa: E402
from thaiwrite.prosody.make_prompts import (AEK_MARK, TOO_MARK,     # noqa: E402
                                            _edit_slots, _resolve, hard_positions)

DATA = HERE / "data"
SLUG = {"khlong_si_suphap": "khlong", "klon_paet": "klon",
        "kap_yani_11": "yani", "intharawichian_chan_11": "chan"}
CLOZE_VARIANTS = {"intharawichian_chan_11": 6}     # thin pool → more variants/stanza
EDIT_VARIANTS = {"intharawichian_chan_11": 8}     # = every light slot
DEFAULT_VARIANTS, BLANKS = 4, 2
BLANK = "____"


class _P:                                          # minimal Poem shim for make_prompts helpers
    def __init__(self, rec):
        self.meter_id = rec["meter_id"]
        self.lines = [tuple(v) for v in rec["lines"]]


def _unit_of(rec, li: int, si: int) -> int:
    """Index of the unit in grader-order วรรค `li` that holds syllable `si`."""
    k = 0
    for ui, u in enumerate(rec["units"][li]):
        if k <= si < k + len(u["syl"]):
            return ui
        k += len(u["syl"])
    raise IndexError((rec["id"], li, si))


def _render(rec, units_by_line, blanks=frozenset()) -> str:
    out = []
    for g in rec["display_order"]:
        out.append("".join(BLANK if (g, ui) in blanks else u["s"]
                           for ui, u in enumerate(units_by_line[g])))
    return "\n".join(out)


def _ref(rec, units_by_line, blanks=frozenset()) -> list[list]:
    """Reference for grading, in DISPLAY order: per วรรค a list of
    [surface, syllables] (known unit) or None (blank)."""
    return [[None if (g, ui) in blanks else [u["s"], u["syl"]]
             for ui, u in enumerate(units_by_line[g])] for g in rec["display_order"]]


def _base(rec, mode, n):
    spec = M.METER_SPECS_BY_ID[rec["meter_id"]]
    return {"id": f"ce-{SLUG[rec['meter_id']]}-{mode}-{n:03d}", "task_mode": mode,
            "meter": spec.name_th, "meter_id": rec["meter_id"], "topic": rec["topic"],
            "source_id": rec["id"], "provenance": rec["provenance"], "in_main": rec["in_main"],
            "display_order": rec["display_order"],
            "source_text": _render(rec, rec["units"]),
            "source_lines": rec["lines"]}


def gen_cloze(pool) -> list[dict]:
    items, ctr = [], {}
    for rec in pool:
        spec = M.METER_SPECS_BY_ID[rec["meter_id"]]
        seen, cands = set(), []
        for li, si, label in hard_positions(spec):
            if not (0 <= li < len(rec["lines"])):
                continue
            idx = _resolve(si, len(rec["lines"][li]))
            if 0 <= idx < len(rec["lines"][li]) and (li, idx) not in seen:
                seen.add((li, idx))
                cands.append((li, idx, label))
        n_var = min(CLOZE_VARIANTS.get(rec["meter_id"], DEFAULT_VARIANTS), len(cands))
        for v in range(n_var):
            picks = [cands[i] for i in range(len(cands)) if i % n_var == v][:BLANKS]
            blanks, info = set(), []
            for li, idx, label in picks:
                ui = _unit_of(rec, li, idx)
                u = rec["units"][li][ui]
                info.append({"line": li, "syllable": idx, "constraint": label,
                             "unit": ui, "gold_surface": u["s"], "gold_syllables": u["syl"]})
                blanks.add((li, ui))
            slug = SLUG[rec["meter_id"]]
            ctr[slug] = ctr.get(slug, 0) + 1
            it = _base(rec, "cloze", ctr[slug])
            it.update({"poem_with_blanks": _render(rec, rec["units"], blanks),
                       "n_blanks": len(blanks), "blank_positions": info,
                       "reference": _ref(rec, rec["units"], blanks)})
            items.append(it)
    return items


def gen_edit(pool) -> list[dict]:
    items, ctr = [], {}
    for rec in pool:
        slots = _edit_slots(_P(rec))
        if not slots:
            continue
        made, cap = 0, EDIT_VARIANTS.get(rec["meter_id"], DEFAULT_VARIANTS)
        for li, si, kind, mark in slots:
            if made >= cap:
                break
            idx = _resolve(si, len(rec["lines"][li]))
            ui = _unit_of(rec, li, idx)
            u = rec["units"][li][ui]
            if u["syl"] != [u["s"]]:                 # respelled multi-syllable unit: skip slot
                continue
            if kind == "tone":
                m = AEK_MARK if mark is M.ToneMark.AEK else TOO_MARK
                if m not in u["s"]:
                    continue
                new = u["s"].replace(m, "")
            else:
                new = "มา"                            # heavy syllable in a light slot
            units = [[dict(x) for x in line] for line in rec["units"]]
            units[li][ui] = {"s": new, "syl": [new]}
            lines = [[s for x in line for s in x["syl"]] for line in units]
            # validity gate: the injection must break the meter (gold segmentation)
            if score_meter([tuple(v) for v in lines], rec["meter_id"])["score"] >= 0.9999:
                continue
            slug = SLUG[rec["meter_id"]]
            ctr[slug] = ctr.get(slug, 0) + 1
            made += 1
            it = _base(rec, "edit", ctr[slug])
            it.update({"input_poem": _render(rec, units),
                       "broken_rules": [f"{kind}_violation_line_{li}_syllable_{idx}"],
                       "fault": {"line": li, "syllable": idx, "kind": kind,
                                 "display_line": rec["display_order"].index(li),
                                 "original": u["s"], "injected": new},
                       "reference": _ref(rec, units)})
            items.append(it)
    return items


def main() -> None:
    pool = json.loads((DATA / "pool.json").read_text(encoding="utf-8"))
    cloze, edit = gen_cloze(pool), gen_edit(pool)
    for name, rows in (("items_cloze.jsonl", cloze), ("items_edit.jsonl", edit)):
        with (DATA / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    for mode, rows in (("cloze", cloze), ("edit", edit)):
        c = Counter((r["meter_id"], r["provenance"], r["in_main"]) for r in rows)
        print(mode, len(rows))
        for k, v in sorted(c.items()):
            print("   ", k, v)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
