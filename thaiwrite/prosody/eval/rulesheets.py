"""Rules-in-prompt ablation: render each meter's formal spec into a Thai rule
sheet appended to the system prompt.

The sheet is generated from `meter_specs.MeterSpec` itself (the same object the
grader iterates), so the prompt states exactly the rules that are graded (mandatory
line counts, tone-mark positions, ครุ/ลหุ patterns, internal rhyme links) and nothing
else. The zero-shot baseline tests whether a model knows the form; this condition
tests whether it can follow the handed-over specification.

  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.rulesheets   # print all 4
"""
from __future__ import annotations

import sys

from thaiwrite.prosody import meter_specs as M

_TONE_TH = {"aek": "ไม้เอก (่)", "too": "ไม้โท (้)"}
_WEIGHT_TH = {"karu": "ครุ", "lahu": "ลหุ"}


def _pos_th(spec: M.MeterSpec, line_index: int, syll_index: int) -> str:
    """Position to Thai phrase at the วรรค grain (matching the one-วรรค-per-line
    layout the prompt requests)."""
    if syll_index == -1:
        return f"คำสุดท้ายของวรรคที่ {line_index + 1}"
    if syll_index < -1:
        return f"คำที่ {syll_index} จากท้ายวรรคที่ {line_index + 1}"
    return f"คำที่ {syll_index + 1} ของวรรคที่ {line_index + 1}"


def _range_th(spec: M.MeterSpec, r: M.PositionRange) -> str:
    if r.is_single:
        return _pos_th(spec, r.line_index, r.start)
    return (f"คำที่ {r.start + 1} ถึงคำที่ {r.end + 1} ของวรรคที่ {r.line_index + 1}")


def rule_sheet(meter_id: str) -> str:
    spec = M.METER_SPECS_BY_ID[meter_id]
    out: list[str] = [f"กติกาฉันทลักษณ์ของ{spec.name_th} (ต่อ 1 บท) มีดังนี้:"]

    # 1. line structure (mandatory วรรค in order, with the spec's own role notes)
    mand = [ln for ln in spec.lines if not ln.optional]
    out.append(f"1) โครงสร้าง: 1 บท มี {len(mand)} วรรค เขียนหนึ่งวรรคต่อหนึ่งบรรทัด ตามลำดับ:")
    for i, ln in enumerate(spec.lines):
        if ln.optional:
            continue
        cnt = (f"{ln.min_count}-{ln.max_count} พยางค์" if ln.min_count != ln.max_count
               else f"{ln.canonical_count} พยางค์")
        label = f" ({ln.note})" if ln.note else ""
        out.append(f"   - วรรคที่ {i + 1}{label}: {cnt}")
    soi = [ln for ln in spec.lines if ln.optional]
    if soi:
        bats = " และ ".join(f"บาทที่ {ln.bat_index + 1}" for ln in soi)
        out.append(f"   - อาจเติมคำสร้อย {soi[0].canonical_count} พยางค์ ท้าย{bats} ได้ (ไม่บังคับ)")

    n = 2
    # 2. tone constraints
    if spec.tone_constraints:
        out.append(f"{n}) บังคับรูปวรรณยุกต์:")
        for mark in ("aek", "too"):
            tcs = [t for t in spec.tone_constraints if t.mark.value == mark]
            if not tcs:
                continue
            spots = ", ".join(_pos_th(spec, t.position.line_index, t.position.syllable_index)
                              for t in tcs)
            out.append(f"   - ต้องมีรูป{_TONE_TH[mark]} ที่ {spots}")
        if meter_id == "khlong_si_suphap":
            out.append("   - ตำแหน่งบังคับเอก ใช้คำตาย (สะกดแม่ กก กด กบ) แทนได้")
        n += 1

    # 3. weight patterns
    if spec.weight_patterns:
        out.append(f"{n}) บังคับครุ-ลหุ (ทุกพยางค์ตามลำดับ):")
        for wp in spec.weight_patterns:
            pat = "-".join(_WEIGHT_TH[w.value] for w in wp.pattern)
            out.append(f"   - วรรคที่ {wp.line_index + 1}: {pat}")
        n += 1

    # 4. internal rhyme links (the graded ones)
    if spec.rhyme_links:
        out.append(f"{n}) บังคับสัมผัส (สัมผัสสระภายในบท):")
        for rl in spec.rhyme_links:
            src = _pos_th(spec, rl.source.line_index, rl.source.syllable_index)
            tgt = " หรือ ".join(_range_th(spec, r) for r in rl.targets)
            out.append(f"   - {src} ต้องสัมผัสกับ {tgt}")

    out.append("จงยึดกติกาข้างต้นอย่างเคร่งครัดทุกข้อ")
    return "\n".join(out)


RULE_SHEETS: dict[str, str] = {mid: rule_sheet(mid) for mid in M.METER_SPECS_BY_ID}


def _selftest() -> None:
    """Every graded rule must be reflected in its sheet (counts, not prose)."""
    for mid, spec in M.METER_SPECS_BY_ID.items():
        sheet = RULE_SHEETS[mid]
        assert sheet.count("วรรคที่") >= len(spec.mandatory_lines), mid
        assert ("ไม้เอก" in sheet) == any(t.mark.value == "aek" for t in spec.tone_constraints), mid
        assert ("ครุ" in sheet) == bool(spec.weight_patterns), mid
        assert sheet.count("ต้องสัมผัสกับ") == len(spec.rhyme_links), mid
    print("rulesheets selftest: OK")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    _selftest()
    for mid, sheet in RULE_SHEETS.items():
        print(f"\n{'=' * 60}\n[{mid}]\n{sheet}")
