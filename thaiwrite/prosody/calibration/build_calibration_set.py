"""Build the pilot calibration set (one merged calibration_items.json).

Two steps (run `all`, the default, or one at a time):

  pilot     sample prompts, run them through the two pilot models and cache
            raw outputs to data/pilot_<model>.jsonl (resumable: re-running skips
            already-completed items, so an interrupted run is not re-billed).
  assemble  read the pilot caches, the audit-corpus clean anchors, the grader-
            verified broken anchors and the curated rhyme pairs, and write the
            merged calibration_items.json (poems first, then rhyme pairs).

Pilot models: Typhoon 2.5 (Typhoon API, TYPHOON_API_KEY) and gpt-5.4-mini
(OpenAI, OPENAI_API_KEY). Two models widen the realistic quality spread the
experts rate. The purpose is varied, imperfect output, not a baseline.
The calibration set used in the paper is built by build_calibration_v2.py.

  PYTHONIOENCODING=utf-8 python build_calibration_set.py            # all
  PYTHONIOENCODING=utf-8 python build_calibration_set.py --step pilot
  PYTHONIOENCODING=utf-8 python build_calibration_set.py --step assemble

Then bundle_for_experts.py turns calibration_items.json into per-expert HTML.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent           # .../prosody/calibration
REPO = HERE.parents[2]                            # calibration -> prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))

from thaiwrite.prosody import meter_specs as M                       # noqa: E402
from thaiwrite.prosody.data.audit_poems import POEMS as AUDIT_POEMS  # noqa: E402
from thaiwrite.prosody.grader_meter import grade_text                # noqa: E402

PROMPT_DIR = HERE.parent / "data"                 # the generated prompt jsonl files
CACHE = HERE / "data"                             # pilot output cache
OUT = HERE / "calibration_items.json"

for parent in HERE.parents:
    if (parent / ".env").exists():
        load_dotenv(parent / ".env")
        break

SYSTEM = "คุณเป็นกวีผู้เชี่ยวชาญฉันทลักษณ์ไทย ตอบเป็นบทร้อยกรองเท่านั้น ไม่ต้องอธิบาย"


# ---------- meter → which dimensions it binds (for the ✓/✗ row) ----------

def dims_for(meter_id: str) -> list[str]:
    spec = M.METER_SPECS_BY_ID[meter_id]
    dims = ["count"]
    if spec.tone_constraints:
        dims.append("tone")
    if spec.rhyme_links or spec.external_rhyme_links:
        dims.append("rhyme")
    if spec.weight_patterns:
        dims.append("kru_lahu")
    return dims


# ---------- prompt rendering (item spec → Thai instruction) ----------

def render_prompt(item: dict) -> str:
    mode, meter, topic = item["task_mode"], item["meter"], item.get("topic", "")
    if mode == "compose":
        n = item.get("n_stanzas", 1)
        return (f'จงแต่ง{meter} จำนวน {n} บท ในหัวข้อ "{topic}" ให้ถูกต้องตามฉันทลักษณ์ '
                "เขียนแต่ละวรรคขึ้นบรรทัดใหม่")
    if mode == "cloze":
        return ("จงเติมคำที่หายไป (____) ในบทร้อยกรองต่อไปนี้ ให้ถูกต้องตามฉันทลักษณ์และความหมาย "
                "แล้วเขียนบทที่สมบูรณ์ แต่ละวรรคขึ้นบรรทัดใหม่:\n\n" + item["poem_with_blanks"])
    if mode == "edit":
        return ("บทร้อยกรองต่อไปนี้มีข้อผิดพลาดด้านฉันทลักษณ์ จงแก้ไขให้ถูกต้อง "
                "แล้วเขียนบทที่แก้แล้วใหม่ทั้งบท แต่ละวรรคขึ้นบรรทัดใหม่:\n\n" + item["input_poem"])
    raise ValueError(f"unknown task_mode {mode!r}")


# ---------- prompt loading + stratified sampling ----------

def load_prompts() -> list[dict]:
    rows = []
    for name in ("prompts_compose_public.jsonl", "prompts_compose_private.jsonl",
                 "prompts_cloze_public.jsonl", "prompts_edit_public.jsonl"):
        p = PROMPT_DIR / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def sample_prompts(rows: list[dict], n_compose_per_meter: int, seed: int) -> list[dict]:
    random.seed(seed)
    by_meter: dict[str, list[dict]] = {}
    for r in rows:
        if r["task_mode"] == "compose":
            by_meter.setdefault(r["meter_id"], []).append(r)
    picked = []
    for mid, lst in by_meter.items():
        random.shuffle(lst)
        picked += lst[:n_compose_per_meter]
    picked += [r for r in rows if r["task_mode"] == "cloze"]
    picked += [r for r in rows if r["task_mode"] == "edit"]
    return picked


# ---------- pilot runners (cache + resume) ----------

def _append(log: Path, rec: dict) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _done_ids(log: Path) -> set[str]:
    from pilot_rerun.clients import load_completed
    return load_completed(log) if log.exists() else set()


def run_typhoon(prompts: list[dict], log: Path) -> None:
    from pilot_rerun.clients import OpenAICompatClient
    key = os.environ.get("TYPHOON_API_KEY") or os.environ.get("THAILLM_API_KEY", "")
    if not key:
        print("  [skip] Typhoon — no TYPHOON_API_KEY / THAILLM_API_KEY"); return
    client = OpenAICompatClient(api_key=key, base_url="https://api.opentyphoon.ai/v1",
                                model="typhoon-v2.5-30b-a3b-instruct", rpm=200, log_path=log)
    done = _done_ids(log)
    todo = [p for p in prompts if p["id"] not in done]
    print(f"  Typhoon 2.5: {len(todo)} to run ({len(done)} cached)")
    for i, item in enumerate(todo, 1):
        call = client.chat("prosody-pilot", item["id"], render_prompt(item),
                           system=SYSTEM, max_tokens=700)
        flag = "ERR" if call.error else "ok"
        print(f"    [{i}/{len(todo)}] {item['id']:<34} {flag}")


def run_gpt54mini(prompts: list[dict], log: Path, reasoning: str | None) -> None:
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY2")
    if not key:
        print("  [skip] gpt-5.4-mini — no OPENAI_API_KEY"); return
    client, model = OpenAI(api_key=key), "gpt-5.4-mini"
    done = _done_ids(log)
    todo = [p for p in prompts if p["id"] not in done]
    print(f"  gpt-5.4-mini: {len(todo)} to run ({len(done)} cached)")
    for i, item in enumerate(todo, 1):
        prompt, t0, resp, err = render_prompt(item), time.time(), "", None
        use_chat = False
        for attempt in range(4):
            try:
                if use_chat:
                    r = client.chat.completions.create(
                        model=model, max_completion_tokens=900,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": prompt}])
                    resp = r.choices[0].message.content or ""
                else:
                    kw = dict(model=model, instructions=SYSTEM, input=prompt, max_output_tokens=900)
                    if reasoning:
                        kw["reasoning"] = {"effort": reasoning}
                    resp = client.responses.create(**kw).output_text
                err = None
                break
            except TypeError as e:        # SDK shape mismatch for this model → use chat
                err, use_chat = f"{type(e).__name__}: {e}", True
            except Exception as e:        # noqa: BLE001, transient API error
                err = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt + 1, 30))
        _append(log, {"benchmark": "prosody-pilot", "model": model, "item_id": item["id"],
                      "prompt": prompt, "response": resp, "latency_s": round(time.time() - t0, 2),
                      "input_tokens": None, "output_tokens": None, "error": err})
        print(f"    [{i}/{len(todo)}] {item['id']:<34} {'ERR' if err else 'ok'}")


# ---------- assemble ----------

def _clean(text: str) -> str:
    """Light cleanup of a raw model poem: strip fences/whitespace, keep วรรค newlines."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip() != "```"]
    return "\n".join(lines).strip()


def collect_pilot(prompt_index: dict[str, dict]) -> list[dict]:
    items = []
    for log in sorted(CACHE.glob("pilot_*.jsonl")):
        tag = log.stem.replace("pilot_", "")
        seen: set[str] = set()
        for line in log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("error") or not rec.get("response"):
                continue
            sid = str(rec["item_id"])
            if sid in seen:
                continue
            seen.add(sid)
            src = prompt_index.get(sid)
            if not src:
                continue
            poem = _clean(rec["response"])
            if not poem:
                continue
            items.append({
                "id": f"cal-{tag}-{sid}", "kind": "poem", "task_mode": src["task_mode"],
                "meter": src["meter"], "meter_id": src["meter_id"], "topic": src.get("topic", ""),
                "applicable_dims": dims_for(src["meter_id"]), "poem_text": poem,
                "_provenance": {"model": rec.get("model"), "src_id": sid, "kind": "pilot"},
            })
    return items


# topic labels for the famous-excerpt clean anchors (so topicality is meaningful)
ANCHOR_TOPIC = {
    "ลิลิตพระลอ": "ความงามและความรัก", "นิราศภูเขาทอง": "การเดินทางและความคิดถึง",
    "พระอภัยมณี": "การพลัดพรากและการผจญภัย", "กาพย์พระไชยสุริยา": "ภัยธรรมชาติและบทเรียนชีวิต",
    "รำลึกพระคุณแม่": "พระคุณแม่",
}


def _poem_text(poem) -> str:
    return "\n".join("".join(s) for s in poem.lines)


def _topic_for(source: str) -> str:
    for key, th in ANCHOR_TOPIC.items():
        if key in source:
            return th
    # fallback: the incipit after the em-dash separator in "<work> <em-dash> <incipit>".
    # Defensively drop a leading "<label>:" prefix (a model preamble can leak into source).
    tail = source.split("—")[-1].strip()
    if ":" in tail:
        tail = tail.split(":", 1)[1].strip()
    return tail[:40] or "บทร้อยกรอง"


def clean_anchors() -> list[dict]:
    out = []
    for i, poem in enumerate(AUDIT_POEMS):
        spec = M.METER_SPECS_BY_ID[poem.meter_id]
        out.append({
            "id": f"cal-clean-{i:02d}", "kind": "poem", "task_mode": "reference",
            "meter": spec.name_th, "meter_id": poem.meter_id, "topic": _topic_for(poem.source),
            "applicable_dims": dims_for(poem.meter_id), "poem_text": _poem_text(poem),
            "_provenance": {"source": poem.source, "kind": "clean_anchor"},
        })
    return out


def broken_anchors() -> list[dict]:
    """Inject a verified meter break into each audit poem (grader confirms score < 1.0)."""
    AEK, TOO = "่", "้"
    out = []
    for i, poem in enumerate(AUDIT_POEMS):
        spec = M.METER_SPECS_BY_ID[poem.meter_id]
        base = [list(v) for v in poem.lines]
        variants: list[tuple[str, list[list[str]]]] = []

        b_count = [v[:] for v in base]                       # drop one syllable → count break
        for v in b_count:
            if len(v) > 1:
                v.pop(); break
        variants.append(("count", b_count))

        if spec.tone_constraints:                            # โคลง: strip a required tone mark
            b = [v[:] for v in base]
            tc = spec.tone_constraints[0]
            li, si = tc.position.line_index, tc.position.syllable_index
            idx = si if si >= 0 else len(b[li]) + si
            if 0 <= li < len(b) and 0 <= idx < len(b[li]):
                b[li][idx] = b[li][idx].replace(AEK if tc.mark is M.ToneMark.AEK else TOO, "")
                variants.append(("tone", b))

        if spec.weight_patterns:                             # ฉันท์: flip a ลหุ→ครุ
            b = [v[:] for v in base]
            wp = spec.weight_patterns[0]
            j = next((k for k, w in enumerate(wp.pattern) if w is M.Weight.LAHU), 0)
            if 0 <= wp.line_index < len(b) and j < len(b[wp.line_index]):
                b[wp.line_index][j] = "มา"
                variants.append(("kru_lahu", b))

        for kind, lines in variants:
            text = "\n".join("".join(s) for s in lines)
            score = grade_text(text, poem.meter_id, eval_rhyme=poem.eval_rhyme)["score"]
            if score < 1.0:                                  # only keep a verified break
                out.append({
                    "id": f"cal-broken-{i:02d}-{kind}", "kind": "poem", "task_mode": "reference",
                    "meter": spec.name_th, "meter_id": poem.meter_id, "topic": _topic_for(poem.source),
                    "applicable_dims": dims_for(poem.meter_id), "poem_text": text,
                    "_provenance": {"source": poem.source, "kind": "broken_anchor",
                                    "break": kind, "grader_score": round(score, 3)},
                })
    return out


# Curated rhyme pairs, labels withheld (experts decide). `_cat` is grading-side
# only (analysis of where the fallback fails); the annotation tool never shows it.
#   en/ev = easy valid/invalid · an = อักษรนำ · ka = การันต์ · sv = สระ-variant
CURATED_RHYME = [
    # easy valid (สัมผัสสระชัดเจน)
    ("ใจ", "ไป", "ev"), ("มา", "ปลา", "ev"), ("รัก", "ภักดิ์", "ev"), ("ดี", "ปี", "ev"),
    ("นอน", "กร", "ev"), ("ทอง", "ของ", "ev"), ("เย็น", "เห็น", "ev"), ("ลม", "ชม", "ev"),
    ("คิด", "ปิด", "ev"), ("เรา", "เขา", "ev"),
    # easy invalid (คนละสระ/ตัวสะกด)
    ("ใจ", "ดี", "ei"), ("มา", "นอน", "ei"), ("ทอง", "เย็น", "ei"), ("ลม", "ปี", "ei"),
    ("คิด", "เรา", "ei"), ("รัก", "ลม", "ei"), ("ไป", "ของ", "ei"), ("เห็น", "ปลา", "ei"),
    ("ชม", "ปิด", "ei"), ("เขา", "กร", "ei"),
    # อักษรนำ: silent ห / อ leads; rime is the second consonant's
    ("หนา", "นา", "an"), ("หมด", "ลด", "an"), ("หลง", "ยง", "an"), ("อย่า", "ยา", "an"),
    ("หนาว", "ดาว", "an"), ("หมาย", "าย", "an"), ("ใหล", "ใด", "an"), ("หนู", "ดู", "an"),
    # การันต์: silent final consonant; rime ignores it
    ("ศักดิ์", "รัก", "ka"), ("สิทธิ์", "ผิด", "ka"), ("โทรศัพท์", "ทับ", "ka"),
    ("จันทร์", "ฉัน", "ka"), ("สงฆ์", "สง", "ka"), ("เล่ห์", "เซ", "ka"),
    ("พันธุ์", "พัน", "ka"), ("เมล์", "เม", "ka"),
    # สระ-variant: same sound, different spelling/short-long traps
    ("น้ำ", "ทำ", "sv"), ("ใคร", "ไกล", "sv"), ("เกียรติ", "เบียด", "sv"),
    ("เธอ", "เจอ", "sv"), ("มือ", "ถือ", "sv"), ("เลือด", "เชือด", "sv"),
    ("เพลง", "เคร่ง", "sv"), ("กลัว", "ตัว", "sv"),
]


def rhyme_pairs() -> list[dict]:
    out = []
    for i, (a, b, cat) in enumerate(CURATED_RHYME):
        out.append({"id": f"cal-rhyme-{i:03d}", "kind": "rhyme_pair",
                    "word_a": a, "word_b": b, "_cat": cat})
    return out


def assemble(shuffle_seed: int) -> None:
    index = {str(p["id"]): p for p in load_prompts()}
    poems = collect_pilot(index) + clean_anchors() + broken_anchors()
    random.Random(shuffle_seed).shuffle(poems)
    pairs = rhyme_pairs()
    items = poems + pairs
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    by_kind: dict[str, int] = {}
    for p in poems:
        by_kind[p["_provenance"]["kind"]] = by_kind.get(p["_provenance"]["kind"], 0) + 1
    print(f"\n  poems:       {len(poems):3d}  ({by_kind})")
    print(f"  rhyme pairs: {len(pairs):3d}")
    print(f"  TOTAL items: {len(items):3d}  → {OUT.relative_to(HERE)}")


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=["pilot", "assemble", "all"], default="all")
    ap.add_argument("--compose-per-meter", type=int, default=5,
                    help="compose prompts sampled per meter for the pilot (default 5 → 20)")
    ap.add_argument("--reasoning", default=None,
                    help="gpt-5.4-mini reasoning effort (minimal/low/medium); default omitted")
    ap.add_argument("--seed", type=int, default=20260524)
    args = ap.parse_args()

    if args.step in ("pilot", "all"):
        prompts = sample_prompts(load_prompts(), args.compose_per_meter, args.seed)
        print(f"Pilot over {len(prompts)} prompts × 2 models:")
        run_typhoon(prompts, CACHE / "pilot_typhoon.jsonl")
        run_gpt54mini(prompts, CACHE / "pilot_gpt54mini.jsonl", args.reasoning)

    if args.step in ("assemble", "all"):
        print("\nAssembling calibration_items.json:")
        assemble(args.seed)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
