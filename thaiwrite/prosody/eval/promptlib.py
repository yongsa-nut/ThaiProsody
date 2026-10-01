"""Shared prompt I/O for the evaluation harness (no calibration dependency).

Holds the symbols the eval pipeline shares with the calibration pipeline: the
system role, the item-to-Thai-instruction renderer, the prompt loader, and the
raw-output cleaner. `clean_poem` additionally strips Qwen3-style
`<think>...</think>` reasoning blocks on top of the fence handling used in
calibration's `_clean`. `to_vrk_lines` normalizes a poem to one วรรค per line
for the positional meter grader.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../prosody/eval
PROMPT_DIR = HERE.parent / "data"                 # generated prompt jsonl files

SYSTEM = "คุณเป็นกวีผู้เชี่ยวชาญฉันทลักษณ์ไทย ตอบเป็นบทร้อยกรองเท่านั้น ไม่ต้องอธิบาย"

#: prompt-set name -> jsonl filename in PROMPT_DIR
SET_FILES = {
    "compose_public": "prompts_compose_public.jsonl",
    "compose_private": "prompts_compose_private.jsonl",
    "cloze": "prompts_cloze_public.jsonl",
    "edit": "prompts_edit_public.jsonl",
}


def render_prompt(item: dict) -> str:
    """Item spec to Thai instruction (verbatim parity with build_calibration_set)."""
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


def load_prompts(sets: list[str]) -> list[dict]:
    """Load the named prompt sets (in order). Missing files are skipped silently."""
    rows: list[dict] = []
    for s in sets:
        fname = SET_FILES.get(s)
        if fname is None:
            raise ValueError(f"unknown set {s!r}; choices: {sorted(SET_FILES)}")
        p = PROMPT_DIR / fname
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


#: markdown emphasis / code / strike markers (stripped inline, not Thai script)
_EMPH = re.compile(r"\*\*+|__+|`+|~~+")
_THAI = re.compile(r"[ก-๙]")

#: a line that is ONLY a structural label/marker, never a วรรค. Anchored so it
#: never matches verse that merely *contains* "บทที่" mid-line. Covers md headers,
#: horizontal rules, stanza labels (บทที่ ๑ / **บทที่ 1** after emphasis-strip),
#: and bare ordinals (1. ๑) (3)).
_LABEL = re.compile(
    r"^(?:"
    r"#{1,6}.*"                                                        # md header
    r"|[-_*=~]{3,}"                                                     # horizontal rule
    r"|\(?\s*(?:บทที่|บท|วรรคที่|วรรค|บาทที่|บาท|ตอนที่|ตอน|ท่อนที่|ท่อน|stanza|verse|part)"
    r"\s*[๐-๙\d]*\s*\)?\s*[:：.)]?"                                      # label (+optional number)
    r"|\(?\s*[๐-๙\d]+\s*\)?\s*[:：.)]"                                   # bare ordinal w/ trailing punct
    r")\s*$",
    re.IGNORECASE,
)


def _is_noise(stripped: str) -> bool:
    """True if a (emphasis-stripped) line is structural noise, not Thai verse.

    Blank lines are kept (display spacing; the grader ignores them). Dropped:
    label/header/rule lines, prose lead-ins ending in a colon (คำตอบ: / ขอแก้ใหม่:),
    and lines with no Thai at all (English asides like "Let me redo this", separators).
    """
    s = stripped.strip()
    if not s:
        return False
    if _LABEL.match(s):
        return True
    if s.endswith(":") or s.endswith("："):     # prose lead-in, never a วรรค
        return True
    if not _THAI.search(s):                     # English aside / separator / bare number
        return True
    return False


def clean_poem(text: str) -> str:
    """Raw model output to poem text: drop reasoning, fences, and non-verse noise.

    Qwen3 "think" models (Pathumma, OpenThaiGPT v7.2) emit `<think>…</think>` then
    the answer; keep only the text after the last `</think>`. An unclosed `<think>`
    means the answer was never produced (truncated reasoning), so the result is empty.

    Chatty models decorate output with markdown stanza headers (`**บทที่ 1**`),
    English asides, and `---` separators. Parsed naively these become bogus วรรค and
    shift every line off its spec slot (e.g. มา↔สงบ paired across the wrong บาท), so
    we strip them here: emphasis markers inline, label/header/rule/aside lines out.
    """
    if not text:
        return ""
    low = text.lower()
    if "</think>" in low:
        text = text[low.rindex("</think>") + len("</think>"):]
    elif "<think>" in low:
        text = text[:low.index("<think>")]   # truncated reasoning, no answer
    out: list[str] = []
    for ln in text.splitlines():
        if ln.strip().startswith("```"):
            continue
        stripped = _EMPH.sub("", ln)
        if _is_noise(stripped):
            continue
        out.append(stripped)
    return "\n".join(out).strip()


def to_vrk_lines(poem: str) -> str:
    """Normalize a cleaned poem to **one วรรค per line** for the grader.

    The meter grader maps poem lines positionally onto `spec.lines`, so it requires
    one วรรค per line. Thai verse delimits วรรค with whitespace (newline, space, or
    tab) and uses no intra-วรรค spaces, so every whitespace run inside a line is a
    วรรค boundary. Models follow the "เขียนแต่ละวรรคขึ้นบรรทัดใหม่" instruction
    unevenly (some use the traditional 2-วรรค-per-line layout with a tab); splitting
    each line on whitespace runs realigns their output to the grader's contract. A
    model that already writes one วรรค per line is unchanged (no intra-line space).
    """
    out: list[str] = []
    for ln in poem.splitlines():
        for chunk in re.split(r"\s+", ln.strip()):
            if chunk:
                out.append(chunk)
    return "\n".join(out)
