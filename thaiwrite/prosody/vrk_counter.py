"""วรรค-level syllable counting prompt and development harness (Anthropic API).

Thai syllable counting is the hard part of meter grading: PyThaiNLP's dict tokenizer
is only ~84% accurate on classical verse (systematic under-count of อักษรนำ /
Sanskrit). The count-of-record used in the paper is produced by `count_api.py`
(Claude Opus 5.5), cached in vrk_count_cache_opus55.json and
vrk_count_cache_opus55_bcd.json; it reuses this module's `PROMPT_HEAD` and
`EXTRA_GOLD`.

This module also holds the harness used to design and validate that prompt
(`VrkCounter`, default model Claude Opus 4.7; commands smoke / validate / compare /
thinktest / run). Its own caches (vrk_count_cache.json, vrk_count_cache_word.json)
are not part of the release, so these commands make API calls
(ANTHROPIC_API_KEY required).

Design:
  * Count one วรรค (a short verse segment) at a time — syllable count is a pure function
    of the วรรค text, so it is cacheable and context-independent. Batches are
    only an API-call optimization; each วรรค's count is keyed by its own text.
  * temperature=0 (falls back to omitting temperature if the model rejects it; the cache
    makes results reproducible regardless).
  * Self-check: the integer count must equal the dot-separated breakdown length; on a
    mismatch the breakdown length wins (more reliable) and the item is flagged.
  * Cache: วรรค text -> {count, syllables}; a re-run with a filled cache makes zero
    API calls.

  PYTHONIOENCODING=utf-8 python vrk_counter.py smoke        # validate vs hand-gold + determinism
  PYTHONIOENCODING=utf-8 python vrk_counter.py run          # count all grading-relevant compose วรรค
  PYTHONIOENCODING=utf-8 python vrk_counter.py run --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                     # prosody -> thaiwrite -> repo root
sys.path.insert(0, str(REPO))

CACHE_PATH = HERE / "vrk_count_cache.json"
CACHE_WORD_PATH = HERE / "vrk_count_cache_word.json"
MODEL = "claude-opus-4-7"
BATCH = 20

CONVENTIONS = """Conventions (Thai prosodic scansion):
1. Count by PRONUNCIATION, not written characters.
2. การันต์ (◌์) silenced final adds no syllable: จันทร์→1, ศักดิ์→1; โทรศัพท์→3.
3. อักษรนำ / อักษรควบไม่แท้ with an inserted half-vowel ARE separate syllables: ขนม→2, สบาย→2, สว่าง→2, ขจัด→2.
4. True consonant cluster (อักษรควบแท้) = ONE syllable: ปลา→1, ครวญ→1, พราว→1.
5. Silent leading ห/อ before a low consonant = ONE syllable: หนา→1, อยู่→1.
6. Sanskrit/Pali by Thai pronunciation: มนุษย์→2, วิทยา→3, ธรรมชาติ→3, นภา→2.
7. Count ALL syllables, including across internal spaces."""

PROMPT_HEAD = ("You are an expert in Thai prosody (ฉันทลักษณ์) and phonology. For each "
               "numbered วรรค (verse segment) below, count the PRONOUNCED syllables (พยางค์) "
               "actually present as written.\n\n" + CONVENTIONS + "\n\n"
               "For each วรรค give the dot-separated syllable breakdown and the integer count "
               "(the count MUST equal the number of dot-separated syllables).\n"
               "Output ONLY a JSON array, one object per วรรค, in order:\n"
               '[{"n":1,"syllables":"a.b.c","count":3}, ...]\n\nวรรค:\n')

# word-decomposition variant: segment into WORDS first, count each word's syllables, sum.
# Forces atomic per-word counting where อักษรนำ/Sanskrit words (นภา=2, ขนม=2) are reliable.
PROMPT_HEAD_WORD = ("You are an expert in Thai prosody (ฉันทลักษณ์) and phonology. For each "
                    "numbered วรรค below: (1) segment it into WORDS, (2) give EACH word's "
                    "syllable count by pronunciation, (3) the total = the sum.\n\n" + CONVENTIONS +
                    "\n\nPay special attention: Sanskrit/Pali and อักษรนำ words usually have MORE "
                    "syllables than they look — นภา=2, นที=2, กรุณา=3, มนุษย์=2, ขนม=2, สว่าง=2. "
                    "Count EACH such word on its own.\n"
                    "Output ONLY a JSON array, one object per วรรค, in order:\n"
                    '[{"n":1,"words":[{"w":"นภา","s":2},{"w":"ลา","s":1}],"count":3}, ...] '
                    "where count = sum of the words' s.\n\nวรรค:\n")


def syllen(s: str) -> int:
    return len([t for t in (s or "").split(".") if t.strip()])


class VrkCounter:
    def __init__(self, model: str = MODEL, temperature: float = 0.0, persist: bool = True,
                 mode: str = "flat", think: bool = False):
        self._load_env()
        from anthropic import Anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=key)
        self.model = model
        self.temperature = temperature
        self.persist = persist          # write cache to disk (off for the determinism check)
        self.mode = mode                # "flat" (syllable breakdown) | "word" (per-word counts)
        self.think = think              # extended thinking (deliberate pass; for arbitration)
        self.cache_path = CACHE_PATH if mode == "flat" else CACHE_WORD_PATH
        self._temp_ok = not think       # thinking forbids temperature
        self._lock = threading.Lock()
        self.cache: dict[str, dict] = (
            json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {})

    @staticmethod
    def _load_env() -> None:
        p = Path.cwd()
        for d in [p, *p.parents]:
            if (d / ".env").exists():
                for line in (d / ".env").read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")
                return

    def _call(self, prompt: str) -> str:
        kw = dict(model=self.model, max_tokens=24000 if self.think else 4096,
                  messages=[{"role": "user", "content": prompt}])
        if self.think:
            # Opus 4.7 thinking API: adaptive type + output_config effort (not enabled/budget)
            kw["thinking"] = {"type": "adaptive"}
            kw["extra_body"] = {"output_config": {"effort": "high"}}
        if self._temp_ok:
            kw["temperature"] = self.temperature
        for attempt in range(4):
            try:
                if self.think:   # streaming required for long (high max_tokens) thinking calls
                    with self.client.messages.stream(**kw) as s:
                        m = s.get_final_message()
                    return "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
                r = self.client.messages.create(**kw)
                return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "temperature" in msg.lower() and "temperature" in kw:
                    if self._temp_ok:
                        self._temp_ok = False
                        print("  [note] model rejected temperature; omitting (cache still reproducible)")
                    kw.pop("temperature", None)
                    continue
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt + 1, 20))
        return ""

    @staticmethod
    def _parse(raw: str) -> list[dict]:
        raw = raw.strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else json.loads(raw)

    def _grade_batch(self, chunk: list[str]) -> dict[str, dict]:
        head = PROMPT_HEAD if self.mode == "flat" else PROMPT_HEAD_WORD
        arr = self._parse(self._call(head + "\n".join(f"{j+1}. {v}" for j, v in enumerate(chunk))))
        by_n = {int(o.get("n", k + 1)): o for k, o in enumerate(arr)}
        out = {}
        for k, v in enumerate(chunk):
            o = by_n.get(k + 1, arr[k] if k < len(arr) else {})
            stated = o.get("count")
            if self.mode == "flat":
                syl = o.get("syllables", "")
                derived = syllen(syl)
            else:
                words = o.get("words", []) or []
                derived = sum(int(w.get("s", 0) or 0) for w in words)
                syl = ".".join(f"{w.get('w','')}({w.get('s','')})" for w in words)
            use = derived if (derived and stated != derived) else (stated if stated is not None else derived)
            out[v] = {"count": use, "syllables": syl,
                      "flag": "selfcheck" if (derived and stated != derived) else ""}
        return out

    def _save(self) -> None:
        if self.persist:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0),
                                       encoding="utf-8")

    def count(self, vrks: list[str], verbose: bool = True, workers: int = 6,
              batch: int = BATCH) -> dict[str, dict]:
        """Return {วรรค: {count, syllables, flag}} for each input, using/filling the cache."""
        uniq, seen = [], set()
        for v in vrks:
            v = v.strip()
            if v and v not in seen and v not in self.cache:
                seen.add(v); uniq.append(v)
        all_uniq = len({v.strip() for v in vrks if v.strip()})
        if verbose:
            print(f"unique requested: {all_uniq} · cached: {all_uniq - len(uniq)} · to call: {len(uniq)}")
        batches = [uniq[i:i + batch] for i in range(0, len(uniq), batch)]
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self._grade_batch, b) for b in batches]
            for fut in as_completed(futs):
                res = fut.result()
                with self._lock:
                    self.cache.update(res)
                    done += len(res)
                    self._save()
                    if verbose and done % (BATCH * 5) < BATCH:
                        print(f"  …{done}/{len(uniq)}")
        return {v.strip(): self.cache[v.strip()] for v in vrks if v.strip() in self.cache}


# ---------- collect grading-relevant วรรค ----------

def grading_vrks(modes={"compose"}) -> list[str]:
    """The วรรค the grader actually checks: the first len(spec.lines) of to_vrk_lines(poem)."""
    import glob
    from thaiwrite.prosody.eval.promptlib import to_vrk_lines
    from thaiwrite.prosody import meter_specs as M
    out, seen = [], set()
    for f in sorted(glob.glob(str(HERE / "eval" / "graded" / "*__run1.jsonl"))):
        if ".nothink" in f:
            continue
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task_mode") not in modes or r.get("eval_status") != "graded":
                continue
            poem = (r.get("poem") or "").strip()
            mid = r.get("meter_id")
            if not poem or mid not in M.METER_SPECS_BY_ID:
                continue
            nslots = len(M.METER_SPECS_BY_ID[mid].lines)
            vrks = [v for v in to_vrk_lines(poem).splitlines() if v.strip()][:nslots]
            for v in vrks:
                v = v.strip()
                if v not in seen:
                    seen.add(v); out.append(v)
    return out


# ---------- hard-case validation ----------

# Sanskrit/poetic words (incl. the นภา family that failed) — hand-counted.
EXTRA_GOLD = {
    "Sanskrit poetic (hard)": [
        ("นภา", 2), ("นที", 2), ("ชลธี", 3), ("อัมพร", 2), ("ทิฆัมพร", 3),
        ("คีรี", 2), ("รัศมี", 3), ("โสภา", 2), ("มาลี", 2), ("สุริยา", 3),
        ("นภาลัย", 3), ("ปักษา", 2), ("วนา", 2), ("สาคร", 2), ("ธารา", 2),
    ],
}


def validate() -> None:
    """Measure Opus accuracy on HARD cases: isolated hard words AND hard words in line
    context (the นภา-in-a-line failure mode). Uses a throwaway (non-persisted) cache."""
    from audit_count import WORD_GOLD  # type: ignore
    cats = {**WORD_GOLD, **EXTRA_GOLD}
    flat = [(w, g, cat) for cat, ws in cats.items() for (w, g) in ws]

    c = VrkCounter(persist=False)
    print(f"(1) ISOLATED words — {len(flat)} hand-counted\n")
    res = c.count([w for w, _, _ in flat], verbose=False, workers=6)
    by_cat = {}
    miss = []
    for w, g, cat in flat:
        o = res.get(w, {}).get("count")
        by_cat.setdefault(cat, []).append(1 if o == g else 0)
        if o != g:
            miss.append((cat, w, g, o, res.get(w, {}).get("syllables", "")))
    for cat, hits in by_cat.items():
        print(f"  {cat:28s} {sum(hits)}/{len(hits)} = {sum(hits)/len(hits)*100:5.0f}%")
    if miss:
        print("\n  isolated misses:")
        for cat, w, g, o, syl in miss:
            print(f"    {w:<12} gold {g} got {o}  [{syl}]  ({cat})")

    # (2) hard words IN LINE CONTEXT — build lines from a hard word + monosyllables.
    MONO = ["ไป", "มา", "ใจ", "ดาว", "ลม", "ฟ้า", "ยาม", "ตา", "ดี", "นาน"]
    hard = [(w, g) for cat, ws in cats.items() for (w, g) in ws
            if cat in ("อักษรนำ (2 syl)", "Pali/Sanskrit polysyllable", "Sanskrit poetic (hard)")]
    import random as _r
    rng = _r.Random(7)
    lines = []
    for i, (w, g) in enumerate(hard):
        ms = [rng.choice(MONO) for _ in range(rng.randint(2, 3))]
        text = w + "".join(ms)
        lines.append((text, g + len(ms)))
    print(f"\n(2) HARD WORD IN LINE CONTEXT — {len(lines)} constructed lines (gold = sum)\n")
    res2 = c.count([t for t, _ in lines], verbose=False, workers=6)
    ok = 0
    lmiss = []
    for text, g in lines:
        o = res2.get(text, {}).get("count")
        ok += (o == g)
        if o != g:
            lmiss.append((text, g, o, res2.get(text, {}).get("syllables", "")))
    print(f"  line accuracy: {ok}/{len(lines)} = {ok/len(lines)*100:.0f}%")
    if lmiss:
        print("  line misses:")
        for text, g, o, syl in lmiss:
            print(f"    {text:<24} gold {g} got {o}  [{syl}]")


HARD_SUBSTR = ["นภา", "นที", "ชลธี", "อัมพร", "ทิฆัมพร", "คีรี", "รัศมี", "สุริยา", "นภาลัย",
               "ปักษา", "วนา", "สาคร", "ธารา", "กรุณา", "มนุษย์", "ธรรมชาติ", "วิทยา",
               "ปรารถนา", "สังเกต", "วิญญาณ", "ศรัทธา", "ขนม", "สนาม", "สบาย", "สวรรค์"]


def compare(n_real: int = 160) -> None:
    """Head-to-head flat vs word-decomposition on (a) gold-anchored hard lines and
    (b) slip-prone real grading วรรค (those containing a hard word). The word mode
    should match gold and reveal flat's inline-under-counts via its per-word breakdown."""
    from audit_count import WORD_GOLD  # type: ignore
    import random as _r
    cats = {**WORD_GOLD, **EXTRA_GOLD}
    hard = [(w, g) for cat, ws in cats.items() for (w, g) in ws
            if cat in ("อักษรนำ (2 syl)", "Pali/Sanskrit polysyllable", "Sanskrit poetic (hard)")]
    MONO = ["ไป", "มา", "ใจ", "ดาว", "ลม", "ฟ้า", "ยาม", "ตา", "ดี", "นาน"]
    rng = _r.Random(7)
    gold_lines = []
    for w, g in hard:
        ms = [rng.choice(MONO) for _ in range(rng.randint(2, 3))]
        gold_lines.append((w + "".join(ms), g + len(ms)))
    gold_lines.append(("นภาลาลับตา", 5))   # the observed wild slip

    real = [v for v in grading_vrks() if any(h in v for h in HARD_SUBSTR)][:n_real]

    allv = [t for t, _ in gold_lines] + real
    flat = VrkCounter(mode="flat", persist=False); flat.cache = {}
    word = VrkCounter(mode="word", persist=False); word.cache = {}
    print(f"counting {len(allv)} วรรค under each prompt …")
    rf = flat.count(allv, verbose=False)
    rw = word.count(allv, verbose=False)

    fok = wok = 0
    print(f"\n(1) GOLD-ANCHORED lines (n={len(gold_lines)}) — accuracy vs known sum:")
    for t, g in gold_lines:
        f, w_ = rf.get(t, {}).get("count"), rw.get(t, {}).get("count")
        fok += (f == g); wok += (w_ == g)
        if f != g or w_ != g:
            print(f"   gold {g}  flat {f}  word {w_}   {t}")
    print(f"   flat {fok}/{len(gold_lines)} = {fok/len(gold_lines)*100:.0f}%  ·  "
          f"word {wok}/{len(gold_lines)} = {wok/len(gold_lines)*100:.0f}%")

    print(f"\n(2) REAL slip-prone วรรค (n={len(real)}) — flat vs word disagreements:")
    dis = 0
    for v in real:
        f, w_ = rf.get(v, {}).get("count"), rw.get(v, {}).get("count")
        if f != w_:
            dis += 1
            print(f"   flat {f} | word {w_}   {v}")
            print(f"      flat: {rf[v]['syllables']}")
            print(f"      word: {rw[v]['syllables']}")
    print(f"   disagreements: {dis}/{len(real)} ({dis/len(real)*100:.0f}%)")


# confirmable hard real lines (each hard word's count is unambiguous):
# นภา=2, สว่าง=2, สวรรค์=2, สาคร=2, อร่าม=2, พิรุณ=2
HARD_REAL = [
    ("สาครไหวคลื่นรอน", 5), ("เดือนเพ็ญลอยนภาสว่าง", 7), ("นภาใสสว่างใจ", 6),
    ("ส่องสว่างกลางนภา", 6), ("แผ่นฟ้าสวรรค์", 4), ("ส่องแสงจ้าอาบดินถิ่นสวรรค์", 8),
    ("สวนงามอร่ามดั่งแดนสวรรค์", 8), ("หยาดพิรุณร่วงพรมจากสรวงสวรรค์", 9),
    ("นภาลาลับตา", 5), ("ค่ำวันเพ็ญเดือนสิบสองนภาสว่าง", 10),
]


def thinktest() -> None:
    """flat (no-think) vs word vs THINKING on the confirmable hard real lines."""
    vrks = [t for t, _ in HARD_REAL]
    counters = {
        "flat":  VrkCounter(mode="flat", persist=False),
        "word":  VrkCounter(mode="word", persist=False),
        "think": VrkCounter(mode="flat", persist=False, think=True),
    }
    res = {}
    for name, c in counters.items():
        c.cache = {}
        res[name] = c.count(vrks, verbose=False, workers=4)
    print(f"\n{'gold':>4}{'flat':>5}{'word':>5}{'think':>6}  วรรค")
    acc = {k: 0 for k in counters}
    for t, g in HARD_REAL:
        cells = {k: res[k].get(t, {}).get("count") for k in counters}
        for k in counters:
            acc[k] += (cells[k] == g)
        print(f"{g:>4}{cells['flat']:>5}{cells['word']:>5}{cells['think']:>6}  {t}")
    n = len(HARD_REAL)
    print(f"\naccuracy: flat {acc['flat']}/{n} · word {acc['word']}/{n} · think {acc['think']}/{n}")


# ---------- smoke ----------

def smoke() -> None:
    from thaiwrite.prosody.calibration.build_calibration_set import AUDIT_POEMS  # type: ignore
    gold = []   # (วรรค text, hand-gold count)
    for p in AUDIT_POEMS:
        for tup in p.lines:
            gold.append(("".join(tup), len(tup)))
    vrks = [g[0] for g in gold]
    print(f"SMOKE: {len(vrks)} hand-counted audit วรรค\n")
    c = VrkCounter()
    res = c.count(vrks)
    ok = 0
    print(f"\n{'gold':>4}{'opus':>5}  วรรค")
    for text, g in gold:
        o = res.get(text, {}).get("count")
        mark = "" if o == g else "  <-- MISMATCH"
        ok += (o == g)
        print(f"{g:>4}{(o if o is not None else '?'):>5}  {text}{mark}")
    print(f"\naccuracy vs hand-gold: {ok}/{len(gold)} = {ok/len(gold)*100:.0f}%")

    # determinism: re-count one batch with cache bypassed, twice, compare
    print("\nDETERMINISM (temp=0, cache bypassed, two fresh calls on 8 วรรค):")
    sample = vrks[:8]
    c2 = VrkCounter(persist=False); c2.cache = {}
    r1 = {k: v["count"] for k, v in c2.count(sample, verbose=False, workers=4).items()}
    c3 = VrkCounter(persist=False); c3.cache = {}
    r2 = {k: v["count"] for k, v in c3.count(sample, verbose=False, workers=4).items()}
    same = sum(1 for k in r1 if r1[k] == r2.get(k))
    print(f"  identical: {same}/{len(r1)}  temp_accepted={c2._temp_ok}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "validate", "compare", "thinktest", "run"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", action="store_true", help="use extended thinking (count-of-record)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.cmd == "smoke":
        smoke(); return
    if args.cmd == "validate":
        validate(); return
    if args.cmd == "compare":
        compare(); return
    if args.cmd == "thinktest":
        thinktest(); return
    vrks = grading_vrks()
    if args.limit:
        vrks = vrks[: args.limit]
    print(f"grading-relevant unique วรรค (compose): {len(vrks)} · think={args.think} batch={args.batch}")
    c = VrkCounter(think=args.think)
    c.count(vrks, workers=args.workers, batch=args.batch)
    flagged = sum(1 for v in vrks if c.cache.get(v.strip(), {}).get("flag"))
    print(f"\ndone. cache: {len(c.cache)} entries · self-check flags: {flagged} · → {c.cache_path.name}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
