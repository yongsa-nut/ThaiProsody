"""Syllable count-of-record through the Claude API (Claude Opus 5.5), reproducible and cached.

Every distinct line piece the grader looks up (whitespace-separated piece of a scored
line) is counted by the same model with the same prompt (vrk_counter.PROMPT_HEAD), in
requests of LINES_PER_REQUEST pieces. Results go to a separate cache file whose entries
record the model and effort, so the count-of-record has one provenance.

  PYTHONIOENCODING=utf-8 python count_api.py plan  --scope A B C D        # sizes + cost estimate, no API spend
  PYTHONIOENCODING=utf-8 python count_api.py pilot --n 200 --efforts low medium high --reference-cache OTHER.json
  PYTHONIOENCODING=utf-8 python count_api.py run   --scope A --effort medium --mode sync
  PYTHONIOENCODING=utf-8 python count_api.py run   --scope B C D --effort medium --mode batch

Scopes: A = composition, native configurations; B = ablations (thinking disabled, rules in
the prompt); C = repeated runs 2-3; D = cloze/edit outputs. `run` writes to --cache (default
vrk_count_cache_opus55_main.json) and skips pieces already in it or in any --also-cached file,
so an interrupted run resumes. Pieces are read from --graded-dir (default eval/graded); to
count the outputs of a newly graded model, point --graded-dir at its directory. Requires
ANTHROPIC_API_KEY (environment or a .env file in a parent directory).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "eval"))

from thaiwrite.prosody import meter_specs as M                        # noqa: E402
from thaiwrite.prosody.vrk_counter import PROMPT_HEAD, EXTRA_GOLD     # noqa: E402
from promptlib import clean_poem, to_vrk_lines                         # noqa: E402

MODEL = "claude-opus-5-5"
LINES_PER_REQUEST = 50
MAX_TOKENS = 16000
# $ per 1M input / output tokens on the standard API; the Batch API is half price.
PRICE = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00)}
OUT_CACHE = HERE / "vrk_count_cache_opus55_main.json"
USAGE_LOG = HERE / "count_api_usage.jsonl"
METER_BY_TAG = {"khlong": "khlong_si_suphap", "klon8": "klon_paet",
                "yani": "kap_yani_11", "chan": "intharawichian_chan_11"}


# ------------------------------------------------------------------ scope collection
def _meter_of(r: dict) -> str | None:
    mid = r.get("meter_id")
    if mid:
        return mid
    m = re.search(r"prosody-(khlong|klon8|yani|chan)-", r.get("item_id", ""))
    return METER_BY_TAG.get(m.group(1)) if m else None


def _pieces(poem: str, meter_id: str | None) -> set[str]:
    nslots = len(M.METER_SPECS_BY_ID[meter_id].lines) + 2 if meter_id in M.METER_SPECS_BY_ID else 99
    raw = [ln.strip() for ln in poem.splitlines() if ln.strip()][:nslots]
    vrk = [v.strip() for v in to_vrk_lines(poem).splitlines() if v.strip()][:nslots]
    return {p for u in raw + vrk for p in u.split() if p.strip()}


def _graded(path: Path) -> set[str]:
    s: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_status") not in (None, "graded"):
            continue
        poem = r.get("poem") or clean_poem(r.get("raw_response") or "")
        s |= _pieces(poem, _meter_of(r))
    return s


def _runs(path: Path) -> set[str]:
    s: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        txt = r.get("response") or r.get("raw_response") or ""
        if txt and r.get("error") in (None, "None", ""):
            s |= _pieces(clean_poem(txt), _meter_of(r))
    return s


def collect(scopes: list[str], graded_dir: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if "A" in scopes or "B" in scopes:
        nat, abl = set(), set()
        for p in sorted(graded_dir.glob("*__run1.jsonl")):
            (abl if (".nothink" in p.name or ".rules" in p.name) else nat).update(_graded(p))
        if "A" in scopes:
            out["A"] = nat
        if "B" in scopes:
            out["B"] = abl
    if "C" in scopes:
        out["C"] = set().union(*[_runs(p) for p in sorted((HERE / "eval" / "runs").glob("*__run[23].jsonl"))])
    if "D" in scopes:
        ce = HERE / "cloze_edit" / "graded"
        out["D"] = set().union(*[_graded(p) for p in sorted(ce.glob("*__run1.jsonl"))]) if ce.exists() else set()
    return out


# ------------------------------------------------------------------ API
def _client():
    for d in [HERE, *HERE.parents]:
        env = d / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    import anthropic
    return anthropic.Anthropic(max_retries=4)


def _params(lines: list[str], effort: str, model: str = MODEL) -> dict:
    prompt = PROMPT_HEAD + "\n".join(f"{i + 1}. {v}" for i, v in enumerate(lines))
    return {"model": model, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": effort}}


def _parse(text: str, lines: list[str]) -> dict[str, dict]:
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return {}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for o in arr:
        try:
            i = int(o["n"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= i < len(lines):
            continue
        syl = str(o.get("syllables", ""))
        n_syl = len([t for t in syl.split(".") if t.strip()])
        cnt = o.get("count")
        flag = not isinstance(cnt, int) or cnt != n_syl
        out[lines[i]] = {"count": n_syl if flag and n_syl else cnt, "syllables": syl,
                         "selfcheck_flag": flag}
    return out


class Counter:
    def __init__(self, effort: str, mode: str, cache_path: Path = OUT_CACHE, persist: bool = True,
                 model: str = MODEL):
        self.client = _client()
        self.effort, self.mode, self.persist, self.model = effort, mode, persist, model
        self.cache_path = cache_path
        self.cache: dict[str, dict] = (json.loads(cache_path.read_text(encoding="utf-8"))
                                       if persist and cache_path.exists() else {})
        self.lock = threading.Lock()
        # tokens are tallied per pricing tier: batch results vs synchronous calls
        self.usage = {"sync_in": 0, "sync_out": 0, "batch_in": 0, "batch_out": 0,
                      "requests": 0, "refusals": 0, "parse_fail": 0}

    def _record(self, res: dict[str, dict], usage, tier: str) -> None:
        with self.lock:
            for k, v in res.items():
                v.update({"model": self.model, "effort": self.effort})
                self.cache[k] = v
            if usage is not None:
                self.usage[f"{tier}_in"] += usage.input_tokens
                self.usage[f"{tier}_out"] += usage.output_tokens
            self.usage["requests"] += 1

    def _save(self) -> None:
        if self.persist:
            with self.lock:
                self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0),
                                           encoding="utf-8")

    def _sync_one(self, lines: list[str]) -> None:
        todo = list(lines)
        for _ in range(3):
            if not todo:
                return
            msg = self.client.messages.create(**{k: v for k, v in _params(todo, self.effort, self.model).items()
                                                 if k != "output_config"},
                                              extra_body={"output_config": {"effort": self.effort}})
            if msg.stop_reason == "refusal":
                with self.lock:
                    self.usage["refusals"] += 1
            text = "".join(b.text for b in msg.content if b.type == "text")
            res = _parse(text, todo)
            if not res:
                with self.lock:
                    self.usage["parse_fail"] += 1
            self._record(res, msg.usage, "sync")
            todo = [v for v in todo if v not in res]

    def count_sync(self, pieces: list[str], workers: int = 8) -> None:
        todo = [p for p in pieces if p not in self.cache]
        chunks = [todo[i:i + LINES_PER_REQUEST] for i in range(0, len(todo), LINES_PER_REQUEST)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self._sync_one, c) for c in chunks]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001 - a failed request is re-run on the next pass
                    print("request failed, will be retried on a re-run:", type(e).__name__, e, flush=True)
                if i % 5 == 0:   # save often: a dropped connection loses at most a few paid requests
                    self._save()
                if i % 20 == 0:
                    print(f"  {i}/{len(chunks)} requests, {self.cost():.2f} USD so far", flush=True)
        self._save()

    def count_batch(self, pieces: list[str], poll_s: int = 60) -> None:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        todo = [p for p in pieces if p not in self.cache]
        chunks = {f"c{i:05d}": todo[i:i + LINES_PER_REQUEST] for i in range(0, len(todo), LINES_PER_REQUEST)}
        reqs = [Request(custom_id=cid,
                        params=MessageCreateParamsNonStreaming(**_params(c, self.effort, self.model)))
                for cid, c in chunks.items()]
        batch = self.client.messages.batches.create(requests=reqs)
        print("batch", batch.id, len(reqs), "requests", flush=True)
        (HERE / "count_api_batches.jsonl").open("a", encoding="utf-8").write(
            json.dumps({"batch": batch.id, "chunks": chunks, "model": self.model, "effort": self.effort},
                       ensure_ascii=False) + "\n")
        self.collect_batch(batch.id, chunks, poll_s)

    def collect_batch(self, batch_id: str, chunks: dict[str, list[str]], poll_s: int = 60,
                      retry_sync: bool = True) -> None:
        """Wait for a submitted batch (surviving network errors) and record its results."""
        import anthropic
        while True:
            try:
                if self.client.messages.batches.retrieve(batch_id).processing_status == "ended":
                    break
            except anthropic.APIConnectionError as e:
                print("poll: connection error, retrying:", e, flush=True)
            time.sleep(poll_s)
        retry = []
        for r in self.client.messages.batches.results(batch_id):
            lines = chunks[r.custom_id]
            if r.result.type != "succeeded":
                retry += lines
                continue
            msg = r.result.message
            res = _parse("".join(b.text for b in msg.content if b.type == "text"), lines)
            self._record(res, msg.usage, "batch")
            retry += [v for v in lines if v not in res]
        self._save()
        if retry and not retry_sync:
            print(f"{len(retry)} pieces left uncounted (canceled or unparsed)", flush=True)
        elif retry:
            print(f"retrying {len(retry)} pieces synchronously", flush=True)
            self.count_sync(retry)

    def cost(self) -> float:
        pi, po = PRICE[self.model]
        u = self.usage
        return (u["sync_in"] * pi + u["sync_out"] * po + (u["batch_in"] * pi + u["batch_out"] * po) / 2) / 1e6


def _log(entry: dict) -> None:
    with USAGE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ commands
def _known(paths: list[str]) -> set[str]:
    """Pieces already counted in other cache files (skipped by run)."""
    out: set[str] = set()
    for p in paths:
        if Path(p).exists():
            out |= set(json.loads(Path(p).read_text(encoding="utf-8")))
    return out


def cmd_plan(a) -> None:
    sets = collect(a.scope, Path(a.graded_dir))
    known = _known(a.also_cached + [a.cache])
    seen: set[str] = set()
    for k in sorted(sets):
        new = sets[k] - seen
        seen |= sets[k]
        print(f"scope {k}: {len(sets[k])} pieces ({len(new)} not in earlier scopes, "
              f"{len(new - known)} of those not yet counted)")
    todo = seen - known
    # cost per piece measured on the direct runs of 2026-09-30 (effort medium): $30.14 / 17,724 and $6.24 / 3,977
    lo, hi = 6.24 / 3977, 30.14 / 17724
    print(f"total distinct: {len(seen)}; to count: {len(todo)} -> direct ${len(todo) * lo:.2f}-{len(todo) * hi:.2f}")


def _gold_lines() -> list[tuple[str, int]]:
    from audit_count import WORD_GOLD  # type: ignore
    words = [(w, g) for ws in {**WORD_GOLD, **EXTRA_GOLD}.values() for (w, g) in ws]
    mono = ["ไป", "มา", "ใจ", "ดาว", "ลม", "ฟ้า", "ยาม", "ตา", "ดี", "นาน"]
    rng = random.Random(7)
    lines = list(words)
    for w, g in words:
        ms = [rng.choice(mono) for _ in range(rng.randint(2, 3))]
        lines.append((w + "".join(ms), g + len(ms)))
    return lines


def cmd_pilot(a) -> None:
    sets = collect(["A"], Path(a.graded_dir))
    # reference counts to compare against (e.g. another model's count cache); optional
    prior = json.loads(Path(a.reference_cache).read_text(encoding="utf-8")) if a.reference_cache else {}
    rng = random.Random(0)
    pool = sorted(p for p in sets["A"] if p in prior) if prior else sorted(sets["A"])
    real = rng.sample(pool, min(a.n, len(pool)))
    gold = _gold_lines()
    report, counts = {}, {}
    for model in a.models:
        for eff in a.efforts:
            c = Counter(eff, a.mode, persist=False, model=model)
            lines = real + [t for t, _ in gold]
            (c.count_batch(lines) if a.mode == "batch" else c.count_sync(lines, workers=a.workers))
            g_ok = sum(1 for t, g in gold if c.cache.get(t, {}).get("count") == g)
            p_ok = sum(1 for t in real if t in prior and c.cache.get(t, {}).get("count") == prior[t]["count"])
            n = len(lines)
            u = c.usage
            rep = {"model": model, "effort": eff, "mode": a.mode,
                   "gold_acc": g_ok / len(gold), "n_gold": len(gold),
                   "agree_reference": (p_ok / len(real)) if prior else None, "n_real": len(real),
                   "counted": sum(1 for t in lines if t in c.cache),
                   "flags": sum(1 for v in c.cache.values() if v.get("selfcheck_flag")),
                   "input_per_line": (u["sync_in"] + u["batch_in"]) / n,
                   "output_per_line": (u["sync_out"] + u["batch_out"]) / n,
                   "usd": c.cost(), "usd_per_1k_lines_batch": None, "usage": u}
            pi, po = PRICE[model]
            rep["usd_per_1k_lines_batch"] = 1000 * (rep["input_per_line"] * pi + rep["output_per_line"] * po) / 2e6
            report[f"{model}/{eff}"] = rep
            counts[f"{model}/{eff}"] = {t: c.cache.get(t, {}).get("count") for t in real}
            _log({"cmd": "pilot", **rep})
            print(json.dumps(rep, ensure_ascii=False), flush=True)
            misses = [(t, g, c.cache.get(t, {}).get("count"), c.cache.get(t, {}).get("syllables"))
                      for t, g in gold if c.cache.get(t, {}).get("count") != g]
            for m in misses[:20]:
                print("   gold miss:", m, flush=True)
    keys = list(counts)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            same = sum(1 for t in real if counts[keys[i]][t] == counts[keys[j]][t])
            report[f"agree {keys[i]} vs {keys[j]}"] = same / len(real)
            print(f"agreement {keys[i]} vs {keys[j]}: {same}/{len(real)}", flush=True)
    (HERE / "count_api_pilot.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                               encoding="utf-8")


def cmd_run(a) -> None:
    sets = collect(a.scope, Path(a.graded_dir))
    known = _known(a.also_cached)
    pieces = sorted(set().union(*sets.values()) - known)
    if known:
        print(f"skipping pieces already counted in {', '.join(a.also_cached)}", flush=True)
    c = Counter(a.effort, a.mode, model=a.model, cache_path=Path(a.cache))
    print(f"{len(pieces)} pieces, {sum(1 for p in pieces if p in c.cache)} already counted", flush=True)
    t0 = time.time()
    if a.mode == "batch":
        c.count_batch(pieces)
    for _ in range(3):   # further passes pick up requests that failed on the network
        todo = [p for p in pieces if p not in c.cache]
        if not todo:
            break
        print(f"pass: {len(todo)} pieces to count", flush=True)
        c.count_sync(todo, workers=a.workers)
    rep = {"cmd": "run", "scope": a.scope, "effort": a.effort, "mode": a.mode, "usage": c.usage,
           "usd": c.cost(), "minutes": (time.time() - t0) / 60, "cache_size": len(c.cache)}
    _log(rep)
    print(json.dumps(rep, ensure_ascii=False))


def cmd_collect(a) -> None:
    """Download the results of an ended batch into the cache; no new model calls."""
    entry = next(json.loads(l) for l in (HERE / "count_api_batches.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip() and json.loads(l)["batch"] == a.batch_id)
    c = Counter(entry["effort"], "batch", model=entry["model"], cache_path=Path(a.cache))
    before = len(c.cache)
    c.collect_batch(a.batch_id, entry["chunks"], retry_sync=False)
    rep = {"cmd": "collect", "batch": a.batch_id, "usage": c.usage, "usd": c.cost(),
           "added": len(c.cache) - before, "cache_size": len(c.cache)}
    _log(rep)
    print(json.dumps(rep, ensure_ascii=False))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--batch-id", required=True)
    col.add_argument("--cache", default=str(OUT_CACHE))
    for name in ("plan", "pilot", "run"):
        s = sub.add_parser(name)
        s.add_argument("--graded-dir", default=str(HERE / "eval" / "graded"))
        s.add_argument("--workers", type=int, default=8)
        if name in ("plan", "run"):
            s.add_argument("--scope", nargs="+", default=["A"])
            s.add_argument("--cache", default=str(OUT_CACHE))
            s.add_argument("--also-cached", nargs="*", default=[],
                           help="other count caches; pieces already in them are not counted again")
        if name == "run":
            s.add_argument("--effort", default="medium")
            s.add_argument("--mode", choices=["sync", "batch"], default="batch")
            s.add_argument("--model", choices=sorted(PRICE), default=MODEL)
        if name == "pilot":
            s.add_argument("--n", type=int, default=200)
            s.add_argument("--efforts", nargs="+", default=["medium"])
            s.add_argument("--models", nargs="+", choices=sorted(PRICE), default=sorted(PRICE))
            s.add_argument("--mode", choices=["sync", "batch"], default="sync")
            s.add_argument("--reference-cache", default=None,
                           help="count cache to compare against (optional)")
    a = ap.parse_args()
    {"plan": cmd_plan, "pilot": cmd_pilot, "run": cmd_run, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
