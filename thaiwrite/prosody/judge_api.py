"""Content judge (topicality, coherence) through the Claude API, reproducible and cached.

Each delivered composition poem is judged on its own, blind to the model that wrote it,
with the rubric in grader_coherence_prompt.txt. Scores are written to
<--out>/<config>.jsonl (default eval/judged_new/; one line per item, with the judged poem
text, model and effort); reproduce/make_paper_tables.py reads them with --judge. When the judge declines
a poem (for example a leaked reasoning trace), the row is marked "unjudgeable" and counts
as neither on-topic nor coherent. Items already in the output directory are skipped, so an
interrupted run resumes. Requires ANTHROPIC_API_KEY (environment or a .env file in a
parent directory).

  PYTHONIOENCODING=utf-8 python judge_api.py plan  --scope missing
  PYTHONIOENCODING=utf-8 python judge_api.py calib --model claude-opus-5-5        # vs expert labels
  PYTHONIOENCODING=utf-8 python judge_api.py run   --scope missing --mode batch
  PYTHONIOENCODING=utf-8 python judge_api.py run   --scope all --mode batch

Scopes: missing = delivered poems with no judgment in eval/judged_archive/; all = every
delivered poem. `--graded-dir` selects the graded files to judge (default eval/graded).
`calib` scores the 132 calibration poems and writes calibration/judge_api_calib_<model>.json,
which make_paper_tables.py includes in the judge-agreement table.
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

import anthropic
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "eval"))

from promptlib import clean_poem  # noqa: E402

MODEL = "claude-opus-5-5"
PRICE = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00)}
MAX_TOKENS = 8000
GRADED = HERE / "eval" / "graded"
OUT = HERE / "eval" / "judged_new"
ARCHIVE = [HERE / "eval" / "judged_archive"]
CAL_ITEMS = HERE / "calibration" / "calibration_items_v2.json"
LABELS_A = HERE / "calibration" / "labels" / "labels_expertA.json"

_raw = (HERE / "grader_coherence_prompt.txt").read_text(encoding="utf-8")
PROMPT_TEMPLATE = _raw.partition("\n")[2].lstrip("\n") if _raw.startswith("# version:") else _raw
SYSTEM = "You are an expert evaluator of Thai poetry. Follow the rubric exactly and output only the requested JSON."


def _client():
    for d in [HERE, *HERE.parents]:
        env = d / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    import anthropic
    return anthropic.Anthropic(max_retries=4)


def _params(topic: str, poem: str, effort: str, model: str) -> dict:
    return {"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM,
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.replace("{topic}", topic).replace("{poem}", poem)}],
            "output_config": {"effort": effort}}


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        t, c = int(o["topicality"]), int(o["coherence"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return {"topicality": min(5, max(1, t)), "coherence": min(5, max(1, c)),
            "rationale": str(o.get("rationale", ""))[:400]}


def _archived(key: str) -> set[str]:
    for d in ARCHIVE:
        p = d / f"{key}.jsonl"
        if p.exists():
            return {str(json.loads(l)["item_id"]) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    return set()


def _done_in(out_dir: Path, key: str) -> set[str]:
    p = out_dir / f"{key}.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["item_id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def collect(scope: str, native_only: bool = False, out_dir: Path | None = None,
            also_done: list[str] = (), graded_dir: Path = GRADED) -> list[dict]:
    items = []
    for p in sorted(Path(graded_dir).glob("*__run1.jsonl")):
        key = p.name.split("__run")[0]
        if native_only and (".nothink" in key or ".rules" in key):
            continue
        done = _archived(key) if scope == "missing" else set()
        for d in ([out_dir] if out_dir is not None else []) + [Path(x) for x in also_done]:
            done |= _done_in(d, key)
        seen = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r.get("item_id", "")
            if "-compose-" not in iid or r.get("eval_status") != "graded" or iid in seen or iid in done:
                continue
            seen.add(iid)
            poem = (r.get("poem") or clean_poem(r.get("raw_response") or "")).strip()
            items.append({"config": key, "item_id": iid, "topic": r.get("topic", ""), "poem": poem})
    return items


class Judge:
    def __init__(self, model: str, effort: str):
        self.client, self.model, self.effort = _client(), model, effort
        self.lock = threading.Lock()
        self.usage = {"sync_in": 0, "sync_out": 0, "batch_in": 0, "batch_out": 0, "fail": 0}

    def _tally(self, usage, tier):
        with self.lock:
            self.usage[f"{tier}_in"] += usage.input_tokens
            self.usage[f"{tier}_out"] += usage.output_tokens

    def one(self, it: dict) -> dict | None:
        for _ in range(3):
            p = _params(it["topic"], it["poem"], self.effort, self.model)
            msg = self._retrying(lambda: self.client.messages.create(
                **{k: v for k, v in p.items() if k != "output_config"},
                extra_body={"output_config": p["output_config"]}))
            self._tally(msg.usage, "sync")
            if msg.stop_reason == "refusal":
                cat = getattr(getattr(msg, "stop_details", None), "category", None)
                return {"topicality": None, "coherence": None, "unjudgeable": True,
                        "reason": f"refusal: {cat}"}
            res = _parse("".join(b.text for b in msg.content if b.type == "text"))
            if res:
                return res
        with self.lock:
            self.usage["fail"] += 1
        return None

    def sync(self, items: list[dict], workers: int = 8, on_result=None) -> dict[int, dict]:
        out = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.one, it): i for i, it in enumerate(items)}
            for k, f in enumerate(as_completed(futs), 1):
                r = f.result()
                if r:
                    out[futs[f]] = r
                    if on_result:
                        on_result(futs[f], r)
                if k % 100 == 0:
                    print(f"{k}/{len(items)} done, ${self.cost():.2f}", flush=True)
        return out

    def _retrying(self, fn, *args, wait_s: int = 60):
        """Call fn until it gets through; a dropped connection must not lose a submitted batch."""
        while True:
            try:
                return fn(*args)
            except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                print(f"poll: {type(e).__name__}, retrying in {wait_s}s", flush=True)
                time.sleep(wait_s)

    def batch(self, items: list[dict], poll_s: int = 30, batch_id: str | None = None,
              retry_sync: bool = True) -> dict[int, dict]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        if batch_id is None:
            reqs = [Request(custom_id=f"j{i:05d}",
                            params=MessageCreateParamsNonStreaming(**_params(it["topic"], it["poem"], self.effort, self.model)))
                    for i, it in enumerate(items)]
            b = self.client.messages.batches.create(requests=reqs)
            batch_id = b.id
            (HERE / f"judge_api_batch_{batch_id}.json").write_text(json.dumps(
                [{"custom_id": f"j{i:05d}", "config": it["config"], "item_id": it["item_id"]}
                 for i, it in enumerate(items)], ensure_ascii=False), encoding="utf-8")
            print("batch", batch_id, len(reqs), "requests", flush=True)
        else:
            b = self._retrying(self.client.messages.batches.retrieve, batch_id)
            n = sum(getattr(b.request_counts, k) for k in ("processing", "succeeded", "errored", "canceled", "expired"))
            if n != len(items):
                raise SystemExit(f"batch {batch_id} has {n} requests but {len(items)} items were collected")
            man = HERE / f"judge_api_batch_{batch_id}.json"
            if man.exists():
                want = [(m["config"], m["item_id"]) for m in json.loads(man.read_text(encoding="utf-8"))]
                if want != [(it["config"], it["item_id"]) for it in items]:
                    raise SystemExit(f"collected items differ from the manifest {man.name}")
            print("reattached", batch_id, n, "requests", flush=True)
        while self._retrying(self.client.messages.batches.retrieve, batch_id).processing_status != "ended":
            time.sleep(poll_s)
        out, retry = {}, []
        for r in self._retrying(lambda i: list(self.client.messages.batches.results(i)), batch_id):
            i = int(r.custom_id[1:])
            if r.result.type == "succeeded":
                msg = r.result.message
                self._tally(msg.usage, "batch")
                if msg.stop_reason == "refusal":
                    cat = getattr(getattr(msg, "stop_details", None), "category", None)
                    out[i] = {"topicality": None, "coherence": None, "unjudgeable": True,
                              "reason": f"refusal: {cat}"}
                    continue
                res = _parse("".join(x.text for x in msg.content if x.type == "text"))
                if res:
                    out[i] = res
                    continue
            retry.append(i)
        if retry and not retry_sync:
            print(f"{len(retry)} poems left unjudged (canceled or unparsed)", flush=True)
        elif retry:
            print(f"retrying {len(retry)} synchronously", flush=True)
            got = self.sync([items[i] for i in retry])
            out.update({retry[k]: v for k, v in got.items()})
        return out

    def cost(self) -> float:
        pi, po = PRICE[self.model]
        u = self.usage
        return (u["sync_in"] * pi + u["sync_out"] * po + (u["batch_in"] * pi + u["batch_out"] * po) / 2) / 1e6


def cmd_plan(a):
    items = collect(a.scope, a.native_only, Path(a.out), a.also_done, a.graded_dir)
    items = [it for it in items if not any(x in it["config"] for x in a.exclude)]
    by = {}
    for it in items:
        by[it["config"]] = by.get(it["config"], 0) + 1
    # cost per poem measured on the direct runs of 2026-09-30 (effort medium): $1.20 / 132 and $5.05 / 471
    lo, hi = 1.20 / 132, 5.05 / 471
    print(json.dumps(by, indent=1), f"\ntotal poems: {len(items)} -> direct ${len(items) * lo:.2f}-{len(items) * hi:.2f}")


def cmd_run(a):
    items = collect(a.scope, a.native_only, Path(a.out), a.also_done, a.graded_dir)
    items = [it for it in items if not any(x in it["config"] for x in a.exclude)]
    j = Judge(a.model, a.effort)
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    wlock = threading.Lock()

    def save_one(i, r):   # sync mode: persist each judgment as it arrives, so an interrupted run resumes
        it = items[i]
        row = {"item_id": it["item_id"], **r, "model": a.model, "effort": a.effort, "poem": it["poem"]}
        with wlock, (OUT / f"{it['config']}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if a.mode == "sync":
        res = j.sync(items, a.workers, on_result=save_one)
        print(json.dumps({"judged": len(res), "of": len(items), "usage": j.usage, "usd": round(j.cost(), 2)}))
        return
    res = j.batch(items, batch_id=a.batch_id, retry_sync=not a.no_retry)
    per: dict[str, dict] = {}
    for i, it in enumerate(items):
        if i in res:
            per.setdefault(it["config"], {})[it["item_id"]] = {
                "item_id": it["item_id"], **res[i], "model": a.model, "effort": a.effort, "poem": it["poem"]}
    for key, rows in per.items():
        p = OUT / f"{key}.jsonl"
        old = {}
        if p.exists():
            old = {json.loads(l)["item_id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
        old.update(rows)
        p.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in old.values()) + "\n", encoding="utf-8")
    print(json.dumps({"judged": len(res), "of": len(items), "usage": j.usage, "usd": round(j.cost(), 2)}))


def _kappa(pairs):
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def cmd_calib(a):
    items = [x for x in json.loads(CAL_ITEMS.read_text(encoding="utf-8")) if x.get("kind") == "poem"]
    labels = json.loads(LABELS_A.read_text(encoding="utf-8"))["labels"]
    items = [x for x in items if isinstance(labels.get(x["id"]), dict)
             and isinstance(labels[x["id"]].get("topicality"), int) and isinstance(labels[x["id"]].get("coherence"), int)]
    j = Judge(a.model, a.effort)
    res = j.sync([{"topic": x.get("topic", ""), "poem": x.get("poem_text", "")} for x in items], a.workers)
    rep = {"model": a.model, "effort": a.effort, "n": len(res), "usd": round(j.cost(), 3)}
    for axis, cut in (("topicality", 3), ("coherence", 2), ("coherence", 3)):
        pairs = [(int(labels[x["id"]][axis] >= cut), int(res[i][axis] >= cut)) for i, x in enumerate(items) if i in res]
        rep[f"kappa_{axis}>={cut}"] = round(_kappa(pairs), 3)
    print(json.dumps(rep))
    (HERE / "calibration" / f"judge_api_calib_{a.model}.json").write_text(json.dumps(
        {"report": rep, "scores": {items[i]["id"]: v for i, v in res.items()}}, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "calib"):
        s = sub.add_parser(name)
        s.add_argument("--model", choices=sorted(PRICE), default=MODEL)
        s.add_argument("--effort", default="medium")
        s.add_argument("--workers", type=int, default=8)
        if name in ("plan", "run"):
            s.add_argument("--scope", choices=["missing", "all"], default="missing")
            s.add_argument("--native-only", action="store_true")
            s.add_argument("--out", default=str(OUT))
            s.add_argument("--graded-dir", default=str(GRADED))
            s.add_argument("--also-done", nargs="*", default=[],
                           help="other judgment directories; poems already judged there are skipped")
            s.add_argument("--exclude", nargs="*", default=[], help="skip configs whose name contains any of these")
        if name == "run":
            s.add_argument("--mode", choices=["sync", "batch"], default="batch")
            s.add_argument("--batch-id", help="reattach to a submitted batch instead of creating one; "
                                              "the collected items must be the ones it was created from")
            s.add_argument("--no-retry", action="store_true",
                           help="batch mode: only download results; do not re-send failed or canceled requests")
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "calib": cmd_calib}[a.cmd](a)


if __name__ == "__main__":
    main()
