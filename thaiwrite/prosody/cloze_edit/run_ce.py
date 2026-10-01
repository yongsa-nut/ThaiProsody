"""Run models on the cloze/edit items (same prompts, system role, providers and
per-model generation settings as the composition runs in eval/registry.py).

Raw generations go to cloze_edit/runs/<model_key>__run1.jsonl (resume-safe: items
already answered successfully are skipped, so an interrupted run never re-requests them).
This script calls model APIs; keys are read from the environment or a .env file.

  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/run_ce.py --models typhoon25 --per-cell 2
  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/run_ce.py --models gpt55-med --concurrency 8
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from dotenv import load_dotenv                                          # noqa: E402

for _parent in HERE.parents:
    if (_parent / ".env").exists():
        load_dotenv(_parent / ".env", override=True)
        break

import os                                                               # noqa: E402
import threading                                                        # noqa: E402
import time                                                             # noqa: E402

import requests                                                         # noqa: E402

from pilot_rerun.clients import load_completed                          # noqa: E402
from thaiwrite.prosody.eval.promptlib import SYSTEM, render_prompt      # noqa: E402
from thaiwrite.prosody.eval.providers import _resolve_key, make_generator  # noqa: E402
from thaiwrite.prosody.eval.registry import ModelSpec, REGISTRY, resolve  # noqa: E402

DATA = HERE / "data"
RUNS = HERE / "runs"

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
#: Alternative hosts for models whose OpenRouter route is unavailable. Same weights as
#: the paper's registry entries; generation settings noted per entry.
EXTRA: dict[str, ModelSpec] = {
    # DeepSeek V4 Pro via the official DeepSeek API (native thinking, same 24k budget).
    "deepseek-v4-pro": ModelSpec(
        "deepseek-v4-pro", "DeepSeek V4 Pro", "openai_compat", "deepseek-v4-pro",
        "DEEPSEEK_API_KEY", base_url="https://api.deepseek.com", rpm=120,
        max_tokens=24000, request_timeout=900, note="official DeepSeek API; thinking native"),
    # DeepSeek V4 Flash (OpenRouter slug of the composition run). OpenRouter serves it
    # without reasoning unless requested; the composition run reasoned natively (median
    # ~9k output tokens), so reasoning is requested explicitly to keep the same mode.
    "deepseek-v4-flash": dataclasses.replace(
        REGISTRY["deepseek-v4-flash"], or_reasoning={"enabled": True},
        note="native thinking requested explicitly"),
    # Gemma 4 31B via the Google AI API, which always runs Gemma 4 in thinking mode
    # (thinking cannot be disabled there): a thinking configuration with a 16k budget.
    "gemma-4-31b-think": ModelSpec(
        "gemma-4-31b-think", "Gemma 4 31B (thinking)", "openai_compat", "gemma-4-31b-it",
        "GEMINI_API_KEY", base_url=_GEMINI_URL, rpm=30, max_tokens=16000,
        request_timeout=900, note="Google AI API; thinking always on"),
}


class CEGenerator:
    """OpenAI-compatible chat call (temperature 0, as in the composition runs) that also
    logs total/reasoning token usage (Gemini's compat endpoint omits thinking tokens
    from completion_tokens, so cost needs total_tokens)."""

    def __init__(self, spec: ModelSpec, log_path: Path):
        self.spec, self.log_path = spec, log_path
        self.key = _resolve_key(spec)
        if not self.key:
            raise RuntimeError(f"{spec.key_env} not set")
        self._lock = threading.Lock()
        self._last = 0.0

    def _pace(self):
        with self._lock:
            gap = 60.0 / self.spec.rpm - (time.monotonic() - self._last)
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()

    def generate(self, benchmark, item_id, prompt, system, max_tokens) -> dict:
        body = {"model": self.spec.model_id, "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}]}
        if self.spec.or_reasoning is not None:
            body["reasoning"] = self.spec.or_reasoning
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   **(self.spec.extra_headers or {})}
        t0 = time.monotonic()
        rec = {"benchmark": benchmark, "model": self.spec.model_id, "item_id": str(item_id),
               "prompt": prompt, "response": "", "latency_s": None, "input_tokens": None,
               "output_tokens": None, "error": None, "reasoning": None, "finish_reason": None,
               "total_tokens": None, "reasoning_tokens": None}
        for attempt in range(6):
            self._pace()
            try:
                r = requests.post(f"{self.spec.base_url.rstrip('/')}/chat/completions",
                                  headers=headers, json=body, timeout=self.spec.request_timeout)
            except requests.RequestException as e:
                rec["error"] = f"request_exception:{e}"
                time.sleep(min(2 ** attempt + 1, 60))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                rec["error"] = f"status_{r.status_code}:{r.text[:200]}"
                time.sleep(min(2 ** attempt * 5, 120))
                continue
            if r.status_code != 200:
                rec["error"] = f"status_{r.status_code}:{r.text[:300]}"
                break
            try:
                d = r.json()
                ch = d["choices"][0]
                msg = ch["message"]
                rec["response"] = msg.get("content") or ""
                rec["reasoning"] = msg.get("reasoning") or msg.get("reasoning_content")
                rec["finish_reason"] = ch.get("finish_reason")
                u = d.get("usage") or {}
                rec["input_tokens"] = u.get("prompt_tokens")
                rec["output_tokens"] = u.get("completion_tokens")
                rec["total_tokens"] = u.get("total_tokens")
                rec["reasoning_tokens"] = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                rec["error"] = None
            except (ValueError, KeyError, IndexError, TypeError):
                rec["error"] = f"bad_response:{r.text[:300]}"
            break
        rec["latency_s"] = round(time.monotonic() - t0, 2)
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def get_specs(keys: list[str]) -> list[ModelSpec]:
    out = []
    for k in keys:
        out.append(EXTRA[k] if k in EXTRA else resolve([k])[0])
    return out


def get_generator(spec: ModelSpec, log: Path):
    if spec.provider == "openai_compat":
        return CEGenerator(spec, log)
    return make_generator(spec, log)


def load_items(modes: list[str]) -> list[dict]:
    rows = []
    for m in modes:
        for line in (DATA / f"items_{m}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--modes", default="cloze,edit")
    ap.add_argument("--per-cell", type=int, default=0,
                    help="subset: first N items per (mode, meter) cell (0 = all)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--errors-only", action="store_true",
                    help="retry only items whose every attempt errored (API/infra); an item "
                         "answered with an empty output (token run-out) counts as done")
    args = ap.parse_args()

    items = load_items([m.strip() for m in args.modes.split(",") if m.strip()])
    if args.per_cell:
        seen, picked = {}, []
        for it in items:
            k = (it["task_mode"], it["meter_id"])
            if seen.get(k, 0) < args.per_cell:
                picked.append(it)
                seen[k] = seen.get(k, 0) + 1
        items = picked
    RUNS.mkdir(parents=True, exist_ok=True)
    for spec in get_specs([k.strip() for k in args.models.split(",") if k.strip()]):
        log = RUNS / f"{spec.key}__run1.jsonl"
        done = load_completed(log)
        if args.errors_only and log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add(str(r["item_id"]))
        todo = [it for it in items if it["id"] not in done]
        print(f"{spec.display}: {len(todo)} to run ({len(done)} cached), "
              f"max_tokens={spec.max_tokens}", flush=True)
        if args.dry_run or not todo:
            continue
        try:
            gen = get_generator(spec, log)
        except RuntimeError as e:
            print(f"  [skip] {spec.display}: {e}")
            continue

        def _run(it):
            return it, gen.generate("prosody-cloze-edit", it["id"], render_prompt(it),
                                    SYSTEM, spec.max_tokens)

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = [ex.submit(_run, it) for it in todo]
            for n, f in enumerate(as_completed(futs), 1):
                it, call = f.result()
                c = call if isinstance(call, dict) else call.__dict__
                flag = f"ERR {str(c['error'])[:80]}" if c["error"] else \
                    f"ok out={c['output_tokens']} tot={c.get('total_tokens')} {c['latency_s']:.0f}s"
                print(f"  [{n}/{len(todo)}] {it['id']:<24} {flag}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
