"""Provider-agnostic generation interface for the evaluation harness.

`make_generator(spec, log_path)` returns an object with
`.generate(benchmark, item_id, prompt, system, max_tokens) -> Call`. All impls
log resume-safe JSONL in the `Call` schema, so `load_completed` gates every path.

- CompatGenerator    : wraps pilot_rerun.clients.OpenAICompatClient (retry,
                       rate-limit, JSONL log already inside `.chat`).
- ResponsesGenerator : gpt-5.x via the OpenAI SDK Responses API with a chat
                       fallback, mirroring build_calibration_set.run_gpt54mini but
                       resume-safe by logging each Call through the same writer.
- AnthropicGenerator : Claude via the Anthropic Messages API.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from pilot_rerun.clients import Call, OpenAICompatClient

from .registry import ModelSpec


#: per-key-env fallback chain (used only when the primary env var is unset)
_KEY_FALLBACK = {
    "OPENAI_API_KEY2": "OPENAI_API_KEY",
    "TYPHOON_API_KEY": "THAILLM_API_KEY",
}


def _resolve_key(spec: ModelSpec) -> str:
    key = os.environ.get(spec.key_env, "")
    if not key and spec.key_env in _KEY_FALLBACK:
        key = os.environ.get(_KEY_FALLBACK[spec.key_env], "")
    return key


class CompatGenerator:
    def __init__(self, spec: ModelSpec, log_path: Path):
        self.spec = spec
        self.model = spec.model_id
        self.log_path = log_path
        self.client = OpenAICompatClient(
            api_key=_resolve_key(spec), base_url=spec.base_url or "",
            model=spec.model_id, rpm=spec.rpm, log_path=log_path,
            extra_headers=spec.extra_headers, request_timeout=spec.request_timeout,
            reasoning=spec.or_reasoning,
        )

    def generate(self, benchmark, item_id, prompt, system, max_tokens) -> Call:
        return self.client.chat(benchmark, item_id, prompt, system=system,
                                max_tokens=max_tokens, temperature=0.0)


class ResponsesGenerator:
    """OpenAI Responses API (chat fallback on SDK shape mismatch); no temperature."""

    def __init__(self, spec: ModelSpec, log_path: Path):
        from openai import OpenAI
        key = _resolve_key(spec)
        if not key:
            raise RuntimeError(f"{spec.key_env} (or OPENAI_API_KEY) not set in environment/.env")
        self.spec = spec
        self.model = spec.model_id
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(api_key=key)
        self._log_lock = threading.Lock()

    def _log(self, call: Call) -> None:
        with self._log_lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(call.__dict__, ensure_ascii=False) + "\n")

    def generate(self, benchmark, item_id, prompt, system, max_tokens) -> Call:
        t0 = time.time()
        resp, err, in_tok, out_tok, finish = "", None, None, None, None
        use_chat = False
        for attempt in range(4):
            try:
                if use_chat:
                    kw = dict(model=self.model, max_completion_tokens=max_tokens,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}])
                    if self.spec.reasoning:
                        kw["reasoning_effort"] = self.spec.reasoning
                    r = self.client.chat.completions.create(**kw)
                    resp = r.choices[0].message.content or ""
                    finish = r.choices[0].finish_reason
                    u = getattr(r, "usage", None)
                    if u is not None:
                        in_tok = getattr(u, "prompt_tokens", None)
                        out_tok = getattr(u, "completion_tokens", None)
                else:
                    kw = dict(model=self.model, instructions=system, input=prompt,
                              max_output_tokens=max_tokens)
                    if self.spec.reasoning:
                        kw["reasoning"] = {"effort": self.spec.reasoning}
                    r = self.client.responses.create(**kw)
                    resp = r.output_text
                    finish = getattr(r, "status", None)
                    inc = getattr(r, "incomplete_details", None)
                    if inc is not None:
                        finish = f"incomplete:{getattr(inc, 'reason', inc)}"
                    u = getattr(r, "usage", None)
                    if u is not None:
                        in_tok = getattr(u, "input_tokens", None)
                        out_tok = getattr(u, "output_tokens", None)
                err = None
                break
            except TypeError as e:        # SDK shape mismatch: switch to chat path
                err, use_chat = f"{type(e).__name__}: {e}", True
            except Exception as e:        # noqa: BLE001 (transient API error)
                err = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt + 1, 30))
        call = Call(benchmark=benchmark, model=self.model, item_id=str(item_id),
                    prompt=prompt, response=resp, latency_s=round(time.time() - t0, 2),
                    input_tokens=in_tok, output_tokens=out_tok, error=err,
                    reasoning=None, finish_reason=finish)
        self._log(call)
        return call


class AnthropicGenerator:
    """Claude via the Anthropic Messages API (system is a top-level param)."""

    def __init__(self, spec: ModelSpec, log_path: Path):
        from anthropic import Anthropic
        key = _resolve_key(spec)
        if not key:
            raise RuntimeError(f"{spec.key_env} not set in environment/.env")
        self.spec = spec
        self.model = spec.model_id
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = Anthropic(api_key=key)
        self._log_lock = threading.Lock()

    def _log(self, call: Call) -> None:
        with self._log_lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(call.__dict__, ensure_ascii=False) + "\n")

    def generate(self, benchmark, item_id, prompt, system, max_tokens) -> Call:
        t0 = time.time()
        resp, err, in_tok, out_tok, finish = "", None, None, None, None
        for attempt in range(4):
            try:
                # temperature is omitted: newer Anthropic models (e.g. Opus 4.7) reject it
                r = self.client.messages.create(
                    model=self.model, max_tokens=max_tokens,
                    system=system, messages=[{"role": "user", "content": prompt}])
                resp = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
                finish = r.stop_reason
                u = getattr(r, "usage", None)
                if u is not None:
                    in_tok = getattr(u, "input_tokens", None)
                    out_tok = getattr(u, "output_tokens", None)
                err = None
                break
            except Exception as e:        # noqa: BLE001 (transient API error)
                err = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt + 1, 30))
        call = Call(benchmark=benchmark, model=self.model, item_id=str(item_id),
                    prompt=prompt, response=resp, latency_s=round(time.time() - t0, 2),
                    input_tokens=in_tok, output_tokens=out_tok, error=err,
                    reasoning=None, finish_reason=finish)
        self._log(call)
        return call


def make_generator(spec: ModelSpec, log_path: Path):
    if spec.provider == "openai_compat":
        return CompatGenerator(spec, log_path)
    if spec.provider == "openai_responses":
        return ResponsesGenerator(spec, log_path)
    if spec.provider == "anthropic":
        return AnthropicGenerator(spec, log_path)
    raise ValueError(f"unknown provider {spec.provider!r}")
