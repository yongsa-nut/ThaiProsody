"""HTTP clients for OpenAI-compatible chat endpoints (OpenRouter, OpenTyphoon,
ThaiLLM gateway), used by the evaluation harness.

Both expose OpenAI-compatible chat completions. Shared features:
  - Rate limit throttling (sleep-based token bucket per client).
  - Exponential-backoff retries on 429/5xx.
  - Per-call JSONL logging so interrupted runs resume without re-charging.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests
from dotenv import load_dotenv


# Load API keys from the nearest .env in the repository root or a parent directory.
for _parent in Path(__file__).resolve().parents:
    if (_parent / ".env").exists():
        load_dotenv(_parent / ".env")
        break


@dataclass
class Call:
    benchmark: str
    model: str
    item_id: str
    prompt: str
    response: str
    latency_s: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    error: Optional[str]
    reasoning: Optional[str] = None   # CoT trace if the model returns one (for analysis)
    finish_reason: Optional[str] = None


class RateLimiter:
    """Minimal spacing-based limiter: pause so requests are at least `interval` apart."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        delta = time.monotonic() - self.last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self.last = time.monotonic()


class OpenAICompatClient:
    base_url: str
    model: str
    limiter: RateLimiter
    log_path: Path

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        rpm: int,
        log_path: Path,
        extra_headers: Optional[dict] = None,
        request_timeout: int = 120,
        reasoning: Optional[dict] = None,
    ):
        if not api_key:
            raise RuntimeError("missing API key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.limiter = RateLimiter(rpm)
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.extra_headers = extra_headers or {}
        self.request_timeout = request_timeout  # raise for slow reasoning models
        self.reasoning = reasoning              # OpenRouter reasoning control, e.g.
        self._log_lock = threading.Lock()       # {"max_tokens": N} to reserve answer budget

    def chat(
        self,
        benchmark: str,
        item_id: str,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> Call:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning is not None:
            body["reasoning"] = self.reasoning
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        t0 = time.monotonic()
        error = None
        response_text = ""
        reasoning_text = None
        finish_reason = None
        in_tok = out_tok = None

        for attempt in range(max_retries):
            self.limiter.wait()
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as e:
                error = f"request_exception:{e}"
                time.sleep(min(2 ** attempt, 30))
                continue

            if r.status_code == 429 or r.status_code >= 500:
                error = f"status_{r.status_code}:{r.text[:200]}"
                time.sleep(min(2 ** attempt + 1, 60))
                continue

            if r.status_code != 200:
                error = f"status_{r.status_code}:{r.text[:500]}"
                break

            try:
                data = r.json()
                choice = data["choices"][0]
                msg = choice["message"]
                response_text = msg.get("content") or ""
                # reasoning-native models (DeepSeek V4, some gateway models) put CoT
                # in a separate field; keep it for error analysis.
                reasoning_text = msg.get("reasoning") or msg.get("reasoning_content")
                finish_reason = choice.get("finish_reason")
            except ValueError:        # non-JSON body (HTML error page / truncated stream)
                error = f"bad_response_body:{r.text[:300]}"
                break
            except (KeyError, IndexError, TypeError):
                error = f"bad_response_shape:{r.text[:300]}"
                break

            usage = data.get("usage") or {}
            in_tok = usage.get("prompt_tokens")
            out_tok = usage.get("completion_tokens")
            error = None
            break

        latency = time.monotonic() - t0
        call = Call(
            benchmark=benchmark,
            model=self.model,
            item_id=str(item_id),
            prompt=prompt,
            response=response_text,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            error=error,
            reasoning=reasoning_text,
            finish_reason=finish_reason,
        )
        self._log(call)
        return call

    def _log(self, call: Call) -> None:
        with self._log_lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(call.__dict__, ensure_ascii=False) + "\n")


def openrouter_gemma4_free(log_path: Path) -> OpenAICompatClient:
    return OpenAICompatClient(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        model="google/gemma-4-31b-it:free",
        rpm=20,
        log_path=log_path,
        extra_headers={
            "HTTP-Referer": "https://github.com/yongsa-nut/ThaiProsody",
            "X-Title": "ThaiProsody",
        },
    )


def openrouter_gemma4_paid(log_path: Path) -> OpenAICompatClient:
    return OpenAICompatClient(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        model="google/gemma-4-31b-it",
        rpm=120,
        log_path=log_path,
        extra_headers={
            "HTTP-Referer": "https://github.com/yongsa-nut/ThaiProsody",
            "X-Title": "ThaiProsody",
        },
    )


def typhoon_v25(log_path: Path) -> OpenAICompatClient:
    return OpenAICompatClient(
        api_key=os.environ.get("THAILLM_API_KEY", ""),
        base_url="https://api.opentyphoon.ai/v1",
        model="typhoon-v2.5-30b-a3b-instruct",
        rpm=200,
        log_path=log_path,
    )


def load_completed(log_path: Path) -> set[str]:
    """Return set of item_ids already logged successfully (for resume)."""
    done = set()
    if not log_path.exists():
        return done
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error") is None and rec.get("response"):
                done.add(str(rec["item_id"]))
    return done
