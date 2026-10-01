"""LLM-as-judge coherence grader: Layer 2 of prosody grading.

A poem can satisfy every meter rule (Layer 1, `grader_meter.py`) and still be
semantically empty. This module rates **content** on two pointwise axes, each an
integer 1-5:
  * topicality: does the poem address the given topic?
  * coherence: are the Thai lines intelligible and connected?
Composite = mean of the two axes. Rubric + hard constraints live in the versioned
`grader_coherence_prompt.txt` (do NOT judge meter/tone/rhyme; do NOT penalize rare
vocabulary; JSON only).

Default backend: OpenAI `gpt-5.5`, reasoning effort `medium`, key `OPENAI_API_KEY2`
(falls back to `OPENAI_API_KEY`). `judge` is a small interface so other backends can
be dropped in (the calibration scripts compare judges against expert labels). The
content judge whose scores are reported in the paper is `judge_api.py` (Claude Opus
5.5 through the Claude API).

Live API test:  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/test_grader_coherence.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from dotenv import load_dotenv

# --- env + prompt loading --------------------------------------------------

def _load_env() -> None:
    """Walk up from this file to the first `.env` (workspace root) and load it."""
    for parent in Path(__file__).resolve().parents:
        env = parent / ".env"
        if env.exists():
            load_dotenv(env)
            return


_load_env()

_PROMPT_PATH = Path(__file__).with_name("grader_coherence_prompt.txt")
_raw_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
if _raw_prompt.startswith("# version:"):
    _first, _, _rest = _raw_prompt.partition("\n")
    PROMPT_VERSION = _first.split(":", 1)[1].strip()
    PROMPT_TEMPLATE = _rest.lstrip("\n")
else:
    PROMPT_VERSION = "unversioned"
    PROMPT_TEMPLATE = _raw_prompt

_SYSTEM = ("You are an expert evaluator of Thai poetry. Follow the rubric exactly "
           "and output only the requested JSON.")

# Strict structured-output schema. Numeric range is enforced by clamping in code
# (OpenAI strict mode does not enforce min/max), so the schema keeps types only.
_SCHEMA = {
    "type": "object",
    "properties": {
        "topicality": {"type": "integer"},
        "coherence": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["topicality", "coherence", "rationale"],
    "additionalProperties": False,
}


# --- result type -----------------------------------------------------------

@dataclass
class CoherenceResult:
    item_id: str
    topicality: Optional[int]
    coherence: Optional[int]
    composite: Optional[float]
    rationale: str
    model: str
    latency_s: float
    error: Optional[str]
    raw: str


def _clamp(v: object) -> int:
    return max(1, min(5, int(v)))


# --- backend seam ----------------------------------------------------------

class Judge(Protocol):
    model: str
    def judge(self, prompt: str, system: str) -> str: ...  # returns raw JSON text


class OpenAIJudge:
    """Default backend — OpenAI Responses API with structured output + reasoning.

    Tries the Responses API first; if its structured-output shape is rejected for
    this model, falls back to chat.completions with `response_format` json_schema.
    """

    def __init__(self, model: str = "gpt-5.5", reasoning: str = "medium",
                 key_env: str = "OPENAI_API_KEY2", max_retries: int = 4,
                 max_output_tokens: int = 4096):
        from openai import OpenAI
        key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(f"{key_env} (or OPENAI_API_KEY) not set in environment/.env")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.reasoning = reasoning
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens

    def _via_responses(self, prompt: str, system: str) -> str:
        resp = self.client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning},
            instructions=system,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
            text={"format": {"type": "json_schema", "name": "coherence",
                             "schema": _SCHEMA, "strict": True}},
        )
        return resp.output_text

    def _via_chat(self, prompt: str, system: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "coherence", "schema": _SCHEMA,
                                             "strict": True}},
        )
        return resp.choices[0].message.content

    def judge(self, prompt: str, system: str) -> str:
        last: Optional[Exception] = None
        use_chat = False
        for attempt in range(self.max_retries):
            try:
                return self._via_chat(prompt, system) if use_chat else self._via_responses(prompt, system)
            except TypeError as exc:        # SDK shape mismatch for this model
                last = exc
                use_chat = True             # switch path, don't burn the backoff
            except Exception as exc:        # noqa: BLE001 (transient API errors)
                last = exc
                time.sleep(min(2 ** attempt + 1, 30))
        raise last if last else RuntimeError("judge failed")


# --- public API ------------------------------------------------------------

def judge_coherence(poem: str, item: dict, judge: Optional[Judge] = None,
                    model: str = "gpt-5.5", reasoning: str = "medium") -> CoherenceResult:
    """Score one poem's content. `item` supplies `topic` (and optional `id`).
    Returns a `CoherenceResult`; on any failure, axes are None and `error` is set.
    """
    judge = judge or OpenAIJudge(model=model, reasoning=reasoning)
    item_id = str(item.get("id", ""))
    prompt = PROMPT_TEMPLATE.replace("{topic}", str(item.get("topic", ""))).replace("{poem}", poem)
    t0 = time.time()
    raw = ""
    try:
        raw = judge.judge(prompt, _SYSTEM)
        data = json.loads(raw)
        top, coh = _clamp(data["topicality"]), _clamp(data["coherence"])
        return CoherenceResult(item_id, top, coh, (top + coh) / 2,
                               str(data.get("rationale", "")), getattr(judge, "model", model),
                               round(time.time() - t0, 2), None, raw)
    except Exception as exc:  # noqa: BLE001
        return CoherenceResult(item_id, None, None, None, "", model,
                               round(time.time() - t0, 2), f"{type(exc).__name__}: {exc}", raw)
