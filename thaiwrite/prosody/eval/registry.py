"""Model registry for the composition evaluation. Adding a model = one ModelSpec line.

`provider` is the only branch point downstream:
  - "openai_compat"    : OpenAI-compatible /chat/completions (ThaiLLM gateway,
                         OpenTyphoon, OpenRouter, Gemini). Needs base_url.
  - "openai_responses" : OpenAI SDK Responses API (gpt-5.x), base_url=None.
  - "anthropic"        : Anthropic Messages API (Claude models).

Keys are referenced by env-var NAME only (never the value). The ThaiLLM gateway
authenticates with THAILLM_API_KEY; OpenTyphoon with TYPHOON_API_KEY.
"""
from __future__ import annotations

from dataclasses import dataclass

# OpenRouter courtesy headers (mirrors pilot_rerun.clients.openrouter_*).
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/yongsa-nut/ThaiProsody",
    "X-Title": "ThaiProsody",
}

_THAI = "THAILLM_API_KEY"
_GATEWAY = "http://thaillm.or.th/api/v1"          # ThaiLLM national gateway (http, as provided)
_OPENAI = "OPENAI_API_KEY"
_OROUTER = "OPENROUTER_API_KEY"
_NOTHINK = {"enabled": False}     # OpenRouter: disable reasoning (avoids CoT eating the budget)
_GEMINI = "GEMINI_API_KEY"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"   # OpenAI-compat
_ANTHROPIC = "ANTHROPIC_API_KEY"
_OR_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    key: str                       # CLI handle + filename stem (e.g. "gpt55-med")
    display: str                   # report label
    provider: str                  # "openai_compat" | "openai_responses" | "anthropic"
    model_id: str                  # provider model string
    key_env: str                   # env var NAME holding the API key
    base_url: str | None = None    # required for openai_compat
    rpm: int = 60
    reasoning: str | None = None   # OpenAI Responses effort: "low"|"medium"|"high"
    or_reasoning: dict | None = None  # OpenRouter reasoning control, e.g. {"max_tokens": 8000}
    extra_headers: dict | None = None
    max_tokens: int = 900          # headroom for 2-stanza poems
    request_timeout: int = 120     # raise for slow reasoning models (openai_compat)
    enabled: bool = True           # disabled specs are kept for reference, not run by default
    note: str = ""


_SPECS: list[ModelSpec] = [
    # --- ThaiLLM national gateway (OpenAI-compatible) -----------------------
    ModelSpec("openthaigpt", "OpenThaiGPT 8B v7.2", "openai_compat",
              "openthaigpt-thaillm-8b-instruct-v7.2", _THAI, base_url=_GATEWAY,
              max_tokens=2048, note="ThaiLLM gateway"),
    ModelSpec("pathumma", "Pathumma Qwen3-8B (think)", "openai_compat",
              "pathumma-thaillm-qwen3-8b-think-3.0.0", _THAI, base_url=_GATEWAY,
              max_tokens=4096, enabled=False,
              note="disabled: with thinking on, budgets <=4096 truncate before </think> and "
                   ">=8192 time out at the gateway (~5 min)"),
    ModelSpec("typhoon-s", "Typhoon-S 8B", "openai_compat",
              "typhoon-s-thaillm-8b-instruct", _THAI, base_url=_GATEWAY,
              max_tokens=2048, note="ThaiLLM gateway"),
    ModelSpec("thalle", "THaLLE 0.2 8B (KBTG)", "openai_compat",
              "thalle-0.2-thaillm-8b-fa", _THAI, base_url=_GATEWAY,
              max_tokens=8192, note="ThaiLLM gateway / KBTG; also a thinking model"),

    # --- Typhoon 2.5 (OpenTyphoon API; also used in the calibration pilot) ----
    # OpenTyphoon authenticates with TYPHOON_API_KEY (the gateway key is not accepted).
    ModelSpec("typhoon25", "Typhoon 2.5 (30B)", "openai_compat",
              "typhoon-v2.5-30b-a3b-instruct", "TYPHOON_API_KEY",
              base_url="https://api.opentyphoon.ai/v1", rpm=200),

    # --- OpenAI (Responses API) ---------------------------------------------
    # Reasoning tokens count against max_output_tokens, so the budget must cover
    # reasoning + the visible poem, else output_text comes back empty
    # (a 4000-token budget left some medium-effort outputs empty).
    ModelSpec("gpt55-low", "GPT-5.5 (low)", "openai_responses", "gpt-5.5",
              _OPENAI, reasoning="low", max_tokens=20000),
    ModelSpec("gpt55-med", "GPT-5.5 (medium)", "openai_responses", "gpt-5.5",
              _OPENAI, reasoning="medium", max_tokens=20000),
    ModelSpec("gpt55-high", "GPT-5.5 (high)", "openai_responses", "gpt-5.5",
              _OPENAI, reasoning="high", max_tokens=24000),
    ModelSpec("gpt54", "GPT-5.4", "openai_responses", "gpt-5.4", _OPENAI),
    ModelSpec("gpt54-mini", "GPT-5.4-mini", "openai_responses", "gpt-5.4-mini", _OPENAI),

    # --- OpenRouter DeepSeek V4 -----------------------------------------------
    # Both are reasoning-native: reasoning_tokens count against max_tokens and the
    # poem (`content`) is emitted only after reasoning finishes, so the budget must
    # be large enough for reasoning to complete, else content is empty (finish=length).
    ModelSpec("deepseek-v4-pro", "DeepSeek V4 Pro", "openai_compat",
              "deepseek/deepseek-v4-pro", _OROUTER, base_url="https://openrouter.ai/api/v1",
              extra_headers=OPENROUTER_HEADERS, rpm=60, max_tokens=24000,
              request_timeout=600, note="thinking ON natively (slow ~3-4min/call)"),
    ModelSpec("deepseek-v4-flash", "DeepSeek V4 Flash", "openai_compat",
              "deepseek/deepseek-v4-flash", _OROUTER, base_url="https://openrouter.ai/api/v1",
              extra_headers=OPENROUTER_HEADERS, rpm=60, max_tokens=24000,
              request_timeout=600, note="thinking ON natively; `:free` variant also exists"),

    # --- Gemini (Google OpenAI-compat endpoint) -----------------------------
    ModelSpec("gemini-3.5-flash", "Gemini 3.5 Flash", "openai_compat",
              "gemini-3.5-flash", _GEMINI, base_url=_GEMINI_URL, rpm=60,
              max_tokens=24000, request_timeout=600, note="Gemini API (OpenAI-compat); thinks"),

    # --- Claude via the Anthropic Messages API --------------------------------
    ModelSpec("sonnet-4.6", "Claude Sonnet 4.6", "anthropic",
              "claude-sonnet-4-6", _ANTHROPIC, rpm=50, max_tokens=2000,
              request_timeout=120),
    ModelSpec("opus-4.7", "Claude Opus 4.7", "anthropic",
              "claude-opus-4-7", _ANTHROPIC, rpm=30, max_tokens=2000,
              request_timeout=180),

    # --- OpenRouter open-weight models ---------------------------------------
    ModelSpec("kimi-k2.6", "Kimi K2.6", "openai_compat", "moonshotai/kimi-k2.6",
              _OROUTER, base_url=_OR_URL, extra_headers=OPENROUTER_HEADERS, rpm=60,
              max_tokens=24000, request_timeout=600, enabled=False,
              note="disabled by default: about half of items exhaust the 24k reasoning budget "
                   "without emitting a poem"),
    # Native config = thinking on (generous budget). `run_models --no-think` runs the
    # ablation (reasoning.enabled=false). In thinking mode, token runout scores 0.
    ModelSpec("glm-5.1", "GLM 5.1", "openai_compat", "z-ai/glm-5.1",
              _OROUTER, base_url=_OR_URL, extra_headers=OPENROUTER_HEADERS, rpm=60,
              max_tokens=24000, request_timeout=600),
    ModelSpec("minimax-m2.7", "MiniMax M2.7", "openai_compat", "minimax/minimax-m2.7",
              _OROUTER, base_url=_OR_URL, extra_headers=OPENROUTER_HEADERS, rpm=60,
              max_tokens=24000, request_timeout=600,
              note="reasoning-only (can't disable); thinking mode, runout=fail"),
    ModelSpec("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B", "openai_compat", "qwen/qwen3.6-35b-a3b",
              _OROUTER, base_url=_OR_URL, extra_headers=OPENROUTER_HEADERS, rpm=60,
              max_tokens=24000, request_timeout=600, note="thinking; <think> block stripped"),
    # Base-model control: the ThaiLLM-gateway 8B models are continued-pretrained
    # from Qwen3-8B (explicit in Pathumma's model id).
    ModelSpec("qwen3-8b", "Qwen3 8B", "openai_compat", "qwen/qwen3-8b",
              _OROUTER, base_url=_OR_URL, extra_headers=OPENROUTER_HEADERS, rpm=60,
              max_tokens=24000, request_timeout=600,
              note="base model of the Thai-CPT 8Bs; hybrid reasoner, run native + --no-think"),
    ModelSpec("gemma-4-31b", "Gemma 4 31B", "openai_compat", "google/gemma-4-31b-it",
              _OROUTER, base_url=_OR_URL, extra_headers=OPENROUTER_HEADERS, rpm=60,
              max_tokens=2000, request_timeout=120, note="non-thinking"),
]

REGISTRY: dict[str, ModelSpec] = {s.key: s for s in _SPECS}


def resolve(keys: list[str] | None = None) -> list[ModelSpec]:
    """Named keys to specs (errors clearly on unknown/disabled/unconfigured).
    No keys: every enabled spec in registry order.
    """
    if not keys:
        return [s for s in _SPECS if s.enabled]
    out: list[ModelSpec] = []
    for k in keys:
        s = REGISTRY.get(k)
        if s is None:
            raise ValueError(f"unknown model key {k!r}; choices: {sorted(REGISTRY)}")
        if not s.enabled:
            raise ValueError(f"model {k!r} is disabled: {s.note or 'not configured'}")
        if not s.model_id:
            raise ValueError(f"model {k!r} has no model_id: {s.note or 'not configured'}")
        if s.provider == "openai_compat" and not s.base_url:
            raise ValueError(f"model {k!r} (openai_compat) has no base_url")
        out.append(s)
    return out
