"""API spend of the cloze/edit runs, from logged token usage.

Prices ($ per 1M tokens, input / output) are the list prices used for the composition
runs (eval/run_cost.py); DeepSeek V4 Pro is priced at the OpenRouter list price and
Gemma 4 31B at its OpenRouter price as an upper bound (the Google AI API did not bill
per token for it). Gemini's OpenAI-compatible endpoint reports thinking tokens only
in total_tokens, so output = total - prompt where total is logged.

  PYTHONIOENCODING=utf-8 python thaiwrite/prosody/cloze_edit/cost_ce.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
PRICE = {"gpt55-med": (5.0, 30.0), "gemini-3.5-flash": (1.5, 9.0),
         "deepseek-v4-pro": (0.435, 0.870), "gemma-4-31b-think": (0.12, 0.37),
         # OpenRouter prices (September 2026). DeepSeek V4 Flash: effective price billed on a probe call
         # (the provider OpenRouter routed to charged ~1.28 $/M output, above the listed
         # 0.15); Gemma / Qwen3: listed prices (matched the billed cost on a probe call).
         "deepseek-v4-flash": (0.03, 1.28), "gemma-4-31b": (0.09, 0.34),
         "qwen3-8b": (0.117, 0.455),
         "typhoon25": None, "typhoon-s": None, "openthaigpt": None}
#: amounts actually billed where known (DeepSeek V4 Pro through the official DeepSeek
#: API; its official price is above the OpenRouter list price used above)
BILLED = {"deepseek-v4-pro": 5.39}


def main() -> dict:
    out, total = {}, 0.0
    for f in sorted(RUNS.glob("*__run1.jsonl")):
        key = f.name.split("__run")[0]
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        tin = tout = 0
        est = []
        for r in recs:
            i = r.get("input_tokens") or 0
            o = r.get("output_tokens") or 0
            if r.get("total_tokens"):
                o = max(o, r["total_tokens"] - i)
            elif key == "gemini-3.5-flash":
                est.append(r)
                continue
            tin += i
            tout += o
        if est:                       # early calls logged without total: use the mean
            n = len(recs) - len(est)
            tout += round(tout / max(n, 1)) * len(est)
            tin += sum(r.get("input_tokens") or 0 for r in est)
        p = PRICE.get(key)
        cost = (tin * p[0] + tout * p[1]) / 1e6 if p else 0.0
        cost = BILLED.get(key, cost)
        total += cost
        out[key] = {"calls": len(recs), "input_tokens": tin, "output_tokens": tout,
                    "usd": round(cost, 2)}
        print(f"{key:20} calls={len(recs):4} in={tin:>9,} out={tout:>11,} ${cost:7.2f}")
    print(f"total ${total:.2f}")
    out["_total_usd"] = round(total, 2)
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
