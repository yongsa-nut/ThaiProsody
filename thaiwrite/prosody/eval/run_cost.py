"""Per-model cost report from captured token counts x pricing.

Prices are USD per 1M tokens (prompt, completion), as listed on OpenRouter /models
in May 2026 (direct-API models are priced at their OpenRouter-equivalent slug as a
close proxy). ThaiLLM-gateway models and OpenTyphoon were free to use.

  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.run_cost                 # all run files
  PYTHONIOENCODING=utf-8 python -m thaiwrite.prosody.eval.run_cost --full 160      # + projection to N items/model
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"

#: (prompt $/Mtok, completion $/Mtok); None = free
PRICE: dict[str, tuple[float, float] | None] = {
    "deepseek-v4-pro": (0.435, 0.870), "deepseek-v4-flash": (0.100, 0.200),
    "qwen3.6-35b-a3b": (0.150, 1.000), "glm-5.1": (0.980, 3.080),
    "minimax-m2.7": (0.279, 1.200), "kimi-k2.6": (0.730, 3.490),
    "gemma-4-31b": (0.120, 0.370),
    "gpt55-low": (5.0, 30.0), "gpt55-med": (5.0, 30.0), "gpt55-high": (5.0, 30.0),
    "gpt54-mini": (0.750, 4.500),
    "gemini-3.5-flash": (1.500, 9.000), "sonnet-4.6": (3.000, 15.000),
    "typhoon25": None, "openthaigpt": None, "pathumma": None,
    "typhoon-s": None, "thalle": None,
}


def _price_for(key: str):
    return PRICE.get(key.replace(".nothink", ""), None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-model cost from token counts × pricing.")
    ap.add_argument("--full", type=int, default=160,
                    help="project total cost for N items/model (default 160 = compose)")
    args = ap.parse_args()

    agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "priced": True})
    for f in sorted(RUNS_DIR.glob("*__run*.jsonl")):
        key = f.name.split("__run")[0]
        price = _price_for(key)
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            a = agg[key]
            a["calls"] += 1
            it = r.get("input_tokens") or 60                      # short prompt estimate
            ot = r.get("output_tokens") or (len(r.get("response") or "") // 3)
            a["in"] += it
            a["out"] += ot
            if price is None:
                a["priced"] = False
            else:
                a["cost"] += (it * price[0] + ot * price[1]) / 1e6

    print(f"{'model':22}{'calls':>6}{'in_tok':>10}{'out_tok':>11}{'$ so far':>10}{'$/call':>9}"
          f"{f'$/{args.full} items':>14}")
    total = 0.0
    for key in sorted(agg):
        a = agg[key]
        if not a["priced"]:
            print(f"{key:22}{a['calls']:>6}{a['in']:>10}{a['out']:>11}{'free':>10}{'free':>9}{'free':>14}")
            continue
        total += a["cost"]
        per = a["cost"] / a["calls"] if a["calls"] else 0
        print(f"{key:22}{a['calls']:>6}{a['in']:>10}{a['out']:>11}"
              f"{a['cost']:>10.4f}{per:>9.4f}{per * args.full:>14.2f}")
    print(f"\nTotal cost of logged calls (priced models): ${total:.4f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
