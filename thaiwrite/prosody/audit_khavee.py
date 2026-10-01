"""khavee capability audit harness.

Measures what `pythainlp.khavee` (5.3.1) can and cannot verify at meter scale,
which determines what the rule grader delegates to khavee and what it implements
itself (see docs/khavee-audit.md). Three sub-audits, kept separate so a tokenizer
error cannot mask a verifier success:

  A. Tokenizer: `syllable_tokenize` (dict, han_solo) vs gold per-วรรค counts.
  B. Verifier: check_aek_too / check_karu_lahu / is_sumpus on GOLD syllables,
     with expectations derived from `meter_specs`.
  C. check_klon: khavee's built-in กลอนแปด validator, characterized.

Run from the repository root:
  python -m thaiwrite.prosody.audit_khavee
or run the file directly: python thaiwrite/prosody/audit_khavee.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# allow `python thaiwrite/prosody/audit_khavee.py` (adds the repo root to sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pythainlp.khavee import KhaveeVerifier
from pythainlp.tokenize import syllable_tokenize

from thaiwrite.prosody import meter_specs as M
from thaiwrite.prosody.data.audit_poems import POEMS, KARU_LAHU_PROBES, Poem

KV = KhaveeVerifier()
ENGINES = ("dict", "han_solo")


# --- position resolution ---------------------------------------------------

def _norm(idx: int, n: int) -> int:
    return idx if idx >= 0 else n + idx


def syll_at(line: tuple[str, ...], idx: int) -> str | None:
    i = _norm(idx, len(line))
    return line[i] if 0 <= i < len(line) else None


def sylls_in_range(line: tuple[str, ...], rng: M.PositionRange) -> list[str]:
    n = len(line)
    s, e = _norm(rng.start, n), _norm(rng.end, n)
    return [line[i] for i in range(s, e + 1) if 0 <= i < n]


# --- A. tokenizer sub-audit ------------------------------------------------

def audit_tokenizer() -> dict:
    rows = []
    totals = {eng: {"count_ok": 0, "exact": 0, "n": 0} for eng in ENGINES}
    for p in POEMS:
        for li, gold in enumerate(p.lines):
            for eng in ENGINES:
                try:
                    got = syllable_tokenize("".join(gold), engine=eng)
                except Exception as exc:  # noqa: BLE001
                    rows.append((p.source, li, eng, "ERROR", str(exc)[:40]))
                    totals[eng]["n"] += 1
                    continue
                count_ok = len(got) == len(gold)
                exact = list(got) == list(gold)
                totals[eng]["n"] += 1
                totals[eng]["count_ok"] += count_ok
                totals[eng]["exact"] += exact
                if not count_ok:
                    rows.append((p.source[:24], li, eng,
                                 f"len {len(got)}≠{len(gold)}", " ".join(got)))
    return {"rows": rows, "totals": totals}


# --- B. verifier sub-audit -------------------------------------------------

def audit_aek_too() -> dict:
    res = {"pass": 0, "pass_dead": 0, "fail": 0, "n": 0, "details": []}
    for p in POEMS:
        spec = M.METER_SPECS_BY_ID[p.meter_id]
        for tc in spec.tone_constraints:
            syll = syll_at(p.lines[tc.position.line_index], tc.position.syllable_index)
            if syll is None:
                continue
            exp = tc.mark.value
            got = KV.check_aek_too(syll)
            got_dead = KV.check_aek_too(syll, dead_syllable_as_aek=True)
            res["n"] += 1
            ok = got == exp
            ok_dead = got_dead == exp
            res["pass"] += ok
            res["pass_dead"] += ok_dead
            if not ok:
                res["fail"] += 1
                res["details"].append((p.meter_id, syll, exp, str(got), str(got_dead)))
    return res


def audit_karu_lahu() -> dict:
    res = {"pass": 0, "fail": 0, "n": 0, "details": []}
    for p in POEMS:
        spec = M.METER_SPECS_BY_ID[p.meter_id]
        for wp in spec.weight_patterns:
            line = p.lines[wp.line_index]
            for i, w in enumerate(wp.pattern):
                syll = syll_at(line, i)
                if syll is None:
                    continue
                exp = w.value
                got = KV.check_karu_lahu(syll)
                res["n"] += 1
                ok = got == exp
                res["pass"] += ok
                if not ok:
                    res["fail"] += 1
                    res["details"].append((p.meter_id, syll, exp, str(got)))
    return res


def audit_karu_lahu_probes() -> dict:
    res = {"pass": 0, "n": 0, "details": []}
    for exp, sylls in KARU_LAHU_PROBES.items():
        for s in sylls:
            got = KV.check_karu_lahu(s)
            res["n"] += 1
            ok = got == exp
            res["pass"] += ok
            if not ok:
                res["details"].append((s, exp, str(got)))
    return res


def audit_rhyme() -> dict:
    res = {"pass": 0, "fail": 0, "n": 0, "details": []}
    for p in POEMS:
        if not p.eval_rhyme:
            continue
        spec = M.METER_SPECS_BY_ID[p.meter_id]
        for rl in spec.rhyme_links:
            src = syll_at(p.lines[rl.source.line_index], rl.source.syllable_index)
            if src is None:
                continue
            cands = []
            for rng in rl.targets:
                cands += sylls_in_range(p.lines[rng.line_index], rng)
            matched = [c for c in cands if KV.is_sumpus(src, c)]
            res["n"] += 1
            ok = bool(matched)
            res["pass"] += ok
            if not ok:
                res["fail"] += 1
            res["details"].append((p.meter_id, src, cands, matched, ok))
    return res


# --- C. check_klon characterization ---------------------------------------

def audit_check_klon() -> list:
    out = []
    for p in POEMS:
        if p.meter_id != "klon_paet":
            continue
        text = " ".join("".join(line) for line in p.lines)
        try:
            r = KV.check_klon(text, k_type=8)
        except Exception as exc:  # noqa: BLE001
            r = f"ERROR: {type(exc).__name__}: {exc}"
        out.append((p.source[:30], r))
    return out


# --- driver ----------------------------------------------------------------

def main() -> None:
    tok = audit_tokenizer()
    aek = audit_aek_too()
    kl = audit_karu_lahu()
    klp = audit_karu_lahu_probes()
    rh = audit_rhyme()
    ck = audit_check_klon()

    print("=" * 70)
    print("A. TOKENIZER (gold per-วรรค count / exact-boundary match)")
    for eng, t in tok["totals"].items():
        print(f"  {eng:9} count {t['count_ok']}/{t['n']}   exact {t['exact']}/{t['n']}")
    if tok["rows"]:
        print("  count mismatches:")
        for src, li, eng, what, got in tok["rows"]:
            print(f"    [{src}] line{li} {eng}: {what} -> {got}")

    print("=" * 70)
    print(f"B1. check_aek_too (โคลง tone positions): "
          f"{aek['pass']}/{aek['n']} plain, {aek['pass_dead']}/{aek['n']} dead_as_aek")
    for m, syll, exp, got, gotd in aek["details"]:
        print(f"    {m}: {syll!r} exp {exp} got {got} / dead {gotd}")

    print(f"B2. check_karu_lahu (ฉันท์ pattern positions): {kl['pass']}/{kl['n']}")
    for m, syll, exp, got in kl["details"]:
        print(f"    {m}: {syll!r} exp {exp} got {got}")

    print(f"B2b. check_karu_lahu (clean labeled probes): {klp['pass']}/{klp['n']}")
    for s, exp, got in klp["details"]:
        print(f"    {s!r} exp {exp} got {got}")

    print(f"B3. is_sumpus (rhyme links): {rh['pass']}/{rh['n']} detected")
    for m, src, cands, matched, ok in rh["details"]:
        flag = "OK" if ok else "MISS"
        print(f"    [{flag}] {m}: {src!r} ~ {cands} -> matched {matched}")

    print("=" * 70)
    print("C. check_klon (built-in กลอนแปด validator):")
    for src, r in ck:
        print(f"    [{src}] -> {r}")
    print("=" * 70)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
