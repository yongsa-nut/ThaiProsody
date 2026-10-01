"""Thai prosody benchmark (classical-meter mechanics).

`meter_specs` holds formal, cited encodings of the four meters the benchmark
covers; `grader_meter` is the rule-based meter grader; `grader_coherence` and
`judge_api` provide LLM content judging; `eval/` runs and grades models on the
composition task and `cloze_edit/` on the cloze and edit tasks.
"""
