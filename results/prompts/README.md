# Prompt corpus is NOT published

The prompts are sliced deterministically from a private documentation tree and from the
llama.cpp source checkout (see `harness/build_prompts.py` and `harness/mkprompts.py`,
seed 20261117). They are not published because the documentation corpus is not public.
Rebuild them with the same inputs and the same seed to reproduce the arms; the exact
sha256 of every prompt file used is in `manifest.json`, and the engine-reported token
counts of every request are in `results/results_raw.jsonl`.
