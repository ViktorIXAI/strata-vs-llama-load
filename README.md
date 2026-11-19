# strata-vs-llama-load — three inference engines, one RTX 5090 Laptop, one IQ3_S MoE

Benchmark of **Basalt 0.4.1** (production brain), **Strata 0.1.62** (fork candidate) and
**llama.cpp b12531** (external reference) on the *same* model — Qwen3.8-Flash-Next at IQ3_S,
a 54.5 GB GGUF on a 24 GB laptop GPU — using identical arms: context ladder, recall probes,
warm/KV-reuse turn, concurrency waves, an adversarial verification of the "500× cached prefill"
claim, and a 30-minute sustained load with 1 Hz power/thermal telemetry. A CPU appendix
([report §10](report.md)) additionally runs the **Microsoft BitNet engine** (llama.cpp fork v9918,
i2_s kernels) on its own *BitNet b1.58-2B-4T* ~2.4 B ternary model — an engine-class reference
explicitly **not** the same model, so its rows are never comparable to the IQ3_S columns.

Full write-up: **[report.md](report.md)**. Raw rows: [`results/`](results/). Harness:
[`harness/`](harness/). Serve logs + telemetry: [`logs/`](logs/).

## Verdict

* **llama.cpp is the wrong tool for this model on 24 GB VRAM — headline finding.**
  `-ngl 99` dies with `cudaMalloc failed: out of memory`; the only build that loads uses
  `--n-cpu-moe 48` (all experts in host RAM). Cold prefill **163–175 tok/s**, decode **21–28 tok/s**,
  **17.4 minutes** to first token at 172 k tokens. With 8 slots its context auto-fits down to
  33,792 tokens/slot, so every 32 k concurrency arm is rejected (`request (39385 tokens) exceeds
  the available context size (33792 tokens)`). Recall is still 5/5 — the problem is wall-clock,
  not correctness.
* **Strata wins cold long-context prefill, concurrency, energy-per-token and cache transparency.**
  Flat **~3.2–3.3 k tok/s** prefill from 39 k to 189 k tokens; best c=8 aggregate (**126.5 tok/s**);
  **1.20 J / answer token** vs Basalt 1.73; stable for 30 min (1285 requests, drift +2 %).
* **Basalt wins single-stream decode** (engine-reported 139–172 tok/s vs Strata 94–156) and has a
  **persistent disk conversation cache** (20 GiB) that survives restarts. Its ladder TTFTs in this
  run are cache-served, not cold — labelled as such in the report.
* Nothing throttled thermally on this laptop GPU: `SW/HW Thermal Slowdown` never fired; only
  `SW Power Cap` (normal at the power limit). Max 74–77 °C.
* The **"500×" cached-prefill number is real only for an identical prompt** (measured 275×, and
  24.9× on a fresh serve), drops to 3× when one line is appended, to 1× when one character in the
  middle changes, and disappears entirely in a wave of unique prompts (c=4 unique: 19.0 agg tok/s
  vs 51.2 shared). At `temperature: 0` the cached path also gave a **different answer** than the
  cold path on the same prompt (974 vs 953 chars, disagreement on a factual probe).

## Headline tables

Context ladder, TTFT (s) — cold prefill where the engine actually had to read the prompt:

| arm (prompt tokens) | Basalt | Strata | llama.cpp |
|---|---|---|---|
| p4k (5.2 k) | 6.10 | 4.80 | 29.6 |
| p32k (39.2 k) | 4.96 *(cached)* | 13.71 | 232.5 |
| p64k_code (71–75 k) | 4.87 *(cached)* | 23.74 | 434.5 |
| p128k (152–173 k) | 4.97 *(cached)* | 47.65 | 1045.3 |
| p184k (184–229 k) | **FAILED 400** | 58.31 | 1125.7 |

Warm turn (50 k conversation, two foreign chats in between): cold → follow-up TTFT —
Basalt 16.53 → 4.39 s (3.8×), Strata 15.72 → 1.43 s (**11.0×**), llama.cpp 283.68 → 1.63 s (173.7×).

Concurrency on 32 k prompts (aggregate tok/s): c=2 → c=4 → c=8:
Basalt 31.0 / 51.0 / **101.6**; Strata 18.8 / 51.2 / **126.5**; llama.cpp 0 / 0 / 0 (context rejected).
Control arm V6 (4 *unique* prompts, c=4): 19.0 — the honest lower bound.

30-minute sustained (c=2): Basalt 829 requests, 153.0 W, 76.64 Wh, 0.481 Wh/1k answer tokens;
Strata 1285 requests, 164.4 W, 82.35 Wh, **0.334 Wh/1k**; llama.cpp 263/264 requests failed and
still burned 70.83 Wh.

## What was cut by the 05:50 deadline

llama.cpp concurrency at a context that fits 32 k, llama-side warm/500× verification, a repeated
answer-agreement study (V5 was one trial), a Strata `reasoning_budget` sweep, and the BitNet CPU
ladder (completed post-deadline at 08:28 — [report §10](report.md)). Failed arms are recorded
verbatim rather than deleted: [report.md §8](report.md).

## Reproduce

```bash
# on the bench box
bash driver.sh                 # parks the keep-up cron, restores the brain on ANY exit
python3 harness/loadbench.py strata|basalt|llama1|llama2|verify500x|verify500x2|bitnet|llamacpu
python3 harness/loadbench.py llamaopt      # follow-on llama.cpp optimization series (OPT_* tags)
```
Prompts are rebuilt deterministically (`harness/build_prompts.py`, seed 20261117) from a private
docs tree + the llama.cpp checkout; only their sha256 manifest is published, so exact token counts
are verifiable in `results/results_raw.jsonl.gz`.

## Hygiene

All published files passed a two-layer gate: pinned **Gitleaks 8.30.1** (sha256-verified binary,
default detectors + custom infrastructure rules, self-tested against planted canaries) **plus** a
custom infrastructure/privacy scanner that traverses archive members by name and content.
Local account paths, LAN addresses, model-store paths and API-key values were replaced by
placeholders before staging; the gate script, the rule file and the raw scan reports are kept
outside this tree.
