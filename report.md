# Three engines, one RTX 5090 Laptop, one IQ3_S MoE — measured on identical arms

Run date: 2026-11-19, 00:16–05:53 CET, single box, detached driver. Driver deadline 05:50;
arms that did not finish before it are listed in §8 as missing, not silently dropped.

Hardware: one RTX 5090 Laptop (24 GB VRAM, power-limited part), 64 GB host RAM
(≈60 GiB usable), AMD Zen server CPU (15 pool workers), PCIe 5.0 x8 (16 GT/s), Linux, CUDA 13.0.
Model: Qwen3.8-Flash-Next at IQ3_S, one 54.5 GB GGUF (mmap, no mlock) shared by all engines;
the production engine used its own pack conversion of the same quantization.

Engines under test (all with the same arms, same prompts, `temperature: 0`, SSE streaming):

| Engine | Build | Serve configuration (essentials) |
|---|---|---|
| **Basalt** (production brain) | 0.4.1, llama.cpp-family server | `ctx=196608`, `kv=int8`, resident-experts + resident-cpu-experts, `resident-budget=46GiB`, `ram-cache=6144MiB`, `conversation-cache-slots=6`, `disk-cache=20480MiB`, `spec=8`, expert-profile, vision=dynamic, port 18084, production config unchanged |
| **Strata** (fork, candidate) | 0.1.62, winning probe config `v1_profile_ctx196k` | `ctx=196608`, `kv=int8`, expert cache 16,888 MiB (8,697/24,576 expert slots resident, `pcie_frac=0.10`), `conversation-cache-mib=2560`, `-slots=4`, `min-free-mib=1024`, `min-tokens=4096`, `prefill_cap=16384`, MTP `mtp_max=4` + lookup 3, `pool_workers=15` |
| **llama.cpp** (external reference) | b12531, build 214, commit `e4b620af1` | `-ngl 99 --flash-attn on --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 --n-cpu-moe 48`; L1: `-np 1 -c 198144` (loaded n_ctx_slot 198144); L2: `-np 8 -c 270336` → **auto-fitted down to n_ctx_slot 33792** |
| BitNet (reference arm) | i1 GGUF, CPU-only | single sustained arm only (§7, §8) |

Prompt set: deterministic seeded corpus (seed 20261117), identical text files for every engine:
`p4k, p32k, p64k_code, p128k, p184k, p40k, p32k_a/b/c, p4k_a, p4k_f1`. Corpus is private
(docs + llama.cpp source slices) — only its sha256 manifest is published (§9).

---

## 1. Verdict

| Category | Winner | Runner-up | llama.cpp |
|---|---|---|---|
| Cold long-context prefill | **Strata ≈ 3.3 k tok/s** | Basalt ≈ 3.0 k tok/s (measured on a 50 k cold prompt) | **163–175 tok/s — 19× slower** |
| Decode (answer tokens/s, single stream) | **Basalt 139–172 tok/s** (engine-reported) | Strata 94–156 | 21–28 |
| Real-task recall at ≥64 k (5 CONTROL markers) | 5/5 all three engines | — | 5/5 (two runs) |
| Max usable context in one slot | Strata/Basalt 196,608 (Basalt rejected p184k, §8) | — | 198,144 with 1 slot; with 8 slots only 33,792 |
| Concurrency c=8 on 32 k prompts | **Strata 126.5 agg tok/s**, Basalt 101.6 | — | could not run (400 Bad Request) |
| 30-min sustained, answer tokens per Wh (GPU) | **Strata 0.334 Wh/1k tokens** | Basalt 0.481 | 368.9 (arm failed; energy spent for nothing) |
| Energy per answer token, J/token | **Strata 1.20** | Basalt 1.73 | n/a (failed arm) |
| KV-cache behaviour transparency | **Strata** (`/metrics` per-agent `requests[]`, park/restore counters) | Basalt (per-slot `pp/c/hit/pcie` lines) | none (`/metrics` → HTTP 501) |

The headline negative result is llama.cpp: on 24 GB VRAM it cannot keep this 54.5 GB MoE on the
GPU at all. `--n-gpu-layers 99` dies with `cudaMalloc failed: out of memory` (54,468,812,928 bytes),
and the only variant that loads is `--n-cpu-moe 48` — i.e. essentially all experts live in host RAM
and are streamed per token. Cold prefill collapses to ~165 tok/s and decode to 21–28 tok/s.
That is a headline finding, not a footnote: **this model in 24 GB VRAM is a
"weights in RAM, experts over PCIe" workload, and the two purpose-built engines exist exactly
for that case.**

Both purpose-built engines are usable for the brain workload, and they win in different places:
Strata on cold prefill, concurrency, energy-per-token and cache transparency; Basalt on single-stream
decode rate and on a persistent (disk-backed) conversation cache that survives restarts.

---

## 2. Single-request context ladder

TTFT = client time to first content/reasoning delta; decode = client `gen/(wall−TTFT)`
(`gen` = `usage.completion_tokens`); D1–D4 = decode rate in the four quarters of one continuous
answer; recall = exact hits out of 5 injected `CONTROL` markers.
See §9.4 for the known inflation of client-side decode vs the engine's own rate.

### Basalt 0.4.1 (production config, port 18084)

| arm | prompt tok (client / engine) | TTFT s | decode tok/s (client / engine) | D1–D4 | recall |
|---|---|---|---|---|---|
| p4k | 5,198 / 5,250 | 6.104 | 352.1 / 171.1 | 124.4 190.7 228.5 187.9 | n/a |
| p32k | 39,228 / 39,280 | 4.963 | 352.1 / 172.7 | 155.3 177.5 193.9 173.8 | n/a |
| p64k_code | 74,962 / 75,023 | 4.868 | 287.6 / 140.4 | 185.9 146.6 127.8 123.4 | 5/5 |
| p128k | 152,589 / 152,642 | 4.973 | 292.1 / 139.0 | 181.5 123.8 108.0 193.7 | 5/5 |
| p184k | — | — | — | — | **FAILED (§8)** |

**Read this table with care.** The engine's own accounting says the p32k/p64k/p128k requests were
served from cache: `pp=39280/39275c`, `pp=75023/75018c`, `pp=152642/152637c` — 99.99 % of the prompt
tokens reused, `ttft=0.2–0.5 s` server-side. Basalt's `disk-cache=20480MiB` +
`ram-cache=6144MiB` conversation cache is **persistent across serve restarts**, and these exact
seeded prompts had been fed to this engine in earlier bench sessions. So the ~5 s TTFT column is
*warm-cache reload*, not cold prefill, and the apparent 7.9 k → 30.7 k tok/s "prefill rate"
(prompt tokens ÷ TTFT) is a cache artifact. Basalt's measured cold prefill in this run:
50,177 tokens in 16.53 s ≈ **3.0 k tok/s** (warm-arm turn 1, engine line `pp=50177/0c`), and
11,496 tokens in 4.66 s ≈ 2.5 k tok/s on the live production engine (§6.7).

### Strata 0.1.62 (`v1_profile_ctx196k`)

| arm | prompt tok (engine) | TTFT s | prefill tok/s (TTFT-derived) | decode tok/s (client) | D1–D4 | recall |
|---|---|---|---|---|---|---|
| p4k | 5,250 | 4.799 | 1,094 (overhead-dominated) | 230.9 | not captured | n/a |
| p32k | 39,280 | 13.707 | 2,866 | 248.8 | not captured | n/a |
| p64k_code | 75,023 | 23.744 | 3,160 | 221.9 | not captured | 5/5 |
| p128k | 152,642 | 47.646 | 3,204 | 240.7 | not captured | 5/5 |
| p184k | 188,724 | 58.312 | 3,236 | 273.4 | not captured | n/a |

Cold prefill is flat at ~3.2–3.3 k tok/s from 39 k to 189 k tokens (engine log progress lines:
`reading the prompt: 13,098 of 39,298 tokens, 5 s so far` — chunked at `prefill_cap=16384`).
`/metrics` confirms the cold ladder: `live.reused=0`, `prefill_tok_s_mean=3530`.

### llama.cpp b12531 (`-np 1 -c 198144`, loaded with `--n-cpu-moe 48`)

Two independent ladder runs (two driver launches) — the repeatability is the evidence that the
numbers are not noise:

| arm | prompt tok (llama tokenizer) | TTFT s run A / run B | prefill tok/s | decode tok/s run A / B | D1–D4 run B | recall |
|---|---|---|---|---|---|---|
| p4k | 5,198 | 30.737 / 29.647 | 169 / 175 | 29.4 / 28.8 | 29.6 29.5 27.2 29.1 | n/a |
| p32k | 39,228 | 229.265 / 232.473 | 171 / 169 | 29.1 / 30.3 | 31.2 30.0 29.5 30.4 | n/a |
| p64k_code | 71,166 | 434.921 / 434.538 | 164 / 164 | 27.9 / 27.4 | 27.5 27.7 27.2 27.0 | 5/5 both |
| p128k | 172,674 | 1042.640 / 1045.299 | 166 / 165 | 28.2 / 27.5 | 26.5 27.6 27.4 28.7 | 5/5 both |
| p184k | 184,098 | — / 1125.666 | — / 164 | — / 25.6 | 27.6 25.7 27.0 22.8 | n/a |

llama.cpp answers the recall probes perfectly at 64 k and 128 k (5/5 twice) — correctness is not
the problem; wall-clock is: **17 minutes to first token at 172 k tokens.**

Tokenizer/template spread on the *same* files is real and changes who can serve what:
p128k = 152,642 tok (Basalt) vs 152,793 (Strata) vs 172,674 (llama.cpp);
p184k = 229,129 (Basalt) vs 188,724 (Strata) vs 184,098 (llama.cpp). The Basalt/Strata divergence
at p184k is unexplained in the artifacts and is flagged, not smoothed over (§8).

---

## 3. Warm / KV-reuse arm (one 50 k conversation, two foreign chats, follow-up)

Protocol: turn 1 on `p40k` (≈50.1 k tokens) cold → two unrelated small chats (2,800 and 4,100
tokens) → follow-up on the *same* conversation.

| engine | cold TTFT s | follow-up TTFT s | speed-up | engine evidence |
|---|---|---|---|---|
| Basalt | 16.529 | 4.394 | **3.8×** | `pp=50286/50172c` (99.8 % reused), `ttft=0.5s` |
| Strata | 15.718 | 1.433 | **11.0×** | `50,172 reused + 9 read in 0 ms`; conversation parked/restored |
| llama.cpp | 283.680 | 1.633 | **173.7×** | single-slot prompt cache; follow-up prefills 49 tokens |

llama.cpp's *ratio* is the biggest because its cold path is catastrophic. Its warm path is genuinely
good — and it is also the only one that is fragile: one slot, evicted by any second conversation.
Strata's parking is what buys the 11× while four other chats are in flight (§5); Basalt's 3.8× is
the weakest of the three and is bounded by reloading a large snapshot from its RAM/disk cache.

---

## 4. Concurrency ladder: 32 k prompts, c = 2/4/8

`agg` = total generated tokens ÷ total wall time. Prompts `p32k_a/b/c` rotate, so from c=2 on the
same prompt text appears more than once — cache reuse is part of what is being measured (see the
V6 control below).

| arm | ok | wall s | agg tok/s | per-req decode mean / p95 | TTFT mean / p95 |
|---|---|---|---|---|---|
| Basalt c=2 | 2/2 | 16.5 | 31.0 | 322.8 / 311.5 | 9.82 / 3.86 |
| Basalt c=4 | 4/4 | 20.1 | 51.0 | 336.7 / 346.2 | 11.37 / 17.62 |
| Basalt c=8 | 8/8 | 20.2 | **101.6** | 321.4 / 363.8 | 11.79 / 17.73 |
| Strata c=2 | 2/2 | 27.3 | 18.8 | 280.8 / 271.2 | 19.66 / 12.92 |
| Strata c=4 | 4/4 | 20.0 | 51.2 | 291.8 / 313.0 | 7.36 / 5.47 |
| Strata c=8 | 8/8 | **16.2** | **126.5** | 313.2 / 343.2 | 8.55 / 13.72 |
| **Strata V6 (4 unique prompts, c=4)** | 4/4 | 53.8 | 19.0 | 141.2 / 151.0 | 32.13 / 38.92 |
| llama.cpp L2 c=2/4/8 | 0/2, 0/4, 0/8 | ~0 | 0.0 | — | — |

Basalt's ladder is again cache-served (its KV reload is ~4–5 s per 39 k prompt), which is why its
TTFT column barely grows with concurrency. Strata's c=2 arm is the honest cold one (19.66 s mean
TTFT = four 39 k prefills queued), and by c=8 it has parked/restored its way to the best aggregate
number of the run.

llama.cpp could not be measured here: with 8 slots the auto-fitted per-slot context is 33,792
tokens, and every 32 k arm is rejected — verbatim:
`request (39385 tokens) exceeds the available context size (33792 tokens), try increasing it`.
Its `/metrics` for per-agent KV inspection answers `HTTP 501 Not Implemented`.

---

## 5. Per-agent KV cache (Strata only — the other two expose nothing comparable)

Engine block from `GET /metrics` during the waves:

```
version 0.1.62   context 196608   kv int8
expert_cache_mib 16888   expert_slots 8697 / experts 24576   pcie_frac 0.10
arena_mib 47962          vram_free_mib 435
conversation_cache_mib 2560   conversation_cache_slots 4
conversation_cache_min_free_mib 1024   conversation_cache_min_tokens 4096
prefill_cap 16384   spec 12   mtp_max 4   lookup 3   pool_workers 15
```

Conversation-cache counters at the end of each wave (`/metrics.conversation_cache`):

| wave | parked | bytes | parks | restores | evictions | requests | reused reqs | reused / prompt tokens | refused |
|---|---|---|---|---|---|---|---|---|---|
| c=2 | 2 | 2,415,303,672 | 7 | 1 | 4 | 14 | 4 | 277,877 / 835,961 = 33.2 % | short 2, budget 2 |
| c=4 | 2 | 1,918,635,568 | 10 | 3 | 5 | 18 | 7 | 396,006 / 993,399 = 39.9 % | short 2, budget 2 |
| c=8 | 1 | 1,564,182,784 | 16 | 9 | 8 | 26 | 14 | 672,733 / 1,310,176 = 51.3 % | short 2, budget 2 |
| V6 unique c=4 | 2 | 1,919,179,312 | 12 | 1 | 10 | 15 | 4 | 144,480 / 551,795 = 26.2 % | short 1 |

Evolution within the c=8 wave (poll cadence ≈4 s): `parked` 2 → 1, `bytes` 1.92 GB → 1.56 GB,
`parks` 10 → 16, `restores` 3 → 9, `evictions` 5 → 8 — i.e. under concurrency the engine parks and
restores constantly instead of thrashing one slot, and reuse share rises with concurrency.

Per-agent view at the end of the c=8 wave (`requests[]`, most recent first, truncated to 12):

```
prompt_tokens  reused   hit_rate  decode_tok_s  drafts_offered/accepted  pcie_share
40,015          40,010    0.934      129.1        199 / 138                0.001
39,425          39,420    0.912      141.6        198 / 153                0.003
39,294          39,289    0.916      142.5        199 / 151                0.003
39,294          39,289    0.921      142.2        198 / 153                0.002
40,015          40,010    0.934      129.9        210 / 138                0.002
39,294          39,289    0.915      137.9        187 / 154                0.003
39,425          39,420    0.902      139.9        200 / 155                0.003
39,312               0    0.958      156.1        102 /  73                0.001   (V6 unique prompt, no reuse)
```

Observations worth keeping:

* `pcie_share` is 0.1–0.3 % of decode time with `pcie_frac=0.10` (10 % of routed experts pinned to
  VRAM). The PCIe link is not the bottleneck in this configuration; the expert cache hit rate
  (81–97 % in the serve log) is what keeps decode at 100–150 tok/s.
* Speculative decoding is accepted at ~70–80 % of offered drafts (e.g. 429 offered / 341 accepted).
* Parking cost is the same order of magnitude as measured directly on this engine in earlier runs
  (its own park lines: 40,101 tokens = 1,084,923,040 B → 27.1 kB/token; 94,042 = 2,263,672,000 →
  24.1 kB/token; 180,703 = 3,588,456,408 → 19.9 kB/token). Dividing the `/metrics` `bytes` counter
  by parked token counts lands in the same 10–27 kB/token band; exact per-snapshot pairing is not
  resolvable from the polled counters, so the figure is treated as approximate. On that basis the
  2,560 MiB budget holds ~110–150 k parked tokens, which is why a 189 k-token session (p184k) is
  never parked.
* The budget guard fires: `refused.budget=2` on all three waves — the engine declined to park
  because a snapshot would have crossed `min-free-mib`, rather than starving the host. Host RAM
  minimum available during the Strata sustained arm: 4.43 GiB with engine RSS 53.5 GiB.

---

## 6. The "500× cached prefill" claim — adversarial verification

Earlier load data implied cached prefill was ~500× faster than cold (≈15 s cold vs ~30 ms cached).
This section is the attempt to break that number. All arms run against a **freshly loaded**
Strata serve (`v1_profile_ctx196k`, 153 s load), one process, one warm cache chain.
Every V/W prompt is generated at run time with a unique `UNIQUE-SESSION-SALT-<10 digits>` token
inside the text, and its sha256 is recorded in the result row, so "the engine has seen this before"
is checkable.

| arm | what it tests | prompt sha256[:12] | TTFT s | result |
|---|---|---|---|---|
| V1 | cold, unique 32 k prompt | `97f65a18162d` | 12.374 | baseline cold: 39,298 tok ≈ 3.2 k tok/s |
| V2 | **identical repeat, immediately after** | `97f65a18162d` | **0.045** | **275× faster than V1** |
| V3 | append a salted line to the same prefix | `4c018b9025a5` | 11.275 | **no reuse** — full 39,298-token read again |
| W1 | cold, unique 32 k prompt (fresh serve) | `534d2212e460` | 12.679 | baseline cold |
| W2 | identical repeat | `534d2212e460` | **0.509** | 24.9× |
| W3 | append **one line** | `242e6715549e` | 4.225 | **partial reuse** — 3× |
| W4 | flip **one character in the middle** | `dcbb78dc4f6f` | 11.415 | prefix broken → full re-prefill |
| V4a | first 32 k chat | `2528bf92405e` | 11.855 | cold |
| V4b ×4 | four big intruder chats (fills all 4 cache slots) | `e45c58b05bc2`, `acc203812552`, `398330e483bb`, `77e638c16449` | 11.6 / 11.6 / 11.5 / 11.5 | each cold, 4.7 GB parked |
| V4c | re-query the first chat after that pressure | `2528bf92405e` | **0.835** | 14.2× — snapshot survived, 10 evictions in the wave |
| V5a / V5b | **answer correctness**, temp 0, cold vs cached | same prompt `e59ae52c7265` | 4.180 / 0.048 | **answers DIFFER** (see below) |
| V6 | **c=4 wave, four unique prompts** (no cross-request sharing) | — | mean 32.13, p95 38.92 | agg 19.0 tok/s vs 51.2 for the shared-prompt c=4 arm |

### What is real

* Reuse is genuine and enormous **for an identical prompt**: 275× (V1→V2), 24.9× (W1→W2).
  The engine's own accounting backs it: cold shows chunked progress
  (`reading the prompt: 13,098 of 39,298 tokens, 5 s so far` … `39,293 of 39,298 tokens, 12 s`),
  the repeat shows a single read line with all-but-a-few tokens accounted and a 128-token answer in
  1 s total.
* It is **prefix-granular, not all-or-nothing**: W3 (append one line) reused 39,285 of 39,290
  tokens → 3×, and W4 (one flipped character in the middle) lost everything → 11.4 s. V3 shows the
  same boundary from the other side: appending a salted *block* produced a full re-read.
* It survives eviction pressure in this shape: after four 39 k intruder conversations filled all
  four `conversation_cache_slots` (10 evictions), the original conversation still came back in
  0.835 s instead of 11.9 s — because the parked snapshot is a RAM snapshot keyed by conversation,
  and the *evicted* ones were the intruders'.

### What is not real

* **Concurrency destroys it.** V6: four *unique* 39 k prompts at c=4 → mean TTFT 32.13 s,
  decode mean 141.2, agg 19.0 tok/s — 2.7× worse than the shared-prompt `S_load c=4` arm
  (7.36 s / 291.8 / 51.2). The published c=4/c=8 numbers therefore include prompt-reuse benefit;
  the unique-prompt arm is the honest lower bound: 4 × 39 k ≈ 156 k cold prefill tokens in 53.8 s
  ≈ 2.9 k tok/s ≈ exactly one stream's prefill rate. There is no prefill parallelism to find.
* **The fast path does not reproduce the cold answer.** V5a/V5b, same prompt, `temperature: 0`:
  cold answer 974 chars / 229 generated tokens, cached answer 953 chars / 222 tokens, first
  divergence at character 412. They do not just paraphrase — they disagree on a fact. The dump
  contains no `CONTROL` record (the only occurrence of the word is the instruction line); the
  *cold* run invented one (`CONTROL = UNIQUE-SESSION-SALT-4628875385`, the salt taken from the
  prompt), the cached run correctly answered *"There are no CONTROL records in the dump."*
  One trial, so this is an existence proof of divergence, not a rate — but it means a
  cached-prefill speed-up cannot be presented as answer-identical at temp 0.

### Verdict on the claim

The ~500× figure is **not a lie, but it is a corner case**: measured here on a never-before-seen
prompt the repeat ratio is 275×, it drops to 24.9×–3× as soon as the prompt changes at all, and it
vanishes completely under a real multi-agent wave. The number that should be quoted for production
is the warm-conversation one from §3 (11× on a 50 k conversation after two foreign chats) and the
0.835 s re-query after four-slot eviction pressure — those are repeatable and useful.

### Live-engine cross-check (production Basalt 0.4.1 on 8080, after the bench)

Unique 11,496-token prompt sent twice: `0 reused` → TTFT 4.66 s (2.5 k tok/s); repeat →
`11,491 reused` → TTFT 0.102 s (**45.7×**); second repeat 0.052 s. Same shape on a different
engine and a different cache implementation.

---

## 7. Sustained 30-minute arm (c=2, 32 k prompts, 192-token answers, 1 Hz telemetry)

| metric | Basalt | Strata | llama.cpp | BitNet i1 (CPU, 15 min) |
|---|---|---|---|---|
| duration s | 1802.7 | 1802.1 | 1804.1 | 918.7 |
| requests done / errors | 829 / 0 | 1285 / 0 | 264 / **263** | 80 / 0 |
| decode mean / p50 / p95 | 359.8 / 366.6 / 382.5 | 314.3 / 314.2 / 347.2 | 14.8 / 14.8 / 14.9 | 6.1 / 6.2 / 6.2 |
| drift, first → last 5-min bucket | 355.9 → 358.6 (+0.8 %) | 309.9 → 316.1 (+2.0 %) | 14.6 → 14.8 (flat) | 6.1 → 6.2 |
| TTFT mean | 4.97 s (falls to 2.17 s as prompts cache) | 2.19 s (stable) | 2.29 → 0.17 s | 6.38 → 0.47 s |
| GPU power mean / max W | 153.0 / 170.4 | 164.4 / 171.6 | 141.3 / 170.3 | 32.4 / 69.1 |
| **GPU energy in window, Wh** | 76.64 | 82.35 | 70.83 | 8.26 |
| **Wh per 1,000 answer tokens** | 0.481 | **0.334** | 368.9 (failed arm) | 0.538 |
| **J per answer token** | 1.73 | **1.20** | 1328 (failed arm) | 1.94 (GPU-only; CPU work unmeasured) |
| max / mean GPU temp | 74 / 72.5 °C | 76 / 74.3 °C | 77 / 73.1 °C | 53 / 50.7 °C |
| throttle flags seen | `0x4` SwPowerCap 1237/1742 samples, `0x400` 505 | `0x4` SwPowerCap 2797/2885, `0x400` 88 | `0x4` 40, `0x400` 1702 | `0x400` only |
| SM clock min | 1867 MHz | 1582 MHz | 1590 MHz | 1612 MHz |
| VRAM peak MiB | 23,574 | 23,682 | 14,216 | 1,004 |
| host RAM min available | 5.94 GiB | 4.43 GiB (RSS 53.5 GiB) | 55.62 GiB | 54.88 GiB |

* No thermal throttling anywhere: the `SW Thermal Slowdown` (0x20) and `HW Thermal Slowdown` (0x40)
  bits never appear. The `0x4 SwPowerCap` bit is this laptop GPU sitting at its power limit — normal,
  not a fault. A further undocumented flag bit (`0x400`) is present on most samples and is reported
  verbatim rather than interpreted.
* Nothing degraded over 30 minutes (drift <2 % for both GPU engines), so this box can run the brain
  load continuously without a thermal cliff.
* llama.cpp's sustained arm burned 70.8 Wh to produce 192 useful tokens because every 32 k request
  was rejected (§8) — the cleanest illustration in the run of "energy without output".

---

## 8. Failed and missing arms, verbatim

| arm | outcome | verbatim |
|---|---|---|
| `B_single/p184k` | FAILED | `engine reported request exceeds the context window: prompt (229129 tokens) leaves no room to answer in the context (196608); requests are never truncated` → `HTTPError 400` |
| `L2_load/load32k_c2/c4/c8` | 0 of 14 requests | `request (39385 tokens) exceeds the available context size (33792 tokens), try increasing it` (277 such lines in the serve log); KV probe `HTTP 501: 'Not Implemented'` |
| `L1/L2` first attempts (00:53 run) | LOAD_FAILED | `error: invalid argument: --fa` (b12528 renamed the flag to `--flash-attn on`; fixed in the next launch) |
| `L2_sust` | 263 of 264 requests errored | same context-size rejection for the 32 k prompts in the loop |
| `S_single/p184k`, first attempt (01:56) | purged | `ttft=None decode=4.3`, `pp=0` — thinking-overflow rows; also `S_load` dec_mean 14.2 (01:57) purged |
| `B_sust` duplicate row (00:45) | purged | `req=538 err=1859328` — harness counter corruption, the clean 00:53 row is used |
| Basalt p4k/p32k/p64k/p128k "cold" ladder | not cold | engine reports 99.99 % reused (persistent disk conversation cache); no cold read of those prompts exists in the captured serve log |
| p184k token counts | unexplained spread | 229,129 (Basalt) vs 188,724 (Strata) vs 184,098 (llama.cpp) on the same file |
| **missing at deadline 05:50** | not run | llama.cpp concurrency at a fitted context that fits 32 k; llama.cpp warm/500× verification; repeated V5 answer-agreement study; Strata `reasoning_budget` sweep; BitNet beyond one sustained arm; D1–D4 quarters for Strata (`/tokenize`-based quarter cut did not capture) |

---

## 9. Appendix

### 9.1 Corpus and prompt manifests
`build_prompts.py` + `mkprompts.py` slice a deterministic corpus (seed 20261117) from a private
docs tree and the llama.cpp checkout. The prompt **texts are not published** (private corpus);
`results/prompts/manifest.json` carries every prompt's sha256 and the engine-reported token counts.
Rebuilding needs the same two input trees.

### 9.2 Exact serve commands

```
# Basalt (production config unchanged except port/host; API key from $HOME/basalt/basalt.key)
$HOME/basalt-041/bin/basalt serve --config <bench>/configs/basalt.json \
    --python $HOME/basalt/.venv/bin/python --vision dynamic --api-key <REDACTED>

# Strata: JSON config published sanitized at harness/serve-configs/strata.json
#   (api_key replaced by <REDACTED>); winning probe variant v1_profile_ctx196k,
#   serve args incl. --expert-profile <model-store>/placement.prof

# llama.cpp (both stages, after the ordered variant probe: moe=None/kv8, None, 12, 24, 48)
$HOME/llama.cpp/build-load/bin/llama-server -m <model-store>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf \
    --alias iq3s-llama -ngl 99 --flash-attn on --reasoning off \
    -c 198144 -np 1 --host 127.0.0.1 --port 18080 \
    --cache-type-k q8_0 --cache-type-v q8_0 --n-cpu-moe 48
```

### 9.3 Client-side timing method
Single SSE client per request (`loadbench.py`): `POST /v1/chat/completions`,
`temperature: 0`, `stream: true`, `stream_options.include_usage: true`; TTFT = time to the first
non-empty `content` **or** `reasoning_content` delta; decode = `gen/(wall−TTFT)`; `gen` from
`usage.completion_tokens`, else engine `/tokenize` of the answer (marked as an estimate).
D1–D4 are decode rates between the 25/50/75/100 % character marks of one continuous answer.

### 9.4 Accounting asymmetries — read before quoting numbers

1. **Thinking tokens.** Strata arms ran with `reasoning_budget_tokens: 192` and its thinking tokens
   are counted in `completion_tokens`; llama.cpp ran `--reasoning off` (answer tokens only);
   Basalt used its production default (banner: `thinking budget 96 tokens, output cap 192`).
   "Answer tokens/s" is therefore not the same quantity across engines.
2. **Client vs engine decode rate.** For the two engines with thinking/speculative accounting the
   client-side rate is 1.6–2.3× the engine's own reported rate (Strata c=8: client 313, engine
   133; Basalt p4k: client 352, engine 171). Wall-clock arithmetic decides it: 8 requests × 256
   tokens at the engine rate = 15.4 s against a measured wall of 16.2 s, while the client rate
   implies 6.5 s. Treat client-side decode as an upper bound; engine-reported rates as the
   conservative number. llama.cpp numbers agree between both methods (21–29 tok/s).
3. **Token counts differ per engine** on identical text (§2) because of tokenizer/template spread —
   the ladder compares the same *text*, not the same token count.
4. **Basalt's ladder is warm**, from its persistent disk conversation cache (§2). Its cold prefill
   numbers in this run come from the warm arm's turn 1 and from the live probe in §6.7.
5. **Energy is GPU-only** (1 Hz `nvidia-smi` integration over each arm's own window); host/CPU/DRAM
   power is not instrumented, so the BitNet CPU arm's energy figure is a floor, and the MoE numbers
   exclude the CPU-side expert-streaming cost.
6. **llama.cpp version drift across launches**: b12528 (build 211, `--fa` failure), b12530 (build
   213), b12531 (build 214, commit `e4b620af1`). Every published llama.cpp number comes from b12531.

### 9.5 Sanitization applied to this publication
Local account paths → `$HOME`, model-store paths → `<model-store>`, LAN addresses →
`[local-network-redacted]`, API-key values → `<REDACTED>` (values were never printed anywhere in
the pipeline; key fields were reported as lengths only). Prompt corpus texts excluded. Raw scan
reports, the gate script and the scanner rule file are kept **outside** this tree by design.
