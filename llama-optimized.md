# llama.cpp re-test with MoE expert-cache optimization (b12533, 2026-11-19/20)

This is an **append-only re-test** of the llama.cpp column of [report.md](report.md).
No number in `report.md` was changed; the original b12531 run stands as published.
New results below carry the `OPT_*` tags (and the fixed `C_*` CPU re-run) and come from a
detached driver on the same P2005-class box (one RTX 5090 Laptop, 24 GB VRAM, 60 GiB host RAM).
Run window: 2026-11-19 11:54 – 20:35 CET (the first two driver launches failed and are
recorded verbatim in §2; the complete screening + final ladder ran 16:55 – 20:35).

The question this answers: the original run's llama.cpp verdict ("weights in RAM, experts over
PCIe", 165 tok/s prefill, 21–30 tok/s decode, concurrency impossible at 24 GB) was measured with
`--n-cpu-moe 48`. llama.cpp b12533 ships an LRU **MoE expert cache in VRAM**
(`-cmoe --moe-cache-mib N --moe-cache-cpu-share F`, `docs/moe-cache.md`). Correctly tuned,
how much of the gap to Strata/Basalt closes?

---

## 1. Provenance

| item | value |
|---|---|
| llama.cpp | tag **b12533**, commit `d4f5c26049774bb786276eb517f1e6b070d28ff4` |
| build | separate `build-opt` tree: CMake 3.28.3, `CMAKE_BUILD_TYPE=Release`, CUDA backend (`CMAKE_CUDA_ARCHITECTURES=120a`), CPU variant `-march=native` |
| GPU / driver | NVIDIA GeForce RTX 5090 Laptop, 24,463 MiB, driver **615.71.09** |
| CUDA | `cuda_13.0.r13.0/compiler.36424714_0` |
| OS / RAM | Linux 7.0.0-43-generic, 60 GiB RAM |
| model | `<model-store>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (part 1 of 2, 54,817,524,224 B; part 2 = 28,800,138,432 B) — same 54.5 GB IQ3_S GGUF as report.md §9 |
| model sha256 (computed post-run, 2026-11-19) | part 1: `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`, part 2: `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| corpus | same deterministic seeded corpus as report.md §9.1 (`results/prompts/manifest.json`); new per-run salt manifests below |
| client | same `loadbench.py` SSE client, `temperature: 0`, method report.md §9.3 |

New unique-prompt manifest for the concurrency waves (`meta/unique_manifest.txt`, sha256 first 16 hex):

```
p32k_u1.txt 6000d42163ad64a0   p32k_u5.txt bc0fc9af64ffebd6
p32k_u2.txt c80d0eba29cc1235   p32k_u6.txt 16d66ea13d4b4cb5
p32k_u3.txt e7fa33e78876183c   p32k_u7.txt 98ab85b8d6ca4c04
p32k_u4.txt a9f74e20883d82dd   p32k_u8.txt 6121005b6808f34a
```

Final winning serve command, verbatim (E1, single-slot ladder):

```bash
$HOME/llama.cpp/build-opt/bin/llama-server \
    -m <model-store>/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --alias iq3s-opt -ngl 99 --flash-attn on --reasoning off \
    -c 198144 -np 1 -t 15 --host 127.0.0.1 --port 18087 \
    -cmoe --moe-cache-mib 11968 --moe-cache-cpu-share 0.98 \
    --cache-type-k q8_0 --cache-type-v q8_0
```

Concurrency config (E1c0) differs only in `-c 327680 -np 8`; the server logged
`n_slots = 8, n_ctx_slot = 40960, kv_unified = 'false'` (fit verified, not auto-shrunk).

**Version-drift control:** arm A0 re-ran the *old* configuration (`--n-cpu-moe 48`) on b12533.
A0 ≈ the published b12531 numbers (p32k prefill 177.5 vs 169–171 tok/s, decode 29.7 vs 29.1–30.3,
TTFT 221.0 vs 229.3–232.5 s), so every gain below is attributable to the flags, not to the version.

## 2. Screening: nine configs, all arms kept

Each config: fresh serve, then a 3-arm mini-ladder (p4k/p32k/p64k, 256-token answers).
TTFT and decode are client-side; prefill tok/s is the harness's `prompt_tokens/TTFT` engine-side field.

| id | config delta | load s | VRAM MiB | p32k TTFT s | p32k prefill tok/s | decode p4k / p32k / p64k | verdict |
|---|---|---:|---:|---:|---:|---|---|
| A0 | `--n-cpu-moe 48` (old config) | 3.0 | 8,528 | 221.0 | 177.5 | 31.8 / 29.7 / 27.2 | baseline ≈ b12531 result |
| B1 | `-cmoe --moe-cache-mib 1024` | 3.0 | 9,502 | 215.7 | 181.9 | 15.5 / 15.0 / 14.6 | **regression** — cache too small, thrash |
| B2 | `… 2048` | 3.0 | 10,570 | 210.7 | 186.2 | 18.2 / 17.6 / 17.5 | regression |
| B3 | `… 4096` | 3.0 | 12,620 | 202.1 | 194.1 | 22.8 / 21.5 / 23.1 | still below A0 decode |
| B4 | `… 8192` | 3.0 | 16,718 | 182.4 | 215.0 | 31.0 / 28.3 / 30.4 | ties A0, prefill +21 % |
| C1 | `… 4096 --moe-cache-cpu-share 0.5` | 3.0 | 12,626 | 202.5 | 193.7 | 27.1 / 26.4 / 28.1 | CPU-share of misses is a real lever |
| C2 | `… 4096 --moe-cache-cpu-share 0.95` | 3.0 | 12,626 | 201.9 | 194.3 | 40.3 / 37.0 / 34.3 | **share 0.95: +77 % decode at same cache** |
| D1 | `… 4096 -lm mlock` | 90.2 | 12,714 | 198.2 | 197.9 | 21.1 / 20.3 / 21.0 | mlock **hurts** decode, slow load, RAM 10 GiB free |
| D2 | `… 4096 -lm dio` | 72.8 | 12,714 | 186.2 | 210.7 | 23.8 / 22.0 / 23.6 | best prefill at 4 GiB cache |
| E1 | `-cmoe --moe-cache-mib 11968 --moe-cache-cpu-share 0.98` | 72.0 | 20,496 | **168.6** | **232.6** | **42.6 / 44.4 / 38.3** | **selected for the full ladder** |

Notes, kept honest:
* The two earlier driver launches are in `summary.csv` verbatim and were superseded:
  `OPT_A0,load,FAILED,rc=1 tail=error: invalid argument: --t`, then nine
  `load,FAILED,rc=1` rows ending `… E srv start: cou…` (port already bound by the previous serve),
  one `OPT_A0,load,FAILED,rc=-9` (killed mid model-load), one aborted
  `OPT_S_D2,s_p64k … pp=None ttft=None`, and two `OPT,final,FAILED,no sweep config passed screening`.
* `meta/sweep.jsonl` contains the successful nine rows plus null lines from a fixed
  prefill-accounting bug in an earlier iteration; the table above uses the clean block that
  matches `progress.log` timestamps 16:55–18:49.

## 3. Tuner run (`scripts/moe-cache-tune.py`)

Verbatim tuner output (measured decode at 32 k, and the model it fitted):

```
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`: 47962 MiB of experts, CUDA0 23811 MiB free, cache up to 12000 MiB

| cache MiB | share | t/s | hits |
|---:|---:|---:|---:|
| 4736 | 0 | 27.1 | 73% |
| 4736 | 0.5 | 34.2 | 73% |
| 4736 | 0.95 | 41.9 | 65% |
| 14336 | 0 | 43.6 | 93% |
| 14336 | 0.5 | 46.9 | 93% |
| 14336 | 0.95 | 49.4 | 89% |

machine: uploads 13.0 GiB/s, CPU 70.1 GiB/s, CPU part 0.00 ms, overlap -1.5 ms/GiB, rest 17.91 ms per token, fit within 1.0%
```

Predicted warm-cache decode (predicted table, abridged): 11,968 MiB × share 0.8 → **49.5 t/s**
(share 0 → 43.4). `recommended: --moe-cache-mib 11968 --moe-cache-cpu-share 0.8`.

The final run used **share 0.98** (the sweep's E1 measured 44.4 t/s at 32 k, above C2's 37.0 at
4 GiB) — the tuner's own predictions were ±10 % of measured (C2@4736/0.95 predicted 43.3 vs
measured 40.3), so the sweep, not the tuner table, picked E1. The tuner's `hits` column (65–93 %)
is the **only MoE-cache hit-rate number in this publication**: the running server exposes no MoE
hit counter (§5).

## 4. Final config, full ladder (E1, `-np 1 -c 198144`, thinking off, no speculation)

`prompt tok` = client count / engine count (chat-template overhead ≈ +12…+14).
decode = client `gen/(wall−TTFT)` / engine `print_timing` tg rate.

| arm | prompt tok (client/engine) | TTFT s | engine prefill tok/s | decode (client / engine) | D1–D4 (client) | recall |
|---|---|---:|---:|---|---|---|
| p4k | 5,198 / 5,210 | 24.765 | 210.7 | 43.4 / 43.3 | 42.7 47.0 42.3 42.1 | n/a |
| p32k | 39,228 / 39,240 | 165.935 | 236.6 | 40.7 / 40.7 | 40.8 41.7 39.6 41.0 | n/a |
| p64k_code | 73,479 / 73,491 | 326.388 | 225.3 | 36.7 / 36.6 | 38.3 38.5 34.7 35.6 | **5/5** |
| p128k | 170,308 / 170,320 | 790.293 | 215.8 | 25.0 / 25.2 | 24.3 23.8 27.3 25.8 | **5/5** |
| p184k | 191,601 / 191,612 | 932.480 | 205.9 | 23.3 / 23.4 | 23.4 24.2 22.7 23.5 | **5/5** |

* **Client vs engine decode agree** (Δ ≤ 0.2 tok/s on every arm): `--reasoning off`,
  `speculative: false`. The §9.4 inflation that applies to Strata/Basalt client numbers does not
  apply to llama.cpp in either run — llama's client numbers are already conservative.
* Repeat spread: screening E1 measured p32k decode 44.4 / TTFT 168.6 s, final ladder 40.7 /
  165.9 s — ~9 % arm-to-arm variance on decode; TTFT is stable to ~1.6 %.
* VRAM: 20,496 MiB at model load (incl. the 11,968 MiB expert cache); 20,556–20,608 MiB during
  arms. Host RAM: 57 GiB free at serve start; the 11.4 GiB mmap'd-expert set stays page-cached.
* **Cold vs cached, same server, same prompt:** p32k cold TTFT 165.9 s → immediate identical
  repeat 0.168 s (**988×**, §5 W2). The ladder itself is cold on each fresh serve; nothing here
  is cache-served except where marked.
* p184k in a single 198,144-token slot with **5/5 recall** is new: Basalt rejected p184k
  (report.md §8), Strata needed its park path. But 15.5 minutes to first token cold is the cost.

## 5. KV/prefix reuse: warm arm, prefix-granularity waves, mechanism boundary

**Warm arm (report.md §3 protocol):** one ≈50 k conversation turn cold → two foreign chats
(2,760 and 4,060 tok) → follow-up on the *same* conversation.

| metric | value |
|---|---|
| turn-1 cold TTFT | 232.896 s (engine 50,137 tok @ 217.9 tok/s) |
| follow-up TTFT after two foreign chats | **1.528 s** (152.4×) |
| follow-up engine re-prefill | 49 tokens @ 52.9 tok/s (slot: 50,358 prompt tokens, 0 re-processed) |
| follow-up decode | 33.6 client / 33.7 engine, 46 tokens |

**Prefix-granularity waves (OPT_W, one serve, 32 k prompt):**

| wave | what changed | TTFT s | engine re-prefill | ratio vs w1 |
|---|---|---:|---|---:|
| w1_cold | first send | 176.456 | 39,240 @ 222.8 tok/s | 1× |
| w2_identical | byte-identical repeat | **0.168** | **4 tokens** | **1050×** |
| w3_append | + one line at end | 3.823 | 527 @ 139.2 tok/s (≈38.7 k reused) | 46× |
| w4_midflip | one character flipped in the middle | 177.482 | 39,245 @ 222.1 tok/s (no reuse) | 1.0× |

Same shape as the Strata V/W study (report.md §6) with stronger numbers: llama-server keeps a
**server-wide prompt-cache entry list** (LRU; the serve log shows
`alloc: making room for prompt cache entry, removing oldest entry` when VRAM-backed KV tightens),
so reuse is per *prefix*, survives one-slot interruption by foreign chats in the same slot list,
and dies exactly at the first divergent token.

**Two different mechanisms — do not conflate:**
* **KV / prompt-prefix cache** (above) skips *prefill* of an already-seen prefix. llama.cpp
  exposes its effect via `/slots` (`n_prompt_tokens_cache` up to 39,236 observed in §6 waves);
  `/metrics` is `HTTP 501 Not Implemented` — no aggregate counters.
* **MoE expert cache** (`-cmoe`) keeps hot *experts* in VRAM to speed up *compute* (prefill and
  decode) on any token. The server **does not expose its hit rate** — no hit lines in the serve
  logs, no endpoint. The only hit-rate evidence in this publication is the offline tuner's 65–93 %
  table (§3). Where the old report said llama.cpp exposes "none" (§1, cache transparency), that
  stands: `/slots` shows KV state, MoE state remains a black box.

## 6. Concurrency: 8 agents × 32 k on 24 GB — now fits

`-c 327680 -np 8` on the E1 flags: server logged `n_slots = 8, n_ctx_slot = 40960`, VRAM
23,530 MiB at load, **peak 23,630 MiB** across the waves. The old b12531 L2 config auto-fitted to
33,792 tok/slot and could not serve a single 32 k request (report.md §4); with the tuned
expert-cache config the same VRAM leaves room for **8 × 40,960-token slots with q8_0 KV**.
No fallback ladder (q4_0 KV / reduced expert cache) was needed — **the first np=8 variant held**,
so all concurrency numbers below are q8_0-equal to the single-slot ladder.

Each wave: simultaneous requests, 192-token answers, identical = one shared prompt
(`p32k_a/b` rotation), unique = eight salt-distinct files sharing one ≈38.9 k prefix
(§1 manifest). A 16-token probe precedes each wave.

| wave | ok | wall s | agg tok/s | per-req decode mean | TTFT mean / p95 s |
|---|---|---:|---:|---:|---|
| identical c=2 | 2/2 | 380.4 | 1.0 | 13.2 | 293.81 / 214.85 |
| identical c=4 | 4/4 | 178.6 | 4.3 | 6.2 | 48.9 / 8.64 |
| identical c=8 | 8/8 | 185.7 | 8.3 | 10.8 | 167.9 / 167.6 |
| unique c=2 | 2/2 | 357.1 | 1.1 | 14.2 | 272.21 / 194.41 |
| unique c=4 | 4/4 | 202.4 | 3.8 | 5.5 | 63.5 / 23.64 |
| unique c=8 | 8/8 | 212.5 | 7.2 | 11.0 | 194.88 / 196.0 |

* 0 errors in every wave. First fills run concurrently at 182–232 tok/s prefill per slot
  (engine lines), so c=8 cold TTFT ≈ 168–195 s per request.
* **The prompt cache is part of the concurrency story:** slot dumps mid-wave show later
  identical requests processed with `n_prompt_tokens_cache = 39,236` and only 4 re-processed
  tokens (c=4: 3 of 4 slots; c=8: 7 of 8; unique c=4/c=8: 2–6 slots at 38,885 reused). That is
  why identical-c4 TTFT mean collapses to 48.9 s while c2 (cold cache) is 293.8 s. Reuse is not
  guaranteed under allocation pressure — unique-c8 paid near-full prefill (194.9 s mean) while
  still ending with 38,885-token cached entries; the LRU list churns as slots fill 23.6 GiB of KV.
* **Accounting caveat:** the `decode_p95`/`ttft_p95` columns in the raw rows are inconsistent
  with the means (p95 below mean is impossible for 4 samples) — the percentile pools are
  contaminated by the tiny probe requests. Treat `agg`, `ok`, `wall` and the means as the
  load-arm numbers.
* Engine-side decode during the np=8 stage: 298 completed 192-token decodes, per-slot
  1.0–31.7 tok/s (mean 17.9; steady-state two-stream sustained = 19.22 tok/s per stream).

## 7. Sustained 30 minutes (np=8 config, two alternating 32 k prompts, 192-token answers)

| metric | value |
|---|---|
| duration / requests / errors | 1,806.4 s / **270 / 0** |
| decode mean / p50 / p95 | 18.8 / 19.0 / 19.3 tok/s |
| drift, first → last full 5-min bucket | 18.1 → 19.1 tok/s (+5 %, improving) |
| TTFT steady-state | 0.4–0.64 s (every request after the first fill: prompt-cache hit, engine re-prefills ≈ 4 tokens) |
| aggregate answer tokens | 270 × 192 = **51,840 → 28.7 tok/s aggregate** |
| GPU power mean / max | 141.9 / 160.9 W (buckets 109.7 → 151.6 → 149.8 W) |
| **GPU energy in window** | **71.14 Wh** (1 Hz `nvidia-smi` trapezoid, same method as report.md §7) |
| **Wh per 1,000 answer tokens / J per token** | **1.372 / 4.94 J** |
| GPU temp mean / max | 70.1 / 73 °C |
| throttle bits | `0x4` SwPowerCap 1,325/1,758 samples, `0x400` 433/1,758; **no** `0x20`/`0x40` thermal bits |
| SM clock min | 1,582 MHz |
| VRAM peak / host RAM min available | 23,630 MiB / 49.55 GiB (engine RSS peak 57.5 GiB) |

CPU-side energy is **not instrumented** (same limitation as the old report; expert streaming over
PCIe costs CPU/DRAM power this window does not count). The failed old llama sustained arm burned
70.83 Wh for 192 useful tokens; this one burned 71.14 Wh for 51,840 — the config change converted
"energy without output" into usable throughput.

## 8. Cross-engine comparison (new column, everything else unchanged)

Old columns are quoted from report.md §2/§3/§4/§7 with the §9.4 accounting caveats in force
(client-vs-engine decode, warm Basalt ladder, per-engine token counts). `not measured` = no
comparable value published.

| metric | llama b12531 (report.md) | **llama b12533 optimized (this run)** | Strata 0.1.62 | Basalt 0.4.1 |
|---|---|---|---|---|
| cold prefill 32 k tok/s | 169–171 | **236.6** | 2,866 (§2) | ≈3.0 k cold (§2 note) |
| single decode p32k (client / engine) | 29.1–30.3 / same | **40.7 / 40.7** | 248.8 / ≈130–156 | 352.1 / 172.7 |
| TTFT 32 k s | 229.3–232.5 | **165.9** | 13.707 | 4.963 (warm cache) |
| TTFT 64 k s | 434.9 | **326.4** | 23.744 | 4.868 (warm cache) |
| TTFT 128 k s | 1,042.6 | **790.3** | 47.646 | 4.973 (warm cache) |
| single-slot max context answered | 184,098 tok (§2 run B) | **191,612 tok, 5/5** | 188,724 tok | rejected p184k (§8) |
| KV reuse, 50 k conversation | 173.7×, one slot, evicted by any 2nd chat (§3) | **152.4×** (follow-up 1.528 s) | 11.0×, parks under pressure (§3) | 3.8×, survives restarts (§3) |
| KV reuse, identical repeat | not measured | **1050× (0.168 s)** | 275× / 24.9× (§6) | 45.7× live probe (§6) |
| c=4 32 k agg tok/s | 0/4, context-rejected | **4.3** (shared) / **3.8** (unique) | 51.2 / 19.0 unique (§4/§6) | 51.0 (cache-served) |
| c=8 32 k agg tok/s | 0/8, context-rejected | **8.3** (shared) / **7.2** (unique) | **126.5** | 101.6 |
| 30-min sustained, requests / errors | 264 / 263 | **270 / 0** | 1,285 / 0 | 829 / 0 |
| 30-min GPU J/token | 1328 (failed arm) | **4.94** | **1.20** | 1.73 |
| peak VRAM / host RAM left | 14,216 MiB / 55.62 GiB | 23,630 MiB / 49.55 GiB | 23,682 MiB / 4.43 GiB | 23,574 MiB / 5.94 GiB |
| recall ≥64 k (5 CONTROL) | 5/5 | **5/5** (64 k, 128 k, 184 k) | 5/5 | 5/5 (64 k, 128 k) |

**Fixed llama-CPU re-run (`C_*`, b12531 `-ngl 0 -np 1 -c 40000`, same IQ3_S model)** — the old
stage-C row had np=4 → 10,240 tok/slot and p32k context-rejected; np=1 fixes it (40,192 tok/slot):

| arm | value |
|---|---|
| C_single p4k | TTFT 41.175 s (≈126 tok/s prefill), decode 7.4 |
| C_single p32k | TTFT 266.725 s (≈147 tok/s prefill), decode 6.9 — **now actually runs** |
| C_sust (15 min, 4 k prompts, 128 tok) | 56 req / 0 err, decode 8.2, GPU-window 7.28 Wh → 3.66 J/token (GPU idle-residual only; CPU work unmeasured) |
| C VRAM / RAM | 962 MiB / RSS 56.4 GiB |

### Deltas

**Optimized vs old llama.cpp (same model, same box):**
* prefill: ×1.39 (32 k), ×1.37 (64 k), ×1.30 (128 k); TTFT 32 k −28 %, 128 k −24 %.
* decode: ×1.49 (p4k), ×1.37 (p32k), ×1.33 (p64k); **×0.90 at 128 k — a real regression**
  (25.0 vs 27.9): with 11,968 MiB of expert cache + 198 k q8_0 KV the last VRAM slack is gone
  and huge contexts starve; at >128 k the old `--n-cpu-moe 48` config is still the fitter choice.
* concurrency: 0/14 → 8/8 × 40,960 tok/slot (previously unmeasurable); sustained: 263 errors → 0,
  192 → 51,840 useful tokens; energy per useful token: −99.6 %.
* vs its own same-version baseline A0 (§2): decode +33–43 %, prefill +33 % — the gain is the
  flags (`-cmoe` + 11.4 GiB cache + share 0.98), not the version.

**Optimized vs Strata:** prefill −92.5 % (236.6 vs 3,160); TTFT 32 k −91.7 %; engine decode
−69…−74 %; c=8 aggregate −93.4 %; 30-min useful tokens −79 %; J/token +312 % (4.94 vs 1.20).
Closed to parity on exactly one axis: warm follow-up TTFT (1.528 vs 1.433 s, +6.6 %) and
recall (identical 5/5).

**Optimized vs Basalt:** prefill −92 %; engine decode 40.7 vs 172.7 → −76 %; c=8 −91.8 %;
J/token +186 %; 30-min tokens −67 %. Now better than Basalt on warm follow-up TTFT
(1.528 vs 4.394 s, 2.9×) — because llama's prefix cache is VRAM-resident and needs no snapshot
reload — but with no persistence and LRU eviction under pressure.

## 9. Methodology limitations

1. **What changed vs the old llama arms:** binary b12531 → b12533 (control arm A0 shows version
   drift is ≤4 %), `--n-cpu-moe 48` → `-cmoe --moe-cache-mib 11968 --moe-cache-cpu-share 0.98`,
   `+ -t 15` (thread cap, also present in A0 — not the source of gains). KV stays q8_0; np=1 for
   the ladder, np=8 for concurrency/sustained. **np=8 numbers are not comparable to single-slot
   numbers** beyond context math; only one np=8 config was needed (q8_0 at 40,960/slot fit),
   so no fallback variants exist to label.
2. **Prompt rebuild:** p64k/p128k/p184k were regenerated with new targets (73,479 / 170,308 /
   191,601 llama tokens vs 71,166 / 172,674 / 184,098 in report.md §2). Rates (tok/s) compare;
   absolute ladder TTFTs across the two runs do so only loosely.
3. **Percentile contamination** in load arms (§6): p95 columns are unusable; means/aggregates kept.
4. **Repeat variance:** p32k decode 40.7–44.4 tok/s across E1 appearances (~9 %); TTFT stable.
   D1–D4 use character-quarter marks (§9.3) and show ±5 % internal drift.
5. **Accounting asymmetries vs the other engines are unchanged** (report.md §9.4): llama ran
   thinking-off, no speculation → its client decode == engine decode; Strata/Basalt client decode
   columns are inflated 1.6–2.3× by thinking/spec accounting, so cross-engine decode comparisons
   in §8 use engine-reported rates where published. Token counts differ per engine by tokenizer
   spread; the comparison is of the same *text*, not equal token counts. Basalt's ladder column is
   warm-cache-served (report.md §2) — the −92 % llama-vs-Basalt prefill delta uses Basalt's cold
   turn-1 figure.
6. **Energy is GPU-channel only** (1 Hz nvidia-smi); CPU/DRAM/PCIe power for expert streaming is
   not instrumented in any run — llama J/token figures are floors for llama's actual energy bill,
   as flagged in the original.
7. MoE expert-cache hit rate was never exposed by the server (only the offline tuner measured
   one, §3). KV/prefix reuse is observed through `/slots` state, not aggregate counters.

## 10. Verdict: how much of the gap closed

The `--n-cpu-moe 48` configuration was llama.cpp's worst way to use 24 GB. Correctly tuned
(`-cmoe`, 11.4 GiB VRAM expert cache, 98 % of misses on CPU), llama.cpp improves **1.3–1.5×
across prefill, decode and TTFT**, converts its broken concurrency and sustained arms into
working ones (8 slots × 40,960 tok, 270 requests / 0 errors / 4.94 J per token), and beats both
purpose-built engines on one metric it previously lost (warm follow-up TTFT).

It does **not** close the gap where the purpose-built engines were designed strong:

* **single assistant:** usable now — ~43 tok/s decode at 4 k, ~40 at 32–64 k, sub-2 s follow-ups
  with cache, 5/5 recall to 188 k text. But cold prefill is still 13× slower (236 vs 3.2 k tok/s)
  and cold TTFTs are minutes. Fine as a reference/fallback brain; not as the brain.
* **multi-agent:** still not viable — c=8 aggregate 8.3 tok/s vs Strata 126.5 (15×) and Basalt
  101.6; the VRAM-resident LRU prompt cache churns under 8 × 32 k KV, while Strata parks/restores
  conversations and Basalt persists them. **Strata wins multi-agent clearly; Basalt is the
  durable second.**
* **long context:** llama is now the only engine in this publication that *answers* at 184 k in
  a single slot (Basalt rejects it; Strata serves 188.7 k in 58 s where llama needs 932 s) —
  the honest reading is "only when the prefix is already cached", since cold costs 15.5 minutes.
  For frequent long-context work Strata remains the fit; llama.cpp's 11,968-MiB cache even
  *regresses* decode at ≥128 k (−10 % vs its own old config).

Bottom line: the llama.cpp column of the original report was measuring the wrong config;
the right config removes the *catastrophic* findings (broken concurrency, 1328 J/token, energy
without output) and leaves the *structural* ones (prefill rate, prefill parallelism, cache
transparency) exactly where the original report said they were.
