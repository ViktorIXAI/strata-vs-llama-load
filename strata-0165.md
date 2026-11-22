# Strata 0.1.65 vs the 2026-11-19 Strata arm, same Qwen pack, same laptop

People said the new Strata did not get faster. This checks that on the same card and the same model as the published ladder, without replacing those numbers.

- Old arm: Strata 0.1.62, published in [strata-vs-llama-load](https://github.com/ViktorIXAI/strata-vs-llama-load) on 2026-11-19. Client decode in that table runs above the engine's own rate; the verdict there put engine-class decode at 94–156 tok/s.
- New arm: Strata terminal 0.2.19, engine `strata` 0.1.65, same IQ3_S Qwen3.8-Flash-Next files, same context cap 196,608, same MTP drafter (DFlash2 was not downloaded, so the launcher stayed on MTP, which is also what the old arm used). Same prompt files. Measured 2026-11-22.

Hardware is the same laptop RTX 5090, 24 GB, 175 W cap.

## Prefill and time to first token

| Arm | Old prompt tokens | Old TTFT s | Old prefill tok/s | New prompt tokens | New TTFT s | New prefill tok/s |
|---|---:|---:|---:|---:|---:|---:|
| p4k | 5,250 | 4.8 | 1,094 | 5,250 | 3.0 | 1,732 |
| p32k | 39,280 | 13.7 | 2,866 | 39,280 | 12.2 | 3,221 |
| p64k | 75,023 | 23.7 | 3,160 | 73,532 | 21.6 | 3,401 |
| p128k | 152,642 | 47.6 | 3,204 | 170,362 | 50.0 | 3,406 |
| p184k | 188,724 | 58.3 | 3,236 | 191,654 | 56.1 | 3,417 |

Long-context prefill moved from about 3.2k to about 3.4k tok/s, roughly +6%. Not the +76% the 0.1.65 notes quote for a desktop RTX 5090. The short p4k arm looks better (+58%) because that arm was overhead-dominated last time.

## Decode

New engine log, answer-phase rate (thinking tokens are separate and are not counted here):

| Arm | Answering tok/s | Done line tok/s |
|---|---:|---:|
| p4k | 138 | 132 |
| p32k | 134 | 135 |
| p128k | 112 | 101 |
| p184k | 109 | 106 |

That sits inside the old engine-class band of 94–156, not above it. The old client column (221–273) is a different instrument and is not beaten by these engine lines.

## Cache return

A 50,177-token chat, then a short other chat, then back: return time to first token **0.16 s**. The old arm's return on the same shape of test was **1.43 s**. That part did get faster.

## Conclusion

On this laptop, on this Qwen pack, the new engine's long-context prefill is a small step up and decode is unchanged. The claim that performance did not really grow is fair for this card. The cache return is the one clear improvement. The desktop +76% prefill figure does not show up here.
