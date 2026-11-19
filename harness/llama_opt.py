# -*- coding: utf-8 -*-
"""llama_opt: optimize llama.cpp (latest build) for the IQ3_S MoE on a 24 GB
RTX 5090 Laptop, then run the full original arm set on the winning config.

Design: quick screening over memory/MoE-cache configs (short 4k/32k/64k arms,
no long loads), rank, then ONE full stage (ladder, warm/KV, concurrency with
per-slot ctx validation, 30-min sustained). Every config — including failures
and OOMs — is recorded verbatim. Nothing from the old run is modified: new
tags are OPT_*.

Mechanisms probed (verified present in b12533 via docs/moe-cache.md):
  -cmoe / --n-cpu-moe     experts stay in host RAM
  --moe-cache-mib N       LRU expert cache in VRAM
  --moe-cache-cpu-share F CPU-compute share of cache misses
  -lm none|mlock|dio      host-expert pinning modes
  scripts/moe-cache-tune.py  auto-tuning run
"""
import time, os, re, subprocess, json, hashlib
from loadbench import (HOME, B, GG1, log, summary, pf, STOP_AT_OK, Engine,
                       start_serve, wait_health, stop_serve, single_arm,
                       warm_arm, load_arm, sustained_arm, engine_line,
                       wait_vram_free, telem_stats)

OPT_BIN = HOME + "/llama.cpp/build-opt/bin/llama-server"
OPT_PORT = 18087
CTX_FULL = 198144
META = B + "/meta"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def gpu_mem_used():
    return sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")


def ram_avail():
    return sh("free -g | awk '/^Mem/{print $7}'")


def serve_cmd(cfg, np_slots=1, ctx=CTX_FULL):
    cmd = [OPT_BIN, "-m", GG1, "--alias", "iq3s-opt", "-ngl", "99",
           "--flash-attn", "on", "--reasoning", "off",
           "-c", str(ctx), "-np", str(np_slots), "--t", "15",
           "--host", "127.0.0.1", "--port", str(OPT_PORT)]
    if cfg.get("ncmoe"):
        cmd += ["--n-cpu-moe", cfg["ncmoe"]]
    if cfg.get("cmoe"):
        cmd += ["-cmoe"]
    if cfg.get("cache_mib"):
        cmd += ["--moe-cache-mib", str(cfg["cache_mib"])]
    if cfg.get("share"):
        cmd += ["--moe-cache-cpu-share", str(cfg["share"])]
    if cfg.get("lm"):
        cmd += ["-lm", cfg["lm"]]
    cmd += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"] if cfg.get("kv8", True) \
        else ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]
    return cmd


def try_config(cfg, np_slots=1, ctx=CTX_FULL, timeout=420):
    """Start serve with cfg; return (proc, logfh, logfile, health, load_s) or None."""
    logfile = f"{B}/logs/OPT.{cfg['id']}.serve.log"
    cmd = serve_cmd(cfg, np_slots, ctx)
    log(f"OPT serve {cfg['id']}: {' '.join(cmd[3:])}")
    p, fh = start_serve(cmd, logfile)
    eng = Engine("llama", f"http://127.0.0.1:{OPT_PORT}", "iq3s-opt")
    h, rc, load_s = wait_health(eng, p, t=timeout)
    if h is None:
        tail = open(logfile, errors="ignore").read()[-400:].replace("\n", " | ")
        summary(("OPT_" + cfg["id"], "load", "FAILED", f"rc={rc} tail={tail[:200]}"))
        try:
            os.killpg(os.getpgid(p.pid), 15)
        except Exception:
            pass
        fh.close()
        wait_vram_free()
        return None
    slot = re.search(r"n_ctx_slot = (\d+)", open(logfile, errors="ignore").read())
    log(f"OPT serve {cfg['id']}: up in {load_s:.0f}s n_ctx_slot={slot.group(1) if slot else '?'} vram={gpu_mem_used()}MiB ram_free={ram_avail()}GiB")
    return p, fh, logfile, eng, load_s


def prefill_rate(r):
    if r and r.get("ttft_s") and r.get("usage", {}).get("prompt_tokens"):
        return r["usage"]["prompt_tokens"] / r["ttft_s"]
    return None


def screen(cfg):
    """Quick screening: p4k 256 tok, p32k 256 tok, p64k 256 tok. Returns dict or None."""
    got = try_config(cfg)
    if not got:
        return None
    p, fh, logfile, eng, load_s = got
    out = {"id": cfg["id"], "cfg": {k: v for k, v in cfg.items() if k != "id"},
           "load_s": round(load_s, 1), "vram_mib": gpu_mem_used(), "ram_free_gib": ram_avail()}
    try:
        eng.stream_chat("Say OK.", 8)  # warmup
        r4 = single_arm(eng, logfile, "s_p4k", pf("p4k"), 256, "OPT_S_" + cfg["id"])
        r32 = single_arm(eng, logfile, "s_p32k", pf("p32k"), 256, "OPT_S_" + cfg["id"])
        r64 = single_arm(eng, logfile, "s_p64k", pf("p64k_code"), 256, "OPT_S_" + cfg["id"])
        for nm, r in (("p4k", r4), ("p32k", r32), ("p64k", r64)):
            if not r:
                out["fail"] = nm
                return out
            out[nm] = {"ttft": r.get("ttft_s"), "decode": r.get("decode_tok_s"),
                       "prefill": round(prefill_rate(r) or 0, 1)}
        m = re.findall(r"speed_tg = ([0-9.]+)", open(logfile, errors="ignore").read())
        out["engine_tg"] = m[-1] if m else None
    finally:
        stop_serve(p, fh)
        wait_vram_free()
    return out


def run_sweep():
    os.makedirs(META, exist_ok=True)
    tp = subprocess.Popen(["python3", B + "/telem.py", f"{B}/logs/OPT_sweep.telem.csv", "llama"])
    configs = [
        {"id": "A0", "ncmoe": "48"},                                   # original arm
        {"id": "B1", "cmoe": True, "cache_mib": 1024},
        {"id": "B2", "cmoe": True, "cache_mib": 2048},
        {"id": "B3", "cmoe": True, "cache_mib": 4096},
        {"id": "B4", "cmoe": True, "cache_mib": 8192},
        {"id": "C1", "cmoe": True, "cache_mib": 4096, "share": 0.5},
        {"id": "C2", "cmoe": True, "cache_mib": 4096, "share": 0.95},
        {"id": "D1", "cmoe": True, "cache_mib": 4096, "lm": "mlock"},
        {"id": "D2", "cmoe": True, "cache_mib": 4096, "lm": "dio"},
    ]
    results = []
    for cfg in configs:
        if not STOP_AT_OK():
            break
        r = screen(cfg)
        results.append(r)
        with open(f"{META}/sweep.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        if r and "fail" not in r and r.get("p32k"):
            log(f"SWEEP {cfg['id']}: prefill32k={r['p32k']['prefill']} dec4k={r['p4k']['decode']} dec32k={r['p32k']['decode']} vram={r['vram_mib']}")
        else:
            log(f"SWEEP {cfg['id']}: FAIL {r.get('fail') if r else 'load'}")
    tp.terminate()
    ok = [r for r in results if r and "fail" not in r]
    ok.sort(key=lambda r: (r["p32k"]["prefill"] + r["p64k"]["prefill"]) / 2, reverse=True)
    log("SWEEP_RANK: " + ", ".join(f"{r['id']}:{r['p32k']['prefill']}/{r['p4k']['decode']}" for r in ok[:4]))
    return [r["id"] for r in ok[:3]]


def run_tuner():
    """scripts/moe-cache-tune.py — auto pick cache size / cpu-share."""
    script = HOME + "/llama.cpp/scripts/moe-cache-tune.py"
    if not os.path.exists(script):
        log("TUNER: script absent")
        return None
    cmd = (f"cd {HOME}/llama.cpp && timeout 3000 python3 scripts/moe-cache-tune.py "
           f"--max-mib 12000 --markdown -- {OPT_BIN} -m {GG1} -ngl 99 -cmoe -t 15 "
           f"-c 32768 --flash-attn on --reasoning off --port {OPT_PORT}")
    log("TUNER: starting moe-cache-tune.py (up to 50 min)")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    open(f"{META}/tuner.out", "w").write(r.stdout + "\n--STDERR--\n" + r.stderr[-4000:])
    m = re.search(r"Recommended:\s*(--moe-cache-mib\s+(\d+)\s+--moe-cache-cpu-share\s+([0-9.]+)[^,\n]*)", r.stdout)
    if m:
        log("TUNER: " + m.group(1))
        summary(("OPT_tuner", "recommended", m.group(2), m.group(3)))
        return {"id": "E1", "cmoe": True, "cache_mib": int(m.group(2)), "share": float(m.group(3))}
    log("TUNER: no recommendation line, see meta/tuner.out")
    return None


def engine_decode_tail(logfile, n=3):
    txt = open(logfile, errors="ignore").read()
    tg = re.findall(r"speed_tg = ([0-9.]+)", txt)
    ps = re.findall(r"speed_ps = ([0-9.]+)", txt)
    return tg[-n:], ps[-n:]


def unique_prompts():
    base = open(pf("p32k_a")).read()
    files, man = [], []
    for i in range(1, 9):
        txt = base + f"\n\nUNIQUE-SESSION-SALT-{i}: treat this conversation as distinct.\n"
        pth = pf(f"p32k_u{i}")
        open(pth, "w").write(txt)
        files.append(pth)
        man.append(f"{os.path.basename(pth)} sha256={hashlib.sha256(txt.encode()).hexdigest()[:16]} tok=+salt")
    open(f"{META}/unique_manifest.txt", "w").write("\n".join(man) + "\n")
    return files


def full_run(cfg):
    log(f"OPT FULL config: {json.dumps(cfg)}")
    summary(("OPT_FINAL", "config", json.dumps({k: v for k, v in cfg.items() if k != "id"})))
    # ---- single ladder, np=1 ----
    got = try_config(cfg)
    if not got:
        summary(("OPT_single", "load", "FAILED", "final config did not load"))
        return
    p, fh, logfile, eng, load_s = got
    try:
        eng.stream_chat("Say OK.", 8)
        for nm, mt in (("p4k", 512), ("p32k", 512), ("p64k_code", 512),
                       ("p128k", 512), ("p184k", 512)):
            if not STOP_AT_OK():
                break
            single_arm(eng, logfile, nm, pf(nm), mt, "OPT_single", wall_limit=2400,
                       recall=True, tele=f"{B}/logs/OPT_ladder.telem.csv")
        tg, ps = engine_decode_tail(logfile)
        summary(("OPT_dec", "engine_speed_tg", ",".join(tg), "speed_ps=" + ",".join(ps)))
        # ---- KV reuse arm ----
        if STOP_AT_OK():
            warm_arm(eng, logfile, "OPT_warm", pf("p40k"), pf("p4k_f1"))
        # ---- prefix-granularity mini-W ----
        if STOP_AT_OK():
            for nm, txt_file, extra in (
                    ("w1_cold", "p32k", None),
                    ("w2_identical", "p32k", None),
                    ("w3_append", "p32k_app1", None),
                    ("w4_midflip", "p32k_flip", None)):
                src = open(pf("p32k")).read()
                if nm == "w3_append":
                    src = src + "\nOne extra line for prefix-granularity probing.\n"
                if nm == "w4_midflip":
                    src = src[: len(src) // 2] + "XZQJ " + src[len(src) // 2 + 5:]
                open(pf(txt_file), "w").write(src)
                r = single_arm(eng, logfile, nm, pf(txt_file), 64, "OPT_W", wall_limit=600)
                if r:
                    log(f"OPT_W/{nm}: ttft={r.get('ttft_s')}s")
    finally:
        stop_serve(p, fh)
        wait_vram_free()

    # ---- concurrency np=8 with validated per-slot ctx ----
    variants = [
        dict(cfg, kv8=True, cache_mib=cfg.get("cache_mib", 4096)),
        dict(cfg, kv8=False, cache_mib=cfg.get("cache_mib", 4096)),
        dict(cfg, kv8=False, cache_mib=min(cfg.get("cache_mib", 4096), 2048)),
    ]
    loaded = None
    for vi, v in enumerate(variants):
        v["id"] = f"{cfg['id']}c{vi}"
        if not STOP_AT_OK():
            break
        got = try_config(v, np_slots=8, ctx=327680, timeout=420)
        if not got:
            continue
        slot = re.search(r"n_ctx_slot = (\d+)", open(got[2], errors="ignore").read())
        nslot = int(slot.group(1)) if slot else 0
        if nslot < 39680:
            log(f"OPT np=8 variant {v['id']}: n_ctx_slot={nslot} < 39680 -> rejected, recorded")
            summary(("OPT_np8", v["id"], "CTX_SLOT_TOO_SMALL", f"n_ctx_slot={nslot}"))
            stop_serve(got[0], got[1])
            wait_vram_free()
            continue
        loaded = (v, got)
        break
    if loaded:
        v, (p, fh, logfile, eng, ls) = loaded
        summary(("OPT_np8", "ok", v["id"], f"n_ctx_slot>=39680 cache_mib={v.get('cache_mib')} kv8={v.get('kv8')}"))
        try:
            eng.stream_chat("Say OK.", 8)
            for c in (2, 4, 8):
                if not STOP_AT_OK():
                    break
                load_arm(eng, logfile, f"load32k_c{c}", [pf("p32k")], c, 192, "OPT_load")
            ufiles = unique_prompts()
            for c in (2, 4, 8):
                if not STOP_AT_OK():
                    break
                load_arm(eng, logfile, f"load32k_u_c{c}", ufiles, c, 192, "OPT_load")
            if STOP_AT_OK():
                sustained_arm(eng, "OPT_sust", 30 * 60, [pf("p32k_a"), pf("p32k_b")], 192,
                              f"{B}/logs/OPT_sust.telem.csv")
        finally:
            stop_serve(p, fh)
            wait_vram_free()
    else:
        summary(("OPT_np8", "all_variants", "FAILED", "8 slots x 40k tokens do not fit 24GB: KV q8/q4 + reduced MoE cache all failed"))
        log("OPT np=8: NO variant fit — recorded as resource limitation")


def run():
    if not os.path.exists(OPT_BIN):
        summary(("OPT", "build", "MISSING", "build-opt/bin/llama-server absent"))
        return
    if not STOP_AT_OK():
        return
    best_ids = run_sweep()
    tuner_cfg = run_tuner()
    if not STOP_AT_OK():
        return
    final = None
    if tuner_cfg:
        r = screen(tuner_cfg)
        with open(f"{META}/sweep.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        if r and "fail" not in r:
            log(f"SWEEP E1(tuned): prefill32k={r['p32k']['prefill']} dec4k={r['p4k']['decode']}")
            ok = {i: None for i in best_ids}
            # rank E1 against the sweep winners
            cands = [(r, tuner_cfg)]
            for line in open(f"{META}/sweep.jsonl"):
                d = json.loads(line)
                if d.get("id") in ok and d.get("p32k"):
                    cands.append((d, {"id": d["id"], **d["cfg"]}))
            cands.sort(key=lambda x: (x[0]["p32k"]["prefill"] + x[0]["p64k"]["prefill"]) / 2, reverse=True)
            final = cands[0][1]
    if final is None and best_ids:
        for line in open(f"{META}/sweep.jsonl"):
            d = json.loads(line)
            if d.get("id") == best_ids[0]:
                final = {"id": d["id"], **d["cfg"]}
                break
    if final is None:
        summary(("OPT", "final", "FAILED", "no sweep config passed screening"))
        return
    log(f"OPT FINALIST: {final['id']}")
    full_run(final)
