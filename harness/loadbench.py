#!/usr/bin/env python3
"""Load benchmark: Strata v0.2.16 (engine 0.1.62) vs latest llama.cpp,
same IQ3_S Qwen3.8-Flash-Next GGUF, same P2005 (RTX 5090 Laptop 24 GB, 60 GB RAM).

Stages (single process, sequential, deadline-checked):
  S  strata serve: single-stream anchor, concurrency ladder, warm/parking, sustained
  L1 llama serve np=1: single-stream anchor, warm-turn behaviour
  L2 llama serve np=8: concurrency ladder, sustained
Ground rule: no invented numbers; every arm is a measured run or a recorded failure.
Client metrics come from the SSE stream; engine-side numbers are parsed from each
engine's own log lines. Token counts always come from the engine /tokenize.
"""
import csv, hashlib, json, os, re, signal, statistics, subprocess, sys, threading, time
import urllib.request

HOME = os.path.expanduser("~")
B = HOME + "/bench-load"
STRATA_EXE = HOME + "/strata-0216/engine/build/cuda-compile/strata"
STRATA_PY = HOME + "/strata-0216/terminal/.venv/bin/python"
STRATA_SRV = HOME + "/strata-0216/terminal/serve/server.py"
GG1 = HOME + "/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
GG2 = HOME + "/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
LLAMA_BIN = HOME + "/llama.cpp/build-load/bin/llama-server"
PROD_CFG = HOME + "/Strata/strata-iq3_s.json"
API_KEY = json.load(open(PROD_CFG)).get("api_key", "")
STRATA_PORT, LLAMA_PORTS = 18081, (18082, 18083)
DONE_RE = re.compile(r"\[server\]\s+\S+\s+done\s+(.*)$")
REUSE_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read")
LL_PP = re.compile(r"prompt eval time = ([\d.]+) ms / (\d+) tokens .*? ([\d.]+) tokens per second", re.S)
LL_TG = re.compile(r"eval time = ([\d.]+) ms / (\d+) tokens .*? ([\d.]+) tokens per second", re.S)
STOP_AT = int(os.environ.get("STOP_AT", "0"))


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)
    with open(B + "/progress.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d_%H:%M:%S ") + m + "\n")


def beat(msg):
    with open(B + "/heartbeat.txt", "w") as f:
        f.write(time.strftime("%Y-%m-%d_%H:%M:%S ") + msg + "\n")


def summary(row):
    with open(B + "/summary.csv", "a") as f:
        f.write(",".join(str(x) for x in row) + "\n")


class Engine:
    def __init__(self, name, base, alias, key="", defaults=None):
        self.name, self.base, self.alias, self.key = name, base, alias, key
        self.defaults = defaults or {}

    def req(self, path, payload=None, timeout=60, method=None):
        data = json.dumps(payload).encode() if payload is not None else None
        hdr = {"Content-Type": "application/json"}
        if self.key:
            hdr["Authorization"] = "Bearer " + self.key
        r = urllib.request.Request(self.base + path, data=data, headers=hdr,
                                   method=method or ("POST" if data else "GET"))
        return json.loads(urllib.request.urlopen(r, timeout=timeout).read())

    def tokenize(self, text):
        try:
            d = self.req("/tokenize", {"content": text, "parse_special": False}, timeout=300)
            return d["tokens"] if isinstance(d, dict) else d
        except Exception:
            try:
                d = self.req("/tokenize", {"content": text}, timeout=300)
                return d["tokens"] if isinstance(d, dict) else d
            except Exception:
                return None

    def health(self, timeout=8):
        return self.req("/health", timeout=timeout)

    def stream_chat(self, prompt, max_tokens, history=None, wall_limit=1500, extra=None):
        msgs = (history or []) + [{"role": "user", "content": prompt}]
        pl = {"model": self.alias, "messages": msgs, "max_tokens": max_tokens,
              "temperature": 0, "stream": True,
              "stream_options": {"include_usage": True}}
        pl.update(self.defaults)
        if extra:
            pl.update(extra)
        t0 = time.time()
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps(pl).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.key}
                                     if self.key else
                                     {"Content-Type": "application/json"}, method="POST")
        ttft, marks, usage, text = None, [], {}, []
        truncated = False
        with urllib.request.urlopen(req, timeout=wall_limit) as r:
            for raw in r:
                if time.time() - t0 > wall_limit:
                    truncated = True
                    break
                if not raw.startswith(b"data:"):
                    continue
                s = raw[5:].strip()
                if s == b"[DONE]":
                    break
                try:
                    d = json.loads(s)
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                ch = (d.get("choices") or [{}])[0]
                dl = ch.get("delta") or {}
                piece = dl.get("content") or dl.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
                    marks.append((time.time() - t0, len("".join(text))))
        wall = time.time() - t0
        gen = usage.get("completion_tokens") or 0
        gen_est = False
        if not gen and text:
            try:
                t2 = self.tokenize("".join(text))
                if t2:
                    gen, gen_est = len(t2), True
            except Exception:
                pass
        dec_w = (wall - ttft) if ttft else wall
        row = {"ttft_s": round(ttft, 3) if ttft else None, "wall_s": round(wall, 2),
               "gen_tokens": gen, "gen_est": gen_est, "usage": usage, "truncated": truncated,
               "decode_tok_s": round(gen / dec_w, 1) if gen and dec_w > 0.5 else None,
               "t_start": t0}
        row["_text"] = "".join(text)
        row["_marks"] = marks
        return row


def gpu():
    q = ("memory.used,power.draw,temperature.gpu,utilization.gpu,clocks.current.sm,"
         "clocks_throttle_reasons.active")
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip().split(", ")
        return {"vram_mib": int(s[0]), "power_w": float(s[1]), "temp_c": int(s[2]),
                "util": int(s[3]), "sm_clock": int(s[4]), "throttle": s[5]}
    except Exception:
        return {}


def telem_stats(path, t0, t1):
    out = {}
    try:
        rows = [r for r in csv.DictReader(open(path)) if r.get("ts") and t0 <= float(r["ts"]) <= t1]
    except Exception:
        return out
    if not rows:
        return out
    def vals(k):
        v = []
        for r in rows:
            try:
                v.append(float(r[k]))
            except Exception:
                pass
        return v
    for k in ("vram_used_mib", "proc_rss_kb", "power_w", "gpu_temp", "sm_clock", "ram_avail_b"):
        v = vals(k)
        if v:
            if k in ("sm_clock", "ram_avail_b"):
                out["min_" + k] = round(min(v), 1)
            elif k == "gpu_temp":
                out["max_temp_c"] = round(max(v), 1)
            elif k == "power_w":
                out["avg_power_w"] = round(sum(v) / len(v), 1)
                out["max_power_w"] = round(max(v), 1)
            else:
                out["peak_" + k] = round(max(v), 1)
    out["throttled_samples"] = sum(1 for r in rows
                                   if r.get("throttle", "0x0").strip() not in ("0x0", "0", ""))
    out["telem_samples"] = len(rows)
    return out


def start_serve(cmd, logfile):
    fh = open(logfile, "w")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return p, fh


def wait_health(eng, p, t=600):
    t0 = time.time()
    while time.time() - t0 < t:
        if p.poll() is not None:
            return None, p.returncode, time.time() - t0
        try:
            d = eng.health()
            if d.get("loaded") or d.get("status") == "ok":
                return d, 0, time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    return None, -1, time.time() - t0


def stop_serve(p, fh):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass
    for _ in range(30):
        if p.poll() is not None:
            break
        time.sleep(1)
    if p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(3)
    fh.close()


def wait_vram_free(min_mib=20000, t=180):
    t0 = time.time()
    while time.time() - t0 < t:
        if gpu().get("vram_mib", 99999) < min_mib and gpu().get("vram_mib") != 99999:
            if gpu().get("vram_mib", 0) < 3000:
                return True
        time.sleep(3)
    return gpu().get("vram_mib", 0) < 5000


def engine_line(logfile, pat):
    out = {}
    try:
        txt = open(logfile, errors="replace").read()
    except Exception:
        return out
    m = None
    for m in DONE_RE.finditer(txt):
        pass
    if m:
        s = m.group(1)
        for k, rx, cast in (("pp", r"pp=(\d+)", int), ("pp_cached", r"pp=\d+/(\d+)c", int),
                            ("decode_tok_s_engine", r"tg=\d+\s+([\d.]+)t/s", float),
                            ("accept_pct", r"acc=(\d+)%", float),
                            ("prefill_tok_s_engine", r"pp=\S+\s+[\d.]+s\s+(\d+)t/s", float)):
            mm = re.search(rx, s)
            if mm:
                out[k] = cast(mm.group(1))
    rr = None
    for rr in REUSE_RE.finditer(txt):
        pass
    if rr:
        out["reused"] = int(rr.group(2))
        out["read"] = int(rr.group(3))
    ll = LL_PP.findall(txt[-8000:])
    if ll:
        out["llama_prefill_ms"], out["llama_pp_tok"], out["llama_prefill_tok_s"] = \
            round(float(ll[-1][0]), 1), int(ll[-1][1]), float(ll[-1][2])
    lt = LL_TG.findall(txt[-8000:])
    if lt:
        out["llama_decode_ms"], out["llama_tg_tok"], out["llama_decode_tok_s"] = \
            round(float(lt[-1][0]), 1), int(lt[-1][1]), float(lt[-1][2])
    return out


def quarters(eng, marks, text):
    q = {}
    if not marks or not text:
        return q
    ntok = 0
    mm = []
    for t, nchar in marks:
        n = eng.tokenize(text[:nchar])
        n = len(n) if n is not None else 0
        if n > ntok:
            mm.append((t, n))
            ntok = n
    if ntok < 8 or len(mm) < 2:
        return q
    bounds = [0, ntok * 0.25, ntok * 0.5, ntok * 0.75, ntok]
    for i in range(4):
        lo, hi = bounds[i], bounds[i + 1]
        a = next((x for x in mm if x[1] >= lo), None)
        b = next((x for x in mm if x[1] >= hi), None)
        if a and b and b[0] > a[0]:
            q[f"D{i+1}_tok_s"] = round((hi - lo) / (b[0] - a[0]), 1)
    return q


def pf(name):
    return f"{B}/prompts/{name}.txt"


KV_HINT_RE = re.compile(r"prompt \d+ tokens = \d+ reused \+ \d+ read|parked \d+ tokens.*bytes=\d+|kv-grow|no VRAM", re.I)


def kv_lines_since(logfile, off):
    try:
        with open(logfile, errors="replace") as f:
            f.seek(off)
            data = f.read()
            new_off = f.tell()
    except Exception:
        return [], off
    return [l.strip()[:220] for l in data.splitlines() if KV_HINT_RE.search(l)], new_off


def kv_poller(eng, sink, stop_ev, t0, interval=5.0):
    while not stop_ev.is_set():
        row = {"t": round(time.time() - t0, 1)}
        for p in ("/metrics", "/slots"):
            try:
                row[p] = eng.req(p, timeout=4)
            except Exception as e:
                row[p] = repr(e)[:60]
        sink.append(row)
        time.sleep(interval)


def single_arm(eng, logfile, name, prompt, max_tokens, tag, wall_limit=1500, recall=False, tele=None):
    t0 = time.time()
    try:
        r = eng.stream_chat(open(prompt).read(), max_tokens, wall_limit=wall_limit)
    except Exception as e:
        summary((tag, name, "FAILED", repr(e)[:120]))
        log(f"{tag}/{name}: FAILED {repr(e)[:160]}")
        return None
    engline = engine_line(logfile, None)
    q = quarters(eng, r["_marks"], r["_text"])
    ptoks = len(eng.tokenize(open(prompt).read()) or [])
    if not ptoks:
        ptoks = engline.get("prompt") or engline.get("prompt_tokens")
    row = {"arm": name, "tag": tag, "prompt_tokens": ptoks,
           "ttft_s": r["ttft_s"], "wall_s": r["wall_s"], "gen_tokens": r["gen_tokens"],
           "decode_tok_s": r["decode_tok_s"], "engine": engline, "quarters": q,
           "telem": (telem_stats(tele, t0, time.time()) if tele else gpu())}
    if recall:
        try:
            mk = json.load(open(B + "/prompts/manifest.json")).get(
                os.path.basename(prompt)[:-4], {}).get("markers", [])
            if mk:
                hist = [{"role": "user", "content": open(prompt).read()},
                        {"role": "assistant", "content": r["_text"]}]
                r2 = eng.stream_chat("List every CONTROL record in the dump above as `id = value`, "
                                     "one per line. If you cannot find one, say so.",
                                     512, history=hist, wall_limit=900)
                hit = {m["id"]: (m["id"] in r2["_text"] and m["value"] in r2["_text"]) for m in mk}
                row["recall"] = {"asked": len(mk), "exact": sum(hit.values()), "detail": hit}
                open(f"{B}/logs/{tag}_{name}.recall.txt", "w").write(r2["_text"])
                log(f"{tag}/{name} recall: {sum(hit.values())}/{len(mk)} exact")
        except Exception as e:
            row["recall"] = {"error": repr(e)[:120]}
    open(f"{B}/logs/{tag}_{name}.answer.txt", "w").write(r["_text"])
    with open(B + "/results_raw.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary((tag, name, "ok", f"pp={ptoks}", f"ttft={r['ttft_s']}",
             f"dec={r['decode_tok_s']}", f"D={[q.get(k) for k in ('D1_tok_s','D2_tok_s','D3_tok_s','D4_tok_s')]}"))
    log(f"{tag}/{name}: pp={ptoks} ttft={r['ttft_s']}s decode={r['decode_tok_s']} "
        f"quarters={[q.get(k) for k in ('D1_tok_s','D2_tok_s','D3_tok_s','D4_tok_s')]}")
    return row


def load_arm(eng, logfile, name, prompt_files, nconc, max_tokens, tag):
    res, lk = [], threading.Lock()

    def work(i):
        pth = prompt_files[i % len(prompt_files)]
        try:
            r = eng.stream_chat(open(pth).read(), max_tokens, wall_limit=1800)
            with lk:
                res.append(r)
        except Exception as e:
            with lk:
                res.append({"error": repr(e)[:120], "t_start": time.time()})

    t0 = time.time()
    off0 = os.path.getsize(logfile) if os.path.exists(logfile) else 0
    kvs, stop_ev = [], threading.Event()
    pt = threading.Thread(target=lambda: kv_poller(eng, kvs, stop_ev, t0), daemon=True)
    pt.start()
    ths = [threading.Thread(target=work, args=(i,)) for i in range(nconc)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    stop_ev.set()
    klines, _ = kv_lines_since(logfile, off0)
    wall = time.time() - t0
    ok = [r for r in res if r.get("gen_tokens")]
    decs = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s")]
    gens = sum(r["gen_tokens"] for r in ok)
    row = {"arm": name, "tag": tag, "concurrency": nconc, "n": len(res), "ok": len(ok),
           "total_wall_s": round(wall, 1), "aggregate_tok_s": round(gens / wall, 1),
           "decode_mean": round(statistics.mean(decs), 1) if decs else None,
           "decode_p95": round(sorted(decs)[max(0, int(len(decs) * 0.95) - 1)], 1) if decs else None,
           "ttft_mean": round(statistics.mean(ttfts), 2) if ttfts else None,
           "ttft_p95": round(sorted(ttfts)[max(0, int(len(ttfts) * 0.95) - 1)], 2) if ttfts else None,
           "kv_slots_timeline": kvs, "kv_engine_lines": klines,
           "errors": [r["error"] for r in res if r.get("error")]}
    with open(B + "/results_raw.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary((tag, name, "ok", f"c={nconc}", f"agg={row['aggregate_tok_s']}",
             f"dec_mean={row['decode_mean']}", f"ttft_mean={row['ttft_mean']}"))
    log(f"{tag}/{name}: c={nconc} ok={len(ok)}/{len(res)} agg={row['aggregate_tok_s']} tok/s "
        f"dec_mean={row['decode_mean']} ttft_mean={row['ttft_mean']}s errors={row['errors'][:1]}")
    if kvs:
        brief = json.dumps(kvs[-1].get("/metrics", kvs[-1].get("/slots")))[:180]
        log(f"{tag}/{name} KV: {brief}")
    return row


def warm_arm(eng, logfile, tag, big_prompt, foreign_file):
    # turn 1 (cold), then two foreign chats, then follow-up on the SAME conversation
    base = open(big_prompt).read()
    off0 = os.path.getsize(logfile) if os.path.exists(logfile) else 0
    cold = single_arm(eng, logfile, "warm_turn1_cold", big_prompt, 128, tag + "_warm", 1500)
    try:
        eng.stream_chat(open(foreign_file).read(), 64, wall_limit=600)
        eng.stream_chat(open(foreign_file).read()[::-1], 64, wall_limit=600)
    except Exception as e:
        log(f"{tag}_warm foreign chats failed: {repr(e)[:100]}")
    ask = ("Using ONLY the dump above: list every CONTROL record as `id = value`, one per line. "
           "Then state the exact token count you saw for the first file header.")
    ans_file = f"{B}/logs/{tag}_warm_warm_turn1_cold.answer.txt"
    ans = open(ans_file).read() if os.path.exists(ans_file) else "OK"
    try:
        r = eng.stream_chat(ask, 256, history=[{"role": "user", "content": base},
                                {"role": "assistant", "content": ans}],
                                wall_limit=900) if cold else None
    except Exception as e:
        r = {"error": repr(e)[:120]}
    engline = engine_line(logfile, None)
    snap = {}
    for p in ("/metrics", "/slots"):
        try:
            snap[p] = eng.req(p, timeout=4)
        except Exception as e:
            snap[p] = repr(e)[:60]
    klines, _ = kv_lines_since(logfile, off0)
    row = {"arm": "warm_followup", "tag": tag, "cold_ttft": (cold or {}).get("ttft_s"),
           "followup_ttft": r.get("ttft_s"), "followup_decode": r.get("decode_tok_s"),
           "followup_gen": r.get("gen_tokens"), "engine_after": engline,
           "kv_slots_after": snap, "kv_engine_lines": klines,
           "wall_s": r.get("wall_s")}
    with open(B + "/results_raw.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary((tag, "warm_followup", "ok", f"cold_ttft={row['cold_ttft']}",
             f"warm_ttft={row['followup_ttft']}", f"reused={engline.get('reused')}",
             f"read={engline.get('read')}"))
    log(f"{tag}/warm: cold_ttft={row['cold_ttft']}s warm_ttft={row['followup_ttft']}s "
        f"reused={engline.get('reused')} read={engline.get('read')}")
    if r.get("_text"):
        open(f"{B}/logs/{tag}_warm_followup.answer.txt", "w").write(r["_text"])
    return row


def sustained_arm(eng, tag, dur_s, prompt_files, max_tokens, telem_csv):
    stop = time.time() + dur_s
    rows, lk = [], threading.Lock()
    tp = subprocess.Popen(["python3", B + "/telem.py", telem_csv, "strata"
                           if eng.name == "strata" else "llama"])

    def worker(idx):
        i = idx
        while time.time() < stop and STOP_AT_OK():
            pth = prompt_files[i % len(prompt_files)]
            try:
                r = eng.stream_chat(open(pth).read(), max_tokens, wall_limit=900)
                with lk:
                    rows.append(r)
            except Exception as e:
                with lk:
                    rows.append({"error": repr(e)[:120], "t_start": time.time()})
            i += 1

    ths = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
    t0 = time.time()
    for t in ths:
        t.start()
    while any(t.is_alive() for t in ths):
        beat(f"sustained {tag} running")
        for t in ths:
            t.join(20)
    tp.terminate()
    wall = time.time() - t0
    ok = [r for r in rows if r.get("gen_tokens")]
    buckets = {}
    for r in ok:
        b = int((r["t_start"] - t0) // 300) * 300
        buckets.setdefault(b, []).append(r)
    prof = []
    for b in sorted(buckets):
        v = buckets[b]
        decs = [x["decode_tok_s"] for x in v if x.get("decode_tok_s")]
        tts = [x["ttft_s"] for x in v if x.get("ttft_s")]
        prof.append({"bucket_s": b, "n": len(v),
                     "dec_mean": round(statistics.mean(decs), 1) if decs else None,
                     "dec_p95": round(sorted(decs)[max(0, int(len(decs) * 0.95) - 1)], 1) if decs else None,
                     "ttft_mean": round(statistics.mean(tts), 2) if tts else None})
    decs = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    row = {"arm": "sustained", "tag": tag, "duration_s": round(wall, 1),
           "requests_done": len(ok), "errors": len(rows) - len(ok),
           "decode_mean": round(statistics.mean(decs), 1) if decs else None,
           "decode_p50": round(statistics.median(decs), 1) if decs else None,
           "decode_p95": round(sorted(decs)[max(0, int(len(decs) * 0.95) - 1)], 1) if decs else None,
           "profile": prof, "telem": telem_stats(telem_csv, t0, time.time()),
           "final_gpu": gpu()}
    with open(B + "/results_raw.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary((tag, "sustained", "ok", f"dur={round(wall)}", f"req={len(ok)}",
             f"dec_mean={row['decode_mean']}", f"dec_p95={row['decode_p95']}",
             f"maxT={row['telem'].get('max_temp_c')}", f"thr={row['telem'].get('throttled_samples')}"))
    log(f"{tag}/sustained: {round(wall)}s done req={len(ok)} err={len(rows)-len(ok)} "
        f"dec_mean={row['decode_mean']} p95={row['decode_p95']} "
        f"max_temp={row['telem'].get('max_temp_c')}C throttled={row['telem'].get('throttled_samples')}")
    return row


def STOP_AT_OK():
    return (not STOP_AT) or time.time() < STOP_AT - 240


def strata_serve_config(ctx=196608, maxq=16, port=STRATA_PORT):
    cfg = {"exe": STRATA_EXE,
           "args": ["--pack", HOME + "/Strata-data/packs/iq3_s",
                    "--native", GG1, "--ple-gguf", GG2,
                    "--expert-profile", HOME + "/Strata/data/expert-profile.bin",
                    "--expert-cache", "auto", "--prefill", "auto:16384",
                    "--spec", "4", "--spec-min-p", "0.5",
                    "--mtp", HOME + "/Strata-data/mtp/rt",
                    "--max-context", str(ctx), "--kv", "int8",
                    "--vram-reserve-mib", "500", "--pool-workers", "15", "--kv-grow",
                    "--conversation-cache-mib", "2560",
                    "--conversation-cache-min-tokens", "4096",
                    "--conversation-cache-park-gain", "256",
                    "--prompt-cache-tail", "--prompt-cache-spread",
                    "--conversation-cache-min-free-mib", "1024"],
           "cwd": HOME + "/strata-0216/terminal",
           "tokenizer": HOME + "/Strata-data/packs/iq3_s/tokenizer",
           "model_name": "iq3s-strata0216", "log": f"{B}/logs/strata.serve.log",
           "port": port, "backend": "cuda13", "host": "127.0.0.1",
           "api_key": API_KEY, "max_queue": maxq}
    p = f"{B}/configs/strata.json"
    json.dump(cfg, open(p, "w"), indent=1)
    return p, cfg["log"]


def llama_serve_cmd(np_slots, ctx, port, extra=None, ngl="99", ncpu_moe=None, kv8=True,
                    binary=None, model=None, alias="iq3s-llama", modern=True):
    binp, mod = binary or LLAMA_BIN, model or GG1
    if not modern:
        cmd = [binp, "-m", mod, "--alias", alias, "-ngl", ngl,
               "-c", str(ctx), "-np", str(np_slots),
               "--host", "127.0.0.1", "--port", str(port)]
        if extra:
            cmd += extra
        return cmd
    cmd = [binp, "-m", mod, "--alias", alias, "-ngl", ngl,
           "--flash-attn", "on", "--reasoning", "off",
           "-c", str(ctx), "-np", str(np_slots),
           "--host", "127.0.0.1", "--port", str(port)]
    if kv8:
        cmd += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    if ncpu_moe:
        cmd += ["--n-cpu-moe", str(ncpu_moe)]
    if extra:
        cmd += extra
    return cmd


def stage_strata():
    if os.path.exists(B + "/strata_stage_done"):
        log("STAGE strata: already completed earlier, skipping")
        return
    log("STAGE strata: probing serve configs (0.2.16 terminal readiness)")
    eng = Engine("strata", f"http://127.0.0.1:{STRATA_PORT}", "iq3s-strata0216", API_KEY,
                 defaults={"reasoning_budget_tokens": 192})
    base_args = None
    variants = [
        ("v1_profile_ctx196k", None),
        ("v2_ctx131k", {"max_context": "131072"}),
        ("v3_ctx131k_draftvocab", {"max_context": "131072", "add": ["--draft-vocab", "en"]}),
        ("v4_spec0", {"drop_spec": True}),
        ("v5_ctx131k_kvq4", {"max_context": "131072", "kv": "q4_0"}),
    ]
    p = fh = None
    h = None
    used = None
    for vname, mut in variants:
        cfg_path, _clog = strata_serve_config()
        cfg = json.load(open(cfg_path))
        if mut:
            a = cfg["args"]
            if mut.get("max_context"):
                i = a.index("--max-context"); a[i + 1] = mut["max_context"]
            if mut.get("kv"):
                i = a.index("--kv"); a[i + 1] = mut["kv"]
            if mut.get("add"):
                a.extend(mut["add"])
            if mut.get("drop_spec"):
                for flag, n in (("--spec", 2), ("--spec-min-p", 2), ("--mtp", 2)):
                    if flag in a:
                        i = a.index(flag); del a[i:i + n]
                cfg["args"] = a
        json.dump(cfg, open(cfg_path, "w"), indent=1)
        logfile = f"{B}/logs/strata.{vname}.serve.log"
        summary(("strata", "cfg_try", vname))
        p, fh = start_serve([STRATA_PY, STRATA_SRV, "--engine", "strata",
                             "--config", cfg_path, "--port", str(STRATA_PORT)], logfile)
        h, rc, load_s = wait_health(eng, p, t=300)
        if h is not None:
            used = vname
            log(f"STAGE strata: loaded in {load_s:.0f}s config={vname}")
            base_args = cfg["args"]
            break
        log(f"STAGE strata: config {vname} failed rc={rc} in {load_s:.0f}s")
        stop_serve(p, fh)
        wait_vram_free()
    if h is None:
        tail = ""
        try:
            tail = open(logfile, errors="replace").read()[-400:]
        except Exception:
            pass
        log(f"STRATA_START_FAILED_ALL tail={tail[:200]}")
        summary(("strata", "serve", "START_FAILED_ALL", 1))
        return
    logfile = f"{B}/logs/strata.{used}.serve.log"
    log(f"STAGE strata: arms with config {used}")
    try:
        eng.stream_chat("Say OK.", 8, wall_limit=120)
    except Exception:
        pass
    # warmup (not recorded)
    try:
        eng.stream_chat("Say OK.", 8, wall_limit=120)
    except Exception:
        pass
    tele = f"{B}/logs/S.telem.csv"
    tp = subprocess.Popen(["python3", B + "/telem.py", tele, "strata"])
    try:
        for nm, tok, rc in (("p4k", 512, False), ("p32k", 512, False), ("p64k_code", 512, True),
                            ("p128k", 256, True), ("p184k", 256, False)):
            if not STOP_AT_OK():
                break
            single_arm(eng, logfile, nm, pf(nm), tok, "S_single", recall=rc, tele=tele)
        if STOP_AT_OK():
            warm_arm(eng, logfile, "S_warm", pf("p40k"), pf("p4k_f1"))
        for c in (2, 4, 8):
            if not STOP_AT_OK():
                break
            load_arm(eng, logfile, f"load32k_c{c}", [pf("p32k_a"), pf("p32k_b"), pf("p32k_c")],
                     c, 256, "S_load")
        if STOP_AT_OK():
            sustained_arm(eng, "S_sust", 30 * 60, [pf("p4k_a"), pf("p32k_a")], 192,
                          f"{B}/logs/S_sust.telem.csv")
    finally:
        tp.terminate()
        stop_serve(p, fh)
        wait_vram_free()
    open(B + "/strata_stage_done", "w").write(used or "unknown")
    log("STAGE strata: done")


def stage_basalt():
    log("STAGE basalt: serve up (production brain v0.4.1, config as-is on port 18084)")
    cfg = json.load(open(HOME + "/basalt-041/basalt-config.json"))
    cfg.setdefault("server", {})["port"] = 18084
    cfg["server"]["host"] = "127.0.0.1"
    alias = cfg["models"][0]["alias"]
    cfgp = f"{B}/configs/basalt.json"
    json.dump(cfg, open(cfgp, "w"), indent=1)
    key = open(HOME + "/basalt/basalt.key").read().strip()
    logfile = f"{B}/logs/basalt.serve.log"
    cmd = [HOME + "/basalt-041/bin/basalt", "serve", "--config", cfgp,
           "--python", HOME + "/basalt/.venv/bin/python", "--vision", "dynamic",
           "--api-key", key]
    p, fh = start_serve(cmd, logfile)
    eng = Engine("basalt", "http://127.0.0.1:18084", alias, key)
    h, rc, load_s = wait_health(eng, p, t=900)
    if h is None:
        tail = open(logfile, errors="replace").read()[-600:]
        stop_serve(p, fh)
        log(f"BASALT_START_FAILED rc={rc} tail={tail[:200]}")
        summary(("basalt", "serve", "START_FAILED", rc))
        return
    log(f"STAGE basalt: loaded in {load_s:.0f}s alias={alias}")
    try:
        eng.stream_chat("Say OK.", 8, wall_limit=120)
    except Exception:
        pass
    tele = f"{B}/logs/B.telem.csv"
    tp = subprocess.Popen(["python3", B + "/telem.py", tele, "basalt"])
    try:
        for nm, tok, rc2 in (("p4k", 512, False), ("p32k", 512, False), ("p64k_code", 512, True),
                             ("p128k", 256, True), ("p184k", 256, False)):
            if not STOP_AT_OK():
                break
            single_arm(eng, logfile, nm, pf(nm), tok, "B_single", recall=rc2, tele=tele)
        if STOP_AT_OK():
            warm_arm(eng, logfile, "B_warm", pf("p40k"), pf("p4k_f1"))
        for c in (2, 4, 8):
            if not STOP_AT_OK():
                break
            load_arm(eng, logfile, f"load32k_c{c}", [pf("p32k_a"), pf("p32k_b"), pf("p32k_c")],
                     c, 256, "B_load")
        if STOP_AT_OK():
            sustained_arm(eng, "B_sust", 30 * 60, [pf("p4k_a"), pf("p32k_a")], 192,
                          f"{B}/logs/B_sust.telem.csv")
    finally:
        tp.terminate()
        stop_serve(p, fh)
        wait_vram_free()
    log("STAGE basalt: done")


def stage_llama(np_slots, ctx, port, tag, arms, binary=None, model=None,
                alias="iq3s-llama", ngl="99", variants=None, modern=True):
    log(f"STAGE {tag}: serve up np={np_slots} ctx={ctx}")
    logfile = f"{B}/logs/{tag}.serve.log"
    eng = Engine("llama", f"http://127.0.0.1:{port}", alias)
    variants = variants or [(None, True), (None, False), ("12", False), ("24", False), ("48", False)]
    p = fh = None
    h = None
    for (moe, kv8) in variants:
        cmd = llama_serve_cmd(np_slots, ctx, port, ncpu_moe=moe, kv8=kv8, ngl=ngl,
                              binary=binary, model=model, alias=alias, modern=modern)
        summary((tag, "load_try", f"moe={moe}", f"kv8={kv8}"))
        p, fh = start_serve(cmd, logfile + (".moe" + str(moe) if moe else ""))
        h, rc, load_s = wait_health(eng, p, t=420)
        if h is not None:
            log(f"STAGE {tag}: loaded in {load_s:.0f}s (n-cpu-moe={moe})")
            break
        log(f"{tag}: load failed n-cpu-moe={moe}, trying next variant")
        stop_serve(p, fh)
        wait_vram_free()
    if h is None:
        tail = ""
        try:
            tail = open(fh.name if fh else logfile, errors="replace").read()[-600:]
        except Exception:
            pass
        summary((tag, "serve", "LOAD_FAILED", tail[:150]))
        log(f"{tag}_LOAD_FAILED tail={tail[:200]}")
        return
    try:
        eng.stream_chat("Say OK.", 8, wall_limit=120)
    except Exception:
        pass
    try:
        for a in arms:
            if not STOP_AT_OK():
                break
            a(eng, logfile or "", tag)
    finally:
        stop_serve(p, fh)
        wait_vram_free()
    log(f"STAGE {tag}: done")


def stage_verify500x():
    # Adversarial validation of the ~500x prefill-cache wins: real on never-seen
    # prompts, or only lucky repeats? Unique salts, prefix-granularity, eviction
    # pressure, answer correctness after reuse, no-sharing concurrency wave.
    import hashlib
    import random
    log("STAGE verify500x: adversarial KV/prefill validation")
    eng = Engine("strata", f"http://127.0.0.1:{STRATA_PORT}", "iq3s-strata0216", API_KEY,
                 defaults={"reasoning_budget_tokens": 192})
    cfg_path, _ = strata_serve_config()
    logfile = f"{B}/logs/strata.verify.serve.log"
    p, fh = start_serve([STRATA_PY, STRATA_SRV, "--engine", "strata",
                         "--config", cfg_path, "--port", str(STRATA_PORT)], logfile)
    h, rc, load_s = wait_health(eng, p, t=300)
    if h is None:
        stop_serve(p, fh)
        log(f"VERIFY_START_FAILED rc={rc}")
        summary(("verify", "serve", "START_FAILED", rc))
        return
    log(f"STAGE verify500x: loaded in {load_s:.0f}s")
    try:
        eng.stream_chat("Say OK.", 8, wall_limit=120)
    except Exception:
        pass
    rnd = random.Random(20261119)

    def salted(base, where):
        t = open(pf(base)).read()
        s = f"UNIQUE-SESSION-SALT-{rnd.randint(10**9, 10**10)}\n"
        return (s + t) if where == "head" else (t + "\n" + s)

    def varm(name, text, max_tokens=128):
        open(f"{B}/prompts/{name}.txt", "w").write(text)
        off = os.path.getsize(logfile) if os.path.exists(logfile) else 0
        try:
            r = eng.stream_chat(text, max_tokens, wall_limit=1500)
        except Exception as e:
            summary(("verify", name, "FAILED", repr(e)[:100]))
            log(f"V/{name}: FAILED {repr(e)[:120]}")
            return None, None
        ls, _ = kv_lines_since(logfile, off)
        imp = [l for l in ls if "reused" in l]
        m = REUSE_RE.search(imp[-1]) if imp else None
        row = {"arm": name, "tag": "V",
               "sha": hashlib.sha256(text.encode()).hexdigest()[:12],
               "ttft_s": r["ttft_s"], "wall_s": r["wall_s"],
               "gen_tokens": r["gen_tokens"], "decode_tok_s": r["decode_tok_s"],
               "prompt_tokens": int(m.group(1)) if m else None,
               "reused": int(m.group(2)) if m else None,
               "read": int(m.group(3)) if m else None,
               "telem": gpu()}
        with open(B + "/results_raw.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        open(f"{B}/logs/{name}.answer.txt", "w").write(r["_text"])
        summary(("verify", name, "ok", f"ttft={r['ttft_s']}", f"reused={row['reused']}",
                 f"read={row['read']}"))
        log(f"V/{name}: ttft={r['ttft_s']}s reused={row['reused']} read={row['read']} "
            f"sha={row['sha']}")
        return r, row

    try:
        # V1/V2: identical unique 32k chat — true cold vs immediately cached
        p1 = salted("p32k", "head")
        _, c1 = varm("v1_cold_unique_32k", p1)
        _, c2 = varm("v2_repeat_same_32k", p1)
        if c1 and c2 and c1["ttft_s"] and c2["ttft_s"]:
            x = c1["ttft_s"] / max(c2["ttft_s"], 0.001)
            log(f"V/cold_vs_cached: {c1['ttft_s']}s -> {c2['ttft_s']}s = {x:.0f}x")
            summary(("verify", "cold_vs_cached", "ok", f"cold={c1['ttft_s']}",
                     f"cached={c2['ttft_s']}", f"x={round(x)}"))
        # V3: same base, salt at the END — prefix reuse must be granular, not all-or-nothing
        if STOP_AT_OK():
            varm("v3_append_salt_32k", salted("p32k", "tail"))
        # V4: eviction pressure — first chat, 4 other big chats (cache slots=4), re-query
        first = salted("p32k_a", "head")
        if STOP_AT_OK():
            varm("v4a_first", first)
            for i in range(4):
                if not STOP_AT_OK():
                    break
                varm(f"v4b_intruder{i}", salted("p32k_b", "head"))
            varm("v4c_requery_after_pressure", first)
        # V5: answer correctness — same ask cold vs cached at temp 0 must be identical
        if STOP_AT_OK():
            ask = first + ("\n\nList every CONTROL record in the dump above as "
                           "`id = value`, one per line.")
            varm("v5a_answer_cold", ask, 512)
            varm("v5b_answer_cached", ask, 512)
            ta = open(f"{B}/logs/v5a_answer_cold.answer.txt", errors="replace").read()
            tb = open(f"{B}/logs/v5b_answer_cached.answer.txt", errors="replace").read()
            same = ta == tb
            log(f"V/answer_identical_cold_vs_cached={same} len={len(ta)}/{len(tb)}")
            summary(("verify", "correctness", "identical" if same else "DIFFERS",
                     f"len={len(ta)}/{len(tb)}"))
        # V6: c=4 wave with four UNIQUE prompts — no cross-agent cache sharing
        if STOP_AT_OK():
            files = []
            for i, base in enumerate(("p32k_a", "p32k_b", "p32k_c", "p32k")):
                fn = f"{B}/prompts/v6u{i}.txt"
                open(fn, "w").write(salted(base, "head"))
                files.append(fn)
            load_arm(eng, logfile, "v6_unique_prompts_c4", files, 4, 256, "V")
    finally:
        stop_serve(p, fh)
        wait_vram_free()
    log("STAGE verify500x: done")


def stage_verify500x2():
    # Prefix-granularity follow-on: cold prompt, then the SAME prompt with one
    # appended line and with one flipped character mid-way. If reuse is
    # prefix-granular, appended pays only for the delta and mid-flip reuses
    # the prefix up to the change; if not, the huge reuse numbers only ever
    # cover byte-identical repeats.
    import hashlib
    log("STAGE verify500x2: prefix-granularity reuse test")
    eng = Engine("strata", f"http://127.0.0.1:{STRATA_PORT}", "iq3s-strata0216", API_KEY,
                 defaults={"reasoning_budget_tokens": 192})
    cfg_path, _ = strata_serve_config()
    logfile = f"{B}/logs/strata.verify2.serve.log"
    p, fh = start_serve([STRATA_PY, STRATA_SRV, "--engine", "strata",
                         "--config", cfg_path, "--port", str(STRATA_PORT)], logfile)
    h, rc, load_s = wait_health(eng, p, t=300)
    if h is None:
        stop_serve(p, fh)
        log(f"VERIFY2_START_FAILED rc={rc}")
        summary(("verify2", "serve", "START_FAILED", rc))
        return
    log(f"STAGE verify500x2: loaded in {load_s:.0f}s")
    try:
        eng.stream_chat("Say OK.", 8, wall_limit=120)
    except Exception:
        pass

    def wvarm(name, text, max_tokens=128):
        open(f"{B}/prompts/{name}.txt", "w").write(text)
        off = os.path.getsize(logfile) if os.path.exists(logfile) else 0
        try:
            r = eng.stream_chat(text, max_tokens, wall_limit=1500)
        except Exception as e:
            summary(("verify2", name, "FAILED", repr(e)[:100]))
            log(f"W/{name}: FAILED {repr(e)[:120]}")
            return None
        ls, _ = kv_lines_since(logfile, off)
        imp = [l for l in ls if "reused" in l]
        m = REUSE_RE.search(imp[-1]) if imp else None
        row = {"arm": name, "tag": "W",
               "sha": hashlib.sha256(text.encode()).hexdigest()[:12],
               "ttft_s": r["ttft_s"], "wall_s": r["wall_s"], "gen_tokens": r["gen_tokens"],
               "reused": int(m.group(2)) if m else None,
               "read": int(m.group(3)) if m else None,
               "reuse_line_raw": imp[-2:]}
        with open(B + "/results_raw.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary(("verify2", name, "ok", f"ttft={r['ttft_s']}", f"reused={row['reused']}",
                 f"read={row['read']}"))
        log(f"W/{name}: ttft={r['ttft_s']}s reused={row['reused']} read={row['read']} "
            f"raw={(imp[-1][:90] if imp else None)}")
        return r

    try:
        t = open(pf("p32k")).read()
        wvarm("w1_cold_original_32k", t)
        if STOP_AT_OK():
            wvarm("w3_append_one_line", t + "\nEXTRA-LINE-7712\n")
        if STOP_AT_OK():
            mid = len(t) // 2
            wvarm("w4_midflip_one_char", t[:mid] + ("X" if t[mid] != "X" else "Y") + t[mid + 1:])
        if STOP_AT_OK():
            wvarm("w2_identical_repeat", t)
    finally:
        stop_serve(p, fh)
        wait_vram_free()
    log("STAGE verify500x2: done")


def stage_bitnet():
    # Microsoft BitNet engine (their llama.cpp fork + i2_s kernels), CPU-only,
    # their own 1.58-bit b1.58-2B-4T model. Different model class from IQ3_S —
    # same harness, ladder truncated to the model's 4096 ctx (recorded as-is).
    port = 18085
    binp = HOME + "/BitNet/build/bin/llama-server"
    gguf = HOME + "/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    if not (os.path.exists(binp) and os.path.exists(gguf)):
        log(f"BITNET_MISSING bin={os.path.exists(binp)} gguf={os.path.exists(gguf)}")
        summary(("bitnet", "serve", "MISSING_INSTALL", 1))
        return
    full = open(pf("p4k")).read()
    a4 = open(pf("p4k_a")).read()
    b4p = pf("p4k_b")
    open(pf("p1k"), "w").write(full[:4000])
    open(pf("p1k_a"), "w").write(a4[:4000])
    open(pf("p1k_b"), "w").write((open(b4p).read() if os.path.exists(b4p) else full[4000:8000])[:4000])
    open(pf("p3k"), "w").write(full[:11500])

    def armsn(eng, logfile, tag):
        tele = f"{B}/logs/N.telem.csv"
        tp = subprocess.Popen(["python3", B + "/telem.py", tele, "bitnet"])
        try:
            single_arm(eng, logfile, "p1k", pf("p1k"), 256, tag + "_single", tele=tele)
            single_arm(eng, logfile, "p3k", pf("p3k"), 512, tag + "_single", tele=tele)
            warm_arm(eng, logfile, tag + "_warm", pf("p3k"), pf("p1k"))
            load_arm(eng, logfile, "load1k_c2", [pf("p1k_a"), pf("p1k_b")], 2, 128, tag + "_load")
            load_arm(eng, logfile, "load1k_c4", [pf("p1k_a"), pf("p1k_b")], 4, 128, tag + "_load")
            sustained_arm(eng, "N_sust", 20 * 60, [pf("p1k_a"), pf("p1k_b")], 128,
                          f"{B}/logs/N_sust.telem.csv")
        finally:
            tp.terminate()
    stage_llama(4, 4096, port, "N", [lambda e, l, t: armsn(e, l, t)],
                binary=binp, model=gguf, alias="bitnet", ngl="0",
                variants=[(None, False)], modern=False)


def stage_llamacpu():
    # Same IQ3_S model as the GPU run, llama.cpp with --n-gpu-layers 0:
    # the honest "no GPU at all" baseline for OUR model.
    port = 18086

    def armc(eng, logfile, tag):
        single_arm(eng, logfile, "p4k", pf("p4k"), 256, tag + "_single", wall_limit=1800)
        single_arm(eng, logfile, "p32k", pf("p32k"), 128, tag + "_single", wall_limit=2400)
        files = [pf("p4k_a")] + ([pf("p4k_b")] if os.path.exists(pf("p4k_b")) else [])
        sustained_arm(eng, "C_sust", 15 * 60, files, 128,
                      f"{B}/logs/C_sust.telem.csv")
    stage_llama(4, 40000, port, "C", [lambda e, l, t: armc(e, l, t)],
                ngl="0", variants=[(None, False)])


def main():
    stage = sys.argv[1]
    os.makedirs(B + "/logs", exist_ok=True)
    os.makedirs(B + "/configs", exist_ok=True)
    if stage == "strata":
        stage_strata()
    elif stage == "basalt":
        stage_basalt()
    elif stage == "llama1":
        def arms(eng, logfile, tag):
            tele = f"{B}/logs/L1.telem.csv"
            tp = subprocess.Popen(["python3", B + "/telem.py", tele, "llama"])
            try:
                for nm, tok, rc in (("p4k", 512, False), ("p32k", 512, False),
                                    ("p64k_code", 512, True), ("p128k", 256, True),
                                    ("p184k", 256, False)):
                    single_arm(eng, logfile, nm, pf(nm), tok, tag + "_single", recall=rc, tele=tele)
                warm_arm(eng, logfile, tag + "_warm", pf("p40k"), pf("p4k_f1"))
            finally:
                tp.terminate()
        stage_llama(1, 198144, LLAMA_PORTS[0], "L1", [lambda e, l, t: arms(e, l, t)])
    elif stage == "llama2":
        def arms2(eng, logfile, tag):
            for c in (2, 4, 8):
                load_arm(eng, logfile, f"load32k_c{c}", [pf("p32k_a"), pf("p32k_b"), pf("p32k_c")],
                         c, 256, tag + "_load")
            sustained_arm(eng, "L2_sust", 30 * 60, [pf("p4k_a"), pf("p32k_a")], 192,
                          f"{B}/logs/L2_sust.telem.csv")
        stage_llama(8, 270336, LLAMA_PORTS[1], "L2", [lambda e, l, t: arms2(e, l, t)])
    elif stage == "verify500x":
        stage_verify500x()
    elif stage == "verify500x2":
        stage_verify500x2()
    elif stage == "bitnet":
        stage_bitnet()
    elif stage == "llamacpu":
        stage_llamacpu()


if __name__ == "__main__":
    main()
