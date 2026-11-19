#!/usr/bin/env python3
"""1 Hz hardware telemetry sampler for the Basalt 27B express bench."""
import csv, os, subprocess, sys, time

OUT = sys.argv[1]
PID_HINT = sys.argv[2] if len(sys.argv) > 2 else "basalt"
PCI = os.environ.get("BENCH_PCI", "0000:01:00.0")

FIELDS = ["ts", "gpu_util", "gpu_temp", "gpu_power_w", "sm_clock", "mem_clock",
          "vram_used_mib", "vram_free_mib", "throttle", "ram_used_b", "ram_avail_b",
          "swap_used_b", "proc_rss_kb", "proc_cpu_pct", "link_speed", "link_width"]
Q = ("utilization.gpu,temperature.gpu,power.draw,clocks.current.sm,"
     "clocks.current.memory,memory.used,memory.free,clocks_throttle_reasons.active")

FH = open(OUT, "w", newline="")
w = csv.DictWriter(FH, fieldnames=FIELDS)
w.writeheader()
FH.flush()
while True:
    row = {"ts": time.time()}
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=" + Q, "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0].split(", ")
        (row["gpu_util"], row["gpu_temp"], row["gpu_power_w"], row["sm_clock"],
         row["mem_clock"], row["vram_used_mib"], row["vram_free_mib"], row["throttle"]) = s
    except Exception:
        pass
    try:
        m = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            m[k] = int(v.strip().split()[0]) * 1024
        row["ram_used_b"] = m["MemTotal"] - m["MemAvailable"]
        row["ram_avail_b"] = m["MemAvailable"]
        row["swap_used_b"] = m["SwapTotal"] - m["SwapFree"]
    except Exception:
        pass
    try:
        tot = 0
        cpu = 0.0
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                with open(f"/proc/{p}/cmdline", "rb") as f:
                    cl = f.read().decode("utf-8", "replace")
                if PID_HINT not in cl or "telem.py" in cl:
                    continue
                with open(f"/proc/{p}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            tot += int(line.split()[1])
                with open(f"/proc/{p}/stat") as f:
                    st = f.read().split()
                    cpu += (int(st[13]) + int(st[14]) + int(st[15]) + int(st[16])) / os.sysconf("SC_CLK_TCK") * 100.0
            except Exception:
                pass
        row["proc_rss_kb"] = tot
        row["proc_cpu_pct"] = round(cpu, 1)
    except Exception:
        pass
    try:
        d = "/sys/bus/pci/devices/" + PCI + "/"
        row["link_speed"] = open(d + "current_link_speed").read().strip()
        row["link_width"] = open(d + "current_link_width").read().strip()
    except Exception:
        pass
    w.writerow(row)
    FH.flush()
    time.sleep(1.0)
