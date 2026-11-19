#!/usr/bin/env bash
# Build the Microsoft BitNet engine (CPU-only, i2_s kernels) on P2005, then
# run the CPU appendix stages. Waits for the GPU bench to finish first so the
# two never compete for RAM/CPU.
B="$HOME/bench-load"
LOG="$B/bitnet-build.log"
cd "$HOME/BitNet" || exit 1

# wait for the GPU bench driver to be gone
while pgrep -f "bench-load/driver[.]sh" >/dev/null || pgrep -f "verify2_chain[.]sh" >/dev/null; do sleep 60; done

{
  echo "== build start $(date -Is)"
  if [ ! -x build/bin/llama-server ]; then
    python3 -m venv .venv 2>/dev/null
    ./.venv/bin/pip install -q -r requirements.txt 2>&1 | tail -3
    ./.venv/bin/python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
  fi
  ls -la build/bin/llama-server 2>/dev/null || echo "BITNET_BUILD_NO_BINARY"
  build/bin/llama-server --version 2>&1 | head -2
} >> "$LOG" 2>&1

cd "$B" || exit 1
STOP_AT_OVERRIDE=$(date -d '2026-11-19 08:20' +%s) STAGES="bitnet llamacpu" exec bash driver.sh
