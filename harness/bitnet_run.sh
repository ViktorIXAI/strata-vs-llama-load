#!/usr/bin/env bash
# CPU-only appendix: Microsoft BitNet engine + llama.cpp CPU on the same box,
# chained after the main GPU bench. Extended deadline, same driver contract.
B="$HOME/bench-load"
cd "$B" || exit 1
STOP_AT_OVERRIDE=$(date -d '2026-11-19 08:20' +%s) STAGES="bitnet llamacpu" exec bash driver.sh
