#!/usr/bin/env bash
# Chains the adversarial verify stage right after the main bench driver exits,
# reusing driver.sh's contract (brain park/restore, STOP_AT guard).
B="$HOME/bench-load"
PID=$(cat "$B/driver.pid" 2>/dev/null)
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do sleep 20; done
cd "$B" || exit 1
STAGES="verify500x" exec bash driver.sh
