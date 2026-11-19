#!/usr/bin/env bash
# Prefix-granularity follow-on for the 500x verification: runs right after the
# main driver exits, with its own extended STOP_AT window.
B="$HOME/bench-load"
PID=$(cat "$B/driver.pid" 2>/dev/null)
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do sleep 20; done
cd "$B" || exit 1
STOP_AT_OVERRIDE=$(date -d '2026-11-19 06:30' +%s) STAGES="verify500x2" exec bash driver.sh
