#!/usr/bin/env bash
# Strata-vs-llama load bench driver, runs DETACHED on P2005.
# Parks the keep-up cron, stops the brain stack, builds latest llama.cpp,
# runs stages, then always restores the brain (trap EXIT).
set -u
B=$HOME/bench-load
cd "$B" || exit 1
echo $$ > "$B/driver.pid"
export STOP_AT=${STOP_AT_OVERRIDE:-$(date -d '2026-11-19 05:50' +%s)}
log(){ echo "$(date '+%Y-%m-%d_%H:%M:%S') $*" >> "$B/progress.log"; }
restore_brain(){
  if [ ! -f "$B/brain_restored" ]; then
    crontab /tmp/ct-load.bak 2>/dev/null
    bash ~/basalt/run-serve.sh >/dev/null 2>&1
    touch "$B/brain_restored"
    log "BRAIN_RESTORE_ISSUED"
  fi
}
trap restore_brain EXIT

log "DRIVER_START stop_at=$(date -d @$STOP_AT '+%H:%M')"
crontab -l > /tmp/ct-load.bak
crontab -l | grep -v keep-up | crontab -
bash ~/basalt/stop-stack.sh >/dev/null 2>&1
log "keep_up_parked stack_stopped vram_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"

# 1) llama.cpp: fetch latest release tag, CUDA build (llama-server target)
cd ~/llama.cpp
tag=""
if timeout 120 git fetch --tags --quiet origin; then tag=$(git tag --sort=-creatordate | head -1); git checkout -q "$tag" 2>/dev/null; fi
log "llama_tag=${tag:-NONE}"
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export PATH=/usr/local/cuda-13.0/bin:$PATH
if cmake -S . -B build-load -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
     -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
     >> "$B/llama-build.log" 2>&1 \
   && cmake --build build-load -j 24 --target llama-server >> "$B/llama-build.log" 2>&1; then
  V=$(~/llama.cpp/build-load/bin/llama-server --version 2>&1 | head -1)
  log "llama_build_ok $V"
else
  cp -a ~/llama.cpp/build ~/llama.cpp/build-load 2>/dev/null || true
  V=$(~/llama.cpp/build-load/bin/llama-server --version 2>&1 | head -1)
  log "llama_build_FAILED fallback=$V"
fi

# 2) prompts (new seed, corpus identical to the published express bench)
if python3 "$B/build_prompts.py" >> "$B/progress.log" 2>&1; then log "prompts_built"; else log "prompts_FAILED"; fi

# 3) stages
for st in ${STAGES:-strata basalt llama1 llama2}; do
  if [ "$(date +%s)" -lt "$STOP_AT" ]; then
    log "STAGE_START $st"
    python3 "$B/loadbench.py" "$st" >> "$B/stage_$st.out" 2>&1
    log "STAGE_END $st rc=$?"
  else
    log "STAGE_SKIPPED $st deadline"
  fi
  beat(){ :; }; echo "$(date -Is) stage=$st" > "$B/heartbeat.txt"
done
# self-heal: if the strata stage failed to serve and time remains, retry once
if grep -qE "STRATA_START_FAILED|strata,serve,START_FAILED" "$B/progress.log" "$B/summary.csv" 2>/dev/null \
   && [ "$(date +%s)" -lt "$((STOP_AT - 7000))" ]; then
  log "STRATA_RETRY start"
  python3 "$B/loadbench.py" strata >> "$B/stage_strata_retry.out" 2>&1
  log "STRATA_RETRY rc=$?"
fi
echo "DONE $(date -Is)" > "$B/done.marker"
log "ALL_DONE"
