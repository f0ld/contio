#!/usr/bin/env bash
# M0 NPU spike: build FastFlowLM from source into a user prefix (no sudo),
# validate the NPU, pull a few models and probe the OpenAI API contio relies on.
#
# Usage:
#   scripts/spikes/npu-flm.sh                    # build + default models
#   scripts/spikes/npu-flm.sh qwen3:8b gpt-oss:20b  # custom model tags
#   SKIP_BUILD=1 scripts/spikes/npu-flm.sh       # reuse existing build
#
# Everything lives under $CONTIO_SPIKE_DIR (default ~/.local/share/contio-spike).
# Results: $CONTIO_SPIKE_DIR/results/<timestamp>/
set -euo pipefail

FLM_REPO=https://github.com/ROCm/FastFlowLM.git
FLM_COMMIT=d4a4553c9b12c64f16d913423d68d0fddd3536b3   # 2026-11-02, FLM 1.0.12
PORT=${PORT:-52626}
EMBED_MODEL=${EMBED_MODEL:-embed-gemma:300m}
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(llama3.2:1b qwen3-it:4b qwen3:8b qwen3.6-moe:35b-a3b)

ROOT=${CONTIO_SPIKE_DIR:-$HOME/.local/share/contio-spike}
SRC=$ROOT/src/FastFlowLM
export FLM_PREFIX=$ROOT/flm
export FLM_MODEL_PATH=$ROOT/models
OUT=$ROOT/results/$(date +%Y%m%d-%H%M%S)
PROBE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/openai_probe.py"
mkdir -p "$ROOT/src" "$FLM_MODEL_PATH" "$OUT"

log() { echo "[spike] $*" | tee -a "$OUT/spike.log"; }

# --- 1. environment snapshot -------------------------------------------------
log "kernel $(uname -r)"
source /opt/xilinx/xrt/setup.sh >/dev/null
xrt-smi examine > "$OUT/xrt-examine.txt" 2>&1 || log "xrt-smi examine FAILED"
grep -E "NPU Firmware|Version  " "$OUT/xrt-examine.txt" | tee -a "$OUT/spike.log" || true
log "memlock: $(ulimit -l)"

# --- 2. build ------------------------------------------------------------------
if [ -z "${SKIP_BUILD:-}" ]; then
  if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin "$FLM_REPO"
  fi
  git -C "$SRC" fetch -q --depth 1 origin "$FLM_COMMIT"
  git -C "$SRC" checkout -q --detach "$FLM_COMMIT"
  git -C "$SRC" submodule update -q --init --recursive --depth 1
  log "building FLM @ ${FLM_COMMIT:0:7} -> $FLM_PREFIX"
  start=$(date +%s)
  (cd "$SRC/src" && ./home_install.sh) > "$OUT/build.log" 2>&1 \
    || { log "BUILD FAILED, see $OUT/build.log"; tail -30 "$OUT/build.log"; exit 1; }
  log "build+install took $(( $(date +%s) - start ))s"
fi

# shellcheck disable=SC1091
source "$FLM_PREFIX/flm_env.sh" >/dev/null
# Upstream home_install.sh (at FLM_COMMIT) hardcodes a developer's
# FLM_MODEL_PATH=/scratch/alfxu into flm_env.sh, so restore ours.
export FLM_MODEL_PATH=$ROOT/models
flm validate --json > "$OUT/validate.json" 2>&1 || log "flm validate reported problems"
cat "$OUT/validate.json" | tee -a "$OUT/spike.log"

# --- 3. pull models --------------------------------------------------------------
for m in "${MODELS[@]}" "$EMBED_MODEL"; do
  log "pull $m"
  start=$(date +%s)
  flm pull "$m" >> "$OUT/pull.log" 2>&1 || { log "pull $m FAILED"; continue; }
  log "pull $m took $(( $(date +%s) - start ))s"
done

# --- 4. probe each model ---------------------------------------------------------
wait_ready() {
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && return 0
    sleep 1
  done
  return 1
}

for m in "${MODELS[@]}"; do
  log "=== $m ==="
  start=$(date +%s.%N)
  flm serve "$m" --port "$PORT" --embed 1 > "$OUT/serve-${m//[:\/]/_}.log" 2>&1 &
  SERVE_PID=$!
  if ! wait_ready; then
    log "server for $m did not come up"; kill "$SERVE_PID" 2>/dev/null || true; continue
  fi
  log "server ready after $(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $start}")s"
  python3 -I "$PROBE" "http://127.0.0.1:$PORT" "$m" --embed-model "$EMBED_MODEL" \
    | tee "$OUT/probe-${m//[:\/]/_}.jsonl"

  # concurrency: can a second FLM process use the NPU while the first holds a model?
  if [ "$m" = "${MODELS[0]}" ]; then
    log "concurrency check: second flm serve on port $((PORT+1))"
    flm serve "$EMBED_MODEL" --port $((PORT+1)) > "$OUT/serve-second.log" 2>&1 &
    SECOND_PID=$!
    sleep 20
    if curl -sf -X POST "http://127.0.0.1:$((PORT+1))/v1/embeddings" \
         -H 'Content-Type: application/json' \
         -d "{\"model\":\"$EMBED_MODEL\",\"input\":[\"hi\"]}" >/dev/null; then
      log "concurrency: second process WORKS"
    else
      log "concurrency: second process FAILED (see serve-second.log)"
    fi
    kill "$SECOND_PID" 2>/dev/null || true; wait "$SECOND_PID" 2>/dev/null || true
  fi

  kill "$SERVE_PID" 2>/dev/null || true; wait "$SERVE_PID" 2>/dev/null || true
done

log "done. results in $OUT"
