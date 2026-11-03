#!/usr/bin/env bash
# M0 GPU/CPU spike: build llama.cpp (Vulkan, optionally ROCm/HIP) into a user
# prefix, download GGUF equivalents of the NPU spike models, and measure them on
# GPU and CPU with llama-bench + the same OpenAI probe as the NPU spike.
#
# Usage:
#   scripts/spikes/gpu-llamacpp.sh                 # all models, all backends
#   MODELS="qwen3-8b" scripts/spikes/gpu-llamacpp.sh
#   BACKENDS="vulkan" scripts/spikes/gpu-llamacpp.sh
#   SKIP_BUILD=1 scripts/spikes/gpu-llamacpp.sh
#
# ROCm is built only if `hipconfig` exists (Fedora: sudo dnf install
# rocm-hip-devel hipblas-devel rocblas-devel). Otherwise it is skipped.
# Everything lives under $CONTIO_SPIKE_DIR (default ~/.local/share/contio-spike).
set -euo pipefail

LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
LLAMA_TAG=b12068
PORT=${PORT:-52636}
CTX=${CTX:-16384}
CPU_THREADS=${CPU_THREADS:-12}   # physical cores (4x Zen 5 + 8x Zen 5c)

# name | HF repo | file  (same models as the NPU spike, Q4 GGUF)
CATALOG=$(cat <<'EOF'
llama3.2-1b|bartowski/Llama-3.2-1B-Instruct-GGUF|Llama-3.2-1B-Instruct-Q4_K_M.gguf
qwen3-4b-it|unsloth/Qwen3-4B-Instruct-2507-GGUF|Qwen3-4B-Instruct-2507-Q4_K_M.gguf
qwen3-8b|unsloth/Qwen3-8B-GGUF|Qwen3-8B-Q4_K_M.gguf
qwen3.6-35b-a3b|unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
EOF
)
MODELS=${MODELS:-$(cut -d'|' -f1 <<<"$CATALOG" | tr '\n' ' ')}
BACKENDS=${BACKENDS:-vulkan rocm cpu}

ROOT=${CONTIO_SPIKE_DIR:-$HOME/.local/share/contio-spike}
SRC=$ROOT/src/llama.cpp
GGUF=$ROOT/gguf
OUT=$ROOT/results/gpu-$(date +%Y%m%d-%H%M%S)
PROBE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/openai_probe.py"
mkdir -p "$ROOT/src" "$GGUF" "$OUT"

log() { echo "[spike] $*" | tee -a "$OUT/spike.log"; }

# --- 1. environment snapshot -------------------------------------------------
log "kernel $(uname -r)"
log "power profile: $(tuned-adm active 2>/dev/null | sed 's/.*: //' || powerprofilesctl get 2>/dev/null || echo unknown)"
log "AC online: $(cat /sys/class/power_supply/ACAD/online 2>/dev/null || echo ?)"
for f in /sys/class/drm/card*/device/mem_info_vram_total; do
  [ -r "$f" ] && log "VRAM $(dirname "$f"): $(( $(cat "$f") / 1024**3 )) GiB, GTT $(( $(cat "${f%vram_total}gtt_total") / 1024**3 )) GiB"
done
log "mesa: $(rpm -q mesa-vulkan-drivers --qf '%{VERSION}\n' 2>/dev/null | head -1)"

# --- 2. build ------------------------------------------------------------------
build() {  # build <name> <cmake args...>
  local name=$1; shift
  log "building llama.cpp $LLAMA_TAG [$name]"
  local start; start=$(date +%s)
  cmake -S "$SRC" -B "$SRC/build-$name" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "$@" \
      > "$OUT/build-$name.log" 2>&1 \
    && cmake --build "$SRC/build-$name" -j"$(nproc)" --target llama-server llama-bench \
      >> "$OUT/build-$name.log" 2>&1 \
    || { log "BUILD $name FAILED, see $OUT/build-$name.log"; tail -20 "$OUT/build-$name.log"; return 1; }
  log "build $name took $(( $(date +%s) - start ))s"
}

if [ -z "${SKIP_BUILD:-}" ]; then
  if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin "$LLAMA_REPO"
  fi
  git -C "$SRC" fetch -q --depth 1 origin tag "$LLAMA_TAG"
  git -C "$SRC" checkout -q --detach "$LLAMA_TAG"
  log "llama.cpp @ $LLAMA_TAG ($(git -C "$SRC" rev-parse --short HEAD))"

  # The Vulkan build also serves as the CPU backend (with --device none).
  build vulkan -DGGML_VULKAN=ON || BACKENDS=${BACKENDS//vulkan/}
  if [[ " $BACKENDS " == *" rocm "* ]]; then
    if command -v hipconfig >/dev/null; then
      HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
        build rocm -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1150 || BACKENDS=${BACKENDS//rocm/}
    else
      log "ROCm SKIPPED: hipconfig not found (sudo dnf install rocm-hip-devel hipblas-devel rocblas-devel)"
      BACKENDS=${BACKENDS//rocm/}
    fi
  fi
else
  [ -x "$SRC/build-rocm/bin/llama-server" ] || BACKENDS=${BACKENDS//rocm/}
fi
log "backends: $BACKENDS"

# --- 3. download models -------------------------------------------------------
for m in $MODELS; do
  IFS='|' read -r _ repo file < <(grep "^$m|" <<<"$CATALOG")
  [ -s "$GGUF/$file" ] && continue
  log "download $m ($file)"
  start=$(date +%s)
  curl -fL --retry 5 -C - -o "$GGUF/$file.part" \
      "https://huggingface.co/$repo/resolve/main/$file" 2>>"$OUT/download.log" \
    || { log "download $m FAILED"; continue; }
  want=$(curl -sf "https://huggingface.co/api/models/$repo/tree/main" | python3 -I -c \
    'import json,sys; print([x["lfs"]["oid"] for x in json.load(sys.stdin) if x["path"]==sys.argv[1]][0])' "$file")
  have=$(sha256sum "$GGUF/$file.part" | cut -d' ' -f1)
  if [ "$want" != "$have" ]; then
    log "download $m CHECKSUM MISMATCH (got $have, want $want), removing"
    rm -f "$GGUF/$file.part"; continue
  fi
  mv "$GGUF/$file.part" "$GGUF/$file"
  log "download $m took $(( $(date +%s) - start ))s (sha256 ok)"
done

# --- 4. measure ---------------------------------------------------------------
backend_args() {  # -> bin dir + device flags
  case $1 in
    vulkan) echo "$SRC/build-vulkan/bin|-ngl 999" ;;
    rocm)   echo "$SRC/build-rocm/bin|-ngl 999" ;;
    cpu)    echo "$SRC/build-vulkan/bin|-ngl 0 --device none -t $CPU_THREADS" ;;
  esac
}

wait_ready() {
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && return 0
    sleep 1
  done
  return 1
}

for m in $MODELS; do
  IFS='|' read -r _ _ file < <(grep "^$m|" <<<"$CATALOG")
  [ -s "$GGUF/$file" ] || { log "skip $m (no model file)"; continue; }
  for b in $BACKENDS; do
    IFS='|' read -r bin flags <<<"$(backend_args "$b")"
    tag="${m}_${b}"
    log "=== $m on $b ==="

    # raw throughput: prompt processing 512/4096, generation 128, flash attention on
    # shellcheck disable=SC2086
    timeout 1800 "$bin/llama-bench" -m "$GGUF/$file" $flags -fa 1 -p 512,4096 -n 128 -r 3 -o json \
        > "$OUT/bench-$tag.json" 2> "$OUT/bench-$tag.log" \
      && python3 -I -c '
import json, sys
for r in json.load(open(sys.argv[1])):
    n_p, n_g, ts = r["n_prompt"], r["n_gen"], r["avg_ts"]
    kind = f"pp{n_p}" if n_p else f"tg{n_g}"
    print(f"[spike]   llama-bench {kind}: {ts:.1f} tok/s")
' "$OUT/bench-$tag.json" | tee -a "$OUT/spike.log" \
      || log "llama-bench $tag FAILED (see bench-$tag.log)"

    # API behaviour + TTFT + power, same probe as the NPU spike
    start=$(date +%s.%N)
    # shellcheck disable=SC2086
    "$bin/llama-server" -m "$GGUF/$file" $flags -fa on -c "$CTX" --jinja \
        --host 127.0.0.1 --port "$PORT" --alias "$m" > "$OUT/serve-$tag.log" 2>&1 &
    SERVE_PID=$!
    if wait_ready; then
      log "server ready after $(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $start}")s"
      python3 -I "$PROBE" "http://127.0.0.1:$PORT" "$m" | tee "$OUT/probe-$tag.jsonl"
    else
      log "server for $tag did not come up (see serve-$tag.log)"
    fi
    kill "$SERVE_PID" 2>/dev/null || true; wait "$SERVE_PID" 2>/dev/null || true
  done
done

log "done. results in $OUT"
