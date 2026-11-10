#!/usr/bin/env bash
# Prompt reading and writing at agent depths (docs/PERFORMANCE.md §2d, 5a).
#
#   scripts/bench/depth.sh <llama.cpp build dir> <model.gguf> [label]
#
# Reads 2048 tokens and writes 128 tokens after 0/32k/64k/96k tokens of
# context, flash attention on, everything on the GPU. Prints a
# summary table and keeps the JSON for comparisons across builds.
# Each depth is filled before it is measured: ~12 min per build for a 35B-A3B.
set -euo pipefail

build=${1:?llama.cpp build dir (with bin/llama-bench)}
model=${2:?model .gguf}
label=${3:-$(basename "$build")}
depths=${DEPTHS:-0,32768,65536,98304}
out=${OUT:-target/bench-depth}

bench="$build/bin/llama-bench"
[[ -x $bench ]] || { echo "no llama-bench in $build/bin" >&2; exit 1; }
mkdir -p "$out"
json="$out/$label.json"

profile=$(busctl get-property net.hadess.PowerProfiles /net/hadess/PowerProfiles \
    net.hadess.PowerProfiles ActiveProfile 2>/dev/null | cut -d'"' -f2)
battery=$(cat /sys/class/power_supply/BAT*/capacity 2>/dev/null | head -1)
ac=$(cat /sys/class/power_supply/AC*/online 2>/dev/null | head -1)
# a laptop under ~15% battery may switch to power saving mid-run and skew the numbers
echo "# $label: ${profile:-?} profile, AC ${ac:-?}, battery ${battery:-?}%, $(date -Iseconds)"
"$bench" -m "$model" -ngl 999 -fa 1 -r 1 -p 2048 -n 128 -d "$depths" -o json >"$json"
python3 - "$json" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
print(f"{'build':<10} {'depth':>6} {'test':>6} {'tok/s':>8}")
for r in rows:
    test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
    print(f"{r['build_commit'][:9]:<10} {r['n_depth']:>6} {test:>6} {r['avg_ts']:>8.1f}")
EOF
echo "→ $json"
