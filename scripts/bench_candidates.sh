#!/usr/bin/env bash
# Thermal/performance A/B for candidate GGUF models, using the SAME invocation
# profile as the ADTC profiler (llama-bench -ngl 0 -p 512 -n 128).
#
# Usage:
#   ./scripts/bench_candidates.sh [model-dir]
#
# For every *.gguf in model/ (default) it reports:
#   - prompt-processing rate (t/s)      -> drives first-token latency
#   - generation rate (t/s)             -> the scored throughput metric
#   - peak CPU package/core temperature -> the >=85C threshold trips "throttled"
#
# Requirements: llama-bench on PATH; lm-sensors or /sys/class/thermal exposure
# (psutil reads both). Temperatures degrade gracefully to n/a on cloud VMs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-$HERE/../model}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

if ! command -v llama-bench > /dev/null 2>&1; then
  echo "error: llama-bench not found on PATH." >&2
  echo "  Install llama.cpp: https://github.com/ggergenov/llama.cpp" >&2
  exit 1
fi

read_peak_temp() {
  # $1 = output file for the observed peak (whole benchmark window)
  (
    peak=0
    while [[ ! -f "$1.stop" ]]; do
      t="$(python3 -c '
import psutil
try:
    temps = psutil.sensors_temperatures() or {}
    vals = [float(e.current) for entries in temps.values() for e in entries if e.current]
    print(max(vals) if vals else 0)
except Exception:
    print(0)
' 2>/dev/null || echo 0)"
      if awk -v a="$t" -v b="$peak" 'BEGIN{exit !(a>b)}'; then peak="$t"; fi
      sleep 0.5
    done
    echo "$peak" > "$1"
  ) &
  echo $!
}

RESULTS="$WORKDIR/results.tsv"
touch "$RESULTS"

shopt -s nullglob
GGUFS=("$MODEL_DIR"/*.gguf)
shopt -u nullglob

if [[ ${#GGUFS[@]} -eq 0 ]]; then
  echo "error: no .gguf files found in $MODEL_DIR" >&2
  exit 1
fi

for gguf in "${GGUFS[@]}"; do
  name="$(basename "$gguf")"
  size_mb="$(du -m "$gguf" | cut -f1)"
  flag="$WORKDIR/$name.flag"
  out="$WORKDIR/$name.temp"

  echo "=== benchmarking $name (${size_mb} MB) ==="
  pid="$(read_peak_temp "$out")"

  if ! rows="$(llama-bench -m "$gguf" -p 512 -n 128 -ngl 0 --output json 2>"$WORKDIR/stderr.log")"; then
    kill "$pid" 2> /dev/null || true
    touch "$flag.stop"
    echo "error: llama-bench failed for $name:" >&2
    tail -5 "$WORKDIR/stderr.log" >&2 || true
    exit 1
  fi

  touch "$flag.stop"
  wait "$pid" 2> /dev/null || true
  peak_temp="$(cat "$out" 2> /dev/null || echo 0)"

  read_line="$(printf '%s' "$rows" | python3 -c '
import json, sys
rows = json.load(sys.stdin)
pp = next((r for r in rows if r.get("n_gen", 0) == 0 and r.get("n_prompt", 0) > 0), None)
tg = next((r for r in rows if r.get("n_gen", 0) > 0), None)
pp_rate = float(pp["avg_ts"]) if pp else 0.0
tg_rate = float(tg["avg_ts"]) if tg else 0.0
ttft_ms = (512 / pp_rate * 1000.0) if pp_rate > 0 else 0.0
print(f"{pp_rate:.2f}\t{tg_rate:.2f}\t{ttft_ms:.0f}")
')"

  IFS=$'\t' read -r pp_rate tg_rate ttft <<<"$read_line"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$size_mb" "$tg_rate" "$ttft" "$peak_temp" >> "$RESULTS"
done

echo
echo "================ RESULTS (profiler-equivalent invocation) ================"
printf '%-42s %8s %10s %10s %12s\n' "model" "MB" "tg(t/s)" "TTFT(ms)" "peakTemp(C)"
printf '%-42s %8s %10s %10s %12s\n' "------------------------------------------" "----" "-------" "---------" "-----------"
while IFS=$'\t' read -r name size tg ttft temp; do
  [[ "$temp" == "0" ]] && temp="n/a"
  printf '%-42s %8s %10s %10s %12s\n' "$name" "$size" "$tg" "$ttft" "$temp"
done < "$RESULTS"
echo "=========================================================================="
echo "Scoring note: the ADTC profiler flags throttled=true when peak temp >= 85 C."
