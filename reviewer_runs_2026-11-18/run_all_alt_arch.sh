#!/usr/bin/env bash
# Sequential loop: 4 heads x 5 seeds on one dataset and one backbone.
# Each run skips itself if its result JSON already exists.
# NOT run in phase 1. Usage:
#   PY=/path/to/python bash run_all_alt_arch.sh <dataset> <backbone> [out-dir] [device]
# Example (full run, after approval):
#   PY=python bash run_all_alt_arch.sh NF-UNSW-NB15-v2 tcn "" cuda
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
DATASET="${1:?dataset required}"
BACKBONE="${2:?backbone required (tcn|resnet1d|mlp)}"
OUT="${3:-$HERE/results/$BACKBONE}"
[ -z "$OUT" ] && OUT="$HERE/results/$BACKBONE"
DEVICE="${4:-cpu}"
ts() { date '+%Y-%m-%d %I:%M:%S %p'; }
fail=0
for head in gap gmp se gcp; do
  for seed in 42 123 456 789 1337; do
    echo "[$(ts)] START $DATASET $BACKBONE $head seed=$seed"
    "$PY" "$HERE/run_alt_arch.py" --dataset "$DATASET" --backbone "$BACKBONE" \
      --head "$head" --seed "$seed" --out-dir "$OUT" --device "$DEVICE"
    rc=$?
    echo "[$(ts)] END $DATASET $BACKBONE $head seed=$seed exit=$rc"
    [ "$rc" -ne 0 ] && fail=$((fail + 1))
  done
done
echo "[$(ts)] ALL DONE failures=$fail"
exit "$fail"
