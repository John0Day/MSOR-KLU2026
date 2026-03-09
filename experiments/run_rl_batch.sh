#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

SEEDS=(42 142 242)
GAMES=300
NUM_SEEDS=10
EVAL_BASE_SEED=42

run_one() {
  local label="$1"
  local episodes="$2"
  local seed="$3"

  local out_dir="$ROOT/experiments/results_${label}_seed${seed}"
  local q_path="$out_dir/q_table.npy"

  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START label=${label} episodes=${episodes} seed=${seed}"
  echo "out_dir=$out_dir"

  mkdir -p "$out_dir"

  if [[ -f "$q_path" ]]; then
    echo "Q-table already exists, skipping training: $q_path"
  else
    "$PY" "$ROOT/run.py" train \
      --episodes "$episodes" \
      --seed "$seed" \
      --out "$out_dir"
  fi

  "$PY" "$ROOT/run.py" eval \
    --q-table "$q_path" \
    --games "$GAMES" \
    --seed "$EVAL_BASE_SEED" \
    --num-seeds "$NUM_SEEDS" \
    --alternate-start \
    --out "$out_dir"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE label=${label} seed=${seed}"
}

for seed in "${SEEDS[@]}"; do
  run_one "a" 20000 "$seed"
done

for seed in "${SEEDS[@]}"; do
  run_one "b" 40000 "$seed"
done

echo "============================================================"
echo "All batch runs finished."
