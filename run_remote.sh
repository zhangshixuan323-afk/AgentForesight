#!/usr/bin/env bash
# 3x A800 shard runner for the AgentForesight online audit.
#
# One process per GPU (CUDA_VISIBLE_DEVICES=i), each auditing a round-robin
# 1/N_GPUS slice of the 83-sample test set, then merging into outputs/final.
# Resumable: re-running continues from the per-sample.jsonl checkpoints.
#
# Usage:
#   MODEL_PATH=/path/to/AgentForesight-7B DATA_DIR=... OUT=... ./run_remote.sh [local|oracle|mock]
#   (backend defaults to "local"; MOCK_MODE=safe|last|random for mock)
set -euo pipefail

BACKEND="${1:-local}"
MODEL_PATH="${MODEL_PATH:-}"
DATA_DIR="${DATA_DIR:-$HOME/AgentForesight/sample100_by_benchmark}"
OUT="${OUT:-$HOME/AgentForesight/outputs}"
N_GPUS="${N_GPUS:-3}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-30720}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

EXTRA=()
if [ "$BACKEND" = "local" ]; then
  if [ -z "$MODEL_PATH" ]; then
    echo "error: MODEL_PATH is required for backend=local" >&2
    exit 1
  fi
  EXTRA+=(--model-path "$MODEL_PATH" --device auto)
elif [ "$BACKEND" = "mock" ]; then
  EXTRA+=(--mock-mode "${MOCK_MODE:-safe}")
fi

echo "backend=$BACKEND  n_gpus=$N_GPUS  data=$DATA_DIR  out=$OUT"
pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
  echo "starting shard $i/$N_GPUS on GPU $i"
  CUDA_VISIBLE_DEVICES=$i python3 -m inference.infer_local \
    --backend "$BACKEND" \
    --data-format sample100 \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT/shard$i" \
    --shard-index "$i" --shard-count "$N_GPUS" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${EXTRA[@]}" &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

dirs=""
for i in $(seq 0 $((N_GPUS - 1))); do
  [ -n "$dirs" ] && dirs="$dirs,"
  dirs="$dirs$OUT/shard$i"
done
echo "merging: $dirs"
python3 -m tools.merge_results --dirs "$dirs" --out "$OUT/final"

exit $fail
