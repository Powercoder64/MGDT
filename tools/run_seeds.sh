#!/usr/bin/env bash
# Trains a config with seeds 0, 1 and 2 and reports mean ± std of the final metrics.
# Usage: bash tools/run_seeds.sh CONFIG [NUM_GPUS] [key=value overrides...]
set -euo pipefail

CONFIG=${1:?usage: bash tools/run_seeds.sh CONFIG [NUM_GPUS] [key=value ...]}
NGPU=${2:-1}
shift $(( $# >= 2 ? 2 : 1 ))
NAME=$(basename "${CONFIG}" .yaml)
cd "$(dirname "$0")/.."

for SEED in 0 1 2; do
  OUT="outputs/seeds/${NAME}/seed${SEED}"
  if [ "${NGPU}" -gt 1 ]; then
    torchrun --nproc_per_node="${NGPU}" tools/train.py --config "${CONFIG}" --seed "${SEED}" --output-dir "${OUT}" --opts "$@"
  else
    python tools/train.py --config "${CONFIG}" --seed "${SEED}" --output-dir "${OUT}" --opts "$@"
  fi
done

python tools/aggregate_results.py outputs/seeds/"${NAME}"/seed*/results.json
