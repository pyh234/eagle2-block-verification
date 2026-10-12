#!/usr/bin/env bash
set -euo pipefail

EA_FILE="${EA_FILE:-/root/autodl-tmp/results/qwen2_eagle2_t0.0_20260608_140503.jsonl}"
BASE_FILE="${BASE_FILE:-/root/autodl-tmp/results/qwen2_vanilla_t0.0_20260608_133513.jsonl}"

cd /root/autodl-tmp/EAGLE-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagle

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/autodl-tmp/results
OUT_FILE="${OUT_FILE:-/root/autodl-tmp/results/speed_qwen2_t0.0_${TS}.txt}"

{
  echo "ea_file   : ${EA_FILE}"
  echo "base_file : ${BASE_FILE}"
  echo
  python -m eagle.evaluation.speed_qwen2 \
    --ea-file "${EA_FILE}" \
    --base-file "${BASE_FILE}"
} > "${OUT_FILE}" 2>&1

echo "Saved to: ${OUT_FILE}"
