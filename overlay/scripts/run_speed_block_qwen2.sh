#!/usr/bin/env bash
# 计算 Block / EAGLE-2 相对 Vanilla 的加速比
#
# 示例（t=0.5 Block vs 同批 Vanilla）：
#   EA_FILE=/root/autodl-tmp/results/qwen2_eagle2_block_t0.5_XXXX.jsonl \
#   BASE_FILE=/root/autodl-tmp/results/qwen2_vanilla_t0.5_XXXX.jsonl \
#   bash scripts/run_speed_block_qwen2.sh
#
# 示例（t=0 Block vs 你自己的 vanilla baseline）：
#   EA_FILE=/root/autodl-tmp/results/qwen2_eagle2_block_t0.0_XXXX.jsonl \
#   bash scripts/run_speed_block_qwen2.sh
#
# BASE_FILE 默认即你自己的 vanilla t=0（2026-06-08）
set -euo pipefail

EA_FILE="${EA_FILE:?set EA_FILE to eagle2/block answer jsonl}"
BASE_FILE="${BASE_FILE:-/root/autodl-tmp/results/qwen2_vanilla_t0.0_20260608_133513.jsonl}"

cd /root/autodl-tmp/EAGLE-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagle

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/autodl-tmp/results
OUT_FILE="${OUT_FILE:-/root/autodl-tmp/results/speed_block_${TS}.txt}"

{
  echo "ea_file   : ${EA_FILE}"
  echo "base_file : ${BASE_FILE}"
  echo
  python -m eagle.evaluation.speed_qwen2 \
    --ea-file "${EA_FILE}" \
    --base-file "${BASE_FILE}"
} > "${OUT_FILE}" 2>&1

echo "Saved to: ${OUT_FILE}"
