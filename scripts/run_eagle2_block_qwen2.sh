#!/usr/bin/env bash
# EAGLE-2 + Block Verification 全量 MT-bench 评测（verify-block 分支）
#
# 后台运行示例：
#   nohup bash scripts/run_eagle2_block_qwen2.sh > /dev/null 2>&1 &
#   tail -f /root/autodl-tmp/logs/eagle2_block_t0.5_*.log
#
# 常用覆盖：
#   TEMPERATURE=0.5 MAX_NEW_TOKEN=1024 nohup bash scripts/run_eagle2_block_qwen2.sh &
set -euo pipefail

export USE_BLOCK_VERIFY=1
TEMPERATURE="${TEMPERATURE:-0.5}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"

cd /root/autodl-tmp/EAGLE-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagle

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/autodl-tmp/results /root/autodl-tmp/logs
ANSWER_FILE="/root/autodl-tmp/results/qwen2_eagle2_block_t${TEMPERATURE}_${TS}.jsonl"
LOG_FILE="/root/autodl-tmp/logs/eagle2_block_t${TEMPERATURE}_${TS}.log"

{
  echo "USE_BLOCK_VERIFY=${USE_BLOCK_VERIFY}"
  echo "TEMPERATURE=${TEMPERATURE}"
  echo "ANSWER_FILE=${ANSWER_FILE}"
  python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"
  python -m eagle.evaluation.gen_ea_answer_qwen2 \
    --temperature "${TEMPERATURE}" \
    --max-new-token "${MAX_NEW_TOKEN}" \
    --answer-file "${ANSWER_FILE}"
  echo "DONE -> ${ANSWER_FILE}"
} >> "${LOG_FILE}" 2>&1

echo "Started. Log: ${LOG_FILE}"
echo "Result: ${ANSWER_FILE}"
