#!/usr/bin/env bash
set -euo pipefail

TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"

cd /root/autodl-tmp/EAGLE-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagle

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/autodl-tmp/results /root/autodl-tmp/logs
ANSWER_FILE="/root/autodl-tmp/results/qwen2_vanilla_t${TEMPERATURE}_${TS}.jsonl"
LOG_FILE="/root/autodl-tmp/logs/vanilla_t${TEMPERATURE}_${TS}.log"

python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')" > "${LOG_FILE}"

python -m eagle.evaluation.gen_baseline_answer_qwen2 \
  --temperature "${TEMPERATURE}" \
  --max-new-token "${MAX_NEW_TOKEN}" \
  --answer-file "${ANSWER_FILE}" >> "${LOG_FILE}" 2>&1
