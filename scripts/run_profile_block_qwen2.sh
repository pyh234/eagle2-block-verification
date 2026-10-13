#!/usr/bin/env bash
# Block Verification —— Profiling 一键脚本（单卡串行，勿并行抢 GPU）
#
# 目的：在【固定草稿树】、采样温度 t=TEMP_SAMPLE 下，分别跑两次带 --enable-profile 的生成：
#   1) EAGLE-2 baseline（不开 block）+ profile  -> prof_eagle2_t{TEMP}_{TS}.jsonl
#   2) EAGLE-2 + Block            + profile      -> prof_block_t{TEMP}_{TS}.jsonl
# 二者只差一个 USE_BLOCK_VERIFY=1。
#
# 注意：
#   - --enable-profile 会插入 torch.cuda.synchronize() 计时，会拖慢墙钟，
#     所以这两个 jsonl 只用来看【各段占比】（verify% = 你加速比的上限），
#     不要拿它们的 tokens/s 当真实加速比（真实加速比看 20260613_145125 那批无 profile 结果）。
#   - 本脚本【不调用】speed_qwen2.py，分析需要自己单独跑。
#   - 固定 --total-token/--depth/--top-k，跳过 EAGLE-2 的“计时自动标定”，保证可比。
# 
#   - 本脚本输出文件名带 prof_ 前缀 + 新时间戳，互不冲突。
#
# 后台运行（推荐，80 题约 13 分钟/次，两次约 25 分钟）：
#   cd /root/autodl-tmp/EAGLE-main
#   nohup bash scripts/run_profile_block_qwen2.sh > /root/autodl-tmp/logs/profile_suite_nohup.log 2>&1 &
#   tail -f /root/autodl-tmp/logs/profile_suite_nohup.log
#
# 可调参数（带默认值，可用环境变量覆盖）：
#   MAX_NEW_TOKEN=1024 TOTAL_TOKEN=60 DEPTH=5 TOP_K=10 TEMP_SAMPLE=0.5 \
#     bash scripts/run_profile_block_qwen2.sh
set -euo pipefail

cd /root/autodl-tmp/EAGLE-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagle

MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"
# 固定草稿树配置：与主实验保持一致 => 占比可比、可复现
TOTAL_TOKEN="${TOTAL_TOKEN:-60}"
DEPTH="${DEPTH:-5}"
TOP_K="${TOP_K:-10}"
# Block 主实验所在的采样温度
TEMP_SAMPLE="${TEMP_SAMPLE:-0.5}"
TREE_ARGS=(--total-token "${TOTAL_TOKEN}" --depth "${DEPTH}" --top-k "${TOP_K}")

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/autodl-tmp/results /root/autodl-tmp/logs
MANIFEST="/root/autodl-tmp/results/profile_manifest_${TS}.txt"

# log 写 stderr，避免污染任何 stdout 捕获（与主套件约定一致）
log() { echo "[$(date '+%F %T')] $*" | tee -a "${MANIFEST}" >&2; }

# EAGLE-2 baseline（不开 block）+ profile
run_profile_eagle2() {
  local temp="$1"
  unset USE_BLOCK_VERIFY
  local out="/root/autodl-tmp/results/prof_eagle2_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/prof_eagle2_t${temp}_${TS}.log"
  log "RUN [profile] EAGLE-2 (no block) t=${temp} total_token=${TOTAL_TOKEN} -> ${out}"
  {
    echo "ENABLE_PROFILE TEMPERATURE=${temp} ${TREE_ARGS[*]}"
    python -m eagle.evaluation.gen_ea_answer_qwen2 \
      --temperature "${temp}" \
      --max-new-token "${MAX_NEW_TOKEN}" \
      "${TREE_ARGS[@]}" \
      --enable-profile \
      --answer-file "${out}"
  } >> "${lg}" 2>&1
  echo "${out}"
}

# EAGLE-2 + Block + profile
run_profile_block() {
  local temp="$1"
  export USE_BLOCK_VERIFY=1
  local out="/root/autodl-tmp/results/prof_block_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/prof_block_t${temp}_${TS}.log"
  log "RUN [profile] EAGLE-2+Block t=${temp} total_token=${TOTAL_TOKEN} -> ${out}"
  {
    echo "USE_BLOCK_VERIFY=1 ENABLE_PROFILE TEMPERATURE=${temp} ${TREE_ARGS[*]}"
    python -m eagle.evaluation.gen_ea_answer_qwen2 \
      --temperature "${temp}" \
      --max-new-token "${MAX_NEW_TOKEN}" \
      "${TREE_ARGS[@]}" \
      --enable-profile \
      --answer-file "${out}"
  } >> "${lg}" 2>&1
  echo "${out}"
}

log "======== Block Verification PROFILING suite (fixed tree, --enable-profile) ========"
log "MAX_NEW_TOKEN=${MAX_NEW_TOKEN} TOTAL_TOKEN=${TOTAL_TOKEN} DEPTH=${DEPTH} TOP_K=${TOP_K} TEMP_SAMPLE=${TEMP_SAMPLE}"

PROF_EA="$(run_profile_eagle2 "${TEMP_SAMPLE}")"
PROF_BLOCK="$(run_profile_block "${TEMP_SAMPLE}")"

log "======== DONE ========"
log "PROF_EA=${PROF_EA}"
log "PROF_BLOCK=${PROF_BLOCK}"
log "Manifest: ${MANIFEST}"
log ""
log "下一步（手动跑分析，本脚本不调用 speed_qwen2.py）："
log "  python -m eagle.evaluation.speed_qwen2 --ea-file ${PROF_EA} --base-file /root/autodl-tmp/results/qwen2_vanilla_t${TEMP_SAMPLE}_20260613_145125.jsonl"
log "  python -m eagle.evaluation.speed_qwen2 --ea-file ${PROF_BLOCK} --base-file /root/autodl-tmp/results/qwen2_vanilla_t${TEMP_SAMPLE}_20260613_145125.jsonl"
