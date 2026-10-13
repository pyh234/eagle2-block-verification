#!/usr/bin/env bash
# Block Verification 消融实验一键串行脚本（单卡请串行，勿并行抢 GPU）
#
# 本脚本用【固定草稿树大小】重新生成全部 baseline 与对照结果，保证各配置严格可比：
#   1) Vanilla AR         t=0      —— 纯自回归基线
#   2) EAGLE-2 (no block) t=0      —— EAGLE 基线
#   3) EAGLE-2 + Block    t=0      —— sanity（t=0 走贪心分支，结果应与 #2 完全一致）
#   4) Vanilla AR         t=TEMP   —— 同温度自回归对照
#   5) EAGLE-2 (no block) t=TEMP   —— 采样基线
#   6) EAGLE-2 + Block    t=TEMP   —— Block 主实验
# 然后自动算 speed：
#   - eagle2 t=0    vs vanilla t=0     （基线加速比）
#   - block  t=0    vs vanilla t=0     （sanity）
#   - eagle2 t=TEMP vs vanilla t=TEMP
#   - block  t=TEMP vs vanilla t=TEMP
#
# 关键：固定 --total-token/--depth/--top-k，跳过 EAGLE-2 在 total_token=-1 时的
# “计时自动标定”（ea_model.py 会用 GPU 实测耗时从 [40,48,50,56,60] 里挑树大小，
# 计时噪声会让不同次运行选出不同的树，导致接受长度/加速比在运行间漂移、互相不可比）。
#
# 后台运行（推荐）：
#   cd /root/autodl-tmp/EAGLE-main
#   nohup bash scripts/run_all_block_experiments.sh > /root/autodl-tmp/logs/block_suite_nohup.log 2>&1 &
#   tail -f /root/autodl-tmp/logs/block_suite_nohup.log
#
# 可调参数（均带默认值，可用环境变量覆盖）：
#   MAX_NEW_TOKEN=1024 TOTAL_TOKEN=60 DEPTH=5 TOP_K=10 TEMP_SAMPLE=0.5 \
#     bash scripts/run_all_block_experiments.sh
set -euo pipefail

cd /root/autodl-tmp/EAGLE-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagle

MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"
# 固定草稿树配置：所有运行共用同一棵树 => 结果可比、可复现
TOTAL_TOKEN="${TOTAL_TOKEN:-60}"
DEPTH="${DEPTH:-5}"
TOP_K="${TOP_K:-10}"
# 采样分支用的温度（Block 主实验在此温度下做）
TEMP_SAMPLE="${TEMP_SAMPLE:-0.5}"
# 统一传给所有生成脚本的树参数（vanilla / eagle2 / block 都加载 EaModel，均接受这些参数）
TREE_ARGS=(--total-token "${TOTAL_TOKEN}" --depth "${DEPTH}" --top-k "${TOP_K}")

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/autodl-tmp/results /root/autodl-tmp/logs
MANIFEST="/root/autodl-tmp/results/block_experiments_manifest_${TS}.txt"

# NOTE: log 必须写到 stderr（不能写 stdout）。下面的 run_* 用 stdout “返回”结果文件路径，
# 并被 $(...) 捕获；若 log 也写 stdout，捕获到的路径会被日志行污染，导致 speed 步骤报
# FileNotFoundError。
log() { echo "[$(date '+%F %T')] $*" | tee -a "${MANIFEST}" >&2; }

run_eagle2_block() {
  local temp="$1"
  export USE_BLOCK_VERIFY=1
  local out="/root/autodl-tmp/results/qwen2_eagle2_block_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/eagle2_block_t${temp}_${TS}.log"
  log "RUN EAGLE-2+Block t=${temp} total_token=${TOTAL_TOKEN} -> ${out}"
  {
    echo "USE_BLOCK_VERIFY=1 TEMPERATURE=${temp} ${TREE_ARGS[*]}"
    python -m eagle.evaluation.gen_ea_answer_qwen2 \
      --temperature "${temp}" \
      --max-new-token "${MAX_NEW_TOKEN}" \
      "${TREE_ARGS[@]}" \
      --answer-file "${out}"
  } >> "${lg}" 2>&1
  echo "${out}"
}

run_vanilla() {
  local temp="$1"
  unset USE_BLOCK_VERIFY
  local out="/root/autodl-tmp/results/qwen2_vanilla_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/vanilla_t${temp}_${TS}.log"
  log "RUN Vanilla t=${temp} -> ${out}"
  {
    python -m eagle.evaluation.gen_baseline_answer_qwen2 \
      --temperature "${temp}" \
      --max-new-token "${MAX_NEW_TOKEN}" \
      "${TREE_ARGS[@]}" \
      --answer-file "${out}"
  } >> "${lg}" 2>&1
  echo "${out}"
}

run_eagle2() {
  local temp="$1"
  unset USE_BLOCK_VERIFY
  local out="/root/autodl-tmp/results/qwen2_eagle2_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/eagle2_t${temp}_${TS}.log"
  log "RUN EAGLE-2 (no block) t=${temp} total_token=${TOTAL_TOKEN} -> ${out}"
  {
    python -m eagle.evaluation.gen_ea_answer_qwen2 \
      --temperature "${temp}" \
      --max-new-token "${MAX_NEW_TOKEN}" \
      "${TREE_ARGS[@]}" \
      --answer-file "${out}"
  } >> "${lg}" 2>&1
  echo "${out}"
}

run_speed() {
  local ea="$1" base="$2" tag="$3"
  local out="/root/autodl-tmp/results/speed_${tag}_${TS}.txt"
  log "SPEED ${tag} -> ${out}"
  {
    echo "ea_file   : ${ea}"
    echo "base_file : ${base}"
    echo "tree      : total_token=${TOTAL_TOKEN} depth=${DEPTH} top_k=${TOP_K}"
    echo
    python -m eagle.evaluation.speed_qwen2 --ea-file "${ea}" --base-file "${base}"
  } > "${out}" 2>&1
  echo "${out}"
}

log "======== Block Verification experiment suite (fixed tree) ========"
log "MAX_NEW_TOKEN=${MAX_NEW_TOKEN} TOTAL_TOKEN=${TOTAL_TOKEN} DEPTH=${DEPTH} TOP_K=${TOP_K} TEMP_SAMPLE=${TEMP_SAMPLE}"

# ---- t=0（贪心）：重新生成 vanilla / eagle2 / block，全部固定树大小 ----
VAN_T0="$(run_vanilla 0.0)"
EA_T0="$(run_eagle2 0.0)"
BLOCK_T0="$(run_eagle2_block 0.0)"

# ---- t=TEMP_SAMPLE（采样）：vanilla / eagle2 / block ----
VAN_TS="$(run_vanilla "${TEMP_SAMPLE}")"
EA_TS="$(run_eagle2 "${TEMP_SAMPLE}")"
BLOCK_TS="$(run_eagle2_block "${TEMP_SAMPLE}")"

# ---- speed 对比（同温度、同树大小，干净可比）----
SPEED_EA_T0="$(run_speed "${EA_T0}" "${VAN_T0}" "eagle2_t0.0_vs_vanilla")"
SPEED_BLOCK_T0="$(run_speed "${BLOCK_T0}" "${VAN_T0}" "block_t0.0_vs_vanilla")"
SPEED_EA_TS="$(run_speed "${EA_TS}" "${VAN_TS}" "eagle2_t${TEMP_SAMPLE}_vs_vanilla")"
SPEED_BLOCK_TS="$(run_speed "${BLOCK_TS}" "${VAN_TS}" "block_t${TEMP_SAMPLE}_vs_vanilla")"

log "======== DONE ========"
log "VAN_T0=${VAN_T0}"
log "EA_T0=${EA_T0}"
log "BLOCK_T0=${BLOCK_T0}"
log "VAN_TS=${VAN_TS}"
log "EA_TS=${EA_TS}"
log "BLOCK_TS=${BLOCK_TS}"
log "SPEED_EA_T0=${SPEED_EA_T0}"
log "SPEED_BLOCK_T0=${SPEED_BLOCK_T0}"
log "SPEED_EA_TS=${SPEED_EA_TS}"
log "SPEED_BLOCK_TS=${SPEED_BLOCK_TS}"
log "Manifest: ${MANIFEST}"
