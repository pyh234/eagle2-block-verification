#!/usr/bin/env bash
# Temperature-ablation suite for Block Verification: t=0.8 (fixed tree).
#
# 复用主消融的固定树配置，在更高采样温度 t=0.8 下重新生成 vanilla / EAGLE-2 /
# Block(向量化) 三组，并对同温度算 speed。用于回答：高温采样下 Block 的接受长度
# 收益与分布偏移是否变大（与已有 t=0 / t=0.5 批次对照）。
#
# 与 scripts/run_all_block_experiments.sh 保持一致：
#   - 固定 --total-token/--depth/--top-k
#   - Block 用环境变量 USE_BLOCK_VERIFY=1 开关，跑的是当前 utils.py；
#   - 不开 --enable-profile
# 本脚本不改任何现有脚本，输出文件名带新时间戳，互不冲突。
#
# 后台运行（推荐，3 次生成约 ~20 分钟 + 2 次 speed）：
#   cd /root/autodl-tmp/EAGLE-main
#   nohup bash scripts/run_temp08_block_experiments.sh > /root/autodl-tmp/logs/temp08_block_suite_nohup.log 2>&1 &
#   tail -f /root/autodl-tmp/logs/temp08_block_suite_nohup.log
#
# 可调参数（带默认值，可用环境变量覆盖）：
#   MAX_NEW_TOKEN=1024 TOTAL_TOKEN=60 DEPTH=5 TOP_K=10 TEMP=0.8 \
#     bash scripts/run_temp08_block_experiments.sh
set -euo pipefail

cd /root/autodl-tmp/EAGLE-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagle

MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"
# 固定草稿树配置：与主实验一致 => 跨温度、跨配置可比
TOTAL_TOKEN="${TOTAL_TOKEN:-60}"
DEPTH="${DEPTH:-5}"
TOP_K="${TOP_K:-10}"
# 本次消融的采样温度
TEMP="${TEMP:-0.8}"
TREE_ARGS=(--total-token "${TOTAL_TOKEN}" --depth "${DEPTH}" --top-k "${TOP_K}")

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/autodl-tmp/results /root/autodl-tmp/logs
MANIFEST="/root/autodl-tmp/results/temp08_block_manifest_${TS}.txt"

# log 写 stderr（不能写 stdout）；run_* 用 stdout “返回”结果路径并被 $(...) 捕获。
log() { echo "[$(date '+%F %T')] $*" | tee -a "${MANIFEST}" >&2; }

run_vanilla() {
  local temp="$1"
  unset USE_BLOCK_VERIFY
  local out="/root/autodl-tmp/results/qwen2_vanilla_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/vanilla_t${temp}_${TS}.log"
  log "RUN Vanilla t=${temp} total_token=${TOTAL_TOKEN} -> ${out}"
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

run_block_vec() {
  local temp="$1"
  export USE_BLOCK_VERIFY=1
  local out="/root/autodl-tmp/results/qwen2_eagle2_block_vec_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/eagle2_block_vec_t${temp}_${TS}.log"
  log "RUN EAGLE-2+Block(vec) t=${temp} total_token=${TOTAL_TOKEN} -> ${out}"
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

log "======== Temperature ablation suite t=${TEMP} (fixed tree) ========"
log "MAX_NEW_TOKEN=${MAX_NEW_TOKEN} TOTAL_TOKEN=${TOTAL_TOKEN} DEPTH=${DEPTH} TOP_K=${TOP_K} TEMP=${TEMP}"

VAN="$(run_vanilla "${TEMP}")"
EA="$(run_eagle2 "${TEMP}")"
BLOCK="$(run_block_vec "${TEMP}")"

SPEED_EA="$(run_speed "${EA}" "${VAN}" "eagle2_t${TEMP}_vs_vanilla")"
SPEED_BLOCK="$(run_speed "${BLOCK}" "${VAN}" "block_vec_t${TEMP}_vs_vanilla")"

log "======== DONE ========"
log "VAN=${VAN}"
log "EA=${EA}"
log "BLOCK=${BLOCK}"
log "SPEED_EA=${SPEED_EA}"
log "SPEED_BLOCK=${SPEED_BLOCK}"
log "Manifest: ${MANIFEST}"
