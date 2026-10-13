#!/usr/bin/env bash
# Vectorized Block Verification validation suite.
#
# This script assumes the no-block / vanilla fixed-tree baselines from
# BASE_TS already exist. It only reruns EAGLE-2 + Block with the current
# utils.py implementation, then writes speed/profile summaries to files.
#
# Runs:
#   1) Block t=0.0             -> sanity; should match EAGLE-2 t=0 behavior.
#   2) Block t=TEMP_SAMPLE     -> real throughput after vectorizing block verify.
#   3) Block t=TEMP_SAMPLE + profiling -> verify/update/target phase breakdown.
#
# Background usage:
#   cd /root/autodl-tmp/EAGLE-main
#   nohup bash scripts/run_vectorized_block_experiments.sh > /root/autodl-tmp/logs/vectorized_block_suite_nohup.log 2>&1 &
#   tail -f /root/autodl-tmp/logs/vectorized_block_suite_nohup.log
#
# Optional overrides:
#   MAX_NEW_TOKEN=1024 TOTAL_TOKEN=60 DEPTH=5 TOP_K=10 TEMP_SAMPLE=0.5 BASE_TS=20260613_145125 \
#     bash scripts/run_vectorized_block_experiments.sh
set -euo pipefail

cd /root/autodl-tmp/EAGLE-main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate eagle

MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-1024}"
TOTAL_TOKEN="${TOTAL_TOKEN:-60}"
DEPTH="${DEPTH:-5}"
TOP_K="${TOP_K:-10}"
TEMP_SAMPLE="${TEMP_SAMPLE:-0.5}"
BASE_TS="${BASE_TS:-20260613_145125}"
TREE_ARGS=(--total-token "${TOTAL_TOKEN}" --depth "${DEPTH}" --top-k "${TOP_K}")

VAN_T0="/root/autodl-tmp/results/qwen2_vanilla_t0.0_${BASE_TS}.jsonl"
VAN_TS="/root/autodl-tmp/results/qwen2_vanilla_t${TEMP_SAMPLE}_${BASE_TS}.jsonl"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p /root/autodl-tmp/results /root/autodl-tmp/logs
MANIFEST="/root/autodl-tmp/results/vectorized_block_manifest_${TS}.txt"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${MANIFEST}" >&2; }

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log "ERROR missing required file: ${path}"
    exit 1
  fi
}

run_block() {
  local temp="$1" tag="$2"
  export USE_BLOCK_VERIFY=1
  local out="/root/autodl-tmp/results/qwen2_eagle2_block_vec_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/eagle2_block_vec_t${temp}_${TS}.log"
  log "RUN vectorized EAGLE-2+Block t=${temp} total_token=${TOTAL_TOKEN} -> ${out}"
  {
    echo "USE_BLOCK_VERIFY=1 TAG=${tag} TEMPERATURE=${temp} ${TREE_ARGS[*]}"
    python -m eagle.evaluation.gen_ea_answer_qwen2 \
      --temperature "${temp}" \
      --max-new-token "${MAX_NEW_TOKEN}" \
      "${TREE_ARGS[@]}" \
      --answer-file "${out}"
  } >> "${lg}" 2>&1
  echo "${out}"
}

run_block_profile() {
  local temp="$1"
  export USE_BLOCK_VERIFY=1
  local out="/root/autodl-tmp/results/prof_block_vec_t${temp}_${TS}.jsonl"
  local lg="/root/autodl-tmp/logs/prof_block_vec_t${temp}_${TS}.log"
  log "RUN [profile] vectorized EAGLE-2+Block t=${temp} total_token=${TOTAL_TOKEN} -> ${out}"
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

log "======== Vectorized Block Verification suite (fixed tree) ========"
log "MAX_NEW_TOKEN=${MAX_NEW_TOKEN} TOTAL_TOKEN=${TOTAL_TOKEN} DEPTH=${DEPTH} TOP_K=${TOP_K} TEMP_SAMPLE=${TEMP_SAMPLE} BASE_TS=${BASE_TS}"
require_file "${VAN_T0}"
require_file "${VAN_TS}"

BLOCK_T0="$(run_block 0.0 "sanity")"
BLOCK_TS="$(run_block "${TEMP_SAMPLE}" "sample")"
PROF_BLOCK_TS="$(run_block_profile "${TEMP_SAMPLE}")"

SPEED_BLOCK_T0="$(run_speed "${BLOCK_T0}" "${VAN_T0}" "block_vec_t0.0_vs_vanilla")"
SPEED_BLOCK_TS="$(run_speed "${BLOCK_TS}" "${VAN_TS}" "block_vec_t${TEMP_SAMPLE}_vs_vanilla")"
SPEED_PROF_BLOCK_TS="$(run_speed "${PROF_BLOCK_TS}" "${VAN_TS}" "prof_block_vec_t${TEMP_SAMPLE}_vs_vanilla")"

log "======== DONE ========"
log "BLOCK_T0=${BLOCK_T0}"
log "BLOCK_TS=${BLOCK_TS}"
log "PROF_BLOCK_TS=${PROF_BLOCK_TS}"
log "SPEED_BLOCK_T0=${SPEED_BLOCK_T0}"
log "SPEED_BLOCK_TS=${SPEED_BLOCK_TS}"
log "SPEED_PROF_BLOCK_TS=${SPEED_PROF_BLOCK_TS}"
log "Manifest: ${MANIFEST}"
