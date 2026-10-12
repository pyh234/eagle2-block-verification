"""Locate where EAGLE-2 and vanilla greedy outputs diverge (token-level).

按道理EAGLE-2 和 Vanilla 在temperature=0下应该无损，但是实际上却发现输出结果不一致，
原因推测是qwen2使用的时候要求用bf16，这导致在碰上概率接近的token的时候由于调用算子不同以及计算精度
导致arg max的结果不同，从而导致输出结果不一致。
该脚本检查了前两次输出结果不一致的情况，给出一个大致统计结果，以此进行后续分析

Usage:
    python -m eagle.evaluation.diff_outputs \
        --ea-file   /root/autodl-tmp/results/qwen2_eagle2_t0.0_20260608_140503.jsonl \
        --base-file /root/autodl-tmp/results/qwen2_vanilla_t0.0_20260608_133513.jsonl \
        --tokenizer /root/autodl-tmp/models/Qwen2-7B-Instruct \
        --out-file  /root/autodl-tmp/results/diff_report.txt
"""
import argparse
import json

import numpy as np
from transformers import AutoTokenizer


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_by_qid(records):
    return {dp["question_id"]: dp for dp in records}


def first_divergence(ids_a, ids_b):
    """Return (k, total_common_len) where k is the first index at which the two
    token-id lists differ. k == n means identical up to the shorter length."""
    n = min(len(ids_a), len(ids_b))
    for k in range(n):
        if ids_a[k] != ids_b[k]:
            return k, n
    return n, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-file", type=str, required=True)
    parser.add_argument("--base-file", type=str, required=True)
    parser.add_argument("--tokenizer", type=str,
                        default="/root/autodl-tmp/models/Qwen2-7B-Instruct")
    parser.add_argument("--out-file", type=str,
                        default="results/diff_report.txt")
    parser.add_argument("--context", type=int, default=6,
                        help="num tokens of context shown around divergence")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    ea = index_by_qid(load_records(args.ea_file))
    base = index_by_qid(load_records(args.base_file))
    common_qids = sorted(set(ea) & set(base))

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 70)
    emit("EAGLE-2 vs Vanilla greedy output divergence report")
    emit(f"  ea-file   : {args.ea_file}")
    emit(f"  base-file : {args.base_file}")
    emit(f"  tokenizer : {args.tokenizer}")
    emit(f"  aligned questions: {len(common_qids)}")
    emit("=" * 70)
    emit("")

    per_turn = []  # dict per turn for stats + example selection

    for qid in common_qids:
        ea_turns = ea[qid]["choices"][0]["turns"]
        base_turns = base[qid]["choices"][0]["turns"]
        n_turns = min(len(ea_turns), len(base_turns))
        for t in range(n_turns):
            ea_ids = tok(ea_turns[t], add_special_tokens=False).input_ids
            base_ids = tok(base_turns[t], add_special_tokens=False).input_ids
            k, common_n = first_divergence(ea_ids, base_ids)
            identical = (k == common_n and len(ea_ids) == len(base_ids))
            frac = k / max(common_n, 1)
            ea_tok = tok.decode([ea_ids[k]]) if k < len(ea_ids) else "<END>"
            base_tok = tok.decode([base_ids[k]]) if k < len(base_ids) else "<END>"
            ctx = tok.decode(ea_ids[max(0, k - args.context):k])
            per_turn.append({
                "qid": qid,
                "turn": t,
                "k": k,
                "common_n": common_n,
                "ea_len": len(ea_ids),
                "base_len": len(base_ids),
                "frac": frac,
                "identical": identical,
                "ea_tok": ea_tok,
                "base_tok": base_tok,
                "ctx": ctx,
            })

    def summarize_subset(title, subset, note=""):
        n_total = len(subset)
        n_identical = sum(1 for x in subset if x["identical"])
        diverged = [x for x in subset if not x["identical"]]
        fracs = np.array([x["frac"] for x in diverged]) if diverged else np.array([])
        ks = np.array([x["k"] for x in diverged]) if diverged else np.array([])
        emit(f"---- {title} ----")
        if note:
            emit(f"  ({note})")
        emit(f"total turns            : {n_total}")
        emit(f"identical turns        : {n_identical} ({100*n_identical/max(n_total,1):.1f}%)")
        emit(f"diverged turns         : {len(diverged)} ({100*len(diverged)/max(n_total,1):.1f}%)")
        if len(diverged):
            emit(f"first-diverge pos k    : mean={ks.mean():.1f}  median={np.median(ks):.0f}  min={ks.min()}  max={ks.max()}")
            emit(f"common-prefix fraction : mean={fracs.mean():.3f}  median={np.median(fracs):.3f}")
        emit("")
        return diverged

    diverged = summarize_subset("Summary (all turns)", per_turn)
    summarize_subset(
        "Summary (turn0 only)",
        [x for x in per_turn if x["turn"] == 0],
        note="turn0 starts from the IDENTICAL prompt -> the clean lossless test")
    summarize_subset(
        "Summary (turn1 only)",
        [x for x in per_turn if x["turn"] == 1],
        note="turn1 inherits turn0's (already diverged) answer as context -> contaminated")
    fracs = np.array([x["frac"] for x in diverged]) if diverged else np.array([])

    # Histogram of common-prefix fraction (where divergence happens)
    if len(diverged):
        emit("---- First-divergence position (fraction of common length) ----")
        bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001]
        labels = ["[0.0,0.1)", "[0.1,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,0.9)", "[0.9,1.0]"]
        hist, _ = np.histogram(fracs, bins=bins)
        for lab, c in zip(labels, hist):
            bar = "#" * c
            emit(f"  {lab:>12} : {c:3d} {bar}")
        emit("")

    # Only turn0 examples give a clean same-prompt fork usable by the 9.3 probe.
    # Skip k==0 (no shared prefix to probe) and sort by earliest divergence.
    turn0_div = [x for x in diverged if x["turn"] == 0 and x["k"] > 0]
    diverged_sorted = sorted(turn0_div, key=lambda x: x["k"])
    emit("---- turn0 earliest-divergence examples (candidates for 9.3 logits probe) ----")
    emit("(both sides share the SAME prompt; context = shared prefix right before the fork)")
    emit("")
    for x in diverged_sorted[:15]:
        emit(f"q{x['qid']} turn{x['turn']}  k={x['k']}  frac={x['frac']:.3f}  "
             f"(ea_len={x['ea_len']}, base_len={x['base_len']})")
        emit(f"    prefix : ...{x['ctx']!r}")
        emit(f"    EAGLE  -> {x['ea_tok']!r}")
        emit(f"    vanilla-> {x['base_tok']!r}")
        emit("")

    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[saved] {args.out_file}")


if __name__ == "__main__":
    main()
