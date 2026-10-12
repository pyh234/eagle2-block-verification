"""Quantify Block Verification's lossiness at t>0 by comparing the realized text
outputs of vanilla / EAGLE-2 / Block(vectorized) on the same MT-bench questions.

Background
----------
At sampling temperature t>0, EAGLE-2 is (in exact arithmetic) distribution-
lossless w.r.t. target AR. Block Verification is NOT: whenever a draft block
equals the target's greedy top-1 continuation, it accepts the whole block and
skips per-token rejection sampling, biasing those spans toward the target's
greedy text. This script measures, pairwise, how much the three configs' token
streams differ, and pairs that with the accept-length gain to give a
"distribution shift vs accept-length gain" trade-off table.

Metrics (per aligned (question, turn), token-level, like diff_outputs.py):
  - first-divergence position k = length of shared token prefix
  - divergence rate = fraction of turns that are NOT token-identical
  - mean common-prefix length (tokens) and fraction, over diverged turns

IMPORTANT CAVEAT (read before interpreting)
-------------------------------------------
At t>0 the three configs consume INDEPENDENT RNG streams (vanilla samples
sequentially; EAGLE/Block draw randoms inside rejection sampling; Block's fast
path skips draws entirely). So a large share of token-level divergence comes
from different RNG draws, NOT from distribution shift. These text metrics are
therefore a COARSE proxy, not a clean distribution distance. The cleaner
lossiness signals are:
  (1) accept-length proximity to the t=0 greedy accept length (printed here), and
  (2) per-step greedy-match fraction (needs logits capture on GPU; not done here).

No GPU needed: this only re-tokenizes already-generated text on CPU.

Usage (t=0.5, runs on existing archived data):
    python -m eagle.evaluation.block_lossiness \
        --vanilla-file /root/autodl-tmp/peiyuhao/results/qwen2_vanilla_t0.5_20260613_145125.jsonl \
        --eagle2-file  /root/autodl-tmp/peiyuhao/results/qwen2_eagle2_t0.5_20260613_145125.jsonl \
        --block-file   /root/autodl-tmp/peiyuhao/results/qwen2_eagle2_block_vec_t0.5_20260614_010110.jsonl \
        --temp 0.5 \
        --out-file /root/autodl-tmp/results/block_lossiness_t0.5.txt
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
    """Return (k, common_n): k is the first index where the two id lists differ;
    k == common_n means identical up to the shorter length."""
    n = min(len(ids_a), len(ids_b))
    for k in range(n):
        if ids_a[k] != ids_b[k]:
            return k, n
    return n, n


def mean_accept_length(records):
    """new_tokens / (idx + 1), averaged over turns. Mirrors speed_qwen2.py.
    Vanilla AR is ~1.0 by construction."""
    ratios = []
    for dp in records:
        for choice in dp["choices"]:
            for tok, idx in zip(choice["new_tokens"], choice["idxs"]):
                steps = idx + 1
                if steps > 0:
                    ratios.append(tok / steps)
    return float(np.mean(ratios)) if ratios else float("nan")


def pairwise_divergence(tok, recs_a, recs_b):
    """Token-level divergence stats over aligned (qid, turn) using choices[0]."""
    a = index_by_qid(recs_a)
    b = index_by_qid(recs_b)
    qids = sorted(set(a) & set(b))

    n_total = 0
    n_identical = 0
    ks_div = []          # first-divergence position over DIVERGED turns
    fracs_div = []       # common-prefix fraction over DIVERGED turns
    for qid in qids:
        a_turns = a[qid]["choices"][0]["turns"]
        b_turns = b[qid]["choices"][0]["turns"]
        for t in range(min(len(a_turns), len(b_turns))):
            a_ids = tok(a_turns[t], add_special_tokens=False).input_ids
            b_ids = tok(b_turns[t], add_special_tokens=False).input_ids
            k, common_n = first_divergence(a_ids, b_ids)
            identical = (k == common_n and len(a_ids) == len(b_ids))
            n_total += 1
            if identical:
                n_identical += 1
            else:
                ks_div.append(k)
                fracs_div.append(k / max(common_n, 1))
    n_div = n_total - n_identical
    return {
        "turns": n_total,
        "identical": n_identical,
        "diverged": n_div,
        "divergence_rate": n_div / max(n_total, 1),
        "mean_prefix_tokens": float(np.mean(ks_div)) if ks_div else float("nan"),
        "median_prefix_tokens": float(np.median(ks_div)) if ks_div else float("nan"),
        "mean_prefix_frac": float(np.mean(fracs_div)) if fracs_div else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-file", required=True)
    parser.add_argument("--eagle2-file", required=True)
    parser.add_argument("--block-file", required=True)
    parser.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen2-7B-Instruct")
    parser.add_argument("--temp", default="?", help="temperature label for the header")
    parser.add_argument("--out-file", default="/root/autodl-tmp/results/block_lossiness.txt")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    van = load_records(args.vanilla_file)
    ea = load_records(args.eagle2_file)
    block = load_records(args.block_file)

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 78)
    emit(f"Block Verification lossiness analysis  (temperature t={args.temp}, fixed tree)")
    emit(f"  vanilla : {args.vanilla_file}")
    emit(f"  eagle2  : {args.eagle2_file}")
    emit(f"  block   : {args.block_file}")
    emit("=" * 78)
    emit("")
    emit("CAVEAT: at t>0, divergence is dominated by independent RNG streams across")
    emit("configs, NOT only distribution shift. Read text metrics as a coarse proxy;")
    emit("the accept-length column is the more interpretable lossiness signal.")
    emit("")

    # ---- accept length per config ----
    al_van = mean_accept_length(van)
    al_ea = mean_accept_length(ea)
    al_block = mean_accept_length(block)
    emit("---- mean accept length (tokens per target forward) ----")
    emit(f"  vanilla AR        : {al_van:.3f}")
    emit(f"  EAGLE-2           : {al_ea:.3f}")
    emit(f"  Block (vectorized): {al_block:.3f}")
    emit(f"  accept-length gain (Block - EAGLE-2): {al_block - al_ea:+.3f}")
    emit("")

    # ---- pairwise text divergence ----
    pairs = [
        ("EAGLE-2 vs vanilla", ea, van),
        ("Block   vs vanilla", block, van),
        ("Block   vs EAGLE-2", block, ea),
    ]
    emit("---- pairwise token-level divergence ----")
    emit(f"  {'pair':<20} {'turns':>5} {'divrate':>8} {'meanPfx(tok)':>13} {'medPfx':>7} {'meanPfxFrac':>12}")
    results = {}
    for name, ra, rb in pairs:
        st = pairwise_divergence(tok, ra, rb)
        results[name] = st
        emit(f"  {name:<20} {st['turns']:>5} {st['divergence_rate']*100:>7.1f}% "
             f"{st['mean_prefix_tokens']:>13.1f} {st['median_prefix_tokens']:>7.0f} "
             f"{st['mean_prefix_frac']:>12.3f}")
    emit("")

    # ---- trade-off framing ----
    bve = results["Block   vs EAGLE-2"]
    emit("---- trade-off: distribution shift vs accept-length gain ----")
    emit(f"  Block buys +{al_block - al_ea:.3f} accept length over EAGLE-2,")
    emit(f"  at the cost of diverging from EAGLE-2 on {bve['divergence_rate']*100:.1f}% of turns")
    emit(f"  (mean shared prefix {bve['mean_prefix_tokens']:.1f} tokens before they differ).")
    emit("  NOTE: part of that divergence is RNG-stream difference, not pure bias;")
    emit("  cross-check with the accept-length proximity to t=0 greedy (4.08).")
    emit("")

    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saved] {args.out_file}")


if __name__ == "__main__":
    main()
