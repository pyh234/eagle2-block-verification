"""Compute EAGLE-2 vs vanilla speedup from two answer .jsonl files.

Both files are produced by gen_ea_answer_qwen2.py / gen_baseline_answer_qwen2.py,
which record per-turn `new_tokens` (number of generated tokens) and `wall_time`.

Usage:
    python -m eagle.evaluation.speed_qwen2 \
        --ea-file mt_bench/qwen2-7b-eagle2-temperature-0.0.jsonl \
        --base-file mt_bench/qwen2-7b-vanilla-temperature-0.0.jsonl
"""
import argparse
import json

import numpy as np


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def per_turn_speeds(records):
    """tokens/s per turn = generated tokens / wall time."""
    speeds = []
    total_tokens = 0
    total_time = 0.0
    for dp in records:
        for choice in dp["choices"]:
            tokens = choice["new_tokens"]
            times = choice["wall_time"]
            for tok, t in zip(tokens, times):
                if t > 0:
                    speeds.append(tok / t)
                    total_tokens += tok
                    total_time += t
    return np.array(speeds), total_tokens, total_time


def mean_accept_length(records):
    """Average accepted length per forward = generated tokens / forward steps.

    For EAGLE, `idxs` stores the number of speculative iterations (forward
    passes of the target model). new_tokens / (idx + 1) approximates the mean
    number of tokens accepted per target forward. For the vanilla baseline this
    is ~1 by construction.
    """
    ratios = []
    for dp in records:
        for choice in dp["choices"]:
            for tok, idx in zip(choice["new_tokens"], choice["idxs"]):
                steps = idx + 1
                if steps > 0:
                    ratios.append(tok / steps)
    if not ratios:
        return float("nan")
    return float(np.mean(ratios))


def profile_summary(records):
    """Aggregate per-phase timing recorded by gen_ea_answer_qwen2.py --enable-profile.

    Returns avg ms/turn for each phase and each phase's fraction of the summed
    (prefill+target+verify+update) time. `verify_frac` is the share of time spent
    in evaluate_posterior -- i.e. the upper bound on any verify-rule speedup.
    """
    fields = {"prefill_ms": [], "target_ms": [], "verify_ms": [], "update_ms": [], "cycles": []}
    key_map = {
        "prefill_ms": "profile_prefill_ms",
        "target_ms": "profile_target_ms",
        "verify_ms": "profile_verify_ms",
        "update_ms": "profile_update_ms",
        "cycles": "profile_cycles",
    }
    for dp in records:
        for choice in dp["choices"]:
            for out_key, in_key in key_map.items():
                fields[out_key].extend(choice.get(in_key, []))
    if not fields["target_ms"]:
        return None
    total = sum(np.sum(fields[k]) for k in ["prefill_ms", "target_ms", "verify_ms", "update_ms"])
    summary = {k: float(np.mean(v)) if v else float("nan") for k, v in fields.items() if k != "cycles"}
    summary["mean_cycles"] = float(np.mean(fields["cycles"])) if fields["cycles"] else float("nan")
    if total > 0:
        for k in ["prefill_ms", "target_ms", "verify_ms", "update_ms"]:
            summary[k.replace("_ms", "_frac")] = float(np.sum(fields[k]) / total)
    return summary


def summarize(name, records):
    speeds, total_tokens, total_time = per_turn_speeds(records)
    mean_speed = speeds.mean() if len(speeds) else float("nan")
    overall_speed = total_tokens / total_time if total_time > 0 else float("nan")
    accept_len = mean_accept_length(records)
    print(f"[{name}]")
    print(f"  turns                 : {len(speeds)}")
    print(f"  total generated tokens: {total_tokens}")
    print(f"  total wall time (s)   : {total_time:.2f}")
    print(f"  mean speed (tokens/s) : {mean_speed:.2f}  (per-turn average)")
    print(f"  overall tokens/s      : {overall_speed:.2f}  (sum tokens / sum time)")
    print(f"  mean accept length    : {accept_len:.2f}  (tokens per target forward)")
    prof = profile_summary(records)
    if prof is not None:
        print(f"  profile prefill ms    : {prof['prefill_ms']:.2f} avg/turn")
        print(f"  profile target ms     : {prof['target_ms']:.2f} avg/turn ({prof.get('target_frac', float('nan')):.1%})")
        print(f"  profile verify ms     : {prof['verify_ms']:.2f} avg/turn ({prof.get('verify_frac', float('nan')):.1%})")
        print(f"  profile update ms     : {prof['update_ms']:.2f} avg/turn ({prof.get('update_frac', float('nan')):.1%})")
        print(f"  profile cycles/turn   : {prof['mean_cycles']:.2f}")
    return mean_speed, overall_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-file", type=str, required=True,
                        help="EAGLE-2 answer jsonl.")
    parser.add_argument("--base-file", type=str, required=True,
                        help="Vanilla AR answer jsonl.")
    args = parser.parse_args()

    ea_records = load_records(args.ea_file)
    base_records = load_records(args.base_file)

    ea_mean, ea_overall = summarize("EAGLE-2", ea_records)
    print()
    base_mean, base_overall = summarize("Vanilla AR", base_records)
    print()

    print("=" * 40)
    print(f"Speedup (per-turn mean) : {ea_mean / base_mean:.2f}x")
    print(f"Speedup (overall)       : {ea_overall / base_overall:.2f}x")
    print("=" * 40)


if __name__ == "__main__":
    main()
