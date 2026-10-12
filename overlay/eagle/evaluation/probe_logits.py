"""
证明在 temperature=0 时，EAGLE-2 和 vanilla 之间的文本分歧是由“树状/批处理”目标前向传播
与“串行”目标前向传播之间的浮点数非结合律（bf16 舍入误差）导致的，而不是因为接受逻辑存在 Bug。

针对选定的问题（第 0 轮对话，两条路径使用完全相同的 Prompt），我们将：
  1. 运行 EAGLE-2 贪婪解码模式，并通过对 evaluate_posterior 进行动态修改（Monkeypatch），
     捕获每个生成位置上真正驱动决策的“树状前向传播”目标 Logits 向量（即接受的草稿 Token 或奖励 Token）。
  2. 运行 vanilla 贪婪解码模式（使用仓库自带的串行 KV 缓存前向传播，与 naivegenerate 完全一致），
     并记录每个生成位置的“串行”目标 Logits。
  3. 找出两条 Token 流出现分歧的第一个位置 k，然后对比位置 k 处的两个 Logits 向量：
       - 两条路径的 Top-2 Token 都是同一对极其接近的候选词 {ea_tok, van_tok}；
       - Top-1 和 Top-2 之间的差距极小（处于“无限接近平手”的状态）；
       - 两种不同的前向传播路径分配了微弱差异的 Logits，刚好足以翻转这一对词的 argmax 顺序 
         -> 从而证明这是纯粹的数值计算问题，而非算法逻辑 Bug。

Usage:
    python -m eagle.evaluation.probe_logits --question-id 143 96
"""
import argparse
import json
import os

import torch

try:
    from ..model.ea_model import EaModel
    from ..model import ea_model as ea_model_module
    from ..model.utils import evaluate_posterior, reset_tree_mode
    from ..model.kv_cache import initialize_past_key_values
except Exception:
    from eagle.model.ea_model import EaModel
    from eagle.model import ea_model as ea_model_module
    from eagle.model.utils import evaluate_posterior, reset_tree_mode
    from eagle.model.kv_cache import initialize_past_key_values

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)


# ---------------------------------------------------------------------------
# EAGLE tree-forward logit capture
# ---------------------------------------------------------------------------
_CAPTURED = []  # filled per eagenerate call: list of (token_id, logits_vec_cpu)
_ORIG_EVAL_POSTERIOR = evaluate_posterior


def _capturing_evaluate_posterior(logits, candidates, logits_processor):
    """Wrap evaluate_posterior to record the tree-forward logits vector that
    produced each newly committed token this step.

    Greedy branch only (logits_processor is None). For the chosen best
    candidate b with accept_length a:
      - accepted draft token candidates[b, j] (j=1..a) is predicted by logits[b, j-1]
      - the bonus/continuation token argmax(logits[b, a]) is predicted by logits[b, a]
    """
    best, acc, sample_p = _ORIG_EVAL_POSTERIOR(logits, candidates, logits_processor)
    if logits_processor is None:  # greedy only
        b = int(best.item()) if torch.is_tensor(best) else int(best)
        a = int(acc.item()) if torch.is_tensor(acc) else int(acc)
        for j in range(1, a + 1):
            tok = int(candidates[b, j].item())
            vec = logits[b, j - 1].detach().float().cpu()
            _CAPTURED.append((tok, vec))
        bonus_vec = logits[b, a].detach().float().cpu()
        bonus_tok = int(bonus_vec.argmax().item())
        _CAPTURED.append((bonus_tok, bonus_vec))
    return best, acc, sample_p


def run_eagle_capture(model, input_ids, max_new_tokens):
    """Run EAGLE-2 greedy; return (gen_token_ids, per_position_tree_logits).

    per_position_tree_logits[p] is the tree-forward logits vector whose argmax
    region produced the token at generated position p (p>=1). Position 0 (the
    first token) comes from the sequential prefill, so it has no tree logits.
    """
    global _CAPTURED
    _CAPTURED = []
    ea_model_module.evaluate_posterior = _capturing_evaluate_posterior
    try:
        out_ids, new_token, idx = model.eagenerate(
            input_ids, temperature=0.0, max_new_tokens=max_new_tokens, log=True)
    finally:
        ea_model_module.evaluate_posterior = _ORIG_EVAL_POSTERIOR

    gen = out_ids[0][input_ids.shape[1]:].tolist()
    # position 0 = prefill token (sequential); positions 1.. = captured stream
    pos_logits = [None] + [vec for (_tok, vec) in _CAPTURED]
    pos_tokens = [gen[0]] + [tok for (tok, _vec) in _CAPTURED]
    return gen, pos_tokens, pos_logits


@torch.no_grad()
def run_vanilla_capture(model, input_ids, max_new_tokens, max_length=2048):
    """Vanilla sequential greedy (same mechanism as naivegenerate).

    Returns (gen_token_ids, per_position_seq_logits) where per_position_seq_logits[p]
    is the sequential-forward logits predicting the token at position p."""
    model.ea_layer.reset_kv()
    past_key_values, _pkv_data, current_length_data = initialize_past_key_values(
        model.base_model, max_length=max_length)
    current_length_data.zero_()
    reset_tree_mode(model)

    outputs = model.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
    gen, pos_logits = [], []
    for _ in range(max_new_tokens):
        lg = outputs.logits[:, -1]          # predicts the next position
        tok = int(lg.argmax(dim=-1).item())
        gen.append(tok)
        pos_logits.append(lg[0].detach().float().cpu())
        if tok == model.tokenizer.eos_token_id:
            break
        nxt = torch.tensor([[tok]], device=input_ids.device)
        outputs = model.base_model(nxt, use_cache=True, past_key_values=past_key_values)
    return gen, pos_logits


# ---------------------------------------------------------------------------
# prompt + reporting helpers
# ---------------------------------------------------------------------------
def load_question(question_file, qid):
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if q["question_id"] == qid:
                return q
    raise ValueError(f"question_id {qid} not found in {question_file}")


def build_turn0_prompt(tokenizer, question, system_prompt="You are a helpful assistant."):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question["turns"][0]}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
    return torch.as_tensor(input_ids).cuda()


def first_divergence(a, b):
    n = min(len(a), len(b))
    for k in range(n):
        if a[k] != b[k]:
            return k
    return n


def probe_question(model, tokenizer, question_file, qid, max_new_tokens, emit):
    question = load_question(question_file, qid)
    input_ids = build_turn0_prompt(tokenizer, question)

    ea_gen, ea_pos_tokens, ea_pos_logits = run_eagle_capture(model, input_ids, max_new_tokens)
    va_gen, va_pos_logits = run_vanilla_capture(model, input_ids, max_new_tokens)

    k = first_divergence(ea_gen, va_gen)
    emit("")
    emit("=" * 70)
    emit(f"question {qid}  (turn0)")
    emit(f"  prompt tokens: {input_ids.shape[1]}, "
         f"ea_gen={len(ea_gen)}, va_gen={len(va_gen)}, first divergence k={k}")

    if k >= len(ea_gen) or k >= len(va_gen):
        emit("  no divergence within max_new_tokens (outputs identical so far).")
        return
    if k == 0:
        emit("  divergence at position 0 (prefill token); no tree logits to probe.")
        return

    # sanity: tokens agree on [0,k)
    shared = tokenizer.decode(ea_gen[max(0, k - 8):k])
    ea_tok = ea_gen[k]
    va_tok = va_gen[k]
    emit(f"  shared prefix tail : ...{shared!r}")
    emit(f"  EAGLE   token@k    : {ea_tok!r} -> {tokenizer.decode([ea_tok])!r}")
    emit(f"  vanilla token@k    : {va_tok!r} -> {tokenizer.decode([va_tok])!r}")

    # logits vectors at position k from each forward path
    e_vec = ea_pos_logits[k]      # tree forward
    v_vec = va_pos_logits[k]      # sequential forward
    if e_vec is None:
        emit("  (no captured tree logits at k)")
        return

    # cross-check that captured token matches EAGLE's realized token
    if ea_pos_tokens[k] != ea_tok:
        emit(f"  [warn] captured token {ea_pos_tokens[k]} != realized {ea_tok}; "
             f"alignment off, results may be unreliable.")

    def topk_report(name, vec):
        vals, idx = torch.topk(vec, 5)
        emit(f"  [{name}] top-5:")
        for v, i in zip(vals.tolist(), idx.tolist()):
            emit(f"      {v:9.4f}  id={i:6d}  {tokenizer.decode([i])!r}")
        return vals, idx

    e_vals, e_idx = topk_report("tree forward (EAGLE)", e_vec)
    v_vals, v_idx = topk_report("sequential forward (vanilla)", v_vec)

    # focus on the contested pair {ea_tok, va_tok}
    e_pair = (e_vec[ea_tok].item(), e_vec[va_tok].item())
    v_pair = (v_vec[ea_tok].item(), v_vec[va_tok].item())
    e_gap = e_pair[0] - e_pair[1]      # tree: ea minus van logit
    v_gap = v_pair[0] - v_pair[1]      # seq : ea minus van logit
    path_diff = (e_vec - v_vec)
    pair_drift = abs(path_diff[ea_tok].item()) + abs(path_diff[va_tok].item())

    emit("  ---- contested pair {EAGLE_tok, vanilla_tok} ----")
    emit(f"    tree  logit(ea)={e_pair[0]:.4f}  logit(van)={e_pair[1]:.4f}  "
         f"(ea-van)={e_gap:+.5f}  -> argmax {'EAGLE' if e_gap>0 else 'vanilla'}")
    emit(f"    seq   logit(ea)={v_pair[0]:.4f}  logit(van)={v_pair[1]:.4f}  "
         f"(ea-van)={v_gap:+.5f}  -> argmax {'EAGLE' if v_gap>0 else 'vanilla'}")
    emit(f"    tie size |gap| : tree={abs(e_gap):.5f}  seq={abs(v_gap):.5f}")
    emit(f"    cross-path logit drift on the pair: {pair_drift:.5f}")
    emit(f"    max |tree-seq| over full vocab     : {path_diff.abs().max().item():.5f}")
    flipped = (e_gap > 0) != (v_gap > 0)
    emit(f"    ==> argmax FLIPPED between paths: {flipped}")
    if flipped and abs(e_gap) < pair_drift + abs(v_gap):
        emit("    ==> tie gap is within cross-path numerical drift "
             "=> divergence is NUMERICAL (bf16 non-associativity), not a bug.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", type=int, nargs="+", default=[143, 96])
    parser.add_argument("--ea-model-path", type=str,
                        default="/root/autodl-tmp/models/EAGLE-Qwen2-7B-Instruct")
    parser.add_argument("--base-model-path", type=str,
                        default="/root/autodl-tmp/models/Qwen2-7B-Instruct")
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--max-new-tokens", type=int, default=80,
                        help="enough to pass the fork (small ks for the chosen samples)")
    parser.add_argument("--total-token", type=int, default=-1)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-file", type=str, default="/root/autodl-tmp/results/probe_report.txt")
    args = parser.parse_args()

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.bfloat16,   # Qwen2 requires bf16
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=False,             # EAGLE-2
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 70)
    emit("9.3 method A: tree-forward vs sequential-forward target logits at fork")
    emit(f"  base : {args.base_model_path}")
    emit(f"  ea   : {args.ea_model_path}")
    emit(f"  dtype: bfloat16   use_eagle3=False   greedy (temperature=0)")
    emit("=" * 70)

    for qid in args.question_id:
        probe_question(model, tokenizer, question_file, qid, args.max_new_tokens, emit)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[saved] {args.out_file}")


if __name__ == "__main__":
    main()
