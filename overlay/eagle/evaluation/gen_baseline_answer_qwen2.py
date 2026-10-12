"""Generate answers with vanilla auto-regressive decoding on a Qwen2 target model.

This is the non-speculative baseline (no draft tree, one token per forward) used
to measure the reference tokens/s for computing EAGLE-2 speedup. It uses the same
model loading path and prompt formatting as gen_ea_answer_qwen2.py so the
comparison is apples-to-apples.

Note: Qwen2 must run in bfloat16 to avoid numerical overflow (see EAGLE README).
"""
import argparse
import json
import os

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

try:
    from ..model.ea_model import EaModel
    from ..model.utils import *
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.utils import *


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args,
):
    questions = load_questions(question_file, question_begin, question_end)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args,
            )
        )

    if use_ray:
        import ray
        ray.get(ans_handles)


def build_prompt(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
    return input_ids


def clean_output(tokenizer, output_ids, stop_token_ids):
    stop_index = [i for i, tid in enumerate(output_ids) if tid in stop_token_ids]
    if len(stop_index) > 0:
        output_ids = output_ids[: stop_index[0]]

    output = tokenizer.decode(
        output_ids,
        spaces_between_special_tokens=False,
    )
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                output = output.replace(special_tok, "")
        else:
            output = output.replace(special_token, "")
    return output.strip()


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args,
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.bfloat16,  # Qwen2 requires bf16
        low_cpu_mem_usage=True,
        device_map="auto",
        use_eagle3=False,
    )

    tokenizer = model.get_tokenizer()
    model.eval()
    print('Check model training state:', model.training)
    print('CUDA VISIBLE DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))

    stop_token_ids = [tokenizer.eos_token_id]
    extra_stop = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(extra_stop, int) and extra_stop >= 0:
        stop_token_ids.append(extra_stop)
    stop_token_ids = list(set(stop_token_ids))

    system_prompt = "You are a helpful assistant."

    question = questions[0]
    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        messages = [{"role": "system", "content": system_prompt}]
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({"role": "user", "content": qs})
            input_ids = build_prompt(tokenizer, messages)

            torch.cuda.synchronize()
            output_ids, new_token, idx = model.naivegenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                max_new_tokens=max_new_token,
                log=True,
            )
            torch.cuda.synchronize()
            output_ids = output_ids[0][len(input_ids[0]):]
            output = clean_output(tokenizer, output_ids, stop_token_ids)
            messages.append({"role": "assistant", "content": output})
    print('Warmup done')

    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [{"role": "system", "content": system_prompt}]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                input_ids = build_prompt(tokenizer, messages)

                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx = model.naivegenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    max_new_tokens=max_new_token,
                    log=True,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time

                output_ids = output_ids[0][len(input_ids[0]):]
                output = clean_output(tokenizer, output_ids, stop_token_ids)

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({"role": "assistant", "content": output})
            choices.append({
                "index": i,
                "turns": turns,
                "idxs": idxs,
                "new_tokens": new_tokens,
                "wall_time": wall_time,
            })

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplicate."""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/root/autodl-tmp/models/EAGLE-Qwen2-7B-Instruct",
        help="EAGLE weights path (loaded but unused for vanilla decoding).",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/root/autodl-tmp/models/Qwen2-7B-Instruct",
        help="The path to the Qwen2 target model.",
    )
    parser.add_argument("--model-id", type=str, default="qwen2-7b-vanilla")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=-1)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}={v}")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args,
    )

    reorg_answer_file(answer_file)
