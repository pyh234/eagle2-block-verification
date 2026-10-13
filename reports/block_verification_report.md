# Block Verification 实验报告

作者：裴宇皓  
项目：机器学习系统大作业 —— 复现并改进 EAGLE-2 投机解码（speculative decoding）  
代码仓库：`/root/autodl-tmp/EAGLE-main`，工作分支 `verify-block`  
本人负责的优化点：**Block Verification（前缀块验证）**

> 说明：本报告中凡涉及**源码与脚本**，一律使用其在代码仓库中的相对路径（如 `eagle/model/utils.py`、`scripts/run_all_block_experiments.sh`），方便后续合并；凡涉及**实验结果/日志**，使用归档后的 `peiyuhao/` 路径，方便复查。

---

## 1. 实验设置与依据

所有实验都在如下统一配置下进行：


| 项             | 取值                                                | 依据 / 原因                                                                                                                                          |
| ------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| target model  | `/root/autodl-tmp/models/Qwen2-7B-Instruct`       | 课程指定的复现对象之一                                                                                                                                      |
| drafter model | `/root/autodl-tmp/models/EAGLE-Qwen2-7B-Instruct` | EAGLE 官方发布的 Qwen2-7B 配套草稿头，EAGLE-2 复用该权重、无需重训                                                                                                    |
| 精度            | `bfloat16`                                        | EAGLE 官方 README 明确要求：当 target 为 Qwen2 时必须用 bf16 而非 fp16，否则会数值溢出（numerical overflow）                                                              |
| 模式            | `use_eagle3=False`                                | 当前代码默认走 EAGLE-3；README 指出，要使用 EAGLE（含 EAGLE-2）权重必须在 `EaModel.from_pretrained` 中显式设 `use_eagle3=False`，否则会按 EAGLE-3 的特征定义加载，与本次复用的 EAGLE-2 草稿头不匹配 |
| 数据集           | MT-bench，80 题，每题 2 轮，共 160 turns                  | EAGLE 仓库自带评测集 `eagle/data/mt_bench/question.jsonl`                                                                                               |
| 草稿树           | `--total-token 60 --depth 5 --top-k 10`（固定）       | 见下方说明                                                                                                                                            |
| 最大生成长度        | `--max-new-token 1024`                            | 与 baseline 对齐                                                                                                                                    |


### 1.1 评测脚本：

原始 EAGLE 仓库只提供了 vicuna / llama2 / llama3 / mixtral / deepseek / qwen3 的评测脚本，**没有 Qwen2 版本**。本次复现的 target 是 Qwen2-7B-Instruct，以下为本次实验 Qwen2 评测与分析脚本（含 vanilla AR baseline），统一位于 `eagle/evaluation/`：


| 脚本                                  | 角色                          | 关键点                                                                                    |
| ----------------------------------- | --------------------------- | -------------------------------------------------------------------------------------- |
| `gen_baseline_answer_qwen2.py`      | vanilla AR baseline（加速比的分母） | 走 `model.naivegenerate`（无草稿树、逐 token 前向），与 EAGLE-2 脚本共用同一套加载 / prompt / 计时             |
| `gen_ea_answer_qwen2.py`            | EAGLE-2 / Block 生成          | 走 `model.eagenerate`；并加入了 `--enable-profile`（profiling，见 §5）                           |
| `speed_qwen2.py`                    | 指标计算                        | 由两侧 jsonl 算 mean speed / overall tokens/s / mean accept length / speedup，并汇总 profiling |
| `diff_outputs.py`、`probe_logits.py` | t=0 分叉分析                    | 见 §3                                                                                   |


为保证 **vanilla** 与 EAGLE-2 脚本对齐：

- 同样用 `EaModel.from_pretrained` 加载（`bfloat16`、`use_eagle3=False`、同一棵固定树参数）——vanilla 虽不用草稿树，仍走同一加载方式；
- 同一 Qwen2 chat template + system prompt（`"You are a helpful assistant."`）+ 停止符（`eos` 与 `<|im_end|>`）；
- 同样先跑 3 次预热再计时，每个 turn 用 `torch.cuda.synchronize()` 包裹 `time.time()` 取墙钟。

报告中几个口径的定义（实现见 `speed_qwen2.py`）：

- **mean accept length** `= new_tokens / (idx + 1)`，即每次 target 前向平均接受的 token 数；vanilla 恒约为 1，EAGLE / Block 实验中则越大越好；
- **overall tokens/s** `= Σtokens / Σtime`（整体吞吐），**mean speed** 是各 turn tokens/s 的算术平均；
- **speedup** `= 修改后吞吐 ÷ 修改前吞吐`。

实验运行：Block 的开关是环境变量 `USE_BLOCK_VERIFY`（在 `eagle/model/utils.py` 中读取）；主消融可以由 `scripts/run_all_block_experiments.sh` 串行跑 vanilla / EAGLE-2 / Block ×｛t=0, t=0.5｝，并对同温度配对算 speed（全部脚本与文件清单见 `peiyuhao/README.md`）。

### 1.2 EAGLE-2 baseline 设置（固定草稿树参数）

EAGLE-2 在 `total_token = -1` 时会在加载阶段做一次**计时自动标定**来选草稿树大小。该逻辑在 `eagle/model/ea_model.py`：

```python
if total_token == -1:
    device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
    cans = [40, 48, 50, 56, 60]
    x = [1, 1.05, 1.07, 1.1, 1.13]
    times = []
    for i in range(len(cans)):
        length = cans[i]
        input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(20):
            ...
            outputs = model.base_model(input_ids)
            ...
        end_time = time.time()
        times.append((end_time - start_time) / x[i])
    total_token = cans[times.index(min(times))]
    model.ea_layer.total_tokens = total_token - 1
```

它用 GPU 实测耗时从 `[40, 48, 50, 56, 60]` 里挑一个最快的树大小。问题在于：这个计时受运行时 GPU 状态影响，**不同次运行可能选出不同的树大小**，从而让接受长度、加速比在运行间漂移、互相不可比。

因为本次工作要做的是“同一棵树下，不同 verify 规则之间的对照消融”，所以我在所有生成中固定 `--total-token 60 --depth 5 --top-k 10`，跳过自动标定。三个生成脚本都经 `EaModel.from_pretrained(total_token=…, depth=…, top_k=…)` 接收同一组树参数（连不使用草稿树的 vanilla 也走同一加载路径），这样 vanilla / EAGLE-2 / Block 各配置严格共享同一棵草稿树，结果可比、可复现。

---

## 2. 已完成工作概览


| 阶段              | 内容                                                                                    | 产出                                        |
| --------------- | ------------------------------------------------------------------------------------- | ----------------------------------------- |
| 环境与 baseline    | 配置 `eagle` 环境，自建 Qwen2 评测 harness（含 vanilla AR baseline），跑通 vanilla AR 与 EAGLE-2（固定树） | `20260613_145125` 批次结果                    |
| t=0 输出差异诊断      | 分析贪心下 vanilla 与 EAGLE-2 输出为何不逐字一致                                                     | `diff_outputs.py` / `probe_logits.py` 及报告 |
| Block v1        | 在采样分支实现前缀块验证，并修复 per-path logits bug                                                  | 提交 `412c663`                              |
| Profiling       | 给主循环加 opt-in 分段计时，定位瓶颈                                                                | `20260613_235811` 批次                      |
| Block v2（向量化优化） | 去掉 per-path Python 循环与 `.item()` 同步                                                   | 提交 `0f6a5a9`，`20260614_010110` 批次         |
| 温度消融            | 固定树补 `t=0.8` 的 vanilla / EAGLE-2 / Block 向量化，看高温下接受长度收益与分布偏移变化                        | `20260616_102407` 批次结果                    |
| 有损性 trade-off   | 对照 t=0.5 / t=0.8 的接受长度收益与文本分叉，量化「收益 vs 分布偏移」                                          | `block_lossiness_t0.5/0.8.txt`            |


最终主结果（固定树、t=0.5、相对同批 vanilla）：


| 设置               | overall speedup | mean accept length |
| ---------------- | --------------- | ------------------ |
| EAGLE-2 baseline | 1.82x           | 3.95               |
| Block v1         | 1.83x           | 4.07               |
| **Block v2**     | **2.05x**       | **4.11**           |


---

## 3. t=0 EAGLE-2 与 vanilla 输出差异诊断

### 3.1 现象与问题

理论上（在精确算术假设下） EAGLE 系列投机解码的输出分布与 target 自回归一致。但在贪心（t=0）下实测发现：EAGLE-2 与 vanilla 的输出并非逐字一致。为确认这不是评测脚本或 verify 逻辑的 bug，我做了两层分析。

- 粗粒度统计脚本：`eagle/evaluation/diff_outputs.py`
- 细粒度 logits 探针脚本：`eagle/evaluation/probe_logits.py`
- 对应结果（归档）：`peiyuhao/analysis/diff_report.txt`、`peiyuhao/analysis/probe_report.txt`

### 3.2 粗粒度：分叉位置统计

`diff_outputs.py` 按 `question_id` 对齐两侧输出，逐 token 找第一个不同的位置 `k`，统计如下：


| 范围       | identical turns  | diverged turns    | first-diverge 位置均值 | 公共前缀占比均值 |
| -------- | ---------------- | ----------------- | ------------------ | -------- |
| 全部 turns | 29 / 160 (18.1%) | 131 / 160 (81.9%) | 72.6 tokens        | 0.247    |
| 仅 turn0  | 20 / 80 (25.0%)  | 60 / 80 (75.0%)   | 107.0 tokens       | 0.326    |
| 仅 turn1  | 9 / 80 (11.2%)   | 71 / 80 (88.8%)   | 43.6 tokens        | 0.181    |


注意 turn0 用的是完全相同的 prompt，是判断 greedy 路径是否分叉的干净对照；turn1 会把 turn0 已分叉的回答当作上下文，统计会被级联差异污染，所以 turn1 的分叉率更高、分叉位置更靠前。

### 3.3 细粒度：分叉点的 target logits 探针

`probe_logits.py` 用同一条 turn0 prompt 跑两遍：一遍是 EAGLE-2 贪心生成，记录真正驱动每个已接受 token 的 **tree-forward（树形/批处理）target logits**；另一遍是 vanilla 的 **sequential（逐 token KV-cache）forward**，记录每个生成位置的 logits。然后在两条 token 流第一个分叉位置 `k`，对比两条 forward 路径在该位置的 top-k logits 以及争议候选对的间隙。

挑了两个分叉早、分叉词是近义词的样例（均为 turn 0 阶段）：

#### 样例 1：q143，`Photosynthesis is a ___`


| forward 路径          | top-1                 | top-2             | (ea−van) |
| ------------------- | --------------------- | ----------------- | -------- |
| tree（EAGLE）         | `fundamental` 19.3750 | `complex` 19.2500 | +0.12500 |
| sequential（vanilla） | `fundamental` 19.2500 | `complex` 19.2500 | +0.00000 |


#### 样例 2：q96，`... It helps ___`


| forward 路径          | top-1               | top-2               | (ea−van) |
| ------------------- | ------------------- | ------------------- | -------- |
| tree（EAGLE）         | `computers` 26.0000 | `the` 25.8750       | +0.12500 |
| sequential（vanilla） | `the` 26.0000       | `computers` 26.0000 | +0.00000 |


两个样例都呈现同一种情况：分叉处争议的两个候选词在 logits 上几乎平手（vanilla 的 sequential 路径上两者甚至完全相等），而两条 forward 路径在这两个词上的差异只有约 0.125。也就是说，分叉点本身就处在"谁排第一都差不多"的临界状态，两种 forward 计算带来的微小的数值差异，就足以把第一名从一个词翻转成另一个词。一旦第一个词被翻转，后续生成就沿着不同的上下文级联放大，最终输出整段不同。

### 3.4 结论

t=0 下 vanilla 与 EAGLE-2 的分叉，主要发生在 top-1/top-2 logits 极接近的位置。Qwen2 用 bf16，EAGLE-2 的树形 batched forward 与 vanilla 的逐 token forward 计算路径不同，浮点运算的非结合性 + bf16 舍入误差足以在平手位翻转 argmax。**这是工程数值差异，不是 verify 逻辑的 bug。** 这一点也提醒：后面分析「Block 的有损性」时，要把这种 t=0 数值分叉，与 Block 在 t>0 采样分支主动引入的分布偏移区分开。

---

## 4. Block Verification 方法与实现

### 4.1 原理

实现位置：`eagle/model/utils.py` 的 `evaluate_posterior()` **采样分支**（`logits_processor is not None`）。开关为环境变量 `USE_BLOCK_VERIFY=1`；t=0 走贪心分支，不受影响。

EAGLE 原始的采样分支是逐 token 的 modified rejection sampling：对每个位置算 `min(1, p/q)`、取随机数判断是否接受，一旦某 token 被拒就停下并对该位置分布做调整。Block Verification 的思路是加一条快路径：**在逐 token 拒绝采样之前，先看当前候选路径后续的 token 是否正好等于 target 的 greedy top-1 续写；如果等于，就一次性接受这一整块前缀，跳过逐 token 随机判断。** 这样可以增加接受长度、减少 target 前向次数。代价是把这部分续写偏向了 target 的贪心结果（详见第 8 节有损性）。

### 4.2 原始采样分支

`eagle/model/utils.py`，逐 token 拒绝采样，无 block 快路径：

```python
else:
    accept_length = 1
    accept_cand = candidates[0][:1]
    best_candidate = 0
    for i in range(1, candidates.shape[1]):
        if i != accept_length:
            break
        adjustflag = False
        is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
        gt_logits = logits[fi, i - 1][None]
        gt_logits = logits_processor(None, gt_logits)[0]
        gtp = torch.softmax(gt_logits, dim=0)
        candidates_set = []
        for j in range(candidates.shape[0]):
            if is_eq[j]:
                x = candidates[j, i]
                xi = x.item()
                ...
                r = random.random()
                px = gtp[xi]
                acp = px / 1.0
                if r <= acp:
                    accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                    accept_length += 1
                    best_candidate = j
                    break
                else:
                    gtp[xi] = 0
                    gtp = gtp / gtp.sum()
                    adjustflag = True
    ...
```

### 4.3 Block v1：加入快路径

最初实现中曾存在 bug：用单条路径的 `logits[fi]` 去校验所有候选路径。但不同候选路径一旦分叉，其在 target 上的上下文就不同，必须各自用 `logits[j]` 取 top-1。修复后在采样分支的 `is_eq`/`fi` 之后、逐 token 拒绝采样之前插入，代码内容如下：

```python
# Block Verification fast path: among candidate paths matching the
# currently accepted prefix, accept the longest prefix that equals the
# target's own greedy (top-1) continuation, skipping per-token RNG.
# NOTE: each path j must be scored against its OWN target logits (logits[j]).
if USE_BLOCK_VERIFY:
    best_ext = 0
    best_j = int(best_candidate)
    for j in range(candidates.shape[0]):
        if not is_eq[j]:
            continue
        tgt_top1 = torch.argmax(
            logits[j, accept_length - 1: candidates.shape[1] - 1], dim=-1
        )
        cand_part = candidates[j, accept_length:]
        n = min(cand_part.shape[0], tgt_top1.shape[0])
        if n == 0:
            continue
        match = (cand_part[:n] == tgt_top1[:n]).int()
        ext = int(match.cumprod(dim=0).sum().item())   # 每条路径一次 .item() 同步
        if ext > best_ext:
            best_ext = ext
            best_j = j
    if best_ext > 0:
        accept_length += best_ext
        accept_cand = candidates[best_j, :accept_length]
        best_candidate = best_j
        continue
```

效果（固定树，相对**同温度同批** vanilla AR；t=0 数据来自 `20260613_145125` 批次）：


| 设置               | t=0 accept | t=0 overall tok/s | t=0 speedup | t=0.5 accept | t=0.5 overall tok/s | t=0.5 speedup |
| ---------------- | ---------- | ----------------- | ----------- | ------------ | ------------------- | ------------- |
| EAGLE-2 baseline | 4.08       | 80.70             | 1.96x       | 3.95         | 71.03               | 1.82x         |
| Block v1         | 4.08       | 80.77             | 1.97x       | 4.07         | 71.37               | 1.83x         |


可得：

- **t=0（贪心）：与 EAGLE-2 完全一致**（accept 4.08、tok/s 与 speedup 同量级）。这符合预期——贪心走 greedy 分支、Block 快路径不激活，证明改动在贪心下无副作用；这一列同时也是三组里加速比最高的（1.96–1.97x），因为贪心下草稿接受长度本就最高。
- **t=0.5（采样）：接受长度升了，端到端却几乎没快**——accept 从 3.95 提到 4.07，但 speedup 只从 1.82x 动到 1.83x、tok/s 71.03→71.37。接受长度升了、target 前向次数本应减少，吞吐却没跟上，猜测这里的收益被添加代码导致的调度成本消耗掉了。

推测：这版用了 per-path 的 Python `for` 循环，且每条路径都有一次 `.item()`（`int(... .item())`）把 GPU 结果拉回 CPU，强制同步；在 `total_token=60` 时一棵树最多约 60 条路径，于是每个解码 step 都有几十次 kernel launch + 几十次同步。

为把「收益究竟被哪一段吃掉」量化清楚，先做一次分段 profiling（§5），再据 profiling 结论做向量化优化（§6）。

## 5. Profiling：定位 v1 为何没加速效果

### 5.1 Profiling 实现

为把「v1 接受长度升了、端到端却几乎没快」的原因量化到具体阶段，给 `eagle/model/ea_model.py` 的 `eagenerate` 主循环加了 opt-in 的分段计时：新增参数 `return_profile_stats=False`，开启时在四个关键调用前后各插一次 `torch.cuda.synchronize()`，再用 `time.perf_counter()` 取墙钟，确保异步 CUDA kernel 的真实耗时被计入对应段。四段与主循环里的调用严格一一对应：


| 计时段     | 包裹的调用（`eagle/model/ea_model.py`）                           | 含义                                    | 频次                |
| ------- | ---------------------------------------------------------- | ------------------------------------- | ----------------- |
| prefill | `initialize_tree(...)`                                     | prompt 预填充 + 建第一棵草稿树                  | 每次生成一次（循环外）       |
| target  | `tree_decoding(...)`                                       | target 大模型对整棵草稿树做一次树形前向，出 logits      | 每个 decode step 一次 |
| verify  | `evaluate_posterior(logits, candidates, logits_processor)` | 验证 / 接受决策——**Block 改动正改在这里**          | 每个 decode step 一次 |
| update  | `update_inference_inputs(...)`                             | 接受 token、更新 KV cache、drafter 生成下一棵草稿树 | 每个 decode step 一次 |
| cycles  | 主循环迭代计数                                                    | 一个 turn 内 decode step 的次数             | —                 |


说明：**verify 段恰好只包住** `evaluate_posterior`**，而 Block 的全部改动都在** `evaluate_posterior` **内**，所以 `verify ms` 就是「Block 这段改动本身的开销」的直接度量——正好用来对照 v1 与后面向量化版各花了多少。

配套改动与注意事项：

- `eagle/evaluation/gen_ea_answer_qwen2.py`：新增 `--enable-profile`，把每轮 `profile_*_ms` 与 `profile_cycles` 写进结果 jsonl；
- `eagle/evaluation/speed_qwen2.py`：新增 `profile_summary()`，按 turn 汇总各段 ms 与占比并打印；
- **opt-in、无副作用**：不传 `--enable-profile` 时 `return_profile_stats=False`，不插入任何 `synchronize`，行为与原版逐字一致；
- **只看占比、不看墙钟**：开 profile 后强制 `synchronize` 会拖慢真实墙钟，因此 profiling 跑只用来读各段占比，真实加速比一律取不带 profile 的批次。

### 5.2 结果（固定树，t=0.5，EAGLE-2 vs Block v1）

结果文件（归档）：`peiyuhao/results/prof_speed_eagle2_t0.5_20260613_235811.txt`、`peiyuhao/results/prof_speed_block_t0.5_20260613_235811.txt`。

在温度为 0.5 的时候运行了 profiling 版本的 EAGLE-2 和 Block v1:


| 设置               | target ms       | verify ms     | update ms       | cycles/turn | accept length |
| ---------------- | --------------- | ------------- | --------------- | ----------- | ------------- |
| EAGLE-2 baseline | 2880.23 (59.3%) | 221.20 (4.6%) | 1682.41 (34.6%) | 90.17       | 3.99          |
| Block v1         | 2806.75 (56.9%) | 404.41 (8.2%) | 1646.10 (33.4%) | 87.24       | 4.10          |


诊断结论：

- **时间开销主体不在 verify**：target（大模型树前向）约 57%–59%，update（KV 更新 + 下一棵草稿树）约 33%–35%，verify 在 baseline 里只占 4.6%。也就是说「单纯把 verify 这段算得更快」的 Amdahl 天花板只有约 4.6% 。
- **Block 的收益主要来源**：接受长度从 3.99 提到 4.10，使每个 turn 的 decode step 从 90.17 降到 87.24（target 前向少跑约 3 次），target、update 都因此省了一些时间（target 2880→2807ms，update 1682→1646ms）。
- **当前存在的问题**：v1 把 verify 从 221ms 增加到了 404ms（+183ms，近翻倍）。这 +183ms 正是 §4.3 那段 per-path Python 循环 + 逐路径 `.item()` 同步的开销，量级上和「少跑约 3 次 target+update」省下的时间相当，因此此时 v1 端到端几乎不快（1.82x→1.83x）。
- **优化方向**：把 verify 这段的实现开销压回 baseline 水平甚至更低。
- **关于 t=0**：t=0 走贪心分支、Block 快路径不激活，EAGLE-2 与 Block 的分段占比理论上一致、也不存在 v1↔向量化的差异，信息量低，故 profiling 只在 t=0.5 做（t=0 仅做 speed 等价性验证，见 §6.3）。

---

## 6. Block 向量化（依据 profiling 的对症优化）

§5 把问题定位到「verify 段的实现开销」。向量化就针对这一段：只改 block 检查的**执行方式**，不改其**接受方式**。

### 6.1 向量化实现

把整段 per-path 循环替换为一次批量张量运算，保留「每条路径用自己的 `logits[j]`」这一要求（下面 `logits` 的第 0 维就是路径维），但只在最后做一次 `.item()`：

```python
if USE_BLOCK_VERIFY:
    tgt_top1 = torch.argmax(
        logits[:, accept_length - 1: candidates.shape[1] - 1], dim=-1
    )
    cand_part = candidates[:, accept_length:]
    ext_all = (cand_part == tgt_top1).int().cumprod(dim=1).sum(dim=1)
    ext_all = ext_all.masked_fill(~is_eq, 0)          # 不匹配前缀的路径置 0
    best_ext_tensor, best_j_tensor = torch.max(ext_all, dim=0)
    best_ext = int(best_ext_tensor.item())            # 整段仅此一次同步
    if best_ext > 0:
        best_j = int(best_j_tensor.item())
        accept_length += best_ext
        accept_cand = candidates[best_j, :accept_length]
        best_candidate = best_j
        continue
```

执行方式的变化：每个解码 step 内，block 检查由「约 60 条路径、每条各做一次 `argmax` + `cumprod` + `.item()`」压缩成「整批一次 `argmax` / `cumprod` / `max`」；GPU 和 CPU 同步从「每条路径一次」降为「整段一次」，即由约几十次 kernel launch + 几十次同步，降到约 4 个 kernel + 1 次同步。这正对应 §5 定位到的 verify 开销来源。

### 6.2 改动说明

向量化与 v1 的接受逻辑完全一致，逐条对应：

- `masked_fill(~is_eq, 0)` 等价于原来 `if not is_eq[j]: continue`（不匹配当前已接受前缀的路径不参与）；
- `cumprod(dim=1).sum(dim=1)` 对每条路径求「从 `accept_length` 起与 target top-1 连续匹配的长度」，与原 `match.cumprod().sum()` 等价；
- `torch.max` 取最长扩展长度及其路径，平局取第一个，与原 `ext > best_ext`（严格大于、保留首个最大）一致；
- `best_ext == 0` 时落回原逐 token 拒绝采样分支，与原逻辑一致。

### 6.3 效果（t = 0.5）

固定树、t=0.5，相对**同温度同批** vanilla AR（39.10 tok/s）。真实吞吐结果文件（归档）：`peiyuhao/results/speed_eagle2_t0.5_vs_vanilla_20260613_145125.txt`、`peiyuhao/results/speed_block_t0.5_vs_vanilla_20260613_145125.txt`、`peiyuhao/results/speed_block_vec_t0.5_vs_vanilla_20260614_010110.txt`；profiling 见 `peiyuhao/results/speed_prof_block_vec_t0.5_vs_vanilla_20260614_010110.txt`。把三组配置放在一起（baseline / v1 / 向量化）——这也是 t=0.5 的端到端汇总：


| 指标                          | EAGLE-2 baseline | Block v1      | Block v2（向量化）    | v1 → v2             |
| --------------------------- | ---------------- | ------------- | ---------------- | ------------------- |
| verify ms / turn            | 221.20 (4.6%)    | 404.41 (8.2%) | **69.91 (1.6%)** | −334.5 ms，占比 −6.6 % |
| target ms / turn            | 2880.23          | 2806.75       | 2647.83          | 随 cycles 略降         |
| cycles / turn               | 90.17            | 87.24         | 87.33            | 与 v1 相当             |
| mean accept length          | 3.99             | 4.10          | 4.08             | 基本不变                |
| overall speedup（无 profile）  | 1.82x            | 1.83x         | **2.05x**        | +0.22x（+12%）        |
| overall tokens/s（无 profile） | 71.03            | 71.37         | **80.05**        | +8.68 tok/s         |


> 注：本表 accept length（3.99 / 4.10 / 4.08）取自带 profile 的批次，而 overall tok/s / speedup 及 §4.3 / §7 中不带 profile 批次的 accept（3.95 / 4.07 / 4.11）取自不带 profile 的运行，二者来自不同次运行，故有零点几的差异，属正常，不影响结论。

分析：

- **接受行为不变**：向量化版与 v1 的 accept length（4.08 vs 4.10）、cycles/turn（87.33 vs 87.24）几乎相同，证实向量化只是换实现、没改 block 的接受结果。
- **verify 开销大幅下降**：verify 从 v1 的 404ms 降到 69.9ms（占比 1.6%，已低于 baseline 的 4.6%）。§5 中被 v1 自己吃掉的接受长度红利，此时不再被 verify 抵消，于是端到端从 1.83x 提升到 **2.05x**（吞吐 +8.68 tok/s）。
- **正确性（t=0 贪心等价性验证）**：t=0 时候也有完整的 vanilla / EAGLE-2 / Block v1 / Block 向量化四组数据（同一棵固定树，分母 vanilla AR 41.09 tok/s）。结果文件（归档）：`peiyuhao/results/speed_eagle2_t0.0_vs_vanilla_20260613_145125.txt`、`peiyuhao/results/speed_block_t0.0_vs_vanilla_20260613_145125.txt`、`peiyuhao/results/speed_block_vec_t0.0_vs_vanilla_20260614_010110.txt`。三组指标完全一致：


| 设置       | overall tok/s | overall speedup | accept length |
| -------- | ------------- | --------------- | ------------- |
| EAGLE-2  | 80.70         | 1.96x           | 4.08          |
| Block v1 | 80.77         | 1.97x           | 4.08          |
| Block v2 | 80.54         | 1.96x           | 4.08          |


  t=0 走贪心分支、Block 快路径不激活，因此 Block 与 EAGLE-2 等价——正面证明改动只作用于 t>0 采样分支、对贪心无副作用。

---

## 7. 温度消融（固定树，t=0 / t=0.5 / t=0.8）

固定草稿树参数不变，在t = 0.8 的采样温度下重跑 vanilla / EAGLE-2 / Block v2（向量化），观察接受长度与吞吐如何随温度变化。t=0.8 批次由脚本 `scripts/run_temp08_block_experiments.sh` 跑（不开 profile）。结果文件（归档）：`peiyuhao/results/speed_eagle2_t0.8_vs_vanilla_20260616_102407.txt`、`peiyuhao/results/speed_block_vec_t0.8_vs_vanilla_20260616_102407.txt`。

### 7.1 t=0.8 单点结果


| 设置            | overall tok/s | overall speedup | accept length |
| ------------- | ------------- | --------------- | ------------- |
| EAGLE-2       | 68.42         | 1.69x           | 3.70          |
| **Block 向量化** | **80.36**     | **1.99x**       | **4.11**      |


### 7.2 跨温度趋势

把 t=0 / t=0.5 / t=0.8 三个温度并排看（加速比各自相对**同温度同批** vanilla，分母不同，故只比同列趋势）：


| 温度    | EAGLE-2 accept | Block v2 accept | Block v2 与EAGLE-2 accept差值 | Block overall tok/s | Block speedup |
| ----- | -------------- | --------------- | -------------------------- | ------------------- | ------------- |
| t=0   | 4.08           | 4.08            | +0.00                      | 80.54               | 1.96x         |
| t=0.5 | 3.95           | 4.11            | +0.16                      | 80.05               | 2.05x         |
| t=0.8 | 3.70           | 4.11            | **+0.41**                  | 80.36               | 1.99x         |


三个现象：

- **EAGLE-2 的接受长度随温度单调下降**（4.08 → 3.95 → 3.70）：温度越高采样越发散，草稿越难被逐 token 接受。
- **Block 的接受长度几乎不随温度变化**（4.08 / 4.11 / 4.11），稳定在 ≈ t=0 贪心水平；其绝对吞吐（t=0.5 80.05、t=0.8 80.36 tok/s）也基本不随温度变。于是 Block 相对 EAGLE-2 的接受长度优势随温度放大（+0.00 → +0.16 → +0.41）。
- Block speedup 在 t=0.8（1.99x）略低于 t=0.5（2.05x），并不是 Block 变慢了：Block 的绝对吞吐其实基本持平甚至略升（80.05 → 80.36 tok/s）。差别来自分母——加速比是相对**同温度**的 vanilla AR，而 vanilla 各温度吞吐有运行间波动（t=0 41.09、t=0.5 39.10、t=0.8 40.43 tok/s），t=0.8 分母偏高于是同样的 Block 吞吐被算出更低的倍数。

Block 接受长度不随温度退化，正是它将输出分布导向贪心结果，导致分布有损性的一种体现。

---

## 8. 有损性与 trade-off 分析

### 8.1 有损性来源

Block Verification 只在 **t>0 采样分支** 生效：把后续等于 target greedy top-1的那段草稿整块接受、跳过逐 token 拒绝采样，本质上把这部分续写偏向了贪心结果。因此 Block **不是严格分布无损**，而是接受长度与采样分布偏移之间的折中。这与第 3 节的 t=0 数值分叉是两回事：那是 bf16 下的数值噪声，这里是算法主动引入的偏移。

### 8.2 trade-off 数据

用 `eagle/evaluation/block_lossiness.py` 在 t=0.5 / t=0.8 两个温度上，对**同温度同批**的 vanilla / EAGLE-2 / Block v2 已生成输出做对照。结果文件（归档）：`peiyuhao/results/block_lossiness_t0.5.txt`、`peiyuhao/results/block_lossiness_t0.8.txt`。

接受长度收益：


| 温度    | vanilla AR | EAGLE-2 | Block 向量化 | Block−EAGLE-2 |
| ----- | ---------- | ------- | --------- | ------------- |
| t=0.5 | 1.000      | 3.945   | 4.113     | **+0.167**    |
| t=0.8 | 1.000      | 3.700   | 4.111     | **+0.411**    |


文本级两两分叉：


| 温度    | 对比对                | 分叉率（turn 不逐字相同占比） | 平均公共前缀 (tokens) | 公共前缀占比 |
| ----- | ------------------ | ----------------- | --------------- | ------ |
| t=0.5 | EAGLE-2 vs vanilla | 96.9%             | 7.5             | 0.059  |
|       | Block vs vanilla   | 96.9%             | 8.9             | 0.068  |
|       | Block vs EAGLE-2   | 95.6%             | 10.3            | 0.077  |
| t=0.8 | EAGLE-2 vs vanilla | 97.5%             | 4.6             | 0.038  |
|       | Block vs vanilla   | 97.5%             | 5.3             | 0.042  |
|       | Block vs EAGLE-2   | 96.2%             | 6.9             | 0.058  |


### 8.3 分析

- **接受长度收益**：它由 `new_tokens/(idx+1)` 算出、不依赖随机数。Block 相对 EAGLE-2 的收益随温度从 **+0.167（t=0.5）扩大到 +0.411（t=0.8）**——温度越高，被 Block 整块拉回贪心的比例越大，**收益与分布偏移同步放大**：高温下 Block 相对更划算，但偏离真实采样分布也更多。
- **文本分叉率不能直接当分布偏移**：注意 `EAGLE-2 vs vanilla` 的分叉率也高达 96.9% / 97.5%，而 EAGLE-2 在精确算术下本应分布无损。这恰恰证明 t>0 下三方各自运行独立的随机数流（vanilla 逐 token 采样、EAGLE/Block 在拒绝采样内抽随机数、Block 快路径完全跳过抽样），token 级分叉绝大部分来自随机数序列不同、而非分布偏移本身。所以分叉率只能作粗粒度参考，不能用来量化 Block 的偏移。
- **方向一致的弱旁证**：在与 vanilla 的平均公共前缀上 Block 始终略长于 EAGLE-2（t=0.5：8.9 vs 7.5；t=0.8：5.3 vs 4.6），方向上与“Block 更贴 target 贪心”一致，但幅度小、且同样受随机数流影响，仅作旁证。同理可以看到 Block v2 与 EAGLE-2 在温度上升的时候公共前缀减少，也与温度越高，分布偏移越大方向一致。
- **猜想**：可以看到在我的实验中温度升高，平均接受长度的大小变化不明显，由此猜测 drafter 小模型本身输出的时候应该是比较“自信”的，否则调高温度，drafter 输出和 target model 贪心结果的匹配度按道理应该也会下降，这时平均接受长度就会下降。

### 8.4 局限性

- 本报告以**接受长度随温度的稳定性**作为有损性的主证据；文本分叉率因受随机数流影响只作粗粒度参考。要把分布偏移做成更干净的可比数值，还可以在生成时捕获 logits、统计每步接受的 token 是否等于 target greedy top-1的占比（§3 probe 思路、需 GPU），本实验未做。
- 收益对 target 规模敏感：verify 那段开销基本是固定的 CPU/同步成本，而「少跑一次 target 前向」省下的时间随 target 变大而变大。当前 7B 上单卡可跑；更大 target（如 13B+）理论上 block 的净收益会更明显，但受单卡显存限制本次未做。

---

## 9. 结果归档结构（peiyuhao/）


| 目录                | 内容                                                                                                                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reports/`        | 本报告                                                                                                                                                                                                           |
| `results/`        | 主实验（`20260613_145125`）、profiling（`20260613_235811`）、向量化（`20260614_010110`）、温度消融（`20260616_102407`）的生成结果、speed 与 profiling 汇总、manifest；以及 trade-off 分析 `block_lossiness_t0.5.txt` / `block_lossiness_t0.8.txt` |
| `logs/`           | 上述各次运行日志                                                                                                                                                                                                      |
| `scripts/`        | 实验脚本快照（对应仓库 `scripts/`）                                                                                                                                                                                       |
| `code_snapshots/` | 关键源码快照（对应仓库 `eagle/model/`、`eagle/evaluation/`）                                                                                                                                                               |
| `analysis/`       | t=0 输出差异分析：`diff_outputs.py` / `probe_logits.py` 及结果报告                                                                                                                                                        |


