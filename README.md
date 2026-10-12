# EAGLE-2 Block Verification 并行验证优化

本项目为机器学习系统课程 Final Project，研究 EAGLE-2 推测解码中的 Block Verification（前缀块验证）策略，并以 Qwen2-7B-Instruct 为目标模型完成实现、性能分析与实验验证。

本仓库仅保留课程项目中实际修改或新增的代码、实验脚本以及结果，不包含完整的 EAGLE 上游仓库、模型权重和数据集。

## 主要工作

- 在 EAGLE-2 的采样验证流程中实现 Block Verification 快路径。
- 将早期 Python 循环实现改为向量化验证，降低 verify 阶段开销。
- 增加 Qwen2 的生成、测速和分段 profiling 脚本。
- 在固定草稿树下比较 vanilla AR、EAGLE-2、Block v1 和向量化 Block。
- 分析采样温度变化时的接受长度、吞吐收益和分布偏移。

## 主要结果

在 Qwen2-7B-Instruct、温度 0.5、固定草稿树配置下：

| 方法 | 相对 vanilla 加速比 | 平均接受长度 |
| --- | ---: | ---: |
| EAGLE-2 baseline | 1.82x | 3.95 |
| Block v1 | 1.83x | 4.07 |
| 向量化 Block | **2.05x** | **4.11** |

verify 阶段占比由 Block v1 的 8.2% 降至向量化版本的 1.6%。需要注意：Block Verification 在温度大于 0 时并非严格分布无损，结果体现的是吞吐收益与采样分布偏移之间的折中。

## 目录结构

```text
mlsys_project/
├─ overlay/                  # 复制到 EAGLE 仓库根目录的代码与脚本
│  ├─ eagle/model/           # Block Verification 与 profiling 相关修改
│  ├─ eagle/evaluation/      # Qwen2 评测、测速和分析脚本
│  └─ scripts/               # 主实验、profiling 与温度消融脚本
├─ results/                  # 原始 JSONL、测速汇总、manifest 与分析输出
├─ docs/                     # 课程项目报告
├─ LICENSE                   # 上游 EAGLE 的 Apache 2.0 许可声明
└─ README.md
```

## 使用方式

1. 按照 [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) 的说明准备 Linux、CUDA、Python 环境以及 Qwen2/EAGLE 模型。
2. 将 `overlay/` 中的内容复制到 EAGLE 仓库根目录，使 `eagle/` 和 `scripts/` 与上游目录合并。
3. 根据本机环境修改脚本和评测代码中的模型路径、结果路径及 Conda 环境名。
4. 运行主实验脚本：

   ```bash
   bash scripts/run_all_block_experiments.sh
   ```

   向量化验证、profiling 和温度消融分别对应 `run_vectorized_block_experiments.sh`、`run_profile_block_qwen2.sh` 和 `run_temp08_block_experiments.sh`。

原实验在 AutoDL 环境中完成，因此代码和结果中保留了 `/root/autodl-tmp/...` 路径。它只是原运行目录，不包含密钥或账号信息；在其他机器运行时需要替换成实际路径。

更完整的方法、实验设置和结果讨论见 [项目报告](docs/block_verification_progress_report.md)。

## 来源与贡献说明

本项目基于 SafeAI Lab 的开源项目 [EAGLE](https://github.com/SafeAILab/EAGLE) 开发。EAGLE 上游代码采用 Apache License 2.0，版权归 SafeAI Lab (SAIL) 所有；相关许可声明保留在本仓库的 `LICENSE` 中。

本仓库展示的是课程项目中针对 EAGLE-2 Block Verification 完成的代码修改、Qwen2 评测工具、实验脚本、profiling、结果归档与分析，不代表完整 EAGLE 系统均由本项目作者原创。`overlay/` 中的文件应结合官方 EAGLE 仓库使用。

