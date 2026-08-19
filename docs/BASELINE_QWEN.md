# Substitute-Baseline Report — Qwen2.5-7B-Instruct on sample100 (P5)

> 本文档记录正式权重到位前的**替身基线**：用 Qwen2.5-7B-Instruct（AgentForesight-7B 的确认基座）跑通全流程，
> 产出指标基线 + 消耗估算方法学与数字。**不是 AgentForesight-7B 的成绩**，最终成绩见 P6 报告。
> 运行环境：远端 `shixuan@gpu`（3×A800 80GB，共享卡，他人任务约占 50GB/卡）；仓库 `~/AgentForesight`；
> venv `~/af`（uv）；替身权重 `~/models/Qwen2.5-7B-Instruct`（hf-mirror.com 镜像下载）。

## 1. 任务口径（与最终评测一致）

- 测试集：**83 条**（100 − 17 条 travelplanner train/validation），全部 unsafe。
- 协议：论文 Definition 2.2 逐前缀在线审计，首个 valid 非 SAFE 即停（`detection_step = d(τ)`）。
- 指标：Step-Acc（≡B.2 Recall_step）、Agent-Acc（严格/宽松，自定义）、ASS↓（over Udet）、Exact-F1（参考）；
  每样本推理时间 + token 统计。
- 压缩：预算 B=30,720 token（32k − 2k 余量）内逐字，超限 L1 结构化摘要 + 窗口，必要时中间段折叠（L2 滚动摘要
  已实现并有 `--l2-summary` 开关，本语料默认预算下不触发，未用于本跑批）。

## 2. 替身指标结果（83 条，3 卡分片并行约 47 分钟）

| 指标 | 值 | 备注 |
|---|---|---|
| Step-Acc | **1.20%**（1/83） | 未微调基座几乎定位不到决策步（预期） |
| ASS↓ | 61.33 | over Udet（82 条报警，1 条全程 SAFE） |
| Agent-Acc 严格 / 宽松 | 38.67% / 49.33% | step 几乎全错时 agent 归因仍 ~39% |
| Exact-F1 | 1.21% | 参考（precision=Recall_step/…，无 safe 样本） |
| format_valid | 82/83 | 1 条解析失败 |
| 未报警样本 | 1 条 | 排除在 ASS 分母外（n_detected=82） |

每样本平均：`num_calls≈12.7`（中位 8）、`gen_time_s≈78.0`、`prompt_tokens≈80,681`、`completion_tokens≈2,152`。
单样本跨调用累计输入 token 最大 **1,143,574**（最长 561 步样本，数百次前缀调用累加）。

## 3. 消耗估算（你的"逐步推理巨大成本"猜想的量化证据）

| 场景 | 调用数 | 输入 token | 输出 token | 单卡时长（微基准） | **偏置校正后** |
|---|---|---|---|---|---|
| 替身实测（早报警） | 1,058 | 6.70M | 0.18M | —（实测 3 卡 ≈ 47 min） | — |
| **perfect（完美检测器）** | 5,965 | 126.16M | 1.19M | 47.26 h | **20.60 h** |
| worst（永不报警） | 12,446 | 602.37M | 2.49M | 144.28 h | 62.88 h |

- **token 消耗由报警时机主导**：模型越准（越接近 gt_step 才报警）→ 调用越多 → 成本越高。替身(12.7 次/样本) vs
  完美检测器(71.9 次/样本)：**token ≈ 18×、时间 ≈ 26×**。正式 AgentForesight-7B 的成本将显著高于替身，
  单卡量级 ~20h（bias-corrected），三卡并行 ~7h——这就是"长程任务逐前缀审计消耗巨大"的直接量化。
- 长度感知成本模型（文献：prefill/decode 分解 + roofline 校验，见 P4 讨论）：`t(k)=input_k/P(input_k)+out/D(ctx_k)`。
- **run-condition 引擎常数**（从实测拟合）：prefill_eff≈4,056、decode_eff≈37.0 tok/s（共享卡争用下的有效值）；
  微基准中位 prefill 5,756 / decode 12.9 tok/s（与跑批时段争用不同 → microbench 残差 +129%，已用校正因子 ×0.436 修正，
  自洽残差 ≈0%）。**GPU 争用是最大不确定源**：正式跑批建议在相对空闲时段/独占卡上重做 calibrate。

## 4. 产物与复现

- 跑批：`outputs/qwen/shard{0,1,2}/`（可续跑）→ 合并 `outputs/qwen/final/results.json`
- 引擎标定：`outputs/calib.json`；精确 tokenizer profile：`outputs/profile_tok.json`；外推+残差：`outputs/qwen/extrapolate.json`
- 复现命令：
  ```bash
  source ~/af/bin/activate
  MODEL_PATH=~/models/Qwen2.5-7B-Instruct OUT=~/AgentForesight/outputs/qwen ./run_remote.sh local
  python3 -m tools.extrapolate --profile outputs/profile_tok.json --calib outputs/calib.json \
      --measured outputs/qwen/final/results.json --output outputs/qwen/extrapolate.json
  ```

## 5. 对最终评测（P6）的影响

1. 预算 B：本跑批出现 `exceeded 32768` 提示（30,720+2,048=32,768 顶满）→ 正式跑批 `--max-input-tokens 28672`。
2. 时间预算：正式模型三卡 ~7h 量级（若与替身同架构同引擎）；建议 tmux + 断点续跑。
3. 指标对照：P6 报告将给出 替身(1.20%/38.67%) vs AgentForesight-7B 的对比，验证微调带来的定位/归因提升。
