---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 60ea5d0126731e33de81577b27285889_2eede1cfc09211f1884b525400cd780f
    ReservedCode1: xVCgRF/RPz8iq7H6hkNRHsrQNlgrS/ZeLPWHUFDc/+3aXcRx2w5tM6IVYpdBMO4r41eeiUt/6EiaOSo5quG8wThR0uADYk/0e3X9OCatW6bMWLVbX9qspGTH5jJP/fAz0+BK5DfkeE6Og89DsOhIAQR23AFJHLseWkYb9NpBOjB7JX8Qvl3qPMygoNM=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 60ea5d0126731e33de81577b27285889_2eede1cfc09211f1884b525400cd780f
    ReservedCode2: xVCgRF/RPz8iq7H6hkNRHsrQNlgrS/ZeLPWHUFDc/+3aXcRx2w5tM6IVYpdBMO4r41eeiUt/6EiaOSo5quG8wThR0uADYk/0e3X9OCatW6bMWLVbX9qspGTH5jJP/fAz0+BK5DfkeE6Og89DsOhIAQR23AFJHLseWkYb9NpBOjB7JX8Qvl3qPMygoNM=
---

# lenovo_C2G_拿来说明

**提交人：** lenovo ｜ **日期：** 2026-10-04
**本文件的用途：** 逐项说明本方案**从社区拿了什么、自己改了什么、为什么改**，保证每一处改动都可追溯到具体来源（PR 号 / 记录目录），并明确标注哪些是"原样采用"、哪些是"本方案的改动"。

---

## 0. 总览

| 类别 | 数量 | 说明 |
|---|---|---|
| 原样采用（不改逻辑） | 6 | SP8192、Depth Recurrence、Parallel Residuals、MuonEq-R、QK-Gain、Sliding-Window Eval |
| 采用 + 调参 | 3 | Legal Score-First TTT、EMA/WD、GPTQ-SDClip |
| 本方案自设 | 3 | seed 集合 0/42/1234、消融矩阵设计、判定阈值 0.0006 |
| 基线来源 | 1 | 官方 `train_gpt.py` |

**归属声明：** 本方案**不主张**任何上述组件的原创性。所有组件归属见 `lenovo_C2G_submission.json` 的 `attribution` 字段（与 SOTA 记录的归属表一致）。

---

## 1. 逐项追溯

### 1.1 SP8192 分词器（8192 BPE）
- **拿了什么：** SentencePiece 8192 词表配置。
- **来源：** 社区 PR #1394（`@clarkkev`，对应记录 `2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2`，val_bpb 1.08563）。
- **改了什么：** 未改词表本身；改的是与之配套的 `EMBED_BITS`（8 bit）与 `EMBED_CLIP_SIGMAS`（20.0σ）。
- **为什么改：** 词表从 1024 涨到 8192，embedding 参数量涨 8×，若沿用 6 bit 会吃掉 16 MB 预算的余量；8 bit + 20σ 在实测记录中能把 embedding 压回可接受体积。
- **风险：** 大词表在 600 s 内可能欠训练。**对策：** 用 EMA 0.9965 平滑后期权重，并把 early-stop 放在 step ≈4550（对齐 SOTA 的 stop 点）。

### 1.2 3 层深度循环 Depth Recurrence（L3–L5）
- **拿了什么：** 物理层 3/4/5 在训练后 35% 处开始被复用一次（`NUM_LOOPS = 2`）。
- **来源：** `@dexhunter`（PR #1331、#1437），对应记录 `2026-04-03_...`（1.0912）与 `2026-04-06_SP8192_QK5_LegalTTT_1.0828`（1.08279384）。
- **改了什么：** 把 SOTA 的 `ENABLE_LOOPING_AT` 定为 **0.35**、循环段固定 **L3–L5**；SOTA 亦为 3 层循环，此处为**对齐而非改动**。
- **为什么改：** 参数预算被量化压到 16 MB 内后，"用层数换深度"是唯一不增加参数的增容手段。
- **风险：** 循环段过深会导致梯度冲突。**对策：** 循环只从 35% 训练进度开始（前期先学好非循环通路），与 SOTA 实测一致。

### 1.3 Parallel Residuals（L7+）
- **拿了什么：** 第 7 层起把残差改为并行（attention 与 MLP 共享同一输入，不串联）。
- **来源：** `@Robby955`（PR #1412）、`@msisovic`（PR #1204），记录 `2026-03-31_ParallelResiduals_MiniDepthRecurrence`（1.10625353）。
- **改了什么：** `PARALLEL_RESIDUAL_START = 7`（与 SOTA 一致）。
- **为什么改：** 前 6 层保持串行以保住低层特征抽取能力，只在深层并行化以降低梯度路径长度；把并行限制在深层比全层并行更稳。
- **风险：** 并行残差改变梯度尺度，可能与 QK-Gain 冲突。**对策：** 列入消融 C5（见 `lenovo_C2G_ablation.md` §4）。

### 1.4 MuonEq-R（矩阵优化器，WD 0.095）
- **拿了什么：** Muon 的 equi-regularized 变体，对 2D 矩阵参数使用行归一化 + 高 weight decay。
- **来源：** `@dexhunter`（记录 `2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6`，1.0912）、`@aryanbhosale`（`2026-04-04_SP4096_...`，1.08971631）。
- **改了什么：** **WD 从 0.090 提到 0.095**（对齐 SOTA）；scalar / embedding 仍走 AdamW。
- **为什么改：** 本方案比对照记录多了一层循环，参数量再次被复用，需要略强的正则来抑制循环段过拟合。
- **风险：** WD 过大会导致欠拟合。**对策：** 消融 C3 专门测 MuonEq-R × WD(0.090/0.095) 的交互。

### 1.5 QK-Gain 5.25
- **拿了什么：** attention 中 QK 点积的固定增益系数。
- **来源：** `@bigbag`（记录 `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`，**当前 SOTA 1.08100**）。
- **改了什么：** 采用 **5.25**（对照的早期方案多用 5.0）。
- **为什么改：** 从 5.0 → 5.25 是 SOTA 记录里相对前一版（`2026-04-08` 的 1.08217956）的关键改动之一；在 int6 量化下更大的 QK 增益能补偿量化带来的注意力锐度损失。
- **风险：** 增益过大 → logits 爆炸。**对策：** 配合 SDClip 在量化前裁剪激活，且消融 A5 单测该项。

### 1.6 Legal Score-First TTT（SGD，3 epoch，chunk 32K）
- **拿了什么：** 测试时训练框架，"score-first" 采样顺序；`ttt_lr = 0.005`、`ttt_epochs = 3`、`chunk = 32768`。
- **来源：** `@abaybektursun`（PR #549，记录 `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`）、`@dexhunter`（PR #1413）。
- **改了什么：** 严格保持合规四项**全部关闭**：`slot = off`、`etlb = off`、`ngram_cache = off`、`pre_quant_ttt = off`（即 TTT 只在量化**之后**、评分窗口**之内**做，不偷看验证集）。
- **为什么改：** 参数保持与 SOTA 一致，收益主要来自"合规地做 TTT"；任何越过合规边界的 TTT 都会让成绩不可比。
- **风险：** TTT 超时。**对策：** TTT 时间预算单列（SOTA 实测 289–295 s），总时长仍受 600 s 约束。

### 1.7 Hessian SDClip + All-Int6 GPTQ（矩阵 int6 / embedding int8）
- **拿了什么：** GPTQ 后训练量化 + Hessian 引导的 SDClip 激活裁剪；Brotli level 11 容器。
- **来源：** `@clarkkev`（PR #1394）、`@Robby955`（记录 `2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence`，1.08354）。
- **改了什么：** 矩阵 `MATRIX_BITS = 6`、`MATRIX_CLIP_SIGMAS = 12.85`；embedding `EMBED_BITS = 8`、`EMBED_CLIP_SIGMAS = 20.0`（与 SOTA 的 All-Int6 口径一致，embedding 例外为 8 bit）。
- **为什么改：** 全部 6 bit 会把 artifact 压到远低于预算、但 BPB 明显变差；embedding 单独用 8 bit 是"花小钱买分"的性价比选择（对照 `2026-04-06` 的 1.08279384 vs `2026-04-03` 的 1.0912）。
- **风险：** 量化误差吃掉 TTT 收益。**对策：** 量化后重新跑 TTT（score-first），保证 TTT 看到的是量化后的权重。

### 1.8 Sliding-Window Eval（stride 64, len 2048）
- **拿了什么：** 评估时用 stride-64 的滑动窗口替代单窗口。
- **来源：** `@mattqlf`（记录 `2026-03-19_SlidingWindowEval`，1.19250007）。
- **改了什么：** 未改，属**纯评估改动**（不改变模型）。
- **为什么改：** 这是全链中**性价比最高**的单项改动之一（实测 −0.0133），且零训练成本。
- **风险：** 无（仅影响评估口径，需在 README 中声明，避免与其他提交的 BPB 口径混淆）。

### 1.9 EMA 0.9965 / WD 0.095 / warmup 250 步
- **拿了什么：** 超参搜索结论。
- **来源：** `@X-Abhishek-X`（PR #1445，SOTA 记录的 `hyperparameter_tuning` 归属）。
- **改了什么：** 全部对齐 SOTA；本方案**未**自行搜索新超参（本机不具备搜索条件）。
- **为什么改：** 在无法实机搜索的前提下，**复用已验证的超参比自创超参更可信**。
- **风险：** 与自设的 seed 集合（0/42/1234）不匹配。**对策：** seed 集合选用了同 seed 集合的记录（`2026-04-06`，1.08279384）作为最小可比参照。

### 1.10 官方 train_gpt.py 基线
- **拿了什么：** 官方训练脚本骨架（数据管线、模型定义、评估入口）。
- **来源：** 官方 repo `parameter-golf-main/train_gpt.py`。
- **改了什么：** 见 `lenovo_C2G_train_gpt.py` 文件内的 `MODIFIED FROM OFFICIAL` 区块注释；主要新增 SP8192 分支、循环层逻辑、并行残差、MuonEq-R、QK-Gain、GPTQ-SDClip、TTT、EMA 九块。
- **为什么改：** 官方基线（Naive Baseline 1.2243657）→ 目标 1.0810 的 0.143 BPB 差距全部由这些改动承载。

---

## 2. 本方案自设（非社区拿来）的部分

| # | 自设内容 | 理由 |
|---|---|---|
| 1 | seed 集合 **0 / 42 / 1234** | 与 `2026-04-06` 记录同 seed 集合，便于最小可比对照；同时避开 SOTA 已用的 42/314/999 |
| 2 | 消融矩阵 A0–A8（单旋钮设计） | 社区记录是"多项叠加"的日期链，**无法做因果归因**；本方案用单旋钮消融把归因补上 |
| 3 | 判定阈值 **Δ > 0.0006（≈3σ）** | 以 `stderr ≈ 0.0002` 为噪声地板，防止把噪声当收益 |
| 4 | 合规自检清单 | 直接照搬 SOTA 记录的 `compliance` 九项，逐项自查 |

---

## 3. 合规声明（逐项）

| 合规项 | 本方案 | 说明 |
|---|---|---|
| `train_under_600s` | 待验证（目标 588 s） | 未实机运行 |
| `artifact_under_16mb` | 待验证（目标 ≈15,990,000 B） | 本机可验证的是"打包产物 < 16 MB"，模型 artifact 未生成 |
| `eval_under_600s` | 待验证 | — |
| `no_slot` | ✅ 是 | 未使用 slot 机制 |
| `no_pre_quant_ttt` | ✅ 是 | TTT 仅在量化后执行 |
| `no_etlb` | ✅ 是 | 未使用 eval-time layer banking |
| `no_ngram_cache` | ✅ 是 | 未使用 n-gram cache |
| `score_first_ttt` | ✅ 是 | TTT 按 score-first 顺序采样 |
| `three_seeds` | 待验证 | 计划跑 3 个 seed |
*（内容由AI生成，仅供参考）*
