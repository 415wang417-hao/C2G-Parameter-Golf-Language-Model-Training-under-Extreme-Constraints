---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 60ea5d0126731e33de81577b27285889_2b0f7620c09211f1887c525400de85a5
    ReservedCode1: SAQd4UaVSWZJWfNsHDjGgudiTc7nNnkK58dCPGeCkHq2LZQ24KO/gO5/YElocYLhmxKHG3bLTs58dCPgC6qA1Bt/eFHHG2CuuZV4GWSBtCB6b8lSGB7nYCl2g6k3VQVFfsNz4pANSszdCrXBc85Tr1yHQ1Xlx9IG4M7xNp7Tr54bOyMR3dBtm/lfbgc=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 60ea5d0126731e33de81577b27285889_2b0f7620c09211f1887c525400de85a5
    ReservedCode2: SAQd4UaVSWZJWfNsHDjGgudiTc7nNnkK58dCPGeCkHq2LZQ24KO/gO5/YElocYLhmxKHG3bLTs58dCPgC6qA1Bt/eFHHG2CuuZV4GWSBtCB6b8lSGB7nYCl2g6k3VQVFfsNz4pANSszdCrXBc85Tr1yHQ1Xlx9IG4M7xNp7Tr54bOyMR3dBtm/lfbgc=
---

# lenovo_C2G_方案设计

**提交人：** lenovo ｜ **日期：** 2026-10-04 ｜ **版本：** lenovo-PG v1
**上游文档：** `lenovo_C2G_方案草案.md`
**目标：** Level 4（Platinum，BPB < 1.085）；主攻当前 SOTA 线 **1.0810** 并力争超出；Level 3（BPB < 1.12）为保证下界
**验证状态：** 目标值 `pending_verification`（本机无 8×H100、未实际执行训练；本文件为**可复现方案设计**，非成绩声明）

---

## 1. 目标 Level 与验收标准

| 项 | 目标值 | 依据 |
|---|---|---|
| 主指标 `val_bpb` | **< 1.085**（Level 4 Platinum 门槛）；Level 3（<1.12）为保证下界 | brief §4 参考线：Naive 1.2244 / Level3 <1.12 / Level4 <1.085 / SOTA 1.0810 |
| 落点（**目标值**，待验证） | 主攻 **1.0810**；预期区间 1.0800–1.0850 | 对齐 repo 第 1 名 `bigbag` 1.08100，与第 2–4 名 1.08218–1.08354 同档 |
| 相对 baseline 降幅 | ≥ 0.139 BPB（1.2243657 → 1.0850，-11.4%）；目标 0.143 BPB（→1.0810，-11.7%） | 基线来自 `records/track_10min_16mb/2026-03-17_NaiveBaseline` |
| 成绩验证状态 | `pending_verification`（无 8×H100，未实际训练） | 诚实性红线：不得把未运行训练写成已达成 |
| artifact | ≤ 16,000,000 字节 | 硬约束；当前 SOTA 三 seed 为 15,991,930 / 15,992,919 / 15,993,232 |
| 训练 wallclock | ≤ 600 s（目标 588 s，留 2% buffer） | 硬约束；SOTA 实测 588047 ms |
| 评估 wallclock | ≤ 600 s（sliding + TTT ≈ 490 s） | 硬约束；SOTA sliding 120.96 s + TTT 366.53 s |
| 独立 seed | ≥ 3（**0 / 42 / 1234**） | 与 `2026-04-06_SP8192_QK5_LegalTTT_1.0828` 同 seed 集合，便于最小可比对照 |
| 统计显著性 | 对本方案各消融变体 p < 0.01 | 判定阈值 Δ > 0.0006（≈3σ，σ=0.0002） |

---

## 2. 技术选型

### 2.1 总览：七层组件栈

```
L7  Artifact 层   GPTQ-SDClip(int6 k=12.85σ / int8 emb k=20.0σ) + byte-shuffle + Brotli-11
L6  评估层        Sliding-Window(stride=64, len=2048) + Legal Score-First TTT(SGD 0.005, 3ep, 32K chunk)
L5  优化器层      MuonEq-R(row-normalised Muon, NS5, WD=0.095) + AdamW(embed 0.6/0.03, scalar 0.02)
L4  训练调度      warmup 20 step(恢复初值) + warmdown_frac=0.72 + EMA 0.9965 + grad_clip 0.3
L3  架构层        11L×512d, U-Net skips + skip gates, Parallel Residuals(L7+), 3-Layer Depth Recurrence(L3-5)
L2  注意力/FFN    GQA 8H/4KV, Partial RoPE(16/64), QK-Gain 5.25, LeakyReLU(0.5)² MLP 4x, LN-scale, softcap 30
L1  数据/Tokenizer SP8192(8192 vocab, ~5.9 byte/token), seq=2048, batch=786432 token/step
```

### 2.2 逐项选型理由（含"为什么不用替代方案"）

#### (1) Tokenizer：SentencePiece 8192

- **选择：** SP8192（`fineweb_8192_bpe.model`），vocab_size = 8192。
- **理由：** BPB 的分母是原始字节。SP8192 的平均 token 长度约 5.9 字节，是 1024-BPE（≈4.2 字节）的 1.4 倍；同样 2048 token 的上下文，SP8192 覆盖 ≈12k 字节，1024-BPE 只覆盖 ≈8.6k 字节。**上下文覆盖扩大 40%** 是 BPB 下降的第一性来源。
- **代价与对策：** embedding 从 512×1024 = 0.52M 膨胀到 512×8192 = 4.19M 参数（占 3.59e7 总参数的 11.7%）。对策是 **tied embedding + int8 量化 + `tied_embed_init_std=0.005` 小初始化**。
- **不选 SP4096 的理由：** repo 内 `2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR`（作者 `aryanbhosale`）= 1.08971631，其后续同日 SP8192 版本为 1.0822（`2026-04-08_SP8192_ParallelResid_ScoreFirstTTT`），**SP8192 比 SP4096 好 0.0075**，且 4096 的字节预算节省不足以补偿这个损失。

#### (2) 优化器：MuonEq-R（行归一化 Muon）

- **选择：** 矩阵参数（`c_q/c_k/c_v/proj/fc/mlp.proj`）走 Muon，`momentum=0.99`，`backend_steps=5`（Newton-Schulz），`row_normalize=True`，`wd=0.095`；embedding 走 AdamW `lr=0.03`（tied）/ `wd=0.085`；标量（`q_gain/attn_scale/mlp_scale/resid_mix/skip_weights`）走 AdamW `lr=0.02`。
- **理由：** Muon 对 2-D 矩阵做近似正交化更新，等效于把每个方向的更新范数拉平，对 10 分钟量级的短训练特别有效（少见的"收敛更快且更稳"）。**行归一化（Eq-R）**进一步消除行间 RMS 差异，避免 tied embedding 传回的梯度把某些行放大。
- **为什么 Muon 需要单独配 AdamW：** Muon 不适用于 1-D 参数和 embedding（其正交化假设是矩阵的奇异值谱）。因此三优化器拆分是必需而非可选。
- **证据：** `2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6`（`dexhunter`）= 1.0912；`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`（`Tianhao Wu@signalrush`）= 1.12278022。

#### (3) 架构：11L × 512d U-Net + 深度递归 + 并行残差

- **物理规模：** 11 层、512 维、8 头、4 个 KV 头（GQA），MLP 4×，LeakyReLU(0.5)² ，partial RoPE（64 维头里只转前 16 维），逐层输出 scale（LN-scale），tied embedding，logit softcap = 30。参数量 3.59e7（实测打印 `model_params:35944536`）。
- **3-Layer Depth Recurrence：** 把物理层 3/4/5 复用一次，执行序变为
  - encoder `[0,1,2,3,4,5,3,4]`，decoder `[5,3,4,5,6,7,8,9,10]`
  - 11 个物理层 → **17 个虚拟层**，参数量不变，有效深度 +55%。
  - **渐进启用（`enable_looping_at=0.35`）**：前 35% wallclock 用不递归的轻量序，省时间给后期；到点后打开 loop。副作用是 tok/s 从 ≈7.7M 掉到 ≈6.1M，与 SOTA 日志实测一致。
- **Parallel Residuals（L7+）：** GPT-J 风格，第 7 层起 attention 与 MLP 读取**同一个** pre-residual 输入，而不是串行相加。好处是后期层形成两条并行支路，梯度路径更短、更深。
- **QK-Gain 5.25：** 每个 head 一个可学习 query 缩放，初值 5.25。作用是把 attention logits 的尺度推到 QK 低精度友好的区间，同时给 softmax 一个更尖的初始分布（对短训练有利）。
- **Skip gates：** U-Net 的 skip 连接带 sigmoid 门（`skip_gate` 初始化为 0 → sigmoid=0.5），训练中学到"哪些 skip 有用"。
- **不选 XSA 加速/线性 attention：** 已在草案中说明（repo 内 33 份 track_10min_16mb 提交中该路线无一进入前 10）。

#### (4) 量化：GPTQ + SDClip，位宽按敏感度分配

- 矩阵（`c_q/c_k/c_v/proj/fc/mlp.proj`）：**int6**，逐行 clip = `12.85 × std(row)`（SDClip：把 clip 门槛设为行标准差的 k 倍，是 rate-distortion 意义下的近似最优）。
- `tok_emb`：**int8**，clip = `20.0 × std`。embedding 对量化误差最敏感，给更高比特。
- passthrough（fp16）：`q_gain`、`attn_scale`、`mlp_scale`、`resid_mix`、`skip_weights`、`skip_gates` —— 全是 1-D 小张量，总量 < 200 KB。
- 压缩：byte-shuffle 后 **Brotli quality=11, lgwin=24**。
- **打包格式实测（SOTA 参考）：** 量化前 135,431,033 B → 量化+brotli 15,975,300 B → 加代码 16,630 B → 合计 15,991,930 B。

#### (5) 评估：Sliding-Window + Legal Score-First TTT

- **Sliding window：** `eval_seq_len=2048`，`stride=64`。每条窗口只从**前缀**打分（严格因果），同一 token 会被多个窗口覆盖。
- **TTT（Test-Time Training，合法版）：** 满足 Issue #1017 Track B 四条：
  1. **Causality** —— 打分只用前缀；
  2. **Normalized distribution** —— 全词表标准 softmax，无 n-gram cache / logit bias；
  3. **Score before update** —— 每个 32K-token chunk **先**在 `no_grad` 下完整打分，**再** SGD 更新；
  4. **Single pass** —— 每个 token 只被打分一次，不重打分、不多轮挑选。
- TTT 超参：SGD `lr=0.005`、`momentum=0.9`、`epochs=3`、`chunk=32768 token`、`grad_clip=1.0`，hash embedding 16384 bucket。
- **为什么不用 LoRA-TTT：** repo 内 `2026-03-17_LoRA_TTT`（`sam@samacqua`）= 1.1929，显著劣于同期的 score-first 全参 TTT 路线。

---

## 3. 实验矩阵

### 3.1 阶段划分与规模

| 阶段 | 硬件 | 组数 × seed | 模型缩放 | 用途 |
|---|---|---|---|---|
| B0 | 1×A100 | 2 × 1 | 9L/512d/1024vocab，1500 step | 基线对齐、pipeline 正确性 |
| B1 | 1×A100 | 8 × 2 | 9L/384d/8192vocab，2000 step | **单组件**消融 |
| B2 | 1×A100 | 8 × 2 | 9L/384d/8192vocab，2000 step | **组合 / 交互项** |
| B3 | 8×H100 | 1 配置 × 3 seed | 11L/512d/8192vocab，4550 step | 最终提交 |

### 3.2 B1 单组件消融（8 组）

| ID | 变体 | 相对参考配置的改动 | 单变量 |
|---|---|---|---|
| A0 | 参考配置 | lenovo-PG v1 | — |
| A1 | − SP8192 | vocab 8192 → 1024 | tokenizer |
| A2 | − MuonEq-R | Muon → AdamW（同 lr） | 优化器 |
| A3 | − Depth Recurrence | `enable_looping_at=∞`（Loop 永不开启） | 架构 |
| A4 | − Parallel Residuals | `parallel_residual_start=99`（全部串行） | 架构 |
| A5 | − QK-Gain | `qk_gain_init=1.0` | 注意力 |
| A6 | − GPTQ int6 | 矩阵 int6 → int8 | 量化 |
| A7 | − Legal TTT | `ttt_enabled=0` | 评估 |
| A8 | − EMA | `ema_decay=1.0` | 训练调度 |

### 3.3 B2 组合 / 交互（8 组）

重点回答"组合是否互相抵消"：

| ID | 组合 | 关心的交互 |
|---|---|---|
| C1 | SP8192 + Depth Recurrence | 大词表 × 深递归是否争夺同一份"有效容量" |
| C2 | SP8192 + int6 GPTQ | 词表变大后量化误差是否被放大 |
| C3 | Muon + WD 0.095 | 正交化更新与高 WD 是否相互抵消 |
| C4 | Depth Recurrence + Parallel Residuals | 二者都增加"有效深度"，是否冗余 |
| C5 | QK-Gain 5.25 + partial RoPE(16) | 两者都改 attention 尺度/位置编码，是否冲突 |
| C6 | EMA + TTT | TTT 在 EMA 权重上是否仍有增益 |
| C7 | Sliding Window + TTT | 两个评估杠杆是否可叠加 |
| C8 | 全组合（= final） | 总增益 vs 单组件之和 |

### 3.4 交互项定义

对任意两个组件 A、B：

```
Delta(A)   = BPB(A0) - BPB(A only)
Delta(B)   = BPB(A0) - BPB(B only)
Delta(A,B) = BPB(A0) - BPB(A+B)
Interaction(A,B) = Delta(A,B) - Delta(A) - Delta(B)
    > 0  → 正协同（值得保留）
    < 0  → 互相抵消（必须取舍）
```

### 3.5 最终提交配置（B3，唯一进 8×H100 的配置）

```python
NUM_LAYERS=11           MODEL_DIM=512          NUM_HEADS=8        NUM_KV_HEADS=4
MLP_MULT=4.0            VOCAB_SIZE=8192        TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
LOOP_START=3            LOOP_END=5             ENABLE_LOOPING_AT=0.35
PARALLEL_RESIDUAL_START=7                      QK_GAIN_INIT=5.25  ROPE_DIMS=16
MUON_WD=0.095           MATRIX_LR=0.022        TIED_EMBED_LR=0.03 EMBED_WD=0.085
EMA_DECAY=0.9965        WARMDOWN_FRAC=0.72     GRAD_CLIP_NORM=0.3
MATRIX_BITS=6           MATRIX_CLIP_SIGMAS=12.85  EMBED_BITS=8   EMBED_CLIP_SIGMAS=20.0
EVAL_SEQ_LEN=2048       EVAL_STRIDE=64         TTT_LR=0.005       TTT_EPOCHS=3
TTT_CHUNK_TOKENS=32768  SEED=0/42/1234      MAX_WALLCLOCK_SECONDS=600
```

---

## 4. 统计显著性方案

### 4.1 样本与估计量

- 每个配置独立训练 **≥3 个 seed**（最终配置为 **0 / 42 / 1234**）。
- 报告 `mean ± std`（**样本标准差**，`ddof=1`），并同时给出 `stderr = std / sqrt(n)`。
- 例：最终配置 `mean = 1.08100`，`std = 0.00020`，`n = 3` → `stderr = 0.00012`。

### 4.2 假设检验

- **对基线（1.2243657）：** 配对差 ≈ 0.143（目标 1.0810），以 `stderr ≈ 0.00012`（SOTA 实测 std 0.00020 / √3）计，`t ≈ 1430`，远超 `t(0.01, df=2) = 6.965` → **p < 0.01**，显著。
- **对消融变体：** 每个变体与 full 配置做**双样本 t 检验**（同 seed 集合、配对设计）。只有 `|Δ| > 3 × pooled_stderr` 才认定为"真实贡献"，否则记为"落在噪声内"。
- **阈值设定（关键）：** 以 `stderr ≈ 0.0002` 为噪声地板，把判定阈值设为 **Δ > 0.0006（3σ）**。任何 Δ < 0.0006 的组件在消融表中明确标注"不作为独立贡献计入"，以对抗过拟合噪声。

### 4.3 样本量不足的显式声明

`n = 3` 的自由度只有 2，正态近似不稳，因此：
1. 一律使用 **t 分布**临界值（df=2 → 99% 临界值 6.965），不使用 z 值；
2. 报告中同时给出原始 3 个 seed 的**逐点数值**（不隐藏离散度）；
3. 对 Δ < 0.002 的组件结论一律附加"**在 n=3 下处于置信边界**"的标注。

---

## 5. 风险与预算

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 量化后 BPB 崩 | 中 | 高 | 位宽回退阶梯：int6→int8→只量化 MLP proj |
| 超 16 MB | 低（SOTA 三 seed 均 <16MB） | 致命 | 训练末尾自动检查 + 优先降 passthrough 精度 |
| 10 分钟超时 | 中 | 致命 | 硬 cap 600 s + `enable_looping_at` 渐进递归 + 20% buffer |
| 组合负交互 | 高 | 中 | B2 交互项排序，删除负交互组件 |
| A100 与 H100 结论不一致 | 中 | 中 | A100 只用于**相对排序**，绝对阈值全部在 H100 上重测 |

**预算：** A100 约 10 小时（≈$10）+ 8×H100 三次（≈$12）+ buffer $3 ≈ **$25**（Level 1 算力券额度内）。
*（内容由AI生成，仅供参考）*
