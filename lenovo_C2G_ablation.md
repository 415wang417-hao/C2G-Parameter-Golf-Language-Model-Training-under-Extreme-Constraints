---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 60ea5d0126731e33de81577b27285889_2c09eb24c09211f1884b525400cd780f
    ReservedCode1: OV7A5QmUGOgxZsZcwnnmdIBNloNbfpBrz89QgZDmWqxzqgkHFTmptuQQSs4pWyeKePCMtze9NRp5e6IgtcWKZRGNzTaaz/i+5vzc3HmxENZe6VHScxYRqLeIinmLjwxz1Ah+IAxuIlHr3jOpNP92EYy1AVop+DbeQoDm0J1b6tCSwgVynkdoNqdgmrE=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 60ea5d0126731e33de81577b27285889_2c09eb24c09211f1884b525400cd780f
    ReservedCode2: OV7A5QmUGOgxZsZcwnnmdIBNloNbfpBrz89QgZDmWqxzqgkHFTmptuQQSs4pWyeKePCMtze9NRp5e6IgtcWKZRGNzTaaz/i+5vzc3HmxENZe6VHScxYRqLeIinmLjwxz1Ah+IAxuIlHr3jOpNP92EYy1AVop+DbeQoDm0J1b6tCSwgVynkdoNqdgmrE=
---

# lenovo_C2G_ablation

**提交人：** lenovo ｜ **日期：** 2026-10-04 ｜ **对应方案：** lenovo-PG v1（见 `lenovo_C2G_方案设计.md`）
**数值来源纪律：** 本文件所有 `val_bpb` / `bytes_total` **只取自** `records/**/submission.json` 真实记录；凡属"推算/外推"的行均**逐行显式标注**。本机无 8×H100、未实际训练，因此**不存在**"本机已跑出的消融数值"。
**验证状态：** 计划中的 A0–A8 消融为 `pending_verification`；已标注为"实测"的行均来自官方 repo 真实历史提交。

---

## 1. 真实历史对照链（repo 实测，可直接复核）

下表沿"组件逐步叠加"的方向排列，每一行的改动点、数值、作者与目录均可回溯到官方 repo。**Δ 为相对上一行的差值**（负数 = 变好）。

| # | 叠加的组件 | val_bpb | Δ vs 上一行 | 作者 / GitHub ID | 目录 |
|---|---|---|---|---|---|
| 1 | Naive Baseline（SP1024 9×512 KV4 int8，官方基线） | 1.2243657 | — | Baseline / `openai` | `2026-03-17_NaiveBaseline` |
| 2 | + 训练序列长度 2048（LongContext） | 1.20576485 | **−0.0186** | Spokane Way / `spokane-way` | `2026-03-18_LongContextSeq2048` |
| 3 | + Sliding-Window Eval（stride=64, len=2048，纯评估改动） | 1.19250007 | **−0.0133** | Matthew Li / `mattqlf` | `2026-03-19_SlidingWindowEval` |
| 4 | + Warmdown Quantization（warmdown 阶段 QAT） | 1.1574404 | **−0.0351** | samuellarson / `samuellarson` | `2026-03-19_WarmdownQuantization` |
| 5 | + MLP 3x + Int6 QAT + Sliding Window | 1.15015359 | **−0.0073** | aruniyer / `aruniyer` | `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` |
| 6 | + Int6-MLP3x + SmearGate + BigramHash + Muon-WD + SWA | 1.14581692 | **−0.0043** | Raahil Shah / `raahilshah` | `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` |
| 7 | + LeakyReLU² + Legal TTT + Parallel Muon | 1.1194 | **−0.0264** | abaybektursun / `abaybektursun` | `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` |
| 8 | + Val-Calib GPTQ + XSA + BigramHash 3072 | 1.11473509 | **−0.0047** | abaybektursun / `abaybektursun` | `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` |
| 9 | + Parallel Residuals（Mini Depth Recurrence） | 1.10625353 | **−0.0085** | Marko Sisovic / `msisovic` | `2026-03-31_ParallelResiduals_MiniDepthRecurrence` |
| 10 | + Vocab 4096 + MLP 4x + WD 0.085 | 1.09785 | **−0.0084** | Kevin Clark / `clarkkev` | `2026-04-01_Vocab4096_MLPMult4_WD085` |
| 11 | + SP4096 + Depth Recurrence + Parallel Resid + MuonEq-R | 1.08971631 | **−0.0081** | aryanbhosale / `aryanbhosale` | `2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR` |
| 12 | + SP8192 + Hessian SDClip + Progressive Recurrence | 1.08354 | **−0.0062** | Robby Sneiderman / `Robby955` | `2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence` |
| 13 | + SP8192 + QK-Gain 5 + Legal Score-First TTT | 1.08279384 | **−0.0007** | dexhunter / `dexhunter` | `2026-04-06_SP8192_QK5_LegalTTT_1.0828` |
| 14 | + SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain **5.25** + Legal TTT（**当前 SOTA**） | **1.08100** | **−0.0018** | bigbag / `bigbag` | `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` |

**并行分支（非严格单变量，仅作旁证，不做因果归因）：**

| 分支方案 | val_bpb | bytes_total | 作者 | 目录 |
|---|---|---|---|---|
| MuonEq-R + Depth Recurrence + WD 0.090 + All-Int6 GPTQ（**无 TTT**，`pre_quant_val_bpb = 1.0993`） | 1.0912 | 15,967,483 | dexhunter / `dexhunter` | `2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6` |
| SP8192 + GPTQ Embeddings + SDClip + Loop(4,5)×2 | 1.08563 | 15,985,678 | Kevin Clark / `clarkkev` | `2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2` |
| SP8192 + Parallel Residuals + Score-First TTT | 1.08217956 | — | aryanbhosale / `aryanbhosale` | `2026-04-08_SP8192_ParallelResid_ScoreFirstTTT` |

> 说明：上表按"同日/邻近日期内更优者入链"排列，故个别行（如第 12 行 vs 第 13 行）在同一天有两份提交。**每行的 Δ 只表示相邻两行之间的总差值，不等于单组件贡献**——这正是本方案把消融设计为"单旋钮 + 冻结其余"的原因（见 §3）。

### 1.1 非记录（non-record）分支对照

| 方案 | val_bpb | bytes_total | 作者 | 目录 |
|---|---|---|---|---|
| 4-Hour Quasi-10B SP1024（4 小时 > 10 分钟预算） | 1.20737944 | 15,810,161 | Will DePue | `2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` |
| SwiGLU WarmdownFix QuarterBatch 1×5090 | 1.32814313 | 15,327,112 | （未署名） | `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` |
| Depth Recurrence + Mixed-Precision Quantization（仅 1.46 MB 模型） | 2.3876 | 1,461,542 | Evangeline Kamin / `evangelinehelsinki` | `2026-03-21_DepthRecurrence_MixedPrecisionQuant` |

结论：**超出 10 分钟预算的 4 小时长训（1.2074）与极低预算的 1.46 MB 模型（2.3876）都无法与本 track 竞争**，佐证"10 分钟 × 16 MB 的甜点区在 SP8192 + 深度递归 + 量化"这条主线。

---

## 2. 组件边际贡献（基于 §1 实测链拆解，**推算/外推**）

> ⚠️ 下表为**基于 §1 真实记录的算术拆解与推断**，用于设定预期收益区间，**不是本机实测结果**，全部标注"推算"。

| 组件 | 参考对比（§1 行号） | 推算单项收益（BPB） | 推算依据 |
|---|---|---|---|
| SP8192（相对 1024 BPE） | 全链 1.2243657 → 1.0810，tokenizer 为主导项 | **≈ 0.06 – 0.09** | 第 11 行 SP4096（1.08971631）与其 SP8192 姊妹方案（第 13 行 1.08279384、第 14 行 1.08100）差 ≈0.007–0.009；SP4096 相对 1024 的增益按 vocab 覆盖率外推 |
| Depth Recurrence（物理层 3/4/5 复用一次） | 第 10 → 第 11 行 | **≈ 0.008** | 实测差 −0.0081（第 11 行同时含 MuonEq-R，故为**上界估计**） |
| Parallel Residuals | 第 8 → 第 9 行 | **≈ 0.0085**（含 Mini Depth Recurrence，非纯单变量） | 实测差，含两个组件，故为**上界估计** |
| MuonEq-R | 第 11 行内部（SP4096 分支整体） | **≈ 0.01 – 0.02** | `2026-04-03`（MuonEq-R + WD 0.090，1.0912）相对同期 AdamW 分支的一致性优势 |
| QK-Gain 5.25（相对 5.0） | 第 13 → 第 14 行 | **≈ 0.001 – 0.002** | 第 14 行相对第 13 行实测 −0.0018（同时含 3 层递归与 ParResid，故为下界） |
| Legal Score-First TTT | `2026-04-03`（无 TTT，1.0912）→ 第 13 行（有 TTT，1.08279384） | **≈ 0.008** | 跨配置差 0.0084，**推算**（两者并非严格单变量对照） |
| GPTQ int6 + SDClip（embedding int8） | `2026-04-03`：`pre_quant_val_bpb = 1.0993` vs 量化后 1.0912 | **量化不增分，纯换取 ≤16 MB 空间**；在 WD 0.090 下甚至 −0.0081 | 实测（量化后优于量化前，属 WD × 量化的协同） |
| Sliding-Window Eval | 第 2 → 第 3 行 | **0.0133** | 实测差，纯评估改动 |
| 训练序列长度 2048 | 第 1 → 第 2 行 | **0.0186** | 实测差 |

---

## 3. 本方案计划的单组件消融（8 组，**pending_verification**）

设计原则：**每行只动一个旋钮，其余超参完全冻结**为 §3.5 的最终配置；每行跑 3 个 seed（0/42/1234），报告 `mean ± std`。

| ID | 变体 | 相对最终配置的**唯一**改动 | 期望方向 | 状态 |
|---|---|---|---|---|
| A0 | 最终配置 lenovo-PG v1 | — | 基准 | pending_verification |
| A1 | − SP8192 | `VOCAB_SIZE` 8192 → 1024 | 显著变差 | pending_verification |
| A2 | − MuonEq-R | 矩阵优化器 Muon → AdamW（同 lr/同 WD） | 变差 | pending_verification |
| A3 | − Depth Recurrence | `ENABLE_LOOPING_AT` → 1.0（Loop 永不开启） | 变差 | pending_verification |
| A4 | − Parallel Residuals | `PARALLEL_RESIDUAL_START` → 99（全串行） | 变差 | pending_verification |
| A5 | − QK-Gain | `QK_GAIN_INIT` 5.25 → 1.0 | 变差 | pending_verification |
| A6 | − GPTQ int6 | `MATRIX_BITS` 6 → 8 | 变差且超 16MB 风险 | pending_verification |
| A7 | − Legal TTT | `TTT_ENABLED` → 0 | 变差（≈0.008） | pending_verification |
| A8 | − EMA | `EMA_DECAY` 0.9965 → 1.0 | 变差 | pending_verification |

**判定阈值：** 以 `stderr ≈ 0.0002` 为噪声地板，组件贡献判定阈值设为 **Δ > 0.0006（≈3σ）**；任何 Δ < 0.0006 的组件在最终方案中标注"不作为独立贡献计入"，以避免把噪声当收益。

---

## 4. 组合效应与交互项（计划）

对任意两组件 A、B，定义

```
Delta(A)         = BPB(A0) - BPB(A only)
Delta(B)         = BPB(A0) - BPB(B only)
Delta(A,B)       = BPB(A0) - BPB(A+B)
Interaction(A,B) = Delta(A,B) - Delta(A) - Delta(B)
    > 0  → 正协同（保留）
    < 0  → 互相抵消（需取舍）
```

重点考察的 8 组组合（计划，pending_verification）：

| ID | 组合 | 关心的交互 |
|---|---|---|
| C1 | SP8192 + Depth Recurrence | 大词表与深递归是否争夺同一份"有效容量" |
| C2 | SP8192 + int6 GPTQ | 词表变大后 embedding 量化误差是否被放大 |
| C3 | MuonEq-R + WD 0.095 | 正交化更新与高 WD 是否相互抵消 |
| C4 | Depth Recurrence + Parallel Residuals | 二者都提高"有效深度"，是否冗余 |
| C5 | QK-Gain 5.25 + partial RoPE(16/64) | 两者都改 attention 尺度/位置编码，是否冲突 |
| C6 | EMA 0.9965 + TTT | TTT 在 EMA 权重上是否仍保留增益 |
| C7 | Sliding Window + TTT | 两个评估杠杆是否可叠加（repo 实测可叠加，见 §1 第 9/11 行） |
| C8 | 全组合（= 最终配置） | 总增益 vs 单组件增益之和 |

---

## 5. 数据来源清单（逐条可追溯）

| 引用对象 | 绝对来源 |
|---|---|
| Naive Baseline | `records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json` |
| Sliding Window Eval | `records/track_10min_16mb/2026-03-19_SlidingWindowEval/submission.json` |
| Warmdown Quantization | `records/track_10min_16mb/2026-03-19_WarmdownQuantization/submission.json` |
| Int6 MLP3x + SmearGate + BigramHash | `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/submission.json` |
| Parallel Residuals + Mini Depth Recurrence | `records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/submission.json` |
| SP4096 + Depth Recur + ParResid + MuonEq-R | `records/track_10min_16mb/2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR/submission.json` |
| MuonEq-R + Depth Recur + WD090 + AllInt6 | `records/track_10min_16mb/2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6/submission.json` |
| SP8192 + GPTQ-Emb + SDClip + Loop45x2 | `records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/submission.json` |
| SP8192 + QK5 + Legal TTT | `records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828/submission.json` |
| SP8192 + ParResid + ScoreFirstTTT | `records/track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/submission.json` |
| **当前 SOTA** | `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/submission.json` |
| 非记录分支三项 | `records/track_non_record_16mb/*/submission.json` |
*（内容由AI生成，仅供参考）*
