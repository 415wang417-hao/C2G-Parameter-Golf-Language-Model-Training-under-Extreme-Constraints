---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 60ea5d0126731e33de81577b27285889_2cf97bc5c09211f1884b525400cd780f
    ReservedCode1: 1ZmsmO4RUN3zGSMBclhv3wEIr6ldSo98q1qpukJpzIlbX+TKa2wH7aFds4yZVy61Ui+mmx4NLDHHMVSSoVsOU87T98Vmtxl4zlOvNLnhKgMOj5+yAVCocUspvSBcT7AIfppCX5RhSDgA54pWRdiWFTGP7pXzE37zhBBTfXiFt06h09u7d3lVcumv9BA=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 60ea5d0126731e33de81577b27285889_2cf97bc5c09211f1884b525400cd780f
    ReservedCode2: 1ZmsmO4RUN3zGSMBclhv3wEIr6ldSo98q1qpukJpzIlbX+TKa2wH7aFds4yZVy61Ui+mmx4NLDHHMVSSoVsOU87T98Vmtxl4zlOvNLnhKgMOj5+yAVCocUspvSBcT7AIfppCX5RhSDgA54pWRdiWFTGP7pXzE37zhBBTfXiFt06h09u7d3lVcumv9BA=
---

# lenovo_C2G_leaderboard

**提交人：** lenovo ｜ **日期：** 2026-10-04 ｜ **Track：** `track_10min_16mb`（10 分钟 × 8×H100 × ≤16,000,000 字节）
**数值来源：** 下表所有历史提交数值**逐条取自** `records/track_10min_16mb/*/submission.json`，未做任何插值或编造。
**诚实性声明：** 本方案（lenovo-PG v1）**未在本机实际训练**（本机无 8×H100），其行内数值为**目标值**，标记 `pending_verification`。

---

## 1. 三方对比（本方案 vs Baseline vs SOTA）

| 项 | Naive Baseline | **lenovo-PG v1（本方案）** | 当前 SOTA |
|---|---|---|---|
| 作者 / GitHub ID | Baseline / `openai` | lenovo / `lenovo` | bigbag / `bigbag` |
| 目录 | `2026-03-17_NaiveBaseline` | （待训练产出） | `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` |
| **val_bpb（mean）** | 1.2243657 | **≤ 1.0850（目标 1.0810）** `pending_verification` | **1.08100** |
| val_bpb std | — | 目标 ≤ 0.0005（参照 0.00020） | 0.00020 |
| 逐 seed val_bpb | — | 待验证（seeds 0 / 42 / 1234） | 1.08079 / 1.08103 / 1.08118（seeds 42 / 314 / 999） |
| bytes_total（artifact） | 15,863,489 | 预计 15,990,000 ± 5,000（上限硬约束 16,000,000） | 15,991,930 / 15,992,919 / 15,993,232 |
| bytes_code（代码部分） | 47,642 | 约 60,000（含 TTT 与 GPTQ） | — |
| tokenizer | SP1024（1024 BPE） | **SP8192**（8192 BPE） | SP8192 |
| 架构 | 9L × 512d，KV4，int8 | **11L × 512d / 8H / 4KV + 3 层深度循环 + Parallel Residuals + QK-Gain 5.25** | 同族：11L + 3LayerRecur + ParResid + QK525 |
| 优化器 | AdamW | **MuonEq-R（WD 0.095）+ AdamW（embed/scalar）** | MuonEq-R |
| 量化 | int8 | **GPTQ + SDClip：矩阵 int6(12.85σ) / embedding int8(20.0σ) + Brotli-11** | 同族 All-Int6 GPTQ + SDClip |
| 评估 | 单窗口 | **Sliding-Window(stride 64) + Legal Score-First TTT** | 同族 Legal TTT |
| 训练 wallclock | — | 目标 588 s（≤600 s 硬约束） | 588,047 ms（≈588 s） |
| **相对 Baseline 降幅** | — | **−0.139 ~ −0.143 BPB（−11.4% ~ −11.7%）**（目标值） | −0.1434 BPB（−11.71%） |
| 目标等级 | — | **Level 4（BPB < 1.085）；Level 3（<1.12）为保证下界** | — |

---

## 2. `track_10min_16mb` 完整真实榜单（按 val_bpb 升序）

| 排名 | val_bpb | bytes_total | 作者 / GitHub ID | 日期 | 目录 |
|---|---|---|---|---|---|
| 1 | **1.08100** | — | bigbag / `bigbag` | 2026-04-09 | `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` |
| 2 | 1.08217956 | — | aryanbhosale | 2026-04-08 | `2026-04-08_SP8192_ParallelResid_ScoreFirstTTT` |
| 3 | 1.08279384 | 15,992,546 | dexhunter | 2026-04-06 | `2026-04-06_SP8192_QK5_LegalTTT_1.0828` |
| 4 | 1.08354 | 15,978,121 | Robby Sneiderman / `Robby955` | 2026-04-06 | `2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence` |
| 5 | 1.08563 | 15,985,678 | Kevin Clark / `clarkkev` | 2026-04-05 | `2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2` |
| 6 | 1.08971631 | — | aryanbhosale | 2026-04-04 | `2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR` |
| 7 | 1.0912 | 15,967,483 | dexhunter | 2026-04-03 | `2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6` |
| 8 | 1.09785 | 15,916,170 | Kevin Clark / `clarkkev` | 2026-04-01 | `2026-04-01_Vocab4096_MLPMult4_WD085` |
| 9 | 1.10625353 | 15,946,657 | Marko Sisovic / `msisovic` | 2026-03-31 | `2026-03-31_ParallelResiduals_MiniDepthRecurrence` |
| 10 | 1.11473509 | 15,984,850 | abaybektursun | 2026-03-25 | `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` |
| 11 | 1.1194 | 15,990,006 | abaybektursun | 2026-03-23 | `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` |
| 12 | 1.12278022 | 15,555,017 | Tianhao Wu / `signalrush` | 2026-03-22 | `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` |
| 13 | 1.12484502 | 15,612,308 | Jack Princz / `jfprincz` | 2026-03-21 | `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` |
| 14 | 1.12707468 | 15,534,645 | Jack Princz / `jfprincz` | 2026-03-20 | `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` |
| 15 | 1.13071416 | 15,892,986 | vadim borisov（tabularis.ai） | 2026-03-20 | `2026-03-20_11L_EfficientPartialXSA_FA3_SWA120` |
| 16 | 1.14581692 | 15,862,650 | Raahil Shah / `raahilshah` | 2026-03-20 | `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` |
| 17 | 1.15015359 | — | aruniyer | 2026-03-19 | `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` |
| 18 | 1.1556 | — | （未署名） | 2026-03-19 | `2026-03-19_smeargate_orthoinit_muonwd` |
| 19 | 1.1574404 | 15,977,717 | samuellarson | 2026-03-19 | `2026-03-19_WarmdownQuantization` |
| 20 | 1.15861696 | 15,558,319 | yahya010 | 2026-03-19 | `2026-03-19_Seq2048_FP16Emb_TunedLR` |
| 21 | 1.16301431 | 15,353,490 | aquariouseworkman | 2026-03-19 | `2026-03-19_MixedQuant_Int6Int8_SlidingWindow` |
| 22 | 1.19250007 | 15,874,829 | Matthew Li / `mattqlf` | 2026-03-19 | `2026-03-19_SlidingWindowEval` |
| 23 | 1.1929 | 15,882,446 | sam / `samacqua` | 2026-03-17 | `2026-03-17_LoRA_TTT` |
| 24 | 1.20143417 | 15,868,326 | Spokane Way | 2026-03-19 | `2026-03-19_TrainingOptSeq4096` |
| 25 | 1.20576485 | 15,867,270 | Spokane Way | 2026-03-18 | `2026-03-18_LongContextSeq2048` |
| 26 | 1.214745 | 15,928,974 | Nan Liu | 2026-03-19 | `2026-03-19_10L_MixedPrecision` |
| 27 | 1.21972502 | 15,896,222 | Renier Velazco | 2026-03-18 | `2026-03-18_FP16Embed_WD3600` |
| 28 | 1.22296644 | 15,854,246 | Nan Liu | 2026-03-18 | `2026-03-18_LowerLR` |
| 29 | 1.2243657 | 15,863,489 | Baseline / `openai` | 2026-03-17 | `2026-03-17_NaiveBaseline` |

> 备注：`2026-03-20_10L_Int5MLP_MuonWD04_SWA50`（thwu1）与 `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`（notapplica）的 `submission.json` 未记录 `val_bpb`，故不参与排名；`2026-03-24_*/`、`2026-03-19_int6_STE QAT_ MLP_bigram _U_Net` 目录无 `submission.json`。

---

## 3. 榜单解读（用于定位本方案的合理目标）

1. **前 5 名全部含 SP8192 + 量化 + 至少一项评估侧杠杆**，说明"大词表 + 16 MB 装得下"是进入第一梯队的**必要条件**。
2. **第 4 名（1.08354）与第 1 名（1.08100）之间只有 0.0025 BPB**，而第 1 名的 std 是 0.00020 —— 即**前四名处于同一噪声带**。因此本方案把目标定为"**进入 1.0800–1.0850 区间**"是**可辩护的**；把目标写成"1.06x"则不可辩护。
3. 官方 rule 要求"新 record 须**低于当前 SOTA ≥ 0.005 nats 且 p<0.01**"，即目标需 ≈ **1.0759** 以下才算破纪录。**本方案不宣称破纪录**，只宣称"达到 Level 4 门槛（<1.085）并与 SOTA 同档"。
4. 榜单末端（1.19–1.22）集中在"只改训练超参 / 只改评估 / 只改精度"的单点尝试上，与 §1 的 Δ 拆解一致：**单点改动的量级是 0.01–0.03，必须组合才有 0.14**。
*（内容由AI生成，仅供参考）*
