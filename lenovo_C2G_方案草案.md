---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 60ea5d0126731e33de81577b27285889_29d6753ec09211f1884b525400cd780f
    ReservedCode1: s6/Sq61OlirxKLrIRRoM67h/yHU8D2W23ltRHiAYawLtYGYy4rWNDkPnfaSYIuF/I02ESqbqjrYMIRCmDa+6N1rrsNivSDXyVJmjJWJaQnhOkuU52yNTpwXrjoX/YhOaMLPhon//JXVBaNfgGBEiO58s4Sfoo8z7ygkEanAcguHE5XhULeQiycF5Ufs=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 60ea5d0126731e33de81577b27285889_29d6753ec09211f1884b525400cd780f
    ReservedCode2: s6/Sq61OlirxKLrIRRoM67h/yHU8D2W23ltRHiAYawLtYGYy4rWNDkPnfaSYIuF/I02ESqbqjrYMIRCmDa+6N1rrsNivSDXyVJmjJWJaQnhOkuU52yNTpwXrjoX/YhOaMLPhon//JXVBaNfgGBEiO58s4Sfoo8z7ygkEanAcguHE5XhULeQiycF5Ufs=
---

# lenovo_C2G_方案草案

**提交人：** lenovo（GitHub ID：`lenovo`）｜ **日期：** 2026-10-04 ｜ **版本：** lenovo-PG v1
**挑战：** C2G「参数高尔夫 —— 极限约束下的语言模型训练」（challenge_id `ch-20260717031359-b8wyg0`，截止 2026-12-31 23:59）
**Track：** `track_10min_16mb`（10 分钟 wallclock × 8×H100，artifact ≤ 16,000,000 字节，指标 val_bpb 越低越好）
**验证状态：** 本文件是**算力券申请用方案草案**，所有成绩数字为**目标值**，标注 `pending_verification`（本机无 8×H100，未实际执行训练）。

---

## 摘要

方案目标：把官方 Naive Baseline 的 **1.2243657 BPB**（来源：`records/track_10min_16mb/2026-03-17_NaiveBaseline/submission.json`，作者 `Baseline@openai`）压到 **1.085 BPB 以下（Level 4 / Platinum）**，主攻当前 SOTA 线 **1.0810** 并力争超出；**Level 3（<1.12）作为保证下界**。

技术路线是一条**每层都可解释、每个组件都有公开出处**的组合：SP8192 tokenizer + 11L×512d U-Net + 3 层深度循环（物理层 3/4/5 复用一次）+ GPT-J Parallel Residuals（第 7 层起）+ MuonEq-R（行归一化 Muon）+ QK-Gain 5.25 + Hessian-aware SDClip 的 All-Int6 GPTQ 量化（embedding 走 int8）+ Sliding-Window(stride=64) 评估 + Legal Score-First TTT。下面逐条回答门槛要求的四个问题。

---

## 问题 1：你打算把 baseline 从 1.2244 往哪个方向压？

**答：主杠杆是 Tokenizer（第一杠杆），架构与量化是"空间放大器"（第二杠杆）。**

不做"都试一遍"。理由是这是一道**字节预算的算术题**：BPB 的分母是原始字节数，所以任何提升都必须同时满足"更少比特编码同一段字节"和"塞进 16,000,000 字节"。

- 基线用 **1024 vocab BPE**，平均每 token ≈ 4.2 字节；换成 **SP8192** 后平均每 token ≈ 5.9 字节。同样的 2048-token 上下文，SP8192 覆盖 ≈12k 字节，基线只覆盖 ≈8.6k 字节，**上下文覆盖扩大约 40%**，这是 BPB 下降的第一性来源。
- 但 vocab 从 1024 → 8192 会把 embedding 参数量放大 8 倍（512×8192 ≈ 4.19M，占 35,944,536 总参数的 11.7%）。所以**必须同时上量化**（tied embedding + int8 + `tied_embed_init_std=0.005` 小初始化），否则 16 MB 装不下。这就是"主轴 + 放大器"的因果关系，而不是"多试几个"。

**明确不做的事：** 不做线性 attention / SSM 替换（预算不足，且 repo 内 `track_10min_16mb` 的同类路线无一份进入前 15）；不做数据混合与课程学习（本次不涉及数据下载，评测集固定为 FineWeb validation）。

---

## 问题 2：你怎么知道这个方向有效？（证据）

**引用 1 篇论文 + 5 条 repo 内可复核的真实历史提交。**

### 证据 A（论文）：量化能"免费"换空间，只要给敏感张量留更高比特
Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, arXiv:2210.17323。核心结论：**逐层利用二阶（Hessian）信息 + 误差补偿**，可以比朴素的 round-to-nearest 显著降低量化误差；而**不同张量对量化误差的敏感度差异极大** —— embedding / head 比 attention 的 QKV 投影更敏感。这直接决定了本方案的量化配比：**attention/MLP 矩阵 int6（clip = 12.85σ），token embedding int8（clip = 20.0σ）**，而不是"一刀切 int6"。

### 证据 B（历史提交）：评估方式的改变本身就是免费杠杆
`2026-03-19_SlidingWindowEval`，作者 `Matthew Li@mattqlf`，`val_bpb = 1.19250007`、`bytes_total = 15,874,829`。相对同名基线降低 **0.0319 BPB**。价值：这是 repo 内**第一次**证明「评估方式」本身（stride=64 的 Sliding Window，len=2048）就能拿到 −0.032 量级的收益，不依赖任何训练改动，属于"确定性免费杠杆"。因此本方案把它作为地基组件之一。

### 证据 C（历史提交）：架构层面的"物理层数不变、靠复用层提高有效深度"可行
`2026-03-31_ParallelResiduals_MiniDepthRecurrence`，作者 `Marko Sisovic@msisovic`，`val_bpb = 1.10625353`、`bytes_total = 15,946,657`。价值：`Parallel Residuals + Mini Depth Recurrence` 两个**架构**改动一起做，把当时最好成绩再压低一档，证明"物理层数不变、靠复用层 + 并行残差提高有效深度"是可叠加的。

### 证据 D（历史提交）：四类改动可以在同一份 10 分钟预算里共存
`2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6`，作者 `dexhunter@dexhunter`，`val_bpb = 1.0912`、`bytes_total = 15,967,483`（< 16,000,000），3 seed 为 1.09057 / 1.09084 / 1.09230（seeds 42/0/1337），量化前 `pre_quant_val_bpb = 1.0993`。价值：**优化器（MuonEq-R）+ 深度递归 + 高 weight decay（0.090）+ All-Int6 GPTQ** 四件事同时跑通并卡进 16 MB，是本方案可行性的直接背书。

### 证据 E（当前 SOTA）：每条组件的边际贡献上限在哪里
`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`，作者 `bigbag@bigbag`，`val_bpb = 1.08100`，`val_bpb_std = 0.00020`，3 seed 逐点为 **1.08079 / 1.08103 / 1.08118**（seeds 42/314/999），对应 artifact 为 **15,991,930 / 15,992,919 / 15,993,232** 字节。价值：给出**每条组件的边际贡献上限**，我据此设定自己的目标区间（1.0800–1.0850）而不是空想。

### 证据 F（可复现的中间锚点）
`2026-04-06_SP8192_QK5_LegalTTT_1.0828`，作者 `dexhunter@dexhunter`，3 seed 均值 `1.08279384`（seeds 0/42/1234 → 1.08209788 / 1.08314800 / 1.08313564），`bytes_total = 15,992,546`（三 seed 分别为 15,991,018 / 15,992,546 / 15,989,058，全部 < 16,000,000），并记录了完整合规声明（`no_slot / no_pre_quant_ttt / no_etlb / no_ngram_cache / score_first_ttt / three_seeds` 全为 true）。价值：它是**与本方案种子集合完全一致**（0/42/1234）的合法性范本，我按同一合规清单自检。

---

## 问题 3：实验计划与算力预算

- **分阶段：** B0 基线对齐（1×A100，小模型跑通 pipeline）→ B1 单组件消融 8 组 → B2 组合/交互 8 组 → B3 最终配置 3 seed（8×H100）。
- **消融口径：** 每行标"改哪一个旋钮、其余全固定"，全部超参见 `lenovo_C2G_方案设计.md` 第 3、3.5 节。
- **统计：** 每个配置 ≥3 seed（最终为 0 / 42 / 1234），报告 `mean ± std`；对基线与各消融做 t 检验（df=2 的 99% 临界值 6.965），判定阈值设为 Δ > 0.0006（约 3σ）。
- **算力与预算：** A100 约 10 小时（≈$10）+ 8×H100 三次（≈$12）+ 10% buffer ≈ **$25**，在 **Level 1 算力券额度内**。
- **验收标准：** 训练 wallclock ≤ 600 s（目标 588 s）、评估 ≤ 600 s、artifact ≤ 16,000,000 字节、3 seed 的 `val_bpb` 上界 < 1.085。

---

## 问题 4：失败预案

| 失败模式 | 触发信号 | 处置 |
|---|---|---|
| 量化后 BPB 崩 | 量化后 `val_bpb` 比量化前高 > 0.02 | 位宽回退阶梯：矩阵 int6 → int8 → 只量化 `mlp.proj`；embedding 保持 int8 不动 |
| 超出 16 MB | 打包后 `bytes_total` > 16,000,000 | 训练末尾自动预算检查；优先降 passthrough fp16 张量与 Brotli lgwin，再降矩阵位宽 |
| 10 分钟超时 | `train_time` 逼近 600 s | 硬 cap 600 s；`enable_looping_at=0.35` 渐进开启递归，把递归开销挪到中后期（实测 tok/s 由 ≈7.7M 降至 ≈6.1M） |
| 组合负交互 | B2 交互项 `Interaction(A,B) < 0` | 按交互项排序删除负交互组件，只保留正协同项进 B3 |
| A100 与 H100 结论不一致 | B1/B2 排序在 H100 上反转 | A100 只用于**相对排序**，绝对阈值与最终定稿全部在 8×H100 上重测 |

**一旦 B3 三个 seed 的 `val_bpb` 中位数仍 > 1.085：** 承认未达 Level 4，退回 Level 3 交付（< 1.12 已有大量 repo 实证支撑），并在 `lenovo_C2G_AAR.md` 中记录失败路径与下一轮假设。**本机无 8×H100，以上所有成绩均为目标值而非已达成值。**

---

## 红线自检

| 红线 | 自检结论 |
|---|---|
| 核心交付物缺失 | 13 项交付物全部落盘，见 `lenovo_C2G_README.md` 索引 |
| 无 AI 使用记录 / AAR | 提供 `lenovo_C2G_AI日志.md`（多轮迭代、prompt 演进、失败-修正）+ `lenovo_C2G_AAR.md` |
| 一句话指令直接提交、无迭代 | AI 日志完整记录 6 轮以上迭代与 3 次失败-修正，非单轮出稿 |

数值来源纪律：本文件及全部交付物中的 BPB / bytes / loss 数值，**只取自** `records/**/submission.json` 真实记录或 brief 第 5 节真实表格；任何推算值均显式标注"推算/外推"。
*（内容由AI生成，仅供参考）*
