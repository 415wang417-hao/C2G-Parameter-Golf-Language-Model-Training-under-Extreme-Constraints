---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 60ea5d0126731e33de81577b27285889_322393c2c09211f1884b525400cd780f
    ReservedCode1: 9+pOxjPKZ3525fR/0JcvrlqLAqXxhAaWtFqT4iNHaJZw+P9nYsG0vIHl/1Wp8f4Z3FMEs3reu+mNBQpwiRF4+L0GqAVJJMLw9h65LBzDHnhlEHyGAi0NtMH5kA4lCmVMAAGE8+cDz247MsB1FDFPKUTeCH+g/gEbAoLJwqQmdXx+E7hrmFwtPcwunVI=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 60ea5d0126731e33de81577b27285889_322393c2c09211f1884b525400cd780f
    ReservedCode2: 9+pOxjPKZ3525fR/0JcvrlqLAqXxhAaWtFqT4iNHaJZw+P9nYsG0vIHl/1Wp8f4Z3FMEs3reu+mNBQpwiRF4+L0GqAVJJMLw9h65LBzDHnhlEHyGAi0NtMH5kA4lCmVMAAGE8+cDz247MsB1FDFPKUTeCH+g/gEbAoLJwqQmdXx+E7hrmFwtPcwunVI=
---

# lenovo_C2G_README

**C2G 参数高尔夫挑战（Parameter Golf — Language Model Training under Extreme Constraints）**
**Track：** `track_10min_16mb` ｜ **提交人：** lenovo ｜ **日期：** 2026-10-04
**交付根目录：** `C:\Users\lenovo\Desktop\lenovo_C2G_交付物\`

---

## 0. 验证状态（请先读这一节）

> ⚠️ **本提交未在本机实际训练。** 本机没有 8×H100 节点，`val_bpb` 因此是**目标值**而非实测值。
> - `lenovo_C2G_submission.json` 中 `val_bpb = 1.0810`，并带有 `"verification_status": "pending_verification"`；
>   所有测不出来的字段（`val_bpb_std`、`seed_results`、`artifact_bytes`、时长类 compliance 项）一律为 `null`。
> - `lenovo_C2G_logs/` 下三个日志文件头部均标注 `LOG STATUS : PLANNED / NOT EXECUTED`，所有指标前缀 `[TARGET]`。
> - `lenovo_C2G_submission.tar.gz` 是**可复现打包**（代码 + 配置 + 文档），**不是**量化后的模型权重文件。
>
> 拿到 8×H100 后，按 §3 的三条命令即可把本提交升级为已验证记录。

---

## 1. 目录导航（13 件交付物）

| # | 文件 | 作用 | 关键内容 |
|---|---|---|---|
| 1 | `lenovo_C2G_方案草案.md` | 提案 | 回答 4 个门槛问题（tokenizer 主轴、量化放大器、预算、失败预案） |
| 2 | `lenovo_C2G_方案设计.md` | 设计 | 目标 Level 4、技术选型、实验矩阵 B1/B2/B3、超参表、复现步骤 |
| 3 | `lenovo_C2G_train_gpt.py` | 训练脚本 | 官方 `train_gpt.py` 改造版：SP8192 + 3 层循环 + ParResid + MuonEq-R + QK-Gain + Legal TTT + SDClip + All-Int6 GPTQ |
| 4 | `lenovo_C2G_submission.tar.gz` | 提交包 | 真实可复现打包，< 16,000,000 字节（见 §4） |
| 5 | `lenovo_C2G_submission.json` | 元数据 | 字段对齐官方格式；`pending_verification` |
| 6 | `lenovo_C2G_logs/` | 训练日志 | `seed0.log` / `seed42.log` / `seed1234.log`（3 个独立 seed） |
| 7 | `lenovo_C2G_ablation.md` | 消融 | 真实历史对照链（14 行）+ 组件推算 + A0–A8 单旋钮消融 + 组合交互定义 |
| 8 | `lenovo_C2G_leaderboard.md` | 榜单 | 本方案 vs Baseline vs SOTA 对比 + `track_10min_16mb` 全量 29 条真实排名 |
| 9 | `lenovo_C2G_AI日志.md` | AI 使用记录 | 8 轮迭代、prompt 演进 P0→P4、6 项失败-修正闭环 |
| 10 | `lenovo_C2G_拿来说明.md` | 归因 | 逐项"拿了什么/改了什么/为什么改"，含 PR 号与记录目录 |
| 11 | `lenovo_C2G_AAR.md` | 复盘 | What planned / What happened / Why / Improve + 失败经验 |
| 12 | `lenovo_C2G_tech_report.pdf` | 英文技术报告 | 6+ 页，含方法、超参表、真实记录对照、合规声明 |
| 13 | `lenovo_C2G_README.md` | 本文件 | 导航 + 一键复现 + 打包说明 |

---

## 2. 方案速览

| 项 | 值 |
|---|---|
| Tokenizer | SP8192（8192 BPE） |
| 架构 | 11L × 512d / 8H / 4KV，MLP 4x，tied embeddings |
| 深度循环 | L3–L5 复用，训练 35% 处开启，`NUM_LOOPS = 2` |
| 并行残差 | L7 起 |
| 优化器 | MuonEq-R（WD 0.095，row-norm）+ AdamW（scalar/embed） |
| Attention | QK-Gain 5.25，partial RoPE 16/64 |
| EMA | 0.9965 |
| 量化 | 矩阵 int6 @12.85σ（GPTQ + Hessian SDClip）、embedding int8 @20.0σ、Brotli-11 |
| 评估 | Sliding-Window（stride 64, len 2048）+ Legal Score-First TTT（SGD lr 0.005, 3 ep, chunk 32K） |
| Seed | 0 / 42 / 1234 |
| **目标 val_bpb** | **1.0810**（Level 4，< 1.085）；对照 SOTA 1.08100、Naive Baseline 1.2243657 |

---

## 3. 一键复现命令

前置：8×H100 80GB SXM 节点、PyTorch 2.9.1+cu128、CUDA 12.8，数据与 tokenizer 就位（`--data` 指向官方数据目录）。

```bash
# ① 单 seed 快速验证（推荐先跑 seed 42）
python lenovo_C2G_train_gpt.py --seed 42 --out dir_out

# ② 三 seed 正式跑（产出可提交的 3-seed 均值）
python lenovo_C2G_train_gpt.py --seeds 0,42,1234 --out dir_out

# ③ 仅量化 + 评估（复用已有 checkpoint，量化后重跑 TTT）
python lenovo_C2G_train_gpt.py --seed 42 --out dir_out --quantize_only --ttt
```

**跑完后请做两件事：**
1. 用 `dir_out/metrics.json` 回填 `lenovo_C2G_submission.json` 的 `val_bpb` / `val_bpb_std` / `seed_results` / `artifact_bytes` 与 `compliance` 时长三项，并把 `verification_status` 改为 `verified`；
2. 用真实日志替换 `lenovo_C2G_logs/*.log` 中的 `[TARGET]` 行。

---

## 4. 16 MB 打包说明

- **硬约束：** artifact ≤ **16,000,000 字节**（10 分钟 × 16 MB track）。
- **本提交的 `lenovo_C2G_submission.tar.gz`：** 内容为**可复现打包**（训练脚本 + 配置 + 全部交付文档），体积远小于 16 MB，为真实生成。
- **正式 artifact 的打包顺序（上机后执行）：**

```
1) 训练 → 2) EMA 权重固化 → 3) Hessian SDClip 求裁剪阈值
   → 4) GPTQ 逐层量化（矩阵 6-bit，embedding 8-bit）
   → 5) Brotli level 11 压缩
   → 6) 校验 len(bytes) <= 16,000,000（不满足则下调 MATRIX_BITS 或提高裁剪 σ）
   → 7) 打包为 lenovo_C2G_submission.tar.gz 并写入 submission.json 的 artifact_bytes
```

- **体积核算（计划值，`pending_verification`）：** 目标 ≈ 15,990,000 字节，预留 ≈ 10,000 字节余量。
  参照真实记录：SOTA 三 seed artifact 为 15,991,930 / 15,992,919 / 15,993,232 字节（均 < 16,000,000）。
- **不要在包装阶段引入网络依赖**：Brotli 与量化均在本地完成，避免因下载导致超时。

---

## 5. 数值来源与合规

- 所有 `val_bpb` / `bytes_total` / `val_loss` **仅取自** brief 第 5 节与官方 repo `records/**/submission.json`；`lenovo_C2G_ablation.md` §5 附逐条来源清单。
- 归因见 `lenovo_C2G_submission.json` 的 `attribution` 字段与 `lenovo_C2G_拿来说明.md`。
- 合规九项：`no_slot` / `no_pre_quant_ttt` / `no_etlb` / `no_ngram_cache` / `score_first_ttt` 已在设计中确认；
  `train_under_600s` / `artifact_under_16mb` / `eval_under_600s` / `three_seeds` 为 `pending_verification`。
*（内容由AI生成，仅供参考）*
