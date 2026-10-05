#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lenovo_C2G_train_gpt.py
=======================
C2G / OpenAI Parameter Golf Challenge -- 10 minute x 16 MB track.

Configuration: "lenovo-PG v1"
    SP8192 tokenizer + 11L x 512d U-Net (8H/4KV)
    + 3-layer Depth Recurrence (L3-5, progressive, enable_looping_at=0.35)
    + Parallel Residuals (L7+)
    + QK-Gain 5.25
    + LLaMA-style partial RoPE (16/64) + layerwise output (LN) scale
    + skip gates (sigmoid-gated U-Net connections)
    + LeakyReLU(0.5)^2 MLP (4x), tied embeddings, logit softcap 30
    + MuonEq-R (row-normalised Muon + weight decay) for matrices,
      AdamW for embedding / scalars
    + EMA 0.9965, warmdown_frac 0.72
    + GPTQ SDClip quantisation: int6 matrices (k=12.85 Sigma), int8 embeddings (k=20.0 Sigma)
      + byte-shuffle + Brotli-11
    + Sliding-window eval (stride 64, seq 2048)
    + Legal Score-First TTT (SGD lr 0.005, momentum 0.9, 3 epochs, 32K-token chunks,
      hash-embedding of the prefix)

Attribution (all borrowed, none invented -- see lenovo_C2G_拿来说明.md):
    * nanoGPT / modded-nanogpt  -- base GPT skeleton, Muon, logit softcap
    * openai/parameter-golf     -- official train scaffold (I/O, BPB eval, packing format)
    * @clarkkev  PR #1394        -- SP8192 + GPTQ embeddings + SDClip + MuonEq-R
    * @dexhunter  PR #1331/#1437 -- depth recurrence (loop layers 3-5)
    * @Robby955   PR #1412 / @msisovic PR #1204 -- parallel residuals
    * @abaybektursun PR #549 / @dexhunter PR #1413 -- score-first TTT
    * @X-Abhishek-X PR #1445     -- WD 0.095 / MLR 0.022 / EMA 0.9965

Single-command reproduction (8 x H100 SXM 80GB):

    torchrun --standalone --nproc_per_node=8 lenovo_C2G_train_gpt.py

Every hyper-parameter can be overridden from the environment, e.g.

    SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 lenovo_C2G_train_gpt.py
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

import sentencepiece as spm

# Brotli is optional at import time so that the script still runs (with LZMA) on boxes
# where the wheel is unavailable.
try:
    import brotli  # type: ignore

    _HAS_BROTLI = True
except Exception:  # pragma: no cover
    _HAS_BROTLI = False

try:
    import lzma  # stdlib fallback
except Exception:  # pragma: no cover
    lzma = None  # type: ignore


# ==========================================================================================
# 1. Hyper-parameters
# ==========================================================================================


def _envi(name: str, default):
    """Read env override with type inference from the default value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Hyperparameters:
    """lenovo-PG v1 configuration.

    Defaults == the submitted configuration (see lenovo_C2G_submission.json).
    """

    # ---- data / tokenizer ---------------------------------------------------------------
    data_dir: str = _envi("DATA_DIR", "./data/")
    datasets_dir: str = _envi("DATASETS_DIR", "./data/datasets/fineweb10B_sp8192")
    tokenizer_path: str = _envi("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    train_files: str = _envi("TRAIN_FILES", "./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin")
    val_files: str = _envi("VAL_FILES", "./data/datasets/fineweb10B_sp8192/fineweb_val_*.bin")
    vocab_size: int = _envi("VOCAB_SIZE", 8192)

    # ---- model ---------------------------------------------------------------------------
    num_layers: int = _envi("NUM_LAYERS", 11)
    model_dim: int = _envi("MODEL_DIM", 512)
    num_heads: int = _envi("NUM_HEADS", 8)
    num_kv_heads: int = _envi("NUM_KV_HEADS", 4)
    mlp_mult: float = _envi("MLP_MULT", 4.0)
    tie_embeddings: bool = _envi("TIE_EMBEDDINGS", True)
    tied_embed_init_std: float = _envi("TIED_EMBED_INIT_STD", 0.005)
    logit_softcap: float = _envi("LOGIT_SOFTCAP", 30.0)
    rope_base: float = _envi("ROPE_BASE", 10000.0)
    rope_dims: int = _envi("ROPE_DIMS", 16)          # partial RoPE: 16 of 64 head dims
    qk_gain_init: float = _envi("QK_GAIN_INIT", 5.25)
    ln_scale: bool = _envi("LN_SCALE", True)
    skip_gates_enabled: bool = _envi("SKIP_GATES_ENABLED", True)

    # ---- depth recurrence (U-Net) --------------------------------------------------------
    loop_start: int = _envi("LOOP_START", 3)
    loop_end: int = _envi("LOOP_END", 5)             # inclusive -> loops physical layers 3,4,5
    num_loops: int = _envi("NUM_LOOPS", 2)
    enable_looping_at: float = _envi("ENABLE_LOOPING_AT", 0.35)   # fraction of the run
    parallel_residual_start: int = _envi("PARALLEL_RESIDUAL_START", 7)

    # ---- attention variant ----------------------------------------------------------------
    xsa_last_n: int = _envi("XSA_LAST_N", 11)        # exclusive self-attention on last N blocks
    xsa_query_chunk: int = _envi("XSA_QUERY_CHUNK", 512)

    # ---- optimisation ----------------------------------------------------------------------
    iterations: int = _envi("ITERATIONS", 20000)
    train_batch_tokens: int = _envi("TRAIN_BATCH_TOKENS", 786432)
    train_seq_len: int = _envi("TRAIN_SEQ_LEN", 2048)
    warmup_steps: int = _envi("WARMUP_STEPS", 20)
    warmdown_frac: float = _envi("WARMDOWN_FRAC", 0.72)
    muon_momentum: float = _envi("MUON_MOMENTUM", 0.99)
    muon_momentum_warmup_start: float = _envi("MUON_MOMENTUM_WARMUP_START", 0.92)
    muon_momentum_warmup_steps: int = _envi("MUON_MOMENTUM_WARMUP_STEPS", 1500)
    muon_backend_steps: int = _envi("MUON_BACKEND_STEPS", 5)
    muon_beta2: float = _envi("MUON_BETA2", 0.95)
    muon_wd: float = _envi("MUON_WD", 0.095)
    muon_row_normalize: bool = _envi("MUON_ROW_NORMALIZE", True)
    matrix_lr: float = _envi("MATRIX_LR", 0.022)
    scalar_lr: float = _envi("SCALAR_LR", 0.02)
    embed_lr: float = _envi("EMBED_LR", 0.6)
    tied_embed_lr: float = _envi("TIED_EMBED_LR", 0.03)
    head_lr: float = _envi("HEAD_LR", 0.008)
    embed_wd: float = _envi("EMBED_WD", 0.085)
    adam_wd: float = _envi("ADAM_WD", 0.02)
    beta1: float = _envi("BETA1", 0.9)
    beta2: float = _envi("BETA2", 0.95)
    adam_eps: float = _envi("ADAM_EPS", 1e-8)
    grad_clip_norm: float = _envi("GRAD_CLIP_NORM", 0.3)
    ema_decay: float = _envi("EMA_DECAY", 0.9965)

    # ---- evaluation ------------------------------------------------------------------------
    val_batch_tokens: int = _envi("VAL_BATCH_TOKENS", 524288)
    eval_seq_len: int = _envi("EVAL_SEQ_LEN", 2048)
    eval_stride: int = _envi("EVAL_STRIDE", 64)
    sliding_window_enabled: bool = _envi("SLIDING_WINDOW_ENABLED", True)
    val_loss_every: int = _envi("VAL_LOSS_EVERY", 4000)

    # ---- quantisation ----------------------------------------------------------------------
    matrix_bits: int = _envi("MATRIX_BITS", 6)
    matrix_clip_sigmas: float = _envi("MATRIX_CLIP_SIGMAS", 12.85)
    embed_bits: int = _envi("EMBED_BITS", 8)
    embed_clip_sigmas: float = _envi("EMBED_CLIP_SIGMAS", 20.0)
    gptq_calibration_batches: int = _envi("GPTQ_CALIBRATION_BATCHES", 64)
    gptq_reserve_seconds: float = _envi("GPTQ_RESERVE_SECONDS", 12.0)
    compressor: str = _envi("COMPRESSOR", "brotli")

    # ---- test-time training ----------------------------------------------------------------
    ttt_enabled: bool = _envi("TTT_ENABLED", True)
    ttt_lr: float = _envi("TTT_LR", 0.005)
    ttt_epochs: int = _envi("TTT_EPOCHS", 3)
    ttt_momentum: float = _envi("TTT_MOMENTUM", 0.9)
    ttt_chunk_tokens: int = _envi("TTT_CHUNK_TOKENS", 32768)
    ttt_hash_embed: bool = _envi("TTT_HASH_EMBED", True)
    ttt_hash_buckets: int = _envi("TTT_HASH_BUCKETS", 16384)
    ttt_clip: float = _envi("TTT_CLIP", 1.0)

    # ---- bookkeeping -------------------------------------------------------------------------
    run_id: str = _envi("RUN_ID", os.environ.get("RUN_ID", "lenovo-pg-v1"))
    seed: int = _envi("SEED", 42)
    max_wallclock_seconds: float = _envi("MAX_WALLCLOCK_SECONDS", 600.0)
    train_log_every: int = _envi("TRAIN_LOG_EVERY", 500)
    model_path: str = _envi("MODEL_PATH", "final_model.pt")
    quantized_model_path: str = _envi("QUANTIZED_MODEL_PATH", "final_model.int6.ptz")


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weights",
    "skip_gates",
    "ln_scale",
)


# ==========================================================================================
# 2. MuonEq-R  (row-normalised Muon with decoupled weight decay)
# ==========================================================================================


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Newton-Schulz iteration approximating the orthogonal polar factor of G."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon with optional row normalisation (MuonEq-R).

    Row normalisation rescales every row of the update to unit RMS before the
    Newton-Schulz orthogonalisation.  It removes the scale asymmetry between the
    attention / MLP matrices introduced by tied embeddings and keeps the effective
    learning rate comparable across a 512-dim 11-layer stack.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        backend_steps: int = 5,
        row_normalize: bool = True,
        wd: float = 0.0,
    ):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, row_normalize=row_normalize, wd=wd),
        )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group.get("wd", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)

                if group["row_normalize"] and g.ndim == 2:
                    # row-wise RMS normalisation (MuonEq-R)
                    rms = g.float().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
                    g = (g.float() / rms).to(g.dtype)

                u = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                # aspect-ratio scaling (same convention as the official scaffold)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u.to(p.dtype), alpha=-lr * scale)
        return loss


# ==========================================================================================
# 3. Tokenizer utilities + BPB LUTs
# ==========================================================================================


def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    """Per-token byte length / leading-space / boundary LUTs used by the BPB evaluator."""
    base_bytes = np.zeros(vocab_size, dtype=np.float64)
    has_leading_space = np.zeros(vocab_size, dtype=np.float64)
    is_boundary = np.zeros(vocab_size, dtype=np.float64)

    for tid in range(vocab_size):
        try:
            piece = sp.IdToPiece(tid)
        except Exception:
            piece = ""
        s = piece.replace("\u2581", " ")
        # The UTF-8 length of the piece *is* its byte contribution.
        base_bytes[tid] = len(s.encode("utf-8"))
        has_leading_space[tid] = 1.0 if piece.startswith("\u2581") else 0.0
        is_boundary[tid] = 1.0 if (piece.startswith("\u2581") or (len(s) > 0 and not s[0].isalnum())) else 0.0

    return (
        torch.tensor(base_bytes, dtype=torch.float64, device=device),
        torch.tensor(has_leading_space, dtype=torch.float64, device=device),
        torch.tensor(is_boundary, dtype=torch.float64, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(Path().glob(pattern)) if "*" in pattern else [Path(pattern)]
    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"no validation shards matched {pattern}")
    tokens = np.concatenate([np.fromfile(str(f), dtype=np.uint16).astype(np.int64) for f in files])
    return torch.from_numpy(tokens)


class DistributedTokenLoader:
    """Round-robin shard reader producing (x, y) token batches."""

    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        files = sorted(Path().glob(pattern)) if "*" in pattern else [Path(pattern)]
        self.files = [f for f in files if f.exists()]
        if not self.files:
            raise FileNotFoundError(f"no training shards matched {pattern}")
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.shards = [np.fromfile(str(f), dtype=np.uint16).astype(np.int64) for f in self.files]
        self.shard_idx = rank % len(self.shards)
        self.pos = 0

    def _next_shard(self):
        self.shard_idx = (self.shard_idx + 1) % len(self.shards)
        self.pos = 0

    def next_batch(self, batch_tokens: int, seq_len: int, grad_accum_steps: int):
        per_rank = batch_tokens // self.world_size
        seqs = per_rank // seq_len
        need = per_rank + 1
        buf = self.shards[self.shard_idx]
        if self.pos + need >= len(buf):
            self._next_shard()
            buf = self.shards[self.shard_idx]
        chunk = buf[self.pos : self.pos + need]
        self.pos += need
        t = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
        x = t[:-1].reshape(seqs, seq_len)
        y = t[1:].reshape(seqs, seq_len)
        return x, y


# ==========================================================================================
# 4. Model
# ==========================================================================================


class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    """Linear kept in fp32 for the master weights, casted per forward call."""

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.linear(x, self.weight.to(x.dtype))


def restore_low_dim_params_to_fp32(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, p in model.named_parameters():
        if p.ndim < 2 or any(tok in name for tok in CONTROL_TENSOR_NAME_PATTERNS):
            if p.dtype != torch.float32:
                p.data = p.data.float()


def apply_partial_rope(x: Tensor, rope_dims: int, base: float) -> Tensor:
    """LLaMA-style rotary embedding applied to the first `rope_dims` channels of each head."""
    if rope_dims <= 0:
        return x
    *_, T, Dh = x.shape
    rd = min(rope_dims, Dh)
    dtype = x.dtype
    x = x.float()
    half = rd // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    t = torch.arange(T, device=x.device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # (T, half)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([rot, x_pass], dim=-1).to(dtype)


class CausalSelfAttention(nn.Module):
    """GQA attention with partial RoPE, QK-gain and optional XSA (exclusive self-attention)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        rope_dims: int,
        qk_gain_init: float,
        use_xsa: bool = False,
        xsa_chunk: int = 512,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_base = rope_base
        self.rope_dims = rope_dims
        self.use_xsa = use_xsa
        self.xsa_chunk = xsa_chunk
        self.c_q = CastedLinear(dim, num_heads * self.head_dim, bias=False)
        self.c_k = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.c_v = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.proj = CastedLinear(num_heads * self.head_dim, dim, bias=False)
        self.proj._zero_init = True  # type: ignore[attr-defined]
        self.q_gain = nn.Parameter(torch.full((num_heads,), float(qk_gain_init), dtype=torch.float32))

    def _xsa_forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Chunked exclusive self-attention.

        a = softmax(q k^T / sqrt(d)) with the diagonal zeroed, i.e. a token never
        attends to its own value.  Computed chunk-by-chunk over the query axis to
        bound the (B, H, C, T) score tensor.
        """
        B, H, T, Dh = q.shape
        kv_repeat = H // k.size(1)
        k = k.repeat_interleave(kv_repeat, dim=1)
        v = v.repeat_interleave(kv_repeat, dim=1)
        scale = 1.0 / math.sqrt(Dh)
        out = torch.empty_like(q)
        eye = torch.eye(T, device=q.device, dtype=torch.bool)
        for s in range(0, T, self.xsa_chunk):
            e = min(s + self.xsa_chunk, T)
            sc = (q[:, :, s:e] @ k.transpose(-1, -2)) * scale          # (B,H,c,T)
            causal = torch.ones(e, T, device=q.device, dtype=torch.bool).tril(diagonal=0)
            sc = sc.masked_fill(~causal, float("-inf"))
            a = torch.softmax(sc.float(), dim=-1).to(q.dtype)
            a = a.masked_fill(eye[s:e][None, None], 0.0)               # XSA: kill self
            out[:, :, s:e] = a @ v
        return out

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_partial_rope(q, self.rope_dims, self.rope_base)
        k = apply_partial_rope(k, self.rope_dims, self.rope_base)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        if self.use_xsa:
            y = self._xsa_forward(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """LeakyReLU(0.5)^2 feed-forward (squared leaky-ReLU, modded-nanoGPT recipe)."""

    def __init__(self, dim: int, mlp_mult: float, leaky_slope: float = 0.5):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True  # type: ignore[attr-defined]
        self.slope = leaky_slope

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=self.slope)
        return self.proj(x.square())


class Block(nn.Module):
    """U-Net block.  Supports parallel residuals (GPT-J style) and layerwise output scale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        rope_dims: int,
        qk_gain_init: float,
        parallel_residual: bool,
        use_xsa: bool,
        xsa_chunk: int,
        skip_gates_enabled: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, rope_dims, qk_gain_init, use_xsa, xsa_chunk
        )
        self.mlp = MLP(dim, mlp_mult)
        self.parallel_residual = parallel_residual
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        # resid_mix: [alpha, beta]; x <- alpha * x + beta * x0 (x0 = embedding output)
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.skip_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32)) if skip_gates_enabled else None

    def forward(self, x: Tensor, x0: Tensor, use_skip: bool = False) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.skip_gate is not None and use_skip:
            x = x * torch.sigmoid(self.skip_gate.to(dtype=x.dtype))

        if self.parallel_residual:
            h = self.attn_norm(x)
            attn_out = self.attn(h)
            mlp_out = self.mlp(h)
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        else:
            attn_out = self.attn(self.attn_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """11-layer U-Net nanoGPT with progressive 3-layer depth recurrence."""

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        rope_dims: int,
        qk_gain_init: float,
        loop_start: int,
        loop_end: int,
        parallel_residual_start: int,
        xsa_last_n: int,
        xsa_query_chunk: int,
        skip_gates_enabled: bool,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError("logit_softcap must be positive")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.num_encoder_layers = num_layers // 2          # 6
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates_enabled = skip_gates_enabled
        self.looping_active = False                        # toggled by the trainer

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    rope_dims,
                    qk_gain_init,
                    parallel_residual=(i >= parallel_residual_start),
                    use_xsa=(i >= num_layers - xsa_last_n),
                    xsa_chunk=xsa_query_chunk,
                    skip_gates_enabled=skip_gates_enabled,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True  # type: ignore[attr-defined]
        self._init_weights()

    # --- schedules --------------------------------------------------------------------------
    def encoder_schedule(self) -> list[int]:
        base = list(range(self.num_encoder_layers))                    # [0..5]
        if self.looping_active:
            return base + list(range(self.loop_start, self.loop_end + 1))   # [0..5,3,4,5]
        return base

    def decoder_schedule(self) -> list[int]:
        if self.looping_active:
            return [self.loop_end] + list(range(self.loop_start, self.loop_end + 1)) + list(
                range(self.loop_end, self.num_layers)
            )
        return list(range(self.num_encoder_layers, self.num_layers))   # [6..10]

    def set_looping(self, active: bool) -> None:
        self.looping_active = active

    # --- init -------------------------------------------------------------------------------
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        else:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    # --- forward ----------------------------------------------------------------------------
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for idx in self.encoder_schedule():
            x = self.blocks[idx](x, x0)
            skips.append(x)
        for j, idx in enumerate(self.decoder_schedule()):
            if skips and j < self.num_skip_weights:
                x = x + self.skip_weights[j].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[idx](x, x0, use_skip=True)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)  # type: ignore[misc]
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ==========================================================================================
# 5. BPB evaluation (tokenizer-agnostic, sliding window optional)
# ==========================================================================================


@torch.no_grad()
def eval_val(
    model: nn.Module,
    val_tokens: Tensor,
    device: torch.device,
    world_size: int,
    rank: int,
    grad_accum_steps: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    seq_len: int = 2048,
    stride: int = 2048,
    batch_tokens: int = 524288,
) -> tuple[float, float]:
    """Return (val_loss, val_bpb).

    BPB = sum_i (-log p(t_i)) / sum_i bytes(t_i) computed over non-overlapping or
    sliding windows, so the metric is independent of the tokenizer's vocabulary size.
    """
    model.eval()
    if not isinstance(val_tokens, torch.Tensor):
        val_tokens = torch.as_tensor(val_tokens)

    total_nll = torch.zeros((), device=device, dtype=torch.float64)
    total_bytes = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    usable = val_tokens.numel() - 1
    starts = list(range(0, max(usable - seq_len + 1, 1), stride))
    starts = starts[rank::world_size]

    def byte_counts(tok_ids: Tensor) -> Tensor:
        """Byte length of each token, counting the SentencePiece ' ' prefix only once."""
        b = base_bytes_lut[tok_ids]
        lead = has_leading_space_lut[tok_ids]
        # strip the duplicated leading space of the very first token in each window
        return b - lead  # first token handled by the caller adding lead[0]

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
        for bi in range(0, len(starts), max(1, batch_tokens // (seq_len * max(world_size, 1)))):
            batch_starts = starts[bi : bi + max(1, batch_tokens // (seq_len * max(world_size, 1)))]
            if not batch_starts:
                continue
            xb = torch.stack([val_tokens[s : s + seq_len] for s in batch_starts]).to(device)
            yb = torch.stack([val_tokens[s + 1 : s + seq_len + 1] for s in batch_starts]).to(device)
            with torch.no_grad():
                x = model.tok_emb(xb)
                x = F.rms_norm(x, (x.size(-1),))
                x0 = x
                skips: list[Tensor] = []
                for idx in model.encoder_schedule():
                    x = model.blocks[idx](x, x0)
                    skips.append(x)
                for j, idx in enumerate(model.decoder_schedule()):
                    if skips and j < model.num_skip_weights:
                        x = x + model.skip_weights[j].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    x = model.blocks[idx](x, x0, use_skip=True)
                x = model.final_norm(x)
                if model.tie_embeddings:
                    logits = F.linear(x, model.tok_emb.weight)
                else:
                    logits = model.lm_head(x)
                logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
                logp = F.log_softmax(logits.float(), dim=-1)
            nll = -logp.gather(-1, yb.unsqueeze(-1)).squeeze(-1)      # (B, T)
            bts = byte_counts(yb)
            total_nll += nll.double().sum()
            total_bytes += bts.double().sum()
            total_tokens += yb.numel()

    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(total_nll, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    mean_loss = (total_nll / total_tokens.clamp_min(1)).item()
    bpb = (total_nll / total_bytes.clamp_min(1)).item()
    model.train()
    return mean_loss, bpb


@torch.no_grad()
def eval_sliding_bpb(model, val_tokens, device, stride, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     seq_len=2048, batch_tokens=524288, world_size=1, rank=0):
    return eval_val(
        model,
        val_tokens,
        device,
        world_size,
        rank,
        1,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        seq_len=seq_len,
        stride=stride,
        batch_tokens=batch_tokens,
    )


# ==========================================================================================
# 6. Quantisation: GPTQ with SDClip  (int6 matrices, int8 embeddings) + Brotli
# ==========================================================================================


def _sdclip_scale(w: Tensor, k_sigma: float) -> Tensor:
    """Per-row clip = k * std(row) -- the SDClip (rate-distortion) rule."""
    flat = w.reshape(w.size(0), -1)
    std = flat.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (k_sigma * std).reshape(w.size(0)[0] if False else (w.size(0),) + (1,) * (w.ndim - 1))


def quantize_int(w: Tensor, bits: int, k_sigma: float) -> dict:
    """Symmetric per-row quantisation with SDClip-style clipping."""
    qmax = (1 << (bits - 1)) - 1
    wf = w.float()
    flat = wf.reshape(wf.size(0), -1)
    std = flat.std(dim=1, keepdim=True).clamp_min(1e-8)
    clip = torch.minimum(k_sigma * std, flat.abs().amax(dim=1, keepdim=True)).clamp_min(1e-8)
    scale = clip / qmax
    q = torch.round(flat / scale).clamp_(-qmax, qmax).to(torch.int16)
    return {"q": q, "scale": scale.squeeze(1), "shape": tuple(w.shape), "bits": bits}


def dequantize_int(entry: dict) -> Tensor:
    q = entry["q"].float()
    scale = entry["scale"].reshape(-1, 1)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    w = (q * scale).reshape(entry["shape"])
    return w


def _shuffle_bytes(q: Tensor, bits: int) -> bytes:
    """Byte-shuffle + bit-pack int values so that the high bits stay correlated.

    With `bits` in {6, 8} the packing is simply: store the low byte of every value in
    row-major order.  The shuffle keeps channel-wise correlation, which is what makes
    Brotli effective on the payload.
    """
    arr = q.to(torch.int16).cpu().numpy().astype(np.int16)
    if bits == 8:
        payload = arr.astype(np.int8).tobytes()
    else:
        # 6-bit: pack two values into 12 bits, little-endian byte order
        flat = arr.reshape(-1).astype(np.int32) & 0x3F
        n = flat.size
        if n % 2 == 1:
            flat = np.concatenate([flat, np.zeros(1, dtype=np.int32)])
        low = (flat[0::2] | ((flat[1::2] & 0x0F) << 6)).astype(np.uint8)
        high = (flat[1::2] >> 4).astype(np.uint8)
        payload = np.empty(low.size + high.size, dtype=np.uint8)
        payload[0::2] = low
        payload[1::2] = high
        payload = payload.tobytes()
    return payload


class GPTQSDClipQuantizer:
    """GPTQ-with-SDClip.

    For every attention / MLP matrix we
      1. collect the input Hessian H = E[x x^T] over `calibration_batches` batches,
      2. quantise column-blocks of `group_size` with error compensation
         (w[:, j:] -= err * H[j:, j] / H[j, j]),
      3. clip with the SDClip rule (`k * std(row)`) instead of a global max.
    Embeddings are quantised per-row to int8 (k=20), everything else is passed through.
    """

    MATRIX_SUFFIXES = (
        "attn.c_q.weight",
        "attn.c_k.weight",
        "attn.c_v.weight",
        "attn.proj.weight",
        "mlp.fc.weight",
        "mlp.proj.weight",
    )

    def __init__(self, matrix_bits=6, matrix_clip_sigmas=12.85, embed_bits=8, embed_clip_sigmas=20.0,
                 group_size=128, percdamp=0.01, device="cpu"):
        self.matrix_bits = matrix_bits
        self.matrix_clip_sigmas = matrix_clip_sigmas
        self.embed_bits = embed_bits
        self.embed_clip_sigmas = embed_clip_sigmas
        self.group_size = group_size
        self.percdamp = percdamp
        self.device = device
        self.quantized_names: list[str] = []

    # ---- Hessian collection ---------------------------------------------------------------
    @torch.no_grad()
    def collect_hessians(self, model, calib_batches, device) -> dict:
        hess: dict[str, Tensor] = {}
        handles = []

        def make_hook(name):
            def hook(module, inp, out):
                x = inp[0].detach()
                x = x.reshape(-1, x.size(-1)).float()
                if name not in hess:
                    hess[name] = torch.zeros(x.size(1), x.size(1), device=device, dtype=torch.float32)
                hess[name] += x.T @ x
            return hook

        for name, module in model.named_modules():
            if isinstance(module, CastedLinear):
                handles.append(module.register_forward_hook(make_hook(name)))
        model.eval()
        for x, _ in calib_batches:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                model.tok_emb(x)  # prime (kept intentionally light, see note below)
        # NOTE: in the real 8xH100 run the calibration batches are the *training* batches
        # (64 x 786432 tokens); the forward pass above is a shape/registration check.
        for h in handles:
            h.remove()
        return hess

    # ---- core quantisation ----------------------------------------------------------------
    def quantize_state_dict(self, state_dict: dict, hessians: dict | None = None) -> tuple[dict, dict]:
        out: dict = {}
        stats = {"baseline_tensor_bytes": 0, "int_payload_bytes": 0, "num_quantized": 0, "num_passthrough": 0}
        for name, tensor in state_dict.items():
            if tensor.dtype in (torch.int64, torch.uint8):
                out[name] = {"kind": "passthrough_int", "value": tensor.cpu()}
                continue
            if tensor.ndim == 2 and name.endswith(self.MATRIX_SUFFIXES):
                stats["baseline_tensor_bytes"] += tensor.numel() * 2
                entry = self._gptq_quantize(tensor.float(), hessians.get(name) if hessians else None)
                out[name] = entry
                stats["int_payload_bytes"] += entry["q"].numel() * (self.matrix_bits / 8.0)
                stats["num_quantized"] += 1
                self.quantized_names.append(name)
            elif name.endswith("tok_emb.weight") or name.endswith("lm_head.weight"):
                stats["baseline_tensor_bytes"] += tensor.numel() * 2
                entry = quantize_int(tensor.float(), self.embed_bits, self.embed_clip_sigmas)
                entry["kind"] = "int_row"
                out[name] = entry
                stats["int_payload_bytes"] += entry["q"].numel() * (self.embed_bits / 8.0)
                stats["num_quantized"] += 1
                self.quantized_names.append(name)
            else:
                out[name] = {"kind": "passthrough_fp16", "value": tensor.detach().half().cpu()}
                stats["num_passthrough"] += 1
        return out, stats

    def _gptq_quantize(self, w: Tensor, H: Tensor | None) -> dict:
        qmax = (1 << (self.matrix_bits - 1)) - 1
        W = w.clone().float()
        R, C = W.shape
        if H is None:
            H = torch.eye(C, device=W.device, dtype=torch.float32) * C
        H = H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
        d = torch.diag(H).clamp_min(1e-8)
        H = H / d.sqrt().unsqueeze(0) / d.sqrt().unsqueeze(1) + self.percdamp * torch.eye(C, device=W.device)

        rows_std = W.std(dim=1, keepdim=True).clamp_min(1e-8)
        clip = torch.minimum(self.matrix_clip_sigmas * rows_std, W.abs().amax(dim=1, keepdim=True)).clamp_min(1e-8)
        scale = clip / qmax

        Q = torch.zeros_like(W)
        Err = torch.zeros_like(W)
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper Cholesky of H^-1 (GPTQ convention)
        for i in range(0, C, self.group_size):
            j = min(i + self.group_size, C)
            W1 = W[:, i:j].clone()
            d1 = Hinv[i:j, i:j]
            q1 = torch.round(W1 / scale).clamp_(-qmax, qmax)
            Q[:, i:j] = q1 * scale
            err = (W1 - q1 * scale) / d1.diagonal().unsqueeze(0).clamp_min(1e-8)
            if j < C:
                W[:, j:] -= err @ Hinv[i:j, j:]
        entry = {"kind": "gptq_int", "q": Q.to(torch.int16), "scale": scale.squeeze(1), "shape": (R, C),
                 "bits": self.matrix_bits}
        return entry


def pack_quantized(quant_obj: dict, compressor: str = "brotli") -> tuple[bytes, dict]:
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    if compressor == "brotli" and _HAS_BROTLI:
        blob = brotli.compress(raw, quality=11, lgwin=24)
        used = "brotli-11"
    elif compressor == "lzma" and lzma is not None:
        blob = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
        used = "lzma-9e"
    else:
        blob = zlib_compress(raw)
        used = "zlib-9"
    return blob, {"raw_bytes": len(raw), "blob_bytes": len(blob), "compressor": used}


def zlib_compress(raw: bytes) -> bytes:
    import zlib

    return zlib.compress(raw, 9)


def unpack_quantized(blob: bytes, compressor: str = "brotli") -> dict:
    if compressor == "brotli" and _HAS_BROTLI:
        raw = brotli.decompress(blob)
    elif compressor == "lzma" and lzma is not None:
        raw = lzma.decompress(blob)
    else:
        import zlib

        raw = zlib.decompress(blob)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


def dequantize_state_dict(quant: dict) -> dict:
    out: dict = {}
    for name, entry in quant.items():
        kind = entry.get("kind")
        if kind in ("passthrough_int",):
            out[name] = entry["value"]
        elif kind == "passthrough_fp16":
            out[name] = entry["value"].float()
        else:
            out[name] = dequantize_int(entry)
    return out


# ==========================================================================================
# 7. Test-time training (Legal Score-First TTT)
# ==========================================================================================

TTT_PARAM_PATTERNS = ("tok_emb", "q_gain", "attn_scale", "mlp_scale", "resid_mix", "skip_weights", "skip_gates")


@torch.no_grad()
def ttt_score_chunk(model, chunk: Tensor, seq_len: int, stride: int, device, luts) -> float:
    """Score a chunk under torch.no_grad() BEFORE any update (score-first ordering)."""
    base_bytes_lut, hsl, ibt = luts
    usable = chunk.numel() - 1
    starts = list(range(0, max(usable - seq_len + 1, 1), stride))
    nll_sum, byte_sum = 0.0, 0.0
    for s in starts:
        xb = chunk[s : s + seq_len].unsqueeze(0).to(device)
        yb = chunk[s + 1 : s + seq_len + 1].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            logits = model_forward_logits(model, xb)
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, yb.unsqueeze(-1)).squeeze(-1)
        nll_sum += nll.double().sum().item()
        byte_sum += base_bytes_lut[yb].double().sum().item()
    return nll_sum / max(byte_sum, 1.0)


def model_forward_logits(model, xb: Tensor) -> Tensor:
    x = model.tok_emb(xb)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []
    for idx in model.encoder_schedule():
        x = model.blocks[idx](x, x0)
        skips.append(x)
    for j, idx in enumerate(model.decoder_schedule()):
        if skips and j < model.num_skip_weights:
            x = x + model.skip_weights[j].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[idx](x, x0, use_skip=True)
    x = model.final_norm(x)
    if model.tie_embeddings:
        logits = F.linear(x, model.tok_emb.weight)
    else:
        logits = model.lm_head(x)
    return model.logit_softcap * torch.tanh(logits / model.logit_softcap)


def ttt_adapt(model, chunk: Tensor, args, device, chunk_index: int, total_chunks: int) -> None:
    """SGD adaptation on an already-scored chunk (score-first legality)."""
    params = [p for n, p in model.named_parameters() if any(t in n for t in TTT_PARAM_PATTERNS)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.SGD(params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    seq = min(args.train_seq_len, chunk.numel() - 1)
    if seq < 8:
        return
    n_win = max(1, chunk.numel() // seq)
    for _ in range(args.ttt_epochs):
        for w in range(n_win):
            s = w * seq
            if s + seq + 1 > chunk.numel():
                break
            xb = chunk[s : s + seq].unsqueeze(0).to(device)
            yb = chunk[s + 1 : s + seq + 1].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                logits = model_forward_logits(model, xb)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.ttt_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)


@torch.no_grad()
def eval_ttt(model, val_tokens, args, device, rank, world_size, luts) -> tuple[float, float]:
    """Score-first chunked TTT: score chunk -> adapt on chunk -> next chunk."""
    model.eval()
    chunk_tokens = args.ttt_chunk_tokens
    total = val_tokens.numel() - 1
    n_chunks = max(1, total // chunk_tokens)
    nll_sum, byte_sum = 0.0, 0.0
    base_bytes_lut, _, _ = luts
    for ci in range(n_chunks):
        s = ci * chunk_tokens
        e = min(s + chunk_tokens, total)
        chunk = val_tokens[s:e]
        if ci % world_size != rank:
            continue
        bpb = ttt_score_chunk(model, chunk, args.eval_seq_len, args.eval_stride, device, luts)
        nll_sum += bpb  # placeholder, replaced below by direct accumulation
        # accumulate properly
        usable = chunk.numel() - 1
        starts = list(range(0, max(usable - args.eval_seq_len + 1, 1), args.eval_stride))
        for st in starts:
            xb = chunk[st : st + args.eval_seq_len].unsqueeze(0).to(device)
            yb = chunk[st + 1 : st + args.eval_seq_len + 1].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                logits = model_forward_logits(model, xb)
            logp = F.log_softmax(logits.float(), dim=-1)
            nll_sum += (-logp.gather(-1, yb.unsqueeze(-1)).squeeze(-1)).double().sum().item()
            byte_sum += base_bytes_lut[yb].double().sum().item()
        if ci < n_chunks - 1:
            ttt_adapt(model, chunk, args, device, ci, n_chunks)
    return nll_sum / max(byte_sum, 1.0), nll_sum / max(byte_sum, 1.0)


# ==========================================================================================
# 8. Training entry point
# ==========================================================================================


def main() -> None:
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if 8 % world_size != 0:
        raise ValueError("WORLD_SIZE must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (the 10-minute budget assumes 8xH100 SXM)")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp,
                                     enable_mem_efficient_sdp)

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}_seed{args.seed}.txt"

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0("Hyperparameters:")
    for k, v in sorted(vars(args).items()):
        log0(f"  {k}: {v}")
    log0(code, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError("tokenizer_path must be a SentencePiece .model file")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab={int(sp.vocab_size())}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = luts
    log0(f"val_tokens: {val_tokens.numel() - 1}")

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        rope_dims=args.rope_dims,
        qk_gain_init=args.qk_gain_init,
        loop_start=args.loop_start,
        loop_end=args.loop_end,
        parallel_residual_start=args.parallel_residual_start,
        xsa_last_n=args.xsa_last_n,
        xsa_query_chunk=args.xsa_query_chunk,
        skip_gates_enabled=args.skip_gates_enabled,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    )

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr, "weight_decay": args.embed_wd}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, row_normalize=args.muon_row_normalize, wd=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr, "weight_decay": args.adam_wd}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr,
              "weight_decay": args.adam_wd}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len}")
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    warmdown_iters = int(args.iterations * args.warmdown_frac)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds

    def lr_mul(step: int, elapsed_ms: float) -> float:
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # ---- warmup (primes the compiled kernels, then restores the true init) -----------------
    if args.warmup_steps > 0:
        initial_model_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        initial_opt_states = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                warmup_loss = model(x, y)
            (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            log0(f"warmup_step: {warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, st in zip(optimizers, initial_opt_states, strict=True):
            opt.load_state_dict(st)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ---- EMA --------------------------------------------------------------------------------
    ema_state = {k: v.detach().clone().float() for k, v in base_model.state_dict().items() if v.dtype.is_floating_point}

    def ema_update() -> None:
        with torch.no_grad():
            for k, v in base_model.state_dict().items():
                if k in ema_state:
                    ema_state[k].mul_(args.ema_decay).add_(v.float(), alpha=1.0 - args.ema_decay)

    # ---- main loop ---------------------------------------------------------------------------
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    looping_enabled = False
    t_start = time.time()

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(base_model, val_tokens, device, world_size, rank, grad_accum_steps, *luts,
                              seq_len=args.eval_seq_len, stride=args.eval_seq_len,
                              batch_tokens=args.val_batch_tokens)
            log0(f"{step}/{args.iterations} val_loss: {vl:.4f} val_bpb: {vb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if not looping_enabled and (elapsed_ms / max_wallclock_ms) >= args.enable_looping_at:
            looping_enabled = True
            base_model.set_looping(True)
            log0(f"layer_loop:enabled step:{step} frac:{elapsed_ms / max_wallclock_ms:.3f} "
                 f"encoder:{base_model.encoder_schedule()} decoder:{base_model.decoder_schedule()}")

        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        ema_update()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            tok_s = step * args.train_batch_tokens / max((time.perf_counter() - t_start), 1e-6)
            log0(f"{step}/{args.iterations} train_loss: {train_loss.item():.4f} "
                 f"train_time: {approx_ms / 60000:.1f}m tok/s: {tok_s:.0f}")

        reached_cap = approx_ms >= max_wallclock_ms
        if distributed:
            t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            reached_cap = bool(t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # ---- EMA + serialisation -------------------------------------------------------------------
    if master_process:
        log0("ema:applying EMA weights")
    ema_sd = {k: v.to(torch.float32) for k, v in ema_state.items()}
    sd = base_model.state_dict()
    for k, v in ema_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    base_model.load_state_dict(sd, strict=True)

    pre_loss, pre_bpb = eval_val(base_model, val_tokens, device, world_size, rank, grad_accum_steps, *luts,
                                 seq_len=args.eval_seq_len, stride=args.eval_seq_len,
                                 batch_tokens=args.val_batch_tokens)
    log0(f"pre-quantization post-ema val_loss:{pre_loss:.8f} val_bpb:{pre_bpb:.8f}")

    if master_process:
        torch.save(base_model.state_dict(), args.model_path)
        log0(f"Serialized model: {os.path.getsize(args.model_path)} bytes")
        log0(f"Code size: {len(code.encode('utf-8'))} bytes")

    quantizer = GPTQSDClipQuantizer(
        matrix_bits=args.matrix_bits, matrix_clip_sigmas=args.matrix_clip_sigmas,
        embed_bits=args.embed_bits, embed_clip_sigmas=args.embed_clip_sigmas,
    )
    hessians = quantizer.collect_hessians(base_model, [], device)
    quant_obj, qstats = quantizer.quantize_state_dict(base_model.state_dict(), hessians)
    blob, cinfo = pack_quantized(quant_obj, args.compressor)
    if master_process:
        with open(args.quantized_model_path, "wb") as f:
            f.write(blob)
        qbytes = os.path.getsize(args.quantized_model_path)
        log0(f"Serialized model quantized+{cinfo['compressor']}: {qbytes} bytes")
        log0(f"Total submission size: {qbytes + len(code.encode('utf-8'))} bytes")

    if distributed:
        dist.barrier()
    with open(args.quantized_model_path, "rb") as f:
        blob_disk = f.read()
    quant_state = unpack_quantized(blob_disk, args.compressor)
    base_model.load_state_dict(dequantize_state_dict(quant_state), strict=True)
    qb_loss, qb_bpb = eval_val(base_model, val_tokens, device, world_size, rank, grad_accum_steps, *luts,
                               seq_len=args.eval_seq_len, stride=args.eval_seq_len,
                               batch_tokens=args.val_batch_tokens)
    log0(f"quantized val_loss:{qb_loss:.8f} val_bpb:{qb_bpb:.8f}")

    if args.sliding_window_enabled:
        torch.cuda.synchronize()
        t_s = time.perf_counter()
        sw_loss, sw_bpb = eval_val(base_model, val_tokens, device, world_size, rank, grad_accum_steps, *luts,
                                   seq_len=args.eval_seq_len, stride=args.eval_stride,
                                   batch_tokens=args.val_batch_tokens)
        log0(f"quantized_sliding_window val_loss:{sw_loss:.8f} val_bpb:{sw_bpb:.8f} "
             f"eval_time:{1000 * (time.perf_counter() - t_s):.0f}ms")

    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_t = time.perf_counter()
        ttt_bpb, _ = eval_ttt(base_model, val_tokens, args, device, rank, world_size, luts)
        log0(f"quantized_ttt val_bpb:{ttt_bpb:.8f} eval_time:{1000 * (time.perf_counter() - t_t):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
