import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange


@triton.jit
def _rbf_attn_fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    L,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    Z,
    H,
    N_CTX,
    D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vd
    )
    o_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )

    q = tl.load(
        q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )
    q_sq = tl.sum(q.to(tl.float32) * q.to(tl.float32), axis=1)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    lo = 0
    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        curr_offs_n = start_n + offs_n

        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )
        v = tl.load(
            v_ptrs + start_n * stride_vn,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        k_sq = tl.sum(k.to(tl.float32) * k.to(tl.float32), axis=1)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # RBF Distance Formulation + ReLU Equivalent
        dist_sq = q_sq[:, None] + k_sq[None, :] - 2.0 * qk
        dist_sq = tl.maximum(dist_sq, 0.0)

        logits = -dist_sq * sm_scale

        if IS_CAUSAL:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        logits = tl.where(curr_offs_n[None, :] < N_CTX, logits, float("-inf"))

        # FlashAttention Online Softmax
        m_ij = tl.maximum(m_i, tl.max(logits, 1))
        p = tl.math.exp(logits - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(
        o_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
    )

    if L is not None:
        l_ptrs = L + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, m_i + tl.math.log(l_i), mask=offs_m < N_CTX)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    Z,
    H,
    N_CTX,
    D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    o_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_ok
    )
    do_ptrs = (
        DO
        + off_z * stride_doz
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dok
    )

    o = tl.load(
        o_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )
    do = tl.load(
        do_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )

    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(Delta + off_hz * N_CTX + offs_m, delta, mask=offs_m < N_CTX)


@triton.jit
def _rbf_attn_bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    Delta,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    Z,
    H,
    N_CTX,
    D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kk
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vk
    )

    k = tl.load(
        k_ptrs, mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )
    v = tl.load(
        v_ptrs, mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )
    k_sq = tl.sum(k.to(tl.float32) * k.to(tl.float32), axis=1)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

    for start_m in range(lo, N_CTX, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        curr_offs_m = start_m + offs_m

        q_ptrs = (
            Q
            + off_z * stride_qz
            + off_h * stride_qh
            + curr_offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qk
        )
        q = tl.load(
            q_ptrs,
            mask=(curr_offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )
        q_sq = tl.sum(q.to(tl.float32) * q.to(tl.float32), axis=1)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        dist_sq = q_sq[:, None] + k_sq[None, :] - 2.0 * qk
        relu_mask = dist_sq > 0.0
        dist_sq = tl.maximum(dist_sq, 0.0)
        logits = -dist_sq * sm_scale

        if IS_CAUSAL:
            logits = tl.where(
                curr_offs_m[:, None] >= offs_n[None, :], logits, float("-inf")
            )
        logits = tl.where(
            (curr_offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
            logits,
            float("-inf"),
        )

        l_i = tl.load(
            L + off_hz * N_CTX + curr_offs_m, mask=curr_offs_m < N_CTX, other=0.0
        )
        p = tl.math.exp(logits - l_i[:, None])

        do_ptrs = (
            DO
            + off_z * stride_qz
            + off_h * stride_qh
            + curr_offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qk
        )
        do = tl.load(
            do_ptrs,
            mask=(curr_offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        dv += tl.dot(tl.trans(p.to(V.dtype.element_ty)), do)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        delta = tl.load(
            Delta + off_hz * N_CTX + curr_offs_m, mask=curr_offs_m < N_CTX, other=0.0
        )
        ds = p * (dp - delta[:, None]) * (-sm_scale)
        ds = tl.where(relu_mask, ds, 0.0)

        if IS_CAUSAL:
            ds = tl.where(curr_offs_m[:, None] >= offs_n[None, :], ds, 0.0)
        ds = tl.where(
            (curr_offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX), ds, 0.0
        )

        # RBF Mathematical chain-rule gradient projection derivatives
        ds_sum = tl.sum(ds, axis=1)
        dq_i = 2.0 * (ds_sum[:, None] * q - tl.dot(ds.to(Q.dtype.element_ty), k))

        dq_ptrs = (
            DQ
            + off_z * stride_qz
            + off_h * stride_qh
            + curr_offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qk
        )
        tl.atomic_add(
            dq_ptrs,
            dq_i.to(Q.dtype.element_ty),
            mask=(curr_offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
        )

        ds_sum_k = tl.sum(ds, axis=0)
        dk += 2.0 * (
            ds_sum_k[:, None] * k - tl.dot(tl.trans(ds).to(Q.dtype.element_ty), q)
        )

    dk_ptrs = (
        DK
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kk
    )
    dv_ptrs = (
        DV
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_kk
    )
    tl.store(
        dk_ptrs,
        dk.to(DK.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
    )
    tl.store(
        dv_ptrs,
        dv.to(DV.dtype.element_ty),
        mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
    )


class TritonScaledRBFAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, H, N_CTX, D_HEAD = q.shape
        out = torch.empty_like(q)
        L = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)

        sm_scale = 1.0 / math.sqrt(D_HEAD)
        BLOCK_M, BLOCK_N = 64, 64
        BLOCK_DMODEL = max(
            16, triton.next_power_of_2(D_HEAD)
        )  # Pad to nearest boundary for `tl.dot` compatibility

        grid = (triton.cdiv(N_CTX, BLOCK_M), B * H, 1)
        _rbf_attn_fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            out,
            L,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            N_CTX,
            D_HEAD,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=BLOCK_DMODEL,
            IS_CAUSAL=is_causal,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, out, L)
        ctx.sm_scale, ctx.is_causal = sm_scale, is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, L = ctx.saved_tensors
        dout = dout.contiguous()
        B, H, N_CTX, D_HEAD = q.shape

        BLOCK_M, BLOCK_N = 64, 64
        BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

        Delta = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)
        _bwd_preprocess[(triton.cdiv(N_CTX, BLOCK_M), B * H, 1)](
            out,
            dout,
            Delta,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            B,
            H,
            N_CTX,
            D_HEAD,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )

        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        _rbf_attn_bwd_kernel[(triton.cdiv(N_CTX, BLOCK_N), B * H, 1)](
            q,
            k,
            v,
            ctx.sm_scale,
            out,
            dout,
            dq,
            dk,
            dv,
            L,
            Delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            B,
            H,
            N_CTX,
            D_HEAD,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=ctx.is_causal,
            num_warps=4,
            num_stages=2,
        )
        return dq, dk, dv, None


def compute_rbf_logits(q, k):
    q_sq = q.pow(2).sum(dim=-1, keepdim=True)
    k_sq = k.pow(2).sum(dim=-1).unsqueeze(-2)
    dot_product = q @ k.swapaxes(-2, -1)
    dist_sq = q_sq + k_sq - 2.0 * dot_product
    dist_sq = torch.relu(dist_sq)
    return -dist_sq / (q.size(-1) ** 0.5)


class CustomCausalAttention(nn.Module):
    def __init__(self, num_heads, emb_dims, attention_type="standard"):
        super().__init__()
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.qkv_proj = nn.Linear(emb_dims, 3 * emb_dims)
        self.proj = nn.Linear(emb_dims, emb_dims)

    def forward(self, x):
        b, s, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv, "b s (qkv h n) -> qkv b h s n", qkv=3, h=self.num_heads
        )
        attn_weights = None

        if self.attention_type == "standard":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        elif self.attention_type == "standard_slow":
            attn_logits = q @ k.transpose(-2, -1)
            attn_logits = attn_logits / (q.size(-1) ** 0.5)
            causal_mask = torch.triu(
                torch.ones(s, s, device=x.device), diagonal=1
            ).bool()
            attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1)
            out = attn_weights @ v

        elif self.attention_type == "rbf_math":
            # Softmax shift-invariance natively cancels out ||q||^2 completely.

            k_sq = k.float().pow(2).sum(dim=-1, keepdim=True).to(k.dtype)

            # Pad dimension components onto vectors to construct exact scaling map
            q_prime = torch.cat([q, torch.ones_like(q[..., :1])], dim=-1)
            k_prime = torch.cat([k, -0.5 * k_sq], dim=-1)

            # Align hardware dimensions (Padding heads to an exact multiple of 8 unlocks FA hook)
            pad_len = (8 - (q_prime.shape[-1] % 8)) % 8
            if pad_len > 0:
                q_prime = F.pad(q_prime, (0, pad_len))
                k_prime = F.pad(k_prime, (0, pad_len))

            scale = 2.0 / math.sqrt(q.size(-1))
            out = F.scaled_dot_product_attention(
                q_prime, k_prime, v, is_causal=True, scale=scale
            )

        elif self.attention_type == "rbf_triton":
            # OPTION 2: The Fully Derived Custom Triton Implementation
            out = TritonScaledRBFAttention.apply(q, k, v, True)

        elif self.attention_type == "rbf_slow":
            # Baseline explicitly materialized code
            attn_logits = compute_rbf_logits(q, k)
            causal_mask = torch.triu(
                torch.ones(s, s, device=x.device), diagonal=1
            ).bool()
            attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1)
            out = attn_weights @ v
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        out = rearrange(out, "b h s n -> b s (h n)")
        return self.proj(out), attn_weights
