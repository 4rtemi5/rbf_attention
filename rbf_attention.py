import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def _get_autotune_configs():
    configs = []
    # Restrict to num_stages=2, 3 to prevent register spilling in heavy backward kernels
    for num_stages in [2, 3]:
        for block_m, block_n, num_warps in [
            (128, 128, 8),
            (128, 64, 4),
            (64, 128, 4),
            (64, 64, 4),
            (32, 64, 2),
            (64, 32, 2),
        ]:
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs


# =========================================================================
# HELPER TRITON KERNEL (Zero-Allocation VRAM Fix)
# =========================================================================


@triton.jit
def _sq_norm_kernel(
    X,
    SQ,
    sm_scale,
    stride_xz,
    stride_xh,
    stride_xn,
    stride_sz,
    stride_sh,
    Z,
    H,
    N_CTX,
    D_HEAD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0) * BLOCK_N
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    x_ptrs = (
        X
        + off_z * stride_xz
        + off_h * stride_xh
        + offs_n[:, None] * stride_xn
        + offs_d[None, :]
    )
    mask_n = offs_n < N_CTX

    if BLOCK_DMODEL == D_HEAD:
        x = tl.load(x_ptrs, mask=mask_n[:, None], other=0.0)
    else:
        x = tl.load(
            x_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    x_f32 = x.to(tl.float32)
    sq = tl.sum(x_f32 * x_f32, axis=1)

    sq_casted = sq.to(X.dtype.element_ty).to(tl.float32)
    sq_scaled = sq_casted * sm_scale

    sq_ptrs = SQ + off_z * stride_sz + off_h * stride_sh + offs_n
    tl.store(sq_ptrs, sq_scaled, mask=mask_n)


# =========================================================================
# MAIN TRITON KERNELS
# =========================================================================


@triton.autotune(configs=_get_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_attn_fwd_kernel(
    Q,
    K,
    V,
    K_sq,
    sm_scale,
    Out,
    L,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ksqz,
    stride_ksqh,
    stride_oz,
    stride_oh,
    stride_om,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
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
        + offs_d[None, :]
    )
    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )
    k_sq_ptrs = K_sq + off_z * stride_ksqz + off_h * stride_ksqh + offs_n
    o_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :]
    )

    mask_m = offs_m < N_CTX
    if BLOCK_DMODEL == D_HEAD:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        q = tl.load(
            q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    lo = 0
    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    k_ptrs += lo * stride_kn
    v_ptrs += lo * stride_vn
    k_sq_ptrs += lo

    for start_n in range(lo, hi, BLOCK_N):
        curr_offs_n = start_n + offs_n
        mask_n = curr_offs_n < N_CTX

        if BLOCK_DMODEL == D_HEAD:
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )

        k_sq_scaled = tl.load(k_sq_ptrs, mask=mask_n, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        logits = qk * (2.0 * sm_scale) - k_sq_scaled[None, :]

        if IS_CAUSAL and start_m * BLOCK_M < start_n + BLOCK_N:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        if start_n + BLOCK_N > N_CTX:
            logits = tl.where(curr_offs_n[None, :] < N_CTX, logits, float("-inf"))

        logits = logits.to(tl.float32)
        m_ij = tl.maximum(m_i, tl.max(logits, 1))

        p = tl.math.exp(logits - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        m_i = m_ij

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        k_sq_ptrs += BLOCK_N

    acc = acc / l_i[:, None]
    if BLOCK_DMODEL == D_HEAD:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])
    else:
        tl.store(
            o_ptrs,
            acc.to(Out.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD),
        )

    if L is not None:
        l_ptrs = L + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, m_i + tl.math.log(l_i), mask=mask_m)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_doz,
    stride_doh,
    stride_dom,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
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
        + offs_d[None, :]
    )
    do_ptrs = (
        DO
        + off_z * stride_doz
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :]
    )

    mask_m = offs_m < N_CTX
    if BLOCK_DMODEL == D_HEAD:
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        o = tl.load(
            o_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )
        do = tl.load(
            do_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(Delta + off_hz * N_CTX + offs_m, delta.to(tl.float32), mask=mask_m)


@triton.autotune(configs=_get_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_attn_bwd_dk_dv_kernel(
    Q,
    K,
    V,
    K_sq,
    sm_scale,
    DO,
    DK,
    DV,
    L,
    Delta,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ksqz,
    stride_ksqh,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    start_n_idx = start_n * BLOCK_N
    offs_n = start_n_idx + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )
    k_sq_ptrs = K_sq + off_z * stride_ksqz + off_h * stride_ksqh + offs_n

    mask_n = offs_n < N_CTX
    if BLOCK_DMODEL == D_HEAD:
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    else:
        k = tl.load(
            k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )
        v = tl.load(
            v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    k_sq_scaled = tl.load(k_sq_ptrs, mask=mask_n, other=0.0)
    k_f32 = k.to(tl.float32)

    dk_dot = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    ds_scaled_sum_k_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    lo = (start_n_idx // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    do_ptrs = (
        DO
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    l_ptrs = L + off_hz * N_CTX + offs_m
    delta_ptrs = Delta + off_hz * N_CTX + offs_m

    q_ptrs += lo * stride_qm
    do_ptrs += lo * stride_qm
    l_ptrs += lo
    delta_ptrs += lo

    for start_m in range(lo, N_CTX, BLOCK_M):
        curr_offs_m = start_m + offs_m
        mask_m = curr_offs_m < N_CTX

        if BLOCK_DMODEL == D_HEAD:
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )
            do = tl.load(
                do_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        logits = qk * (2.0 * sm_scale) - k_sq_scaled[None, :]

        if IS_CAUSAL and start_m < start_n_idx + BLOCK_N:
            logits = tl.where(
                curr_offs_m[:, None] >= offs_n[None, :], logits, float("-inf")
            )
        if start_m + BLOCK_M > N_CTX or start_n_idx + BLOCK_N > N_CTX:
            logits = tl.where(mask_m[:, None] & mask_n[None, :], logits, float("-inf"))

        logits = logits.to(tl.float32)
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        p = tl.math.exp(logits - l_i[:, None])

        dv += tl.dot(tl.trans(p.to(V.dtype.element_ty)), do)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        delta = tl.load(delta_ptrs, mask=mask_m, other=0.0)

        d_logits = p * (dp - delta[:, None])
        ds_scaled = d_logits * (2.0 * sm_scale)
        ds_scaled_f16 = ds_scaled.to(Q.dtype.element_ty)

        ds_scaled_sum_k_acc += tl.sum(ds_scaled_f16.to(tl.float32), axis=0)
        dk_dot += tl.dot(tl.trans(ds_scaled_f16), q)

        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_qm
        l_ptrs += BLOCK_M
        delta_ptrs += BLOCK_M

    dk_dot_f16 = dk_dot.to(K.dtype.element_ty).to(tl.float32)
    d_k_sq_f16 = ds_scaled_sum_k_acc.to(K.dtype.element_ty).to(tl.float32)

    dk = dk_dot_f16 - d_k_sq_f16[:, None] * k_f32

    dk_ptrs = (
        DK
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    dv_ptrs = (
        DV
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )

    if BLOCK_DMODEL == D_HEAD:
        tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])
    else:
        tl.store(
            dk_ptrs,
            dk.to(DK.dtype.element_ty),
            mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD),
        )
        tl.store(
            dv_ptrs,
            dv.to(DV.dtype.element_ty),
            mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD),
        )


@triton.autotune(configs=_get_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_attn_bwd_dq_kernel(
    Q,
    K,
    V,
    K_sq,
    sm_scale,
    DO,
    DQ,
    L,
    Delta,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_ksqz,
    stride_ksqh,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    start_m_idx = start_m * BLOCK_M
    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    do_ptrs = (
        DO
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )

    mask_m = offs_m < N_CTX
    if BLOCK_DMODEL == D_HEAD:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        q = tl.load(
            q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )
        do = tl.load(
            do_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    l_i = tl.load(L + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0)
    delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0)

    dq_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = tl.minimum(N_CTX, start_m_idx + BLOCK_M) if IS_CAUSAL else N_CTX

    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )
    k_sq_ptrs = K_sq + off_z * stride_ksqz + off_h * stride_ksqh + offs_n

    for start_n in range(0, hi, BLOCK_N):
        curr_offs_n = start_n + offs_n
        mask_n = curr_offs_n < N_CTX

        if BLOCK_DMODEL == D_HEAD:
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )

        k_sq_scaled = tl.load(k_sq_ptrs, mask=mask_n, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        logits = qk * (2.0 * sm_scale) - k_sq_scaled[None, :]

        if IS_CAUSAL and start_m_idx < start_n + BLOCK_N:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        if start_m_idx + BLOCK_M > N_CTX or start_n + BLOCK_N > N_CTX:
            logits = tl.where(mask_m[:, None] & mask_n[None, :], logits, float("-inf"))

        logits = logits.to(tl.float32)
        p = tl.math.exp(logits - l_i[:, None])

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        d_logits = p * (dp - delta[:, None])
        ds_scaled = d_logits * (2.0 * sm_scale)
        ds_scaled_f16 = ds_scaled.to(Q.dtype.element_ty)

        dq_acc += tl.dot(ds_scaled_f16, k)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        k_sq_ptrs += BLOCK_N

    dq_ptrs = (
        DQ
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    if BLOCK_DMODEL == D_HEAD:
        tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty), mask=mask_m[:, None])
    else:
        tl.store(
            dq_ptrs,
            dq_acc.to(DQ.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD),
        )


# =========================================================================
# TORCH.LIBRARY CUSTOM OP MIGRATION (PyTorch 2.4+ Native Fusion)
# =========================================================================


# --- 1. Scaled RBF Attention ---
@torch.library.custom_op("rbf_attn::scaled_fwd", mutates_args=())
def rbf_scaled_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, N_CTX, D_HEAD = q.shape

    out = torch.empty_like(q)
    L = q.new_empty((B, H, N_CTX), dtype=torch.float32)
    k_sq_scaled = q.new_empty((B, H, N_CTX), dtype=torch.float32)

    sm_scale = 1.0 / math.sqrt(D_HEAD)
    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

    grid_sq = (triton.cdiv(N_CTX, 128), B * H, 1)
    _sq_norm_kernel[grid_sq](
        k,
        k_sq_scaled,
        sm_scale,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=128,
    )

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)  # noqa: E731
    _rbf_attn_fwd_kernel[grid](
        q,
        k,
        v,
        k_sq_scaled,
        sm_scale,
        out,
        L,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )
    return out, L, k_sq_scaled


@rbf_scaled_fwd.register_fake
def _(q, k, v, is_causal):
    B, H, N_CTX, D_HEAD = q.shape
    return (
        # FIX 1: Use .new_empty() to predict contiguous strides,
        # instead of empty_like() which inherits non-contiguous strides.
        q.new_empty(q.shape),
        q.new_empty((B, H, N_CTX), dtype=torch.float32),
        k.new_empty((B, H, N_CTX), dtype=torch.float32),
    )


@torch.library.custom_op("rbf_attn::scaled_bwd", mutates_args=())
def rbf_scaled_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    L: torch.Tensor,
    k_sq_scaled: torch.Tensor,
    dout: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # FIX 2: Ensure all tensors are completely contiguous!
    # The backward kernels use `stride_qz` to index into `DO`, meaning
    # Q and DO must have perfectly identical memory alignments.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out, dout = out.contiguous(), dout.contiguous()

    B, H, N_CTX, D_HEAD = q.shape

    PREPROCESS_BLOCK_M = 64
    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

    Delta = q.new_empty((B, H, N_CTX), dtype=torch.float32)
    _bwd_preprocess[(triton.cdiv(N_CTX, PREPROCESS_BLOCK_M), B * H, 1)](
        out,
        dout,
        Delta,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        B,
        H,
        N_CTX,
        BLOCK_M=PREPROCESS_BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        D_HEAD=D_HEAD,
    )

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    grid_dk_dv = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_N"]), B * H, 1)  # noqa: E731
    _rbf_attn_bwd_dk_dv_kernel[grid_dk_dv](
        q,
        k,
        v,
        k_sq_scaled,
        sm_scale,
        dout,
        dk,
        dv,
        L,
        Delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )

    grid_dq = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)  # noqa: E731
    _rbf_attn_bwd_dq_kernel[grid_dq](
        q,
        k,
        v,
        k_sq_scaled,
        sm_scale,
        dout,
        dq,
        L,
        Delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )
    return dq, dk, dv


@rbf_scaled_bwd.register_fake
def _(q, k, v, out, L, k_sq_scaled, dout, is_causal, sm_scale):
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def rbf_scaled_setup_context(ctx, inputs, output):
    q, k, v, is_causal = inputs
    out, L, k_sq_scaled = output
    ctx.save_for_backward(q, k, v, out, L, k_sq_scaled)
    ctx.is_causal = is_causal
    ctx.sm_scale = 1.0 / math.sqrt(q.shape[-1])


def rbf_scaled_backward(ctx, dout, dL, dk_sq_scaled):
    q, k, v, out, L, k_sq_scaled = ctx.saved_tensors
    dq, dk, dv = torch.ops.rbf_attn.scaled_bwd(
        q, k, v, out, L, k_sq_scaled, dout, ctx.is_causal, ctx.sm_scale
    )
    return dq, dk, dv, None


torch.library.register_autograd(
    "rbf_attn::scaled_fwd", rbf_scaled_backward, setup_context=rbf_scaled_setup_context
)


# -------------------------------------------------------------------------
# Clean Python Wrappers
# -------------------------------------------------------------------------
def run_triton_rbf(q, k, v, is_causal=True):
    return torch.ops.rbf_attn.scaled_fwd(q, k, v, is_causal)[0]


def run_triton_non_softmax_rbf(q, k, v, is_causal=True):
    return torch.ops.rbf_attn.non_softmax_fwd(q, k, v, is_causal)[0]


# =========================================================================
# NON-SOFTMAX KERNELS & WRAPPERS
# =========================================================================


@torch.library.custom_op("rbf_attn::non_softmax_fwd", mutates_args=())
def rbf_non_softmax_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, N_CTX, D_HEAD = q.shape
    out = torch.empty_like(q)

    sm_scale = 1.0 / math.sqrt(D_HEAD)
    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

    q_sq_scaled = q.new_empty((B, H, N_CTX), dtype=torch.float32)
    k_sq_scaled = k.new_empty((B, H, N_CTX), dtype=torch.float32)

    grid_sq = (triton.cdiv(N_CTX, 128), B * H, 1)
    _sq_norm_kernel[grid_sq](
        q,
        q_sq_scaled,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_sq_scaled.stride(0),
        q_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=128,
    )
    _sq_norm_kernel[grid_sq](
        k,
        k_sq_scaled,
        sm_scale,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=128,
    )

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)  # noqa: E731
    _rbf_non_softmax_fwd_kernel[grid](
        q,
        k,
        v,
        q_sq_scaled,
        k_sq_scaled,
        sm_scale,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        q_sq_scaled.stride(0),
        q_sq_scaled.stride(1),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )
    return out, q_sq_scaled, k_sq_scaled


@rbf_non_softmax_fwd.register_fake
def _(q, k, v, is_causal):
    B, H, N_CTX, D_HEAD = q.shape
    return (
        q.new_empty(q.shape),
        q.new_empty((B, H, N_CTX), dtype=torch.float32),
        k.new_empty((B, H, N_CTX), dtype=torch.float32),
    )


@torch.library.custom_op("rbf_attn::non_softmax_bwd", mutates_args=())
def rbf_non_softmax_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_sq_scaled: torch.Tensor,
    k_sq_scaled: torch.Tensor,
    dout: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Ensure backward alignments match exactly
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    dout = dout.contiguous()

    B, H, N_CTX, D_HEAD = q.shape

    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    grid_dk_dv = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_N"]), B * H, 1)  # noqa: E731
    _rbf_non_softmax_bwd_dk_dv_kernel[grid_dk_dv](
        q,
        k,
        v,
        q_sq_scaled,
        k_sq_scaled,
        sm_scale,
        dout,
        dk,
        dv,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        q_sq_scaled.stride(0),
        q_sq_scaled.stride(1),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )

    grid_dq = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)  # noqa: E731
    _rbf_non_softmax_bwd_dq_kernel[grid_dq](
        q,
        k,
        v,
        q_sq_scaled,
        k_sq_scaled,
        sm_scale,
        dout,
        dq,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        q_sq_scaled.stride(0),
        q_sq_scaled.stride(1),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )
    return dq, dk, dv


@rbf_non_softmax_bwd.register_fake
def _(q, k, v, q_sq_scaled, k_sq_scaled, dout, is_causal, sm_scale):
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


def rbf_non_softmax_setup_context(ctx, inputs, output):
    q, k, v, is_causal = inputs
    out, q_sq_scaled, k_sq_scaled = output
    ctx.save_for_backward(q, k, v, q_sq_scaled, k_sq_scaled)
    ctx.is_causal = is_causal
    ctx.sm_scale = 1.0 / math.sqrt(q.shape[-1])


def rbf_non_softmax_backward(ctx, dout, dq_sq, dk_sq):
    q, k, v, q_sq_scaled, k_sq_scaled = ctx.saved_tensors
    dq, dk, dv = torch.ops.rbf_attn.non_softmax_bwd(
        q, k, v, q_sq_scaled, k_sq_scaled, dout, ctx.is_causal, ctx.sm_scale
    )
    return dq, dk, dv, None


torch.library.register_autograd(
    "rbf_attn::non_softmax_fwd",
    rbf_non_softmax_backward,
    setup_context=rbf_non_softmax_setup_context,
)

# =========================================================================
# NON-SOFTMAX KERNELS & WRAPPERS
# =========================================================================


@triton.autotune(configs=_get_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_non_softmax_fwd_kernel(
    Q,
    K,
    V,
    Q_sq,
    K_sq,
    sm_scale,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_qsqz,
    stride_qsqh,
    stride_ksqz,
    stride_ksqh,
    stride_oz,
    stride_oh,
    stride_om,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
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
        + offs_d[None, :]
    )
    o_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :]
    )
    q_sq_ptrs = Q_sq + off_z * stride_qsqz + off_h * stride_qsqh + offs_m

    mask_m = offs_m < N_CTX
    if BLOCK_DMODEL == D_HEAD:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        q = tl.load(
            q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    q_sq_scaled = tl.load(q_sq_ptrs, mask=mask_m, other=0.0)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    lo = 0
    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )
    k_sq_ptrs = K_sq + off_z * stride_ksqz + off_h * stride_ksqh + offs_n

    k_ptrs += lo * stride_kn
    v_ptrs += lo * stride_vn
    k_sq_ptrs += lo

    for start_n in range(lo, hi, BLOCK_N):
        curr_offs_n = start_n + offs_n
        mask_n = curr_offs_n < N_CTX

        if BLOCK_DMODEL == D_HEAD:
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )

        k_sq_scaled = tl.load(k_sq_ptrs, mask=mask_n, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        logits = qk * (2.0 * sm_scale) - q_sq_scaled[:, None] - k_sq_scaled[None, :]

        if IS_CAUSAL and start_m * BLOCK_M < start_n + BLOCK_N:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        if start_n + BLOCK_N > N_CTX or start_m * BLOCK_M + BLOCK_M > N_CTX:
            logits = tl.where(mask_m[:, None] & mask_n[None, :], logits, float("-inf"))

        p = tl.math.exp(logits)
        acc += tl.dot(p.to(V.dtype.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        k_sq_ptrs += BLOCK_N

    if BLOCK_DMODEL == D_HEAD:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])
    else:
        tl.store(
            o_ptrs,
            acc.to(Out.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD),
        )


@triton.autotune(configs=_get_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_non_softmax_bwd_dk_dv_kernel(
    Q,
    K,
    V,
    Q_sq,
    K_sq,
    sm_scale,
    DO,
    DK,
    DV,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_qsqz,
    stride_qsqh,
    stride_ksqz,
    stride_ksqh,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    start_n_idx = start_n * BLOCK_N
    offs_n = start_n_idx + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )
    k_sq_ptrs = K_sq + off_z * stride_ksqz + off_h * stride_ksqh + offs_n

    mask_n = offs_n < N_CTX
    if BLOCK_DMODEL == D_HEAD:
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    else:
        k = tl.load(
            k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )
        v = tl.load(
            v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    k_sq_scaled = tl.load(k_sq_ptrs, mask=mask_n, other=0.0)
    k_f32 = k.to(tl.float32)

    dk_dot = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    S_colsum_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    lo = (start_n_idx // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    do_ptrs = (
        DO
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    q_sq_ptrs = Q_sq + off_z * stride_qsqz + off_h * stride_qsqh + offs_m

    q_ptrs += lo * stride_qm
    do_ptrs += lo * stride_qm
    q_sq_ptrs += lo

    for start_m in range(lo, N_CTX, BLOCK_M):
        curr_offs_m = start_m + offs_m
        mask_m = curr_offs_m < N_CTX

        if BLOCK_DMODEL == D_HEAD:
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )
            do = tl.load(
                do_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )

        q_sq_scaled = tl.load(q_sq_ptrs, mask=mask_m, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        logits = qk * (2.0 * sm_scale) - q_sq_scaled[:, None] - k_sq_scaled[None, :]

        if IS_CAUSAL and start_m < start_n_idx + BLOCK_N:
            logits = tl.where(
                curr_offs_m[:, None] >= offs_n[None, :], logits, float("-inf")
            )
        if start_m + BLOCK_M > N_CTX or start_n_idx + BLOCK_N > N_CTX:
            logits = tl.where(mask_m[:, None] & mask_n[None, :], logits, float("-inf"))

        p = tl.math.exp(logits)

        dv += tl.dot(tl.trans(p.to(V.dtype.element_ty)), do)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        S_unscaled = p * dp
        S_scaled = S_unscaled * (2.0 * sm_scale)
        S_scaled_f16 = S_scaled.to(Q.dtype.element_ty)

        S_colsum_acc += tl.sum(S_scaled_f16.to(tl.float32), axis=0)
        dk_dot += tl.dot(tl.trans(S_scaled_f16), q)

        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_qm
        q_sq_ptrs += BLOCK_M

    dk_dot_f16 = dk_dot.to(K.dtype.element_ty).to(tl.float32)
    S_colsum_f16 = S_colsum_acc.to(K.dtype.element_ty).to(tl.float32)

    dk = dk_dot_f16 - S_colsum_f16[:, None] * k_f32

    dk_ptrs = (
        DK
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    dv_ptrs = (
        DV
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )

    if BLOCK_DMODEL == D_HEAD:
        tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])
    else:
        tl.store(
            dk_ptrs,
            dk.to(DK.dtype.element_ty),
            mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD),
        )
        tl.store(
            dv_ptrs,
            dv.to(DV.dtype.element_ty),
            mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD),
        )


@triton.autotune(configs=_get_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_non_softmax_bwd_dq_kernel(
    Q,
    K,
    V,
    Q_sq,
    K_sq,
    sm_scale,
    DO,
    DQ,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_qsqz,
    stride_qsqh,
    stride_ksqz,
    stride_ksqh,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    start_m_idx = start_m * BLOCK_M
    offs_m = start_m_idx + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    do_ptrs = (
        DO
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    q_sq_ptrs = Q_sq + off_z * stride_qsqz + off_h * stride_qsqh + offs_m

    mask_m = offs_m < N_CTX
    if BLOCK_DMODEL == D_HEAD:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        q = tl.load(
            q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )
        do = tl.load(
            do_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    q_sq_scaled = tl.load(q_sq_ptrs, mask=mask_m, other=0.0)
    q_f32 = q.to(tl.float32)

    dq_dot_unscaled = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    S_rowsum_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    hi = tl.minimum(N_CTX, start_m_idx + BLOCK_M) if IS_CAUSAL else N_CTX

    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :]
    )
    k_sq_ptrs = K_sq + off_z * stride_ksqz + off_h * stride_ksqh + offs_n

    for start_n in range(0, hi, BLOCK_N):
        curr_offs_n = start_n + offs_n
        mask_n = curr_offs_n < N_CTX

        if BLOCK_DMODEL == D_HEAD:
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
            )

        k_sq_scaled = tl.load(k_sq_ptrs, mask=mask_n, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        logits = qk * (2.0 * sm_scale) - q_sq_scaled[:, None] - k_sq_scaled[None, :]

        if IS_CAUSAL and start_m_idx < start_n + BLOCK_N:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        if start_m_idx + BLOCK_M > N_CTX or start_n + BLOCK_N > N_CTX:
            logits = tl.where(mask_m[:, None] & mask_n[None, :], logits, float("-inf"))

        p = tl.math.exp(logits)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        S_unscaled = p * dp
        S_scaled = S_unscaled * (2.0 * sm_scale)
        S_scaled_f16 = S_scaled.to(Q.dtype.element_ty)

        S_rowsum_acc += tl.sum(S_scaled_f16.to(tl.float32), axis=1)
        dq_dot_unscaled += tl.dot(S_scaled_f16, k)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        k_sq_ptrs += BLOCK_N

    dq_dot_f16 = dq_dot_unscaled.to(Q.dtype.element_ty).to(tl.float32)
    S_rowsum_f16 = S_rowsum_acc.to(Q.dtype.element_ty).to(tl.float32)

    dq = dq_dot_f16 - S_rowsum_f16[:, None] * q_f32

    dq_ptrs = (
        DQ
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )

    if BLOCK_DMODEL == D_HEAD:
        tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])
    else:
        tl.store(
            dq_ptrs,
            dq.to(DQ.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD),
        )


@torch.library.custom_op("rbf_attn::non_softmax_fwd", mutates_args=())
def rbf_non_softmax_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, N_CTX, D_HEAD = q.shape
    out = torch.empty_like(q)

    sm_scale = 1.0 / math.sqrt(D_HEAD)
    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

    q_sq_scaled = q.new_empty((B, H, N_CTX), dtype=torch.float32)
    k_sq_scaled = k.new_empty((B, H, N_CTX), dtype=torch.float32)

    grid_sq = (triton.cdiv(N_CTX, 128), B * H, 1)
    _sq_norm_kernel[grid_sq](
        q,
        q_sq_scaled,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q_sq_scaled.stride(0),
        q_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=128,
    )
    _sq_norm_kernel[grid_sq](
        k,
        k_sq_scaled,
        sm_scale,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=128,
    )

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)  # noqa: E731
    _rbf_non_softmax_fwd_kernel[grid](
        q,
        k,
        v,
        q_sq_scaled,
        k_sq_scaled,
        sm_scale,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        q_sq_scaled.stride(0),
        q_sq_scaled.stride(1),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )
    return out, q_sq_scaled, k_sq_scaled


@rbf_non_softmax_fwd.register_fake
def _(q, k, v, is_causal):
    B, H, N_CTX, D_HEAD = q.shape
    return (
        torch.empty_like(q),
        q.new_empty((B, H, N_CTX), dtype=torch.float32),
        k.new_empty((B, H, N_CTX), dtype=torch.float32),
    )


@torch.library.custom_op("rbf_attn::non_softmax_bwd", mutates_args=())
def rbf_non_softmax_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_sq_scaled: torch.Tensor,
    k_sq_scaled: torch.Tensor,
    dout: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = dout.contiguous()
    B, H, N_CTX, D_HEAD = q.shape

    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))
    dq = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk = torch.empty_like(k, memory_format=torch.contiguous_format)
    dv = torch.empty_like(v, memory_format=torch.contiguous_format)

    grid_dk_dv = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_N"]), B * H, 1)  # noqa: E731
    _rbf_non_softmax_bwd_dk_dv_kernel[grid_dk_dv](
        q,
        k,
        v,
        q_sq_scaled,
        k_sq_scaled,
        sm_scale,
        dout,
        dk,
        dv,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        q_sq_scaled.stride(0),
        q_sq_scaled.stride(1),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )

    grid_dq = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)  # noqa: E731
    _rbf_non_softmax_bwd_dq_kernel[grid_dq](
        q,
        k,
        v,
        q_sq_scaled,
        k_sq_scaled,
        sm_scale,
        dout,
        dq,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        q_sq_scaled.stride(0),
        q_sq_scaled.stride(1),
        k_sq_scaled.stride(0),
        k_sq_scaled.stride(1),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )
    return dq, dk, dv


@rbf_non_softmax_bwd.register_fake
def _(q, k, v, q_sq_scaled, k_sq_scaled, dout, is_causal, sm_scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def rbf_non_softmax_setup_context(ctx, inputs, output):
    q, k, v, is_causal = inputs
    out, q_sq_scaled, k_sq_scaled = output
    ctx.save_for_backward(q, k, v, q_sq_scaled, k_sq_scaled)
    ctx.is_causal = is_causal
    ctx.sm_scale = 1.0 / math.sqrt(q.shape[-1])


def rbf_non_softmax_backward(ctx, dout, dq_sq, dk_sq):
    q, k, v, q_sq_scaled, k_sq_scaled = ctx.saved_tensors
    dq, dk, dv = torch.ops.rbf_attn.non_softmax_bwd(
        q, k, v, q_sq_scaled, k_sq_scaled, dout, ctx.is_causal, ctx.sm_scale
    )
    return dq, dk, dv, None


torch.library.register_autograd(
    "rbf_attn::non_softmax_fwd",
    rbf_non_softmax_backward,
    setup_context=rbf_non_softmax_setup_context,
)

# =========================================================================
# UTILITIES & POSITIONAL ENCODINGS
# =========================================================================


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    return cos, sin


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)


def get_unrotated_sinusoids(seq_len, dim, device, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.sin(), freqs.cos()), dim=-1)


def compute_rbf_logits(q, k):
    q_f32, k_f32 = q.float(), k.float()
    q_sq = q_f32.pow(2).sum(dim=-1, keepdim=True)
    k_sq = k_f32.pow(2).sum(dim=-1).unsqueeze(-2)
    dot_product = q_f32 @ k_f32.transpose(-2, -1)
    dist_sq = q_sq + k_sq - 2.0 * dot_product
    return (-dist_sq / (q.size(-1) ** 0.5)).to(q.dtype)


_CAUSAL_MASK_CACHE = {}


def get_causal_mask(seq_len, device):
    key = (seq_len, str(device))
    if key not in _CAUSAL_MASK_CACHE:
        _CAUSAL_MASK_CACHE[key] = torch.ones(
            seq_len, seq_len, device=device, dtype=torch.bool
        ).triu_(1)
    return _CAUSAL_MASK_CACHE[key]


_FLEX_MASK_CACHE = {}


def _causal_mask_fn(b, h, q_idx, k_idx):
    return q_idx >= k_idx


def get_causal_mask_flex(seq_len, device):
    dev_str = str(torch.tensor([], device=device).device)
    key = (seq_len, dev_str)
    if key not in _FLEX_MASK_CACHE:
        _FLEX_MASK_CACHE[key] = create_block_mask(
            _causal_mask_fn,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=dev_str,
        )
    return _FLEX_MASK_CACHE[key]


def rbf_flex_attention(q, k, v, is_causal=True):
    b, h, s, d = q.shape
    sm_scale = 1.0 / (d**0.5)

    k_sq_scaled = (torch.sum(k * k, dim=-1, dtype=torch.float32) * sm_scale).to(k.dtype)
    torch._dynamo.graph_break()

    def rbf_score_mod(score, b, h, q_idx, k_idx):
        return (2.0 * score) - k_sq_scaled[b, h, k_idx]

    block_mask = None
    if is_causal:
        block_mask = get_causal_mask_flex(s, q.device)

    return flex_attention(q, k, v, score_mod=rbf_score_mod, block_mask=block_mask)


class CustomCausalAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        emb_dims,
        max_seq_len=2048,
        use_rope=True,
        attention_type="standard",
        use_qk_norm=False,
        apply_xsa=False,
        num_registers=0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.use_qk_norm = use_qk_norm
        self.apply_xsa = apply_xsa
        self.num_registers = num_registers
        self.head_dim = emb_dims // num_heads

        self.qkv_proj = nn.Linear(emb_dims, 3 * emb_dims)
        self.proj = nn.Linear(emb_dims, emb_dims)

        self.positional_encoding_type = "none"
        if use_rope:
            if attention_type.startswith("standard"):
                self.positional_encoding_type = "rope"
            elif attention_type.startswith("rbf"):
                self.positional_encoding_type = "susie"

        if self.positional_encoding_type == "rope":
            cos, sin = precompute_freqs_cis(self.head_dim, max_seq_len)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        elif self.positional_encoding_type == "susie":
            self.pos_dim = self.head_dim
            self.pos_weight = nn.Parameter(
                torch.full((1, num_heads, 1, self.pos_dim // 2), 0.5)
            )
            if self.num_registers > 0:
                self.reg_pos_emb = nn.Parameter(
                    torch.randn(1, num_heads, self.num_registers, self.pos_dim) * 0.02
                )
            susie_cache = get_unrotated_sinusoids(
                max_seq_len, self.pos_dim, device="cpu"
            )
            self.register_buffer("susie_cache", susie_cache, persistent=False)

    def forward(self, x):
        b, s, _ = x.shape
        attn_weights = None
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv, "b s (qkv h n) -> qkv b h s n", qkv=3, h=self.num_heads
        )
        if self.use_qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))

        if self.positional_encoding_type == "susie":
            pos_weight = self.pos_weight.to(q.dtype)  # [1, H, 1, D/2]

            if self.num_registers > 0:
                text_len = s - self.num_registers
                pos_emb_seq = self.susie_cache[:text_len].to(
                    device=q.device, dtype=q.dtype
                )

                # BUG FIX: Replaced .repeat() with highly efficient broadcasting, but swapped .view() for .reshape()
                # because expanding axes natively via broadcasting can disrupt contiguous memory layouts.
                pos_emb_seq = pos_emb_seq.view(1, 1, text_len, 2, self.pos_dim // 2)
                pos_emb_seq = (pos_emb_seq * pos_weight.unsqueeze(-2)).reshape(
                    1, self.num_heads, text_len, self.pos_dim
                )
                pos_emb_reg = self.reg_pos_emb.to(q.dtype)
                pos_emb = torch.cat([pos_emb_reg, pos_emb_seq], dim=2)
            else:
                pos_emb = self.susie_cache[:s].to(device=q.device, dtype=q.dtype)

                pos_emb = pos_emb.view(1, 1, s, 2, self.pos_dim // 2)
                pos_emb = (pos_emb * pos_weight.unsqueeze(-2)).reshape(
                    1, self.num_heads, s, self.pos_dim
                )

            q, k = q + pos_emb, k + pos_emb

        elif self.positional_encoding_type == "rope":
            if self.num_registers > 0:
                q_reg, k_reg = (
                    q[:, :, : self.num_registers, :],
                    k[:, :, : self.num_registers, :],
                )
                q_text, k_text = (
                    q[:, :, self.num_registers :, :],
                    k[:, :, self.num_registers :, :],
                )
                text_len = s - self.num_registers
                q_text, k_text = apply_rotary_pos_emb(
                    q_text, k_text, self.cos[:text_len, :], self.sin[:text_len, :]
                )
                q = torch.cat([q_reg, q_text], dim=2)
                k = torch.cat([k_reg, k_text], dim=2)
            else:
                q, k = apply_rotary_pos_emb(q, k, self.cos[:s, :], self.sin[:s, :])

        if self.attention_type == "standard":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.attention_type == "standard_slow":
            attn_logits = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            causal_mask = get_causal_mask(s, x.device)
            attn_weights = F.softmax(
                attn_logits.masked_fill_(causal_mask, float("-inf")), dim=-1
            )
            out = attn_weights @ v
        elif self.attention_type == "rbf_math":
            k_sq = torch.sum(k * k, dim=-1, keepdim=True, dtype=torch.float32).to(
                k.dtype
            )
            q_prime = torch.cat([q, torch.ones_like(q[..., :1])], dim=-1)
            k_prime = torch.cat([k, -0.5 * k_sq], dim=-1)
            pad_len = (8 - (q_prime.shape[-1] % 8)) % 8
            if pad_len > 0:
                q_prime = F.pad(q_prime, (0, pad_len))
                k_prime = F.pad(k_prime, (0, pad_len))
            v_pad_len = q_prime.shape[-1] - v.shape[-1]
            v_prime = F.pad(v, (0, v_pad_len)) if v_pad_len > 0 else v
            scale = 2.0 / math.sqrt(q.size(-1))
            out = F.scaled_dot_product_attention(
                q_prime, k_prime, v_prime, is_causal=True, scale=scale
            )
            if v_pad_len > 0:
                out = out[..., :-v_pad_len]
        elif self.attention_type == "rbf_triton":
            out = run_triton_rbf(q, k, v, is_causal=True)
        elif self.attention_type == "rbf_slow":
            attn_logits = compute_rbf_logits(q, k)
            causal_mask = get_causal_mask(s, x.device)
            attn_weights = F.softmax(
                attn_logits.masked_fill_(causal_mask, float("-inf")), dim=-1
            )
            out = attn_weights @ v
        elif self.attention_type == "rbf_flex":
            out = rbf_flex_attention(q, k, v, is_causal=True)
        elif self.attention_type == "rbf_non_softmax_slow":
            attn_logits = compute_rbf_logits(q, k)
            causal_mask = get_causal_mask(s, x.device)
            attn_weights = torch.exp(
                attn_logits.masked_fill_(causal_mask, float("-inf"))
            )
            out = attn_weights @ v
        elif self.attention_type == "rbf_non_softmax":
            out = run_triton_non_softmax_rbf(q, k, v, is_causal=True)

        if self.apply_xsa:
            v_n = F.normalize(v, dim=-1)
            out = out - (out * v_n).sum(dim=-1, keepdim=True) * v_n

        out = rearrange(out, "b h s n -> b s (h n)")
        return self.proj(out), attn_weights


# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================
BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 64
SEQ_LENS = [1024, 2048, 4096, 8192]
DEVICE = "cuda"


def rbf_math_forward(q, k, v, is_causal=True):
    k_sq = torch.sum(k * k, dim=-1, keepdim=True, dtype=torch.float32).to(k.dtype)
    q_prime = torch.cat([q, torch.ones_like(q[..., :1])], dim=-1)
    k_prime = torch.cat([k, -0.5 * k_sq], dim=-1)

    pad_len = (8 - (q_prime.shape[-1] % 8)) % 8
    if pad_len > 0:
        q_prime = F.pad(q_prime, (0, pad_len))
        k_prime = F.pad(k_prime, (0, pad_len))

    v_pad_len = q_prime.shape[-1] - v.shape[-1]
    v_prime = F.pad(v, (0, v_pad_len)) if v_pad_len > 0 else v

    scale = 2.0 / math.sqrt(q.size(-1))
    out = F.scaled_dot_product_attention(
        q_prime, k_prime, v_prime, is_causal=is_causal, scale=scale
    )
    return out[..., :-v_pad_len] if v_pad_len > 0 else out


def run_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def run_sdpa_qk_norm(q, k, v):
    q = F.rms_norm(q, (HEAD_DIM,))
    k = F.rms_norm(k, (HEAD_DIM,))
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def run_triton_rbf_bench(q, k, v):
    return run_triton_rbf(q, k, v, is_causal=True)


def profile_memory(func, *args, **kwargs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    out = func(*args, **kwargs)
    peak_mb = (torch.cuda.max_memory_allocated() - base_mem) / (1024 * 1024)
    del out
    return peak_mb


def run_attention_benchmarks():
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")
    print(
        f"{'Seq Len':<10} | {'Method':<25} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Peak VRAM (MB)':<15}"
    )
    print("-" * 80)

    method_names = [
        "SDPA Baseline",
        "SDPA QK-Norm",
        "Naive RBF Math",
        "RBF Triton",
        "RBF Flex-Attention",
    ]
    results = {name: {"fwd": [], "bwd": [], "mem": []} for name in method_names}

    for seq_len in SEQ_LENS:
        compiled_sdpa = torch.compile(run_sdpa)
        compiled_sdpa_qk_norm = torch.compile(run_sdpa_qk_norm)
        compiled_naive = torch.compile(rbf_math_forward)
        compiled_flex = torch.compile(rbf_flex_attention)
        compiled_triton = torch.compile(run_triton_rbf_bench)

        methods = [
            ("SDPA Baseline", compiled_sdpa),
            ("SDPA QK-Norm", compiled_sdpa_qk_norm),
            ("Naive RBF Math", compiled_naive),
            ("RBF Triton", compiled_triton),
            ("RBF Flex-Attention", compiled_flex),
        ]

        q = torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            seq_len,
            HEAD_DIM,
            device=DEVICE,
            dtype=torch.float16,
            requires_grad=True,
        )
        k = torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            seq_len,
            HEAD_DIM,
            device=DEVICE,
            dtype=torch.float16,
            requires_grad=True,
        )
        v = torch.randn(
            BATCH_SIZE,
            NUM_HEADS,
            seq_len,
            HEAD_DIM,
            device=DEVICE,
            dtype=torch.float16,
            requires_grad=True,
        )
        dout = torch.randn_like(q)

        for name, compiled_fn in methods:
            torch._dynamo.reset()
            try:
                for _ in range(3):
                    q.grad, k.grad, v.grad = None, None, None
                    out = compiled_fn(q, k, v)
                    out.backward(dout)
                torch.cuda.empty_cache()

                with torch.no_grad():
                    fwd_ms = triton.testing.do_bench(
                        lambda: compiled_fn(q, k, v), quantiles=None
                    )

                q.grad, k.grad, v.grad = None, None, None
                mem_mb = profile_memory(compiled_fn, q, k, v)

                def fwd_bwd():
                    out_bwd = compiled_fn(q, k, v)
                    out_bwd.backward(dout)

                fwd_bwd_ms = triton.testing.do_bench(
                    fwd_bwd, quantiles=None, grad_to_none=[q, k, v]
                )
                bwd_ms = fwd_bwd_ms - fwd_ms

                torch.cuda.empty_cache()
                print(
                    f"{seq_len:<10} | {name:<25} | {fwd_ms:<10.3f} | {bwd_ms:<10.3f} | {mem_mb:<15.2f}"
                )

                results[name]["fwd"].append(fwd_ms)
                results[name]["bwd"].append(bwd_ms)
                results[name]["mem"].append(mem_mb)

            except Exception:
                print(
                    f"{seq_len:<10} | {name:<25} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<15}"
                )
                results[name]["fwd"].append(float("nan"))
                results[name]["bwd"].append(float("nan"))
                results[name]["mem"].append(float("nan"))

        print("-" * 80)
    return results


def plot_attention_results(results, filename="attention_profiling_results.png"):
    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("fwd", "Forward Time (ms)"),
        ("bwd", "Backward Time (ms)"),
        ("mem", "Peak Forward Activations (MB)"),
    ]

    for ax, (metric_key, title) in zip(axes, metrics):
        for name, data in results.items():
            ax.plot(SEQ_LENS, data[metric_key], marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(title)
        ax.set_xticks(SEQ_LENS)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    filepath = os.path.join("outputs", filename)
    plt.savefig(filepath)
    print(f"\nSaved benchmark plots to '{filepath}'")


def run_layer_benchmarks():
    print(f"Benchmarking Layers on: {torch.cuda.get_device_name(0)}")
    print(
        f"{'Seq Len':<10} | {'Method':<38} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Peak VRAM (MB)':<15}"
    )
    print("-" * 95)

    configs = [
        (
            "SDPA + RoPE",
            {
                "attention_type": "standard",
                "use_rope": True,
                "use_qk_norm": False,
                "apply_xsa": False,
            },
        ),
        (
            "SDPA + RoPE + QK-Norm",
            {
                "attention_type": "standard",
                "use_rope": True,
                "use_qk_norm": True,
                "apply_xsa": False,
            },
        ),
        (
            "SDPA + RoPE + QK-Norm + XSA",
            {
                "attention_type": "standard",
                "use_rope": True,
                "use_qk_norm": True,
                "apply_xsa": True,
            },
        ),
        # (
        #     "RBF Flex + SuSiE",
        #     {
        #         "attention_type": "rbf_flex",
        #         "use_rope": True,
        #         "use_qk_norm": False,
        #         "apply_xsa": False,
        #     },
        # ),
        # (
        #     "RBF Flex + SuSiE + XSA",
        #     {
        #         "attention_type": "rbf_flex",
        #         "use_rope": True,
        #         "use_qk_norm": False,
        #         "apply_xsa": True,
        #     },
        # ),
        (
            "RBF Triton + SuSiE",
            {
                "attention_type": "rbf_triton",
                "use_rope": True,
                "use_qk_norm": False,
                "apply_xsa": False,
            },
        ),
        (
            "RBF Triton + SuSiE + XSA",
            {
                "attention_type": "rbf_triton",
                "use_rope": True,
                "use_qk_norm": False,
                "apply_xsa": True,
            },
        ),
    ]

    results = {name: {"fwd": [], "bwd": [], "mem": []} for name, _ in configs}
    emb_dims = NUM_HEADS * HEAD_DIM

    for seq_len in SEQ_LENS:
        x = torch.randn(
            BATCH_SIZE,
            seq_len,
            emb_dims,
            device=DEVICE,
            dtype=torch.float16,
            requires_grad=True,
        )
        dout = torch.randn_like(x)

        for name, kwargs in configs:
            torch._dynamo.reset()

            layer = CustomCausalAttention(
                num_heads=NUM_HEADS,
                emb_dims=emb_dims,
                max_seq_len=max(SEQ_LENS),
                **kwargs,
            ).to(DEVICE, dtype=torch.float16)

            compiled_layer = torch.compile(layer)

            try:
                # Warmup
                for _ in range(3):
                    x.grad = None
                    out, _ = compiled_layer(x)
                    out.backward(dout)
                torch.cuda.empty_cache()

                # Forward benchmark
                with torch.no_grad():
                    fwd_ms = triton.testing.do_bench(
                        lambda: compiled_layer(x)[0], quantiles=None
                    )

                # Memory profiling
                x.grad = None

                def wrapper_fwd(x_input):
                    return compiled_layer(x_input)[0]

                mem_mb = profile_memory(wrapper_fwd, x)

                # Forward + Backward benchmark
                def fwd_bwd():
                    out_bwd, _ = compiled_layer(x)
                    out_bwd.backward(dout)

                fwd_bwd_ms = triton.testing.do_bench(
                    fwd_bwd, quantiles=None, grad_to_none=[x]
                )
                bwd_ms = fwd_bwd_ms - fwd_ms

                torch.cuda.empty_cache()
                print(
                    f"{seq_len:<10} | {name:<38} | {fwd_ms:<10.3f} | {bwd_ms:<10.3f} | {mem_mb:<15.2f}"
                )

                results[name]["fwd"].append(fwd_ms)
                results[name]["bwd"].append(bwd_ms)
                results[name]["mem"].append(mem_mb)

            except Exception:
                print(
                    f"{seq_len:<10} | {name:<38} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<15}"
                )
                results[name]["fwd"].append(float("nan"))
                results[name]["bwd"].append(float("nan"))
                results[name]["mem"].append(float("nan"))

        print("-" * 95)
    return results


def plot_layer_results(results, filename="layer_profiling_results.png"):
    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("fwd", "Layer Forward Time (ms)"),
        ("bwd", "Layer Backward Time (ms)"),
        ("mem", "Layer Peak Forward Activations (MB)"),
    ]

    for ax, (metric_key, title) in zip(axes, metrics):
        for name, data in results.items():
            ax.plot(SEQ_LENS, data[metric_key], marker="o", label=name)
        ax.set_title(title)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(title)
        ax.set_xticks(SEQ_LENS)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    filepath = os.path.join("outputs", filename)
    plt.savefig(filepath)
    print(f"\nSaved layer benchmark plots to '{filepath}'")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Pre-computing Flex-Attention block masks...")
    for seq_len in SEQ_LENS:
        get_causal_mask_flex(seq_len, DEVICE)

    # run attention benchmarks
    attention_results = run_attention_benchmarks()
    plot_attention_results(attention_results)

    # run layer benchmnarks
    layer_results = run_layer_benchmarks()
    plot_layer_results(layer_results)
