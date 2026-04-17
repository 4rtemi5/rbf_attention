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


def _get_fwd_autotune_configs():
    configs = []
    # Forward passes use fewer registers, can safely use 3-5 stages to hide latency
    for num_stages in [3, 4, 5]:
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


def _get_bwd_autotune_configs():
    configs = []
    # 128x128 halves HBM memory reads. With our register-dropping tricks, it now perfectly fits!
    for num_stages in [1, 2]:
        for block_m, block_n, num_warps in [
            (128, 128, 8),
            (128, 64, 4),
            (128, 64, 8),
            (64, 128, 4),
            (64, 128, 8),
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
# MAIN TRITON KERNELS
# =========================================================================


@triton.autotune(configs=_get_fwd_autotune_configs(), key=["N_CTX"])
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
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vn,
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
    start_m_linear = tl.program_id(0)
    # STRAGGLER FIX: Schedule the heaviest causal blocks first!
    start_m = (
        ((N_CTX + BLOCK_M - 1) // BLOCK_M) - 1 - start_m_linear
        if IS_CAUSAL
        else start_m_linear
    )

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

    # MATH HOISTING: Bake the scaling and log2 conversions directly into Q
    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E
    q = (q * (2.0 * sm_scale_log2)).to(q.dtype)

    m_i_log2 = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    lo = 0
    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    k_ptrs += lo * stride_kn
    v_ptrs += lo * stride_vn

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

        k_f32 = k.to(tl.float32)
        k_sq_log2 = tl.sum(k_f32 * k_f32, axis=1) * sm_scale_log2

        # Eliminates ALL scalar math inside the inner loop completely!
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk -= k_sq_log2[None, :]

        if IS_CAUSAL and start_m * BLOCK_M < start_n + BLOCK_N:
            qk = tl.where(offs_m[:, None] >= curr_offs_n[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where(curr_offs_n[None, :] < N_CTX, qk, float("-inf"))

        m_ij_log2 = tl.maximum(m_i_log2, tl.max(qk, 1))
        qk -= m_ij_log2[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i_log2 - m_ij_log2)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        m_i_log2 = m_ij_log2

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

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
        # Store L perfectly natively in base-2 Log Space
        tl.store(L + off_hz * N_CTX + offs_m, m_i_log2 + tl.math.log2(l_i), mask=mask_m)


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


@triton.autotune(configs=_get_bwd_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_attn_bwd_dk_dv_kernel(
    Q,
    K,
    V,
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
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dvz,
    stride_dvh,
    stride_dvn,
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

    # MATH HOISTING
    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E
    k_sq_log2 = tl.sum(k.to(tl.float32) * k.to(tl.float32), axis=1) * sm_scale_log2

    # Overwrite K in place so the compiler knows to drop it
    k = (k * (2.0 * sm_scale_log2)).to(k.dtype)

    dk_dot = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    ds_sum_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
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
        + off_z * stride_doz
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :]
    )
    l_ptrs = L + off_hz * N_CTX + offs_m
    delta_ptrs = Delta + off_hz * N_CTX + offs_m

    q_ptrs += lo * stride_qm
    do_ptrs += lo * stride_dom
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

        qk_T = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        qk_T += tl.dot(k, tl.trans(q))
        qk_T -= k_sq_log2[:, None]

        if IS_CAUSAL and start_m < start_n_idx + BLOCK_N:
            qk_T = tl.where(
                offs_n[:, None] <= curr_offs_m[None, :], qk_T, float("-inf")
            )
        if start_m + BLOCK_M > N_CTX or start_n_idx + BLOCK_N > N_CTX:
            qk_T = tl.where(mask_n[:, None] & mask_m[None, :], qk_T, float("-inf"))

        # Pure base-2 loads!
        l_i_log2 = tl.load(l_ptrs, mask=mask_m, other=0.0)
        qk_T -= l_i_log2[None, :]
        p_T = tl.math.exp2(qk_T)

        dv += tl.dot(p_T.to(V.dtype.element_ty), do)

        dp_T = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        dp_T += tl.dot(v, tl.trans(do))
        delta = tl.load(delta_ptrs, mask=mask_m, other=0.0)

        dp_T -= delta[None, :]
        dp_T *= p_T

        ds_sum_acc += tl.sum(dp_T, axis=1)
        dk_dot += tl.dot(dp_T.to(Q.dtype.element_ty), q)

        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        l_ptrs += BLOCK_M
        delta_ptrs += BLOCK_M

    # Reload the Unscaled K at the end! This allowed us to free massive registers inside the loop!
    if BLOCK_DMODEL == D_HEAD:
        k_orig = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    else:
        k_orig = tl.load(
            k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    dk = (dk_dot - ds_sum_acc[:, None] * k_orig.to(tl.float32)) * (2.0 * sm_scale)

    dk_ptrs = (
        DK
        + off_z * stride_dkz
        + off_h * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :]
    )
    dv_ptrs = (
        DV
        + off_z * stride_dvz
        + off_h * stride_dvh
        + offs_n[:, None] * stride_dvn
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


@triton.autotune(configs=_get_bwd_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_attn_bwd_dq_kernel(
    Q,
    K,
    V,
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
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    start_m_linear = tl.program_id(0)
    # STRAGGLER FIX
    start_m = (
        ((N_CTX + BLOCK_M - 1) // BLOCK_M) - 1 - start_m_linear
        if IS_CAUSAL
        else start_m_linear
    )

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
        + off_z * stride_doz
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
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

    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E
    q = (q * (2.0 * sm_scale_log2)).to(q.dtype)

    l_i_log2 = tl.load(L + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0)
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

        k_f32 = k.to(tl.float32)
        k_sq_log2 = tl.sum(k_f32 * k_f32, axis=1) * sm_scale_log2

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk -= k_sq_log2[None, :]

        if IS_CAUSAL and start_m_idx < start_n + BLOCK_N:
            qk = tl.where(offs_m[:, None] >= curr_offs_n[None, :], qk, float("-inf"))
        if start_m_idx + BLOCK_M > N_CTX or start_n + BLOCK_N > N_CTX:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        qk -= l_i_log2[:, None]
        p = tl.math.exp2(qk)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        dp -= delta[:, None]
        dp *= p

        dq_acc += tl.dot(dp.to(Q.dtype.element_ty), k)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    dq = dq_acc * (2.0 * sm_scale)

    dq_ptrs = (
        DQ
        + off_z * stride_dqz
        + off_h * stride_dqh
        + offs_m[:, None] * stride_dqm
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


# =========================================================================
# TORCH.LIBRARY CUSTOM OP MIGRATION (PyTorch 2.4+ Native Fusion)
# BUST CACHE: Migrated to v5 to invalidate previous stale graphs
# =========================================================================


@torch.library.custom_op("rbf_attn_v5::scaled_fwd", mutates_args=())
def rbf_scaled_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, N_CTX, D_HEAD = q.shape

    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    L = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)

    sm_scale = 1.0 / math.sqrt(D_HEAD)
    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)  # noqa: E731
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
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
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
    return out, L


@rbf_scaled_fwd.register_fake
def _(q, k, v, is_causal):
    B, H, N_CTX, D_HEAD = q.shape
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32),
    )


@torch.library.custom_op("rbf_attn_v5::scaled_bwd", mutates_args=())
def rbf_scaled_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    L: torch.Tensor,
    dout: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N_CTX, D_HEAD = q.shape

    PREPROCESS_BLOCK_M = 64
    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

    Delta = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)
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

    dq = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk = torch.empty_like(k, memory_format=torch.contiguous_format)
    dv = torch.empty_like(v, memory_format=torch.contiguous_format)

    grid_dk_dv = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_N"]), B * H, 1)  # noqa: E731
    _rbf_attn_bwd_dk_dv_kernel[grid_dk_dv](
        q,
        k,
        v,
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
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
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
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )
    return dq, dk, dv


@rbf_scaled_bwd.register_fake
def _(q, k, v, out, L, dout, is_causal, sm_scale):
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
    )


def rbf_scaled_setup_context(ctx, inputs, output):
    q, k, v, is_causal = inputs
    out, L = output
    ctx.save_for_backward(q, k, v, out, L)
    ctx.is_causal = is_causal
    ctx.sm_scale = 1.0 / math.sqrt(q.shape[-1])


def rbf_scaled_backward(ctx, dout, dL):
    q, k, v, out, L = ctx.saved_tensors
    dq, dk, dv = torch.ops.rbf_attn_v5.scaled_bwd(
        q, k, v, out, L, dout, ctx.is_causal, ctx.sm_scale
    )
    return dq, dk, dv, None


torch.library.register_autograd(
    "rbf_attn_v5::scaled_fwd",
    rbf_scaled_backward,
    setup_context=rbf_scaled_setup_context,
)


# -------------------------------------------------------------------------
# Clean Python Wrappers
# -------------------------------------------------------------------------
def run_triton_rbf(q, k, v, is_causal=True):
    return torch.ops.rbf_attn_v5.scaled_fwd(q, k, v, is_causal)[0]


def run_triton_non_softmax_rbf(q, k, v, is_causal=True):
    return torch.ops.rbf_attn_v5.non_softmax_fwd(q, k, v, is_causal)


# =========================================================================
# NON-SOFTMAX KERNELS
# =========================================================================


@triton.autotune(configs=_get_fwd_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_non_softmax_fwd_kernel(
    Q,
    K,
    V,
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
    start_m_linear = tl.program_id(0)
    start_m = (
        ((N_CTX + BLOCK_M - 1) // BLOCK_M) - 1 - start_m_linear
        if IS_CAUSAL
        else start_m_linear
    )
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

    mask_m = offs_m < N_CTX
    if BLOCK_DMODEL == D_HEAD:
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        q = tl.load(
            q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E
    q_sq_log2 = tl.sum(q.to(tl.float32) * q.to(tl.float32), axis=1) * sm_scale_log2
    q = (q * (2.0 * sm_scale_log2)).to(q.dtype)

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

    k_ptrs += lo * stride_kn
    v_ptrs += lo * stride_vn

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

        k_f32 = k.to(tl.float32)
        k_sq_log2 = tl.sum(k_f32 * k_f32, axis=1) * sm_scale_log2

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        qk -= q_sq_log2[:, None]
        qk -= k_sq_log2[None, :]

        if IS_CAUSAL and start_m * BLOCK_M < start_n + BLOCK_N:
            qk = tl.where(offs_m[:, None] >= curr_offs_n[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX or start_m * BLOCK_M + BLOCK_M > N_CTX:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        p = tl.math.exp2(qk)
        acc += tl.dot(p.to(V.dtype.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    if BLOCK_DMODEL == D_HEAD:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])
    else:
        tl.store(
            o_ptrs,
            acc.to(Out.dtype.element_ty),
            mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD),
        )


@triton.autotune(configs=_get_bwd_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_non_softmax_bwd_dk_dv_kernel(
    Q,
    K,
    V,
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
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dvz,
    stride_dvh,
    stride_dvn,
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

    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E
    k_sq_log2 = tl.sum(k.to(tl.float32) * k.to(tl.float32), axis=1) * sm_scale_log2
    k = (k * (2.0 * sm_scale_log2)).to(k.dtype)

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
        + off_z * stride_doz
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :]
    )

    q_ptrs += lo * stride_qm
    do_ptrs += lo * stride_dom

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

        q_sq_log2 = tl.sum(q.to(tl.float32) * q.to(tl.float32), axis=1) * sm_scale_log2

        qk_T = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        qk_T += tl.dot(k, tl.trans(q))

        qk_T -= q_sq_log2[None, :]
        qk_T -= k_sq_log2[:, None]

        if IS_CAUSAL and start_m < start_n_idx + BLOCK_N:
            qk_T = tl.where(
                offs_n[:, None] <= curr_offs_m[None, :], qk_T, float("-inf")
            )
        if start_m + BLOCK_M > N_CTX or start_n_idx + BLOCK_N > N_CTX:
            qk_T = tl.where(mask_n[:, None] & mask_m[None, :], qk_T, float("-inf"))

        p_T = tl.math.exp2(qk_T)

        dv += tl.dot(p_T.to(V.dtype.element_ty), do)

        dp_T = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        dp_T += tl.dot(v, tl.trans(do))

        dp_T *= p_T
        S_colsum_acc += tl.sum(dp_T, axis=1)
        dk_dot += tl.dot(dp_T.to(Q.dtype.element_ty), q)

        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom

    if BLOCK_DMODEL == D_HEAD:
        k_orig = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    else:
        k_orig = tl.load(
            k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    dk = (dk_dot - S_colsum_acc[:, None] * k_orig.to(tl.float32)) * (2.0 * sm_scale)

    dk_ptrs = (
        DK
        + off_z * stride_dkz
        + off_h * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :]
    )
    dv_ptrs = (
        DV
        + off_z * stride_dvz
        + off_h * stride_dvh
        + offs_n[:, None] * stride_dvn
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


@triton.autotune(configs=_get_bwd_autotune_configs(), key=["N_CTX"])
@triton.jit
def _rbf_non_softmax_bwd_dq_kernel(
    Q,
    K,
    V,
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
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    start_m_linear = tl.program_id(0)
    start_m = (
        ((N_CTX + BLOCK_M - 1) // BLOCK_M) - 1 - start_m_linear
        if IS_CAUSAL
        else start_m_linear
    )

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
        + off_z * stride_doz
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
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

    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E
    q_sq_log2 = tl.sum(q.to(tl.float32) * q.to(tl.float32), axis=1) * sm_scale_log2
    q = (q * (2.0 * sm_scale_log2)).to(q.dtype)

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

        k_sq_log2 = tl.sum(k.to(tl.float32) * k.to(tl.float32), axis=1) * sm_scale_log2

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        qk -= q_sq_log2[:, None]
        qk -= k_sq_log2[None, :]

        if IS_CAUSAL and start_m_idx < start_n + BLOCK_N:
            qk = tl.where(offs_m[:, None] >= curr_offs_n[None, :], qk, float("-inf"))
        if start_m_idx + BLOCK_M > N_CTX or start_n + BLOCK_N > N_CTX:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        p = tl.math.exp2(qk)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        dp *= p
        S_rowsum_acc += tl.sum(dp, axis=1)
        dq_dot_unscaled += tl.dot(dp.to(Q.dtype.element_ty), k)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    if BLOCK_DMODEL == D_HEAD:
        q_orig = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    else:
        q_orig = tl.load(
            q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D_HEAD), other=0.0
        )

    dq = (dq_dot_unscaled - S_rowsum_acc[:, None] * q_orig.to(tl.float32)) * (
        2.0 * sm_scale
    )

    dq_ptrs = (
        DQ
        + off_z * stride_dqz
        + off_h * stride_dqh
        + offs_m[:, None] * stride_dqm
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


@torch.library.custom_op("rbf_attn_v5::non_softmax_fwd", mutates_args=())
def rbf_non_softmax_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
) -> torch.Tensor:
    B, H, N_CTX, D_HEAD = q.shape
    out = torch.empty_like(q, memory_format=torch.contiguous_format)

    sm_scale = 1.0 / math.sqrt(D_HEAD)
    BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)  # noqa: E731
    _rbf_non_softmax_fwd_kernel[grid](
        q,
        k,
        v,
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
    return out


@rbf_non_softmax_fwd.register_fake
def _(q, k, v, is_causal):
    return torch.empty_like(q, memory_format=torch.contiguous_format)


@torch.library.custom_op("rbf_attn_v5::non_softmax_bwd", mutates_args=())
def rbf_non_softmax_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    is_causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
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
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        B,
        H,
        N_CTX,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
        D_HEAD=D_HEAD,
    )
    return dq, dk, dv


@rbf_non_softmax_bwd.register_fake
def _(q, k, v, dout, is_causal, sm_scale):
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
    )


def rbf_non_softmax_setup_context(ctx, inputs, output):
    q, k, v, is_causal = inputs
    ctx.save_for_backward(q, k, v)
    ctx.is_causal = is_causal
    ctx.sm_scale = 1.0 / math.sqrt(q.shape[-1])


def rbf_non_softmax_backward(ctx, dout):
    q, k, v = ctx.saved_tensors
    dq, dk, dv = torch.ops.rbf_attn_v5.non_softmax_bwd(
        q, k, v, dout, ctx.is_causal, ctx.sm_scale
    )
    return dq, dk, dv, None


torch.library.register_autograd(
    "rbf_attn_v5::non_softmax_fwd",
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

    k_sq_scaled = torch.sum(k * k, dim=-1, dtype=torch.float32) * sm_scale

    torch._dynamo.graph_break()  # graph break necessary due to compiler bug :(

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
            weight = torch.cat([self.pos_weight, self.pos_weight], dim=-1).to(q.dtype)

            if self.num_registers > 0:
                text_len = s - self.num_registers
                pos_emb_seq = (
                    self.susie_cache[:text_len]
                    .view(1, 1, text_len, self.pos_dim)
                    .to(device=q.device, dtype=q.dtype)
                    * weight
                )
                pos_emb_reg = self.reg_pos_emb.to(q.dtype)
                pos_emb = torch.cat([pos_emb_reg, pos_emb_seq], dim=2)
            else:
                pos_emb = (
                    self.susie_cache[:s]
                    .view(1, 1, s, self.pos_dim)
                    .to(device=q.device, dtype=q.dtype)
                    * weight
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


def run_attention_benchmarks(method_names):
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")
    print(
        f"{'Seq Len':<10} | {'Method':<25} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Peak VRAM (MB)':<15}"
    )
    print("-" * 80)
    results = {name: {"fwd": [], "bwd": [], "mem": []} for name in method_names}

    for seq_len in SEQ_LENS:
        compiled_methods = {
            "SDPA Baseline": torch.compile(run_sdpa),
            "SDPA QK-Norm": torch.compile(run_sdpa_qk_norm),
            "Naive RBF Math": torch.compile(rbf_math_forward),
            "RBF Triton": torch.compile(run_triton_rbf_bench),
            "RBF Flex-Attention": torch.compile(rbf_flex_attention),
        }

        methods = [(name, compiled_methods[name]) for name in method_names]

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


def run_layer_benchmarks(test_configs):
    print(f"Benchmarking Layers on: {torch.cuda.get_device_name(0)}")
    print(
        f"{'Seq Len':<10} | {'Method':<38} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Peak VRAM (MB)':<15}"
    )
    print("-" * 95)

    base_configs = {
        "SDPA + RoPE": (
            "SDPA + RoPE",
            {
                "attention_type": "standard",
                "use_rope": True,
                "use_qk_norm": False,
                "apply_xsa": False,
            },
        ),
        "SDPA + RoPE + QK-Norm": (
            "SDPA + RoPE + QK-Norm",
            {
                "attention_type": "standard",
                "use_rope": True,
                "use_qk_norm": True,
                "apply_xsa": False,
            },
        ),
        "SDPA + RoPE + QK-Norm + XSA": (
            "SDPA + RoPE + QK-Norm + XSA",
            {
                "attention_type": "standard",
                "use_rope": True,
                "use_qk_norm": True,
                "apply_xsa": True,
            },
        ),
        "RBF Flex + SuSiE": (
            "RBF Flex + SuSiE",
            {
                "attention_type": "rbf_flex",
                "use_rope": True,
                "use_qk_norm": False,
                "apply_xsa": False,
            },
        ),
        "RBF Flex + SuSiE + XSA": (
            "RBF Flex + SuSiE + XSA",
            {
                "attention_type": "rbf_flex",
                "use_rope": True,
                "use_qk_norm": False,
                "apply_xsa": True,
            },
        ),
        "RBF Triton + SuSiE": (
            "RBF Triton + SuSiE",
            {
                "attention_type": "rbf_triton",
                "use_rope": True,
                "use_qk_norm": False,
                "apply_xsa": False,
            },
        ),
        "RBF Triton + SuSiE + XSA": (
            "RBF Triton + SuSiE + XSA",
            {
                "attention_type": "rbf_triton",
                "use_rope": True,
                "use_qk_norm": False,
                "apply_xsa": True,
            },
        ),
    }

    configs = [base_configs[name] for name in test_configs]

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

    attention_methods = [
        "SDPA Baseline",
        "SDPA QK-Norm",
        # "Naive RBF Math",
        "RBF Triton",
        # "RBF Flex-Attention",
    ]

    print("Pre-computing Flex-Attention block masks...")
    for seq_len in SEQ_LENS:
        get_causal_mask_flex(seq_len, DEVICE)

    # run attention benchmarks
    attention_results = run_attention_benchmarks(attention_methods)
    plot_attention_results(attention_results)

    # run layer benchmnarks
    test_configs = [
        # "SDPA + RoPE",
        "SDPA + RoPE + QK-Norm",
        # "SDPA + RoPE + QK-Norm + XSA",
        # "RBF Flex + SuSiE",
        # "RBF Flex + SuSiE + XSA",
        "RBF Triton + SuSiE",
        # "RBF Triton + SuSiE + XSA",
    ]
    layer_results = run_layer_benchmarks(test_configs)
    plot_layer_results(layer_results)
