import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange


def _get_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=2),
    ]


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["N_CTX", "D_HEAD"],
)
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

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    lo = 0
    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n_idx = tl.multiple_of(start_n, BLOCK_N)
        curr_offs_n = start_n_idx + offs_n

        k = tl.load(
            k_ptrs + start_n_idx * stride_kn,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )
        v = tl.load(
            v_ptrs + start_n_idx * stride_vn,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        k_f32 = k.to(tl.float32)
        k_sq = tl.sum(k_f32 * k_f32, axis=1)

        # Exact PyTorch SDPA testing matching (FP16 k_sq truncation explicitly enforced)
        k_sq_f16 = k_sq.to(K.dtype.element_ty).to(tl.float32)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # Exact matching with PyTorch SDPA scaling sequence
        logits = (qk - 0.5 * k_sq_f16[None, :]) * (2.0 * sm_scale)

        if IS_CAUSAL and start_m * BLOCK_M < start_n_idx + BLOCK_N:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        if start_n_idx + BLOCK_N > N_CTX:
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
    tl.store(Delta + off_hz * N_CTX + offs_m, delta.to(tl.float32), mask=offs_m < N_CTX)


# =========================================================================
# FA2 BACKWARD KERNEL 1: Accumulate dK and dV cleanly (Threadblock over Keys)
# =========================================================================
@triton.autotune(
    configs=_get_autotune_configs(),
    key=["N_CTX", "D_HEAD"],
)
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

    start_n_idx = start_n * BLOCK_N
    offs_n = start_n_idx + tl.arange(0, BLOCK_N)
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

    k_f32 = k.to(tl.float32)
    k_sq = tl.sum(k_f32 * k_f32, axis=1)
    k_sq_f16 = k_sq.to(K.dtype.element_ty).to(tl.float32)

    # [OPTIMIZATION] Setup pure accumulators completely outside the loop
    dk_dot = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    ds_scaled_sum_k_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    lo = (start_n_idx // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

    for start_m in range(lo, N_CTX, BLOCK_M):
        start_m_idx = start_m
        curr_offs_m = start_m_idx + offs_m

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

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        logits = (qk - 0.5 * k_sq_f16[None, :]) * (2.0 * sm_scale)

        if IS_CAUSAL and start_m_idx < start_n_idx + BLOCK_N:
            logits = tl.where(
                curr_offs_m[:, None] >= offs_n[None, :], logits, float("-inf")
            )
        if start_m_idx + BLOCK_M > N_CTX or start_n_idx + BLOCK_N > N_CTX:
            logits = tl.where(
                (curr_offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
                logits,
                float("-inf"),
            )

        logits = logits.to(tl.float32)
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

        d_logits = p * (dp - delta[:, None])
        ds_scaled = (d_logits * (2.0 * sm_scale)).to(tl.float32)

        # [OPTIMIZATION] Maintain clean Tensor Core instructions
        ds_scaled_sum_k_acc += tl.sum(ds_scaled, axis=0)
        dk_dot += tl.dot(tl.trans(ds_scaled).to(Q.dtype.element_ty), q)

    # [OPTIMIZATION] Apply vector scaling EXACTLY ONCE outside the loop
    dk = dk_dot - ds_scaled_sum_k_acc[:, None] * k_f32

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
        + offs_d[None, :] * stride_vk
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


# =========================================================================
# FA2 BACKWARD KERNEL 2: Accumulate dQ cleanly (Threadblock over Queries)
# =========================================================================
@triton.autotune(
    configs=_get_autotune_configs(),
    key=["N_CTX", "D_HEAD"],
)
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
        + offs_d[None, :] * stride_qk
    )
    do_ptrs = (
        DO
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )

    q = tl.load(
        q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )
    do = tl.load(
        do_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )

    l_i = tl.load(L + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = tl.minimum(N_CTX, start_m_idx + BLOCK_M) if IS_CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK_N):
        start_n_idx = start_n
        curr_offs_n = start_n_idx + offs_n

        k_ptrs = (
            K
            + off_z * stride_kz
            + off_h * stride_kh
            + curr_offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kk
        )
        v_ptrs = (
            V
            + off_z * stride_vz
            + off_h * stride_vh
            + curr_offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vk
        )

        k = tl.load(
            k_ptrs,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )
        v = tl.load(
            v_ptrs,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        k_f32 = k.to(tl.float32)
        k_sq = tl.sum(k_f32 * k_f32, axis=1)
        k_sq_f16 = k_sq.to(K.dtype.element_ty).to(tl.float32)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        logits = (qk - 0.5 * k_sq_f16[None, :]) * (2.0 * sm_scale)

        if IS_CAUSAL:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        logits = tl.where(
            (offs_m[:, None] < N_CTX) & (curr_offs_n[None, :] < N_CTX),
            logits,
            float("-inf"),
        )

        logits = logits.to(tl.float32)
        p = tl.math.exp(logits - l_i[:, None])

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        d_logits = p * (dp - delta[:, None])
        ds_scaled = (d_logits * (2.0 * sm_scale)).to(tl.float32)

        dq += tl.dot(ds_scaled.to(Q.dtype.element_ty), k)

    dq_ptrs = (
        DQ
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    tl.store(
        dq_ptrs,
        dq.to(DQ.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
    )


class TritonScaledRBFAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, H, N_CTX, D_HEAD = q.shape
        out = torch.empty_like(q)
        L = torch.empty((B, H, N_CTX), device=q.device, dtype=torch.float32)

        sm_scale = 1.0 / math.sqrt(D_HEAD)
        BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

        grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)
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
            BLOCK_DMODEL=BLOCK_DMODEL,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(q, k, v, out, L)
        ctx.sm_scale, ctx.is_causal = sm_scale, is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, L = ctx.saved_tensors
        dout = dout.contiguous()
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
            out.stride(3),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            B,
            H,
            N_CTX,
            D_HEAD,
            BLOCK_M=PREPROCESS_BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
        )

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        # Launch distinct kernels across their respective parallelization dimensions to bypass atomics
        grid_dk_dv = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_N"]), B * H, 1)
        _rbf_attn_bwd_dk_dv_kernel[grid_dk_dv](
            q,
            k,
            v,
            ctx.sm_scale,
            dout,
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
            BLOCK_DMODEL=BLOCK_DMODEL,
            IS_CAUSAL=ctx.is_causal,
        )

        grid_dq = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)
        _rbf_attn_bwd_dq_kernel[grid_dq](
            q,
            k,
            v,
            ctx.sm_scale,
            dout,
            dq,
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
            BLOCK_DMODEL=BLOCK_DMODEL,
            IS_CAUSAL=ctx.is_causal,
        )
        return dq, dk, dv, None


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["N_CTX", "D_HEAD"],
)
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

    # PRE-COMPUTE scaled Q norm to fixe numerical instability while preserving performance
    q_f32 = q.to(tl.float32)
    q_sq_scaled = tl.sum(q_f32 * q_f32, axis=1) * sm_scale
    sm_scale_x2 = 2.0 * sm_scale

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    lo = 0
    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    # Set pointer bases cleanly outside the loop
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

    for start_n in range(lo, hi, BLOCK_N):
        start_n_idx = tl.multiple_of(start_n, BLOCK_N)
        curr_offs_n = start_n_idx + offs_n

        # Fast 1D scalar pointer advancements
        k = tl.load(
            k_ptrs + start_n_idx * stride_kn,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )
        v = tl.load(
            v_ptrs + start_n_idx * stride_vn,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        # PRE-COMPUTE scaled K norm
        k_f32 = k.to(tl.float32)
        k_sq_scaled = tl.sum(k_f32 * k_f32, axis=1) * sm_scale

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        # Fused Logits (Inherently bounds <= 0, protecting against exp() fp32 overflow limits)
        logits = qk * sm_scale_x2 - q_sq_scaled[:, None] - k_sq_scaled[None, :]

        if IS_CAUSAL and start_m * BLOCK_M < start_n_idx + BLOCK_N:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        if start_n_idx + BLOCK_N > N_CTX or start_m * BLOCK_M + BLOCK_M > N_CTX:
            logits = tl.where(
                (offs_m[:, None] < N_CTX) & (curr_offs_n[None, :] < N_CTX),
                logits,
                float("-inf"),
            )

        p = tl.math.exp(logits)
        acc += tl.dot(p.to(V.dtype.element_ty), v)

    tl.store(
        o_ptrs,
        acc.to(Out.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
    )


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["N_CTX", "D_HEAD"],
)
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

    start_n_idx = start_n * BLOCK_N
    offs_n = start_n_idx + tl.arange(0, BLOCK_N)
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

    k_f32 = k.to(tl.float32)
    k_sq_scaled = tl.sum(k_f32 * k_f32, axis=1) * sm_scale
    sm_scale_x2 = 2.0 * sm_scale

    # Setup pure accumulators completely outside the loop
    dk_dot = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    S_colsum_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    lo = (start_n_idx // BLOCK_M) * BLOCK_M if IS_CAUSAL else 0

    # Set pointer bases cleanly outside the loop
    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    do_ptrs = (
        DO
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )

    for start_m in range(lo, N_CTX, BLOCK_M):
        start_m_idx = start_m
        curr_offs_m = start_m_idx + offs_m

        q = tl.load(
            q_ptrs + start_m_idx * stride_qm,
            mask=(curr_offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        q_f32 = q.to(tl.float32)
        q_sq_scaled = tl.sum(q_f32 * q_f32, axis=1) * sm_scale

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        logits = qk * sm_scale_x2 - q_sq_scaled[:, None] - k_sq_scaled[None, :]

        if IS_CAUSAL and start_m_idx < start_n_idx + BLOCK_N:
            logits = tl.where(
                curr_offs_m[:, None] >= offs_n[None, :], logits, float("-inf")
            )
        if start_m_idx + BLOCK_M > N_CTX or start_n_idx + BLOCK_N > N_CTX:
            logits = tl.where(
                (curr_offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX),
                logits,
                float("-inf"),
            )

        p = tl.math.exp(logits)

        do = tl.load(
            do_ptrs + start_m_idx * stride_qm,
            mask=(curr_offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        dv += tl.dot(tl.trans(p.to(V.dtype.element_ty)), do)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        S = (p * dp * (-sm_scale_x2)).to(tl.float32)

        # Tensor core accumulation inside loop
        S_colsum_acc += tl.sum(S, axis=0)
        dk_dot += tl.dot(tl.trans(S).to(Q.dtype.element_ty), q)

    # Apply mathematical hoist ONCE
    dk = S_colsum_acc[:, None] * k_f32 - dk_dot

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
        + offs_d[None, :] * stride_vk
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


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["N_CTX", "D_HEAD"],
)
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
        + offs_d[None, :] * stride_qk
    )
    do_ptrs = (
        DO
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )

    q = tl.load(
        q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )
    do = tl.load(
        do_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD), other=0.0
    )

    q_f32 = q.to(tl.float32)
    q_sq_scaled = tl.sum(q_f32 * q_f32, axis=1) * sm_scale
    sm_scale_x2 = 2.0 * sm_scale

    # Pure accumulators completely outside the loop
    dq_dot = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    S_rowsum_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    hi = tl.minimum(N_CTX, start_m_idx + BLOCK_M) if IS_CAUSAL else N_CTX

    # Set pointer bases outside the loop
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

    for start_n in range(0, hi, BLOCK_N):
        start_n_idx = start_n
        curr_offs_n = start_n_idx + offs_n

        k = tl.load(
            k_ptrs + start_n_idx * stride_kn,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )
        v = tl.load(
            v_ptrs + start_n_idx * stride_vn,
            mask=(curr_offs_n[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
            other=0.0,
        )

        k_f32 = k.to(tl.float32)
        k_sq_scaled = tl.sum(k_f32 * k_f32, axis=1) * sm_scale

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))

        logits = qk * sm_scale_x2 - q_sq_scaled[:, None] - k_sq_scaled[None, :]

        # Boundary tracking directly mirrored from dK_dV logic.
        if IS_CAUSAL and start_m_idx < start_n_idx + BLOCK_N:
            logits = tl.where(
                offs_m[:, None] >= curr_offs_n[None, :], logits, float("-inf")
            )
        if start_m_idx + BLOCK_M > N_CTX or start_n_idx + BLOCK_N > N_CTX:
            logits = tl.where(
                (offs_m[:, None] < N_CTX) & (curr_offs_n[None, :] < N_CTX),
                logits,
                float("-inf"),
            )

        p = tl.math.exp(logits)

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        S = (p * dp * (-sm_scale_x2)).to(tl.float32)

        # Pure tensor core accumulation inside loop
        S_rowsum_acc += tl.sum(S, axis=1)
        dq_dot += tl.dot(S.to(K.dtype.element_ty), k)

    # Apply mathematical hoist outside the loop
    dq = S_rowsum_acc[:, None] * q_f32 - dq_dot

    dq_ptrs = (
        DQ
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    tl.store(
        dq_ptrs,
        dq.to(DQ.dtype.element_ty),
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < D_HEAD),
    )


class TritonNonSoftmaxRBFAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, H, N_CTX, D_HEAD = q.shape
        out = torch.empty_like(q)

        sm_scale = 1.0 / math.sqrt(D_HEAD)
        BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

        grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)
        _rbf_non_softmax_fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            out,
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
            BLOCK_DMODEL=BLOCK_DMODEL,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(q, k, v)
        ctx.sm_scale, ctx.is_causal = sm_scale, is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        dout = dout.contiguous()
        B, H, N_CTX, D_HEAD = q.shape

        BLOCK_DMODEL = max(16, triton.next_power_of_2(D_HEAD))

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        grid_dk_dv = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_N"]), B * H, 1)
        _rbf_non_softmax_bwd_dk_dv_kernel[grid_dk_dv](
            q,
            k,
            v,
            ctx.sm_scale,
            dout,
            dk,
            dv,
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
            BLOCK_DMODEL=BLOCK_DMODEL,
            IS_CAUSAL=ctx.is_causal,
        )

        grid_dq = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H, 1)
        _rbf_non_softmax_bwd_dq_kernel[grid_dq](
            q,
            k,
            v,
            ctx.sm_scale,
            dout,
            dq,
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
            BLOCK_DMODEL=BLOCK_DMODEL,
            IS_CAUSAL=ctx.is_causal,
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
        assert (
            attention_type
            in [
                "standard",  # Standard Attention using F.scaled_dot_product_attention
                "standard_slow",  # Explicit baseline implementation of Attention
                "rbf_math",  # Mathematical equivalent of rbf-attention using F.scaled_dot_product_attention
                "rbf",  # Triton Implementation of RBF-Attention
                "rbf_slow",  # Baseline implementation of RBF-Attention
                "rbf_non_softmax_slow",  # Non-Softmax RBF-Attention
                "rbf_non_softmax",  # Triton Implementation of Non-Softmax RBF-Attention
            ]
        ), f"Unknown attention type: {attention_type}"
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
            # Standard Attention using F.scaled_dot_product_attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        elif self.attention_type == "standard_slow":
            # Explicit baseline implementation of Attention
            attn_logits = q @ k.transpose(-2, -1)
            attn_logits = attn_logits / (q.size(-1) ** 0.5)
            causal_mask = torch.triu(
                torch.ones(s, s, device=x.device), diagonal=1
            ).bool()
            attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1)
            out = attn_weights @ v

        elif self.attention_type == "rbf_math":
            # Mathematical equivalent of rbf-attention using F.scaled_dot_product_attention
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

        elif self.attention_type == "rbf":
            # Triton Implementation of RBF-Attention
            out = TritonScaledRBFAttention.apply(q, k, v, True)

        elif self.attention_type == "rbf_slow":
            # Baseline implementation of RBF-Attention
            attn_logits = compute_rbf_logits(q, k)
            causal_mask = torch.triu(
                torch.ones(s, s, device=x.device), diagonal=1
            ).bool()
            attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1)
            out = attn_weights @ v

        elif self.attention_type == "rbf_non_softmax_slow":
            # Non-Softmax RBF-Attention
            attn_logits = compute_rbf_logits(q, k)
            causal_mask = torch.triu(
                torch.ones(s, s, device=x.device), diagonal=1
            ).bool()
            attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
            attn_weights = torch.exp(attn_logits)
            out = attn_weights @ v
        elif self.attention_type == "rbf_non_softmax":
            # Triton Implementation of Non-Softmax RBF-Attention
            out = TritonNonSoftmaxRBFAttention.apply(q, k, v, True)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        out = rearrange(out, "b h s n -> b s (h n)")
        return self.proj(out), attn_weights
