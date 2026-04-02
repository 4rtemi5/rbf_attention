import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from rbf_attention import (
    TritonScaledRBFAttention,
)

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 64
SEQ_LENS = [1024, 2048, 4096, 8192]
DEVICE = "cuda"


def rbf_math_forward(q, k, v):
    """The PyTorch unrolled version for baseline comparison."""
    k_sq = k.float().pow(2).sum(dim=-1, keepdim=True).to(k.dtype)
    q_prime = torch.cat([q, torch.ones_like(q[..., :1])], dim=-1)
    k_prime = torch.cat([k, -0.5 * k_sq], dim=-1)

    pad_len = (8 - (q_prime.shape[-1] % 8)) % 8
    if pad_len > 0:
        q_prime = F.pad(q_prime, (0, pad_len))
        k_prime = F.pad(k_prime, (0, pad_len))

    # [FIX 1]: Pad `v` to perfectly match `q_prime` and `k_prime`'s final dimension.
    # Mismatched dimensions force SDPA to drop out of FlashAttention
    # and use the notoriously slow Math backend.
    v_pad_len = q_prime.shape[-1] - v.shape[-1]
    if v_pad_len > 0:
        v_prime = F.pad(v, (0, v_pad_len))
    else:
        v_prime = v

    scale = 2.0 / math.sqrt(q.size(-1))
    out = F.scaled_dot_product_attention(
        q_prime, k_prime, v_prime, is_causal=True, scale=scale
    )

    # Strip the padding off the value dimension of the outputs
    if v_pad_len > 0:
        out = out[..., :-v_pad_len]

    return out


def rbf_non_softmax_math_forward(q, k, v):
    """The PyTorch unrolled version for the non-softmax baseline comparison."""
    sm_scale = 1.0 / math.sqrt(q.size(-1))
    q_sq = q.float().pow(2).sum(dim=-1, keepdim=True).to(q.dtype) * sm_scale
    k_sq = k.float().pow(2).sum(dim=-1).unsqueeze(-2).to(k.dtype) * sm_scale
    qk = q @ k.transpose(-2, -1)

    logits = qk * (2.0 * sm_scale) - q_sq - k_sq

    s = q.size(2)
    causal_mask = torch.triu(
        torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1
    )
    logits = logits.masked_fill(causal_mask, float("-inf"))

    p = torch.exp(logits)
    return p @ v


# [FIX 2]: Globally cache Flex-Attention's block mask.
# Continuously redefining `causal_mask` inside the function causes cache misses,
# making `create_block_mask` run synchronously on the CPU every forward pass.
_FLEX_MASK_CACHE = {}


def causal_mask_fn(b, h, q_idx, k_idx):
    return q_idx >= k_idx


def rbf_flex_attention(q, k, v, is_causal=True):
    b, h, s, d = q.shape
    sm_scale = 1.0 / (d**0.5)

    k_sq_scaled = k.pow(2).sum(dim=-1) * sm_scale

    def rbf_score_mod(score, b, h, q_idx, k_idx):
        return (2.0 * score) - k_sq_scaled[b, h, k_idx]

    block_mask = None
    if is_causal:
        key = (s, q.device)
        if key not in _FLEX_MASK_CACHE:
            # B=None, H=None natively broadcasts the mask across batches and heads
            _FLEX_MASK_CACHE[key] = create_block_mask(
                causal_mask_fn, B=None, H=None, Q_LEN=s, KV_LEN=s, device=q.device.type
            )
        block_mask = _FLEX_MASK_CACHE[key]

    return flex_attention(q, k, v, score_mod=rbf_score_mod, block_mask=block_mask)


def profile_memory(func, *args, **kwargs):
    """Measures strictly the peak incremental VRAM usage triggered by a function call."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # [FIX 3]: Track base memory prior to execution. We subtract this from the peak
    # to isolate true algorithm footprint from the pre-allocated input tensors.
    base_mem = torch.cuda.memory_allocated()
    out = func(*args, **kwargs)
    peak_mb = (torch.cuda.max_memory_allocated() - base_mem) / (1024 * 1024)

    del out  # Safely clear the graph out of scope
    return peak_mb


# ==========================================
# Wrappers for Compilation
# ==========================================
def run_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def run_triton_rbf(q, k, v):
    return TritonScaledRBFAttention.apply(q, k, v, True)


def run_benchmarks():
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")
    print(
        f"Fixed Dimensions: BATCH={BATCH_SIZE}, HEADS={NUM_HEADS}, HEAD_DIM={HEAD_DIM}\n"
    )
    print(
        f"{'Seq Len':<10} | {'Method':<25} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Peak VRAM (MB)':<15}"
    )
    print("-" * 80)

    results = {
        "SDPA Baseline": {"fwd": [], "bwd": [], "mem": []},
        "Naive RBF Math": {"fwd": [], "bwd": [], "mem": []},
        "RBF Triton": {"fwd": [], "bwd": [], "mem": []},
        "RBF Flex-Attention": {"fwd": [], "bwd": [], "mem": []},
    }

    for seq_len in SEQ_LENS:
        # Reset Dynamo cache to prevent dynamic shape generalization
        torch._dynamo.reset()

        # Compile FRESH for strictly static shapes
        compiled_sdpa = torch.compile(run_sdpa)
        compiled_naive = torch.compile(rbf_math_forward)
        compiled_triton = torch.compile(run_triton_rbf)
        compiled_flex = torch.compile(rbf_flex_attention)

        methods = [
            ("SDPA Baseline", compiled_sdpa),
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
            try:
                # WARMUP
                for _ in range(3):
                    q.grad, k.grad, v.grad = None, None, None
                    out = compiled_fn(q, k, v)
                    out.backward(dout)

                torch.cuda.empty_cache()

                # Measure Forward Time
                fwd_ms = triton.testing.do_bench(
                    lambda: compiled_fn(q, k, v), quantiles=None
                )

                # Measure Memory Overhead
                q.grad, k.grad, v.grad = None, None, None
                mem_mb = profile_memory(compiled_fn, q, k, v)

                # Measure Forward + Backward Time
                def fwd_bwd():
                    out_bwd = compiled_fn(q, k, v)
                    out_bwd.backward(dout)

                # [FIX 4]: Pass grad_to_none explicitly. Removes python overhead from the
                # benchmarking timing event block while securely preventing memory leaks.
                fwd_bwd_ms = triton.testing.do_bench(
                    fwd_bwd, quantiles=None, grad_to_none=[q, k, v]
                )

                # Derive Backward by cleanly subtracting Forward
                bwd_ms = fwd_bwd_ms - fwd_ms

                torch.cuda.empty_cache()

                print(
                    f"{seq_len:<10} | {name:<25} | {fwd_ms:<10.3f} | {bwd_ms:<10.3f} | {mem_mb:<15.2f}"
                )

                results[name]["fwd"].append(fwd_ms)
                results[name]["bwd"].append(bwd_ms)
                results[name]["mem"].append(mem_mb)

            except Exception as e:
                print(
                    f"{seq_len:<10} | {name:<25} | {'ERROR':<10} | {'ERROR':<10} | {'ERROR':<15}"
                )
                print(f"   -> {type(e).__name__}: {str(e)[:100]}...")

                results[name]["fwd"].append(float("nan"))
                results[name]["bwd"].append(float("nan"))
                results[name]["mem"].append(float("nan"))
                torch.cuda.empty_cache()

        print("-" * 80)

    return results


def plot_results(results, filename="attention_profiling_results.png"):
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


if __name__ == "__main__":
    # Ensure TF32 is enabled for fair comparisons
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results = run_benchmarks()
    plot_results(results, filename="attention_profiling_results.png")
