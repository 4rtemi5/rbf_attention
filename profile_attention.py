import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import triton

from triton_rbf_attention import (
    TritonNonSoftmaxRBFAttention,
    TritonScaledRBFAttention,
)

# ==========================================
# Configuration
# ==========================================
BATCH_SIZE = 4
NUM_HEADS = 8
HEAD_DIM = 64
# Sweeping from 1K to 8K context lengths
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

    scale = 2.0 / math.sqrt(q.size(-1))
    return F.scaled_dot_product_attention(
        q_prime, k_prime, v, is_causal=True, scale=scale
    )


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


def profile_memory(func, *args, **kwargs):
    """Measures peak VRAM usage of a function call."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = func(*args, **kwargs)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return peak_mb


def run_benchmarks():
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")
    print(
        f"Fixed Dimensions: BATCH={BATCH_SIZE}, HEADS={NUM_HEADS}, HEAD_DIM={HEAD_DIM}\n"
    )
    print(
        f"{'Seq Len':<10} | {'Method':<15} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Peak VRAM (MB)':<15}"
    )
    print("-" * 65)

    results = {
        "SDPA Baseline": {"fwd": [], "bwd": [], "mem": []},
        "RBF Math": {"fwd": [], "bwd": [], "mem": []},
        "RBF Triton": {"fwd": [], "bwd": [], "mem": []},
        "Non-Softmax Math": {"fwd": [], "bwd": [], "mem": []},
        "Non-Softmax Triton": {"fwd": [], "bwd": [], "mem": []},
    }

    for seq_len in SEQ_LENS:
        # 1. Initialize Tensors
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

        methods = [
            (
                "SDPA Baseline",
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            ),
            ("RBF Math", lambda: rbf_math_forward(q, k, v)),
            ("RBF Triton", lambda: TritonScaledRBFAttention.apply(q, k, v, True)),
            ("Non-Softmax Math", lambda: rbf_non_softmax_math_forward(q, k, v)),
            (
                "Non-Softmax Triton",
                lambda: TritonNonSoftmaxRBFAttention.apply(q, k, v, True),
            ),
        ]

        for name, fn in methods:
            try:
                # Clear grads
                q.grad, k.grad, v.grad = None, None, None

                # Measure Forward Time
                fwd_ms = triton.testing.do_bench(fn, quantiles=None)

                # Measure Memory (Forward only to isolate materialization footprint)
                mem_mb = profile_memory(fn)

                # Measure Backward Time
                out = fn()
                bwd_ms = triton.testing.do_bench(
                    lambda: out.backward(dout, retain_graph=True), quantiles=None
                )

                # --- FIX: CLEAR THE AUTOGRAD GRAPH MEMORY LEAK ---
                del out
                torch.cuda.empty_cache()

                print(
                    f"{seq_len:<10} | {name:<15} | {fwd_ms:<10.3f} | {bwd_ms:<10.3f} | {mem_mb:<15.2f}"
                )

                results[name]["fwd"].append(fwd_ms)
                results[name]["bwd"].append(bwd_ms)
                results[name]["mem"].append(mem_mb)

            except RuntimeError:
                # Catch OOM errors for math version at high sequence lengths
                print(
                    f"{seq_len:<10} | {name:<15} | {'OOM':<10} | {'OOM':<10} | {'OOM':<15}"
                )
                results[name]["fwd"].append(float("nan"))
                results[name]["bwd"].append(float("nan"))
                results[name]["mem"].append(float("nan"))

                # Cleanup in case of OOM
                torch.cuda.empty_cache()

        print("-" * 65)

    return results


def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("fwd", "Forward Time (ms)"),
        ("bwd", "Backward Time (ms)"),
        ("mem", "Peak VRAM (MB)"),
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
    plt.savefig("outputs/attention_profiling_results.png")
    print("\nSaved benchmark plots to 'outputs/attention_profiling_results.png'")


if __name__ == "__main__":
    # Ensure TF32 is enabled for fair comparisons
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results = run_benchmarks()
    plot_results(results)
