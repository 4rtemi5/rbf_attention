import math

import torch
import torch.nn.functional as F

from triton_rbf_attention import TritonScaledRBFAttention


def rbf_math_forward(q, k, v):
    """
    Extracts the PyTorch-native math version from your CustomCausalAttention class
    so we can test it directly on Q, K, V tensors without the projection layers.
    """
    k_sq = k.float().pow(2).sum(dim=-1, keepdim=True).to(k.dtype)

    # Pad dimension components onto vectors to construct exact scaling map
    q_prime = torch.cat([q, torch.ones_like(q[..., :1])], dim=-1)
    k_prime = torch.cat([k, -0.5 * k_sq], dim=-1)

    # Align hardware dimensions
    pad_len = (8 - (q_prime.shape[-1] % 8)) % 8
    if pad_len > 0:
        q_prime = F.pad(q_prime, (0, pad_len))
        k_prime = F.pad(k_prime, (0, pad_len))

    scale = 2.0 / math.sqrt(q.size(-1))

    # Use PyTorch's native SDPA for the baseline
    out = F.scaled_dot_product_attention(
        q_prime, k_prime, v, is_causal=True, scale=scale
    )
    return out


def run_equivalence_test(dtype=torch.float16):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("Triton requires a CUDA device. Please run this on a GPU.")
        return

    # Configuration
    BATCH = 2
    HEADS = 4
    SEQ_LEN = 128
    HEAD_DIM = 64
    DTYPE = dtype

    print(
        f"Testing Equivalence -> B={BATCH}, H={HEADS}, SEQ={SEQ_LEN}, DIM={HEAD_DIM}, DTYPE={DTYPE}\n"
    )

    # 1. Initialize identical tensors for the Math Baseline
    q_math = torch.randn(
        BATCH, HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=DTYPE, requires_grad=True
    )
    k_math = torch.randn_like(q_math, requires_grad=True)
    v_math = torch.randn_like(q_math, requires_grad=True)

    # 2. Clone tensors for the Triton version (to keep gradients isolated)
    q_triton = q_math.clone().detach().requires_grad_(True)
    k_triton = k_math.clone().detach().requires_grad_(True)
    v_triton = v_math.clone().detach().requires_grad_(True)

    # 3. Create a random upstream gradient to simulate backpropagation
    dout = torch.randn_like(q_math)

    # ==========================================
    # FORWARD PASS
    # ==========================================
    out_math = rbf_math_forward(q_math, k_math, v_math)
    out_triton = TritonScaledRBFAttention.apply(q_triton, k_triton, v_triton, True)

    # ==========================================
    # BACKWARD PASS
    # ==========================================
    out_math.backward(dout)
    out_triton.backward(dout)

    # ==========================================
    # COMPARISON
    # ==========================================
    def compare_tensors(name, t1, t2, atol=None, rtol=1e-3):
        if atol is None:
            atol = 3e-2 if t1.dtype == torch.float32 else 1e-2
        # Calculate maximum absolute difference
        max_diff = (t1 - t2).abs().max().item()

        # Check if they are close within tolerances
        is_close = torch.allclose(t1, t2, atol=atol, rtol=rtol)

        status = "✅ PASS" if is_close else "❌ FAIL"
        print(f"{name:<10} | {status} | Max Diff: {max_diff:.6f}")
        return is_close

    print("--- Forward Pass ---")
    compare_tensors("Output", out_math, out_triton)

    print("\n--- Backward Pass (Gradients) ---")
    compare_tensors("dQ", q_math.grad, q_triton.grad)
    compare_tensors("dK", k_math.grad, k_triton.grad)
    compare_tensors("dV", v_math.grad, v_triton.grad)


if __name__ == "__main__":
    # Ensure TF32 is allowed for fair internal comparisons if your GPU supports it
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_equivalence_test(dtype=torch.float16)
    run_equivalence_test(dtype=torch.float32)
