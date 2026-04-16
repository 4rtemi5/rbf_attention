import math

import torch
import torch.nn.functional as F

from rbf_attention import rbf_flex_attention, run_triton_rbf


def rbf_math_forward(q, k, v, is_causal=True):
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
        q_prime, k_prime, v, is_causal=is_causal, scale=scale
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
        f"\nTesting Equivalence -> B={BATCH}, H={HEADS}, SEQ={SEQ_LEN}, DIM={HEAD_DIM}, DTYPE={DTYPE}"
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

    # 3. Clone tensors for the Flex Attention version (to keep gradients isolated)
    q_flex = q_math.clone().detach().requires_grad_(True)
    k_flex = k_math.clone().detach().requires_grad_(True)
    v_flex = v_math.clone().detach().requires_grad_(True)

    # 4. Create a random upstream gradient to simulate backpropagation
    dout = torch.randn_like(q_math)

    # ==========================================
    # FORWARD PASS
    # ==========================================
    out_math = rbf_math_forward(q_math, k_math, v_math, is_causal=True)
    out_triton = run_triton_rbf(q_triton, k_triton, v_triton, True)
    out_flex = torch.compile(rbf_flex_attention)(q_flex, k_flex, v_flex, is_causal=True)

    # ==========================================
    # BACKWARD PASS
    # ==========================================
    out_math.backward(dout)
    out_triton.backward(dout)
    out_flex.backward(dout)

    # ==========================================
    # COMPARISON
    # ==========================================
    def compare_tensors(name, t_math, t_triton, t_flex, atol=None, rtol=1e-3):
        if atol is None:
            atol = 3e-2 if t_math.dtype == torch.float32 else 1e-2

        # Calculate maximum absolute differences against the math baseline
        diff_triton = (t_math - t_triton).abs().max().item()
        diff_flex = (t_math - t_flex).abs().max().item()

        # Check tolerances
        is_close_triton = torch.allclose(t_math, t_triton, atol=atol, rtol=rtol)
        is_close_flex = torch.allclose(t_math, t_flex, atol=atol, rtol=rtol)

        status_triton = "✅ PASS" if is_close_triton else "❌ FAIL"
        status_flex = "✅ PASS" if is_close_flex else "❌ FAIL"

        # Format rows with clear columns for both Triton and Flex
        print(
            f"{name:<7} | {status_triton} (Max Diff: {diff_triton:.6f}) | {status_flex} (Max Diff: {diff_flex:.6f})"
        )
        return is_close_triton, is_close_flex

    # Print Table Headers
    print(f"\n{'-' * 75}")
    print(
        f"{'Tensor':<7} | {'Triton vs Math Baseline':<29} | {'Flex vs Math Baseline':<29}"
    )
    print(f"{'-' * 75}")

    print("--- Forward Pass ---")
    compare_tensors("Output", out_math, out_triton, out_flex)

    print("\n--- Backward Pass (Gradients) ---")
    compare_tensors("dQ", q_math.grad, q_triton.grad, q_flex.grad)
    compare_tensors("dK", k_math.grad, k_triton.grad, k_flex.grad)
    compare_tensors("dV", v_math.grad, v_triton.grad, v_flex.grad)
    print(f"{'-' * 75}")


if __name__ == "__main__":
    # Ensure TF32 is allowed for fair internal comparisons if your GPU supports it
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_equivalence_test(dtype=torch.float16)
    run_equivalence_test(dtype=torch.float32)
