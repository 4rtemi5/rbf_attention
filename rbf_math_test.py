import math

import torch
import torch.nn.functional as F


def compute_rbf_logits_slow(q, k):
    q_sq = q.pow(2).sum(dim=-1, keepdim=True)
    k_sq = k.pow(2).sum(dim=-1).unsqueeze(-2)
    dot_product = q @ k.swapaxes(-2, -1)
    dist_sq = q_sq + k_sq - 2.0 * dot_product
    # Note: relu() omitted here to show pure mathematical float64 equivalency
    return -dist_sq / math.sqrt(q.size(-1))


def test_equivalence():
    b, h, s, d = 2, 4, 128, 64

    # 1. Initialize strictly identical inputs requiring gradients
    torch.manual_seed(42)
    q_slow = torch.randn(b, h, s, d, dtype=torch.float64, requires_grad=True)
    k_slow = torch.randn(b, h, s, d, dtype=torch.float64, requires_grad=True)
    v_slow = torch.randn(b, h, s, d, dtype=torch.float64, requires_grad=True)

    q_math = q_slow.clone().detach().requires_grad_(True)
    k_math = k_slow.clone().detach().requires_grad_(True)
    v_math = v_slow.clone().detach().requires_grad_(True)

    # Random incoming gradient to test Backpropagation
    grad_out = torch.randn(b, h, s, d, dtype=torch.float64)

    # ==========================================
    # PASS 1: ORIGINAL "SLOW" CODE
    # ==========================================
    attn_logits = compute_rbf_logits_slow(q_slow, k_slow)
    causal_mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
    attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
    attn_weights = F.softmax(attn_logits, dim=-1)
    out_slow = attn_weights @ v_slow

    out_slow.backward(grad_out)

    # ==========================================
    # PASS 2: THE "RBF_MATH" TRICK
    # ==========================================
    k_sq = k_math.pow(2).sum(dim=-1, keepdim=True)

    # Append the dimension trackers
    q_prime = torch.cat([q_math, torch.ones_like(q_math[..., :1])], dim=-1)
    k_prime = torch.cat([k_math, -0.5 * k_sq], dim=-1)

    # Pad features to a multiple of 8 with zeros (Doesn't change dot product: 0*0=0)
    pad_len = (8 - (q_prime.shape[-1] % 8)) % 8
    if pad_len > 0:
        q_prime = F.pad(q_prime, (0, pad_len))
        k_prime = F.pad(k_prime, (0, pad_len))

    # NOTE: PyTorch SDPA inherently supports the Value (V) tensor having a
    # different embedding dimension than Q and K, so no V padding is needed!
    scale = 2.0 / math.sqrt(d)

    out_math = F.scaled_dot_product_attention(
        q_prime, k_prime, v_math, is_causal=True, scale=scale
    )

    out_math.backward(grad_out)

    # ==========================================
    # VERIFY DIFFERENCES
    # ==========================================
    print("--- FORWARD PASS ---")
    print(f"Max diff (Outputs): {(out_slow - out_math).abs().max().item():.15f}")

    print("\n--- BACKWARD PASS (GRADIENTS) ---")
    print(f"Max diff (d_q): {(q_slow.grad - q_math.grad).abs().max().item():.15f}")
    print(f"Max diff (d_k): {(k_slow.grad - k_math.grad).abs().max().item():.15f}")
    print(f"Max diff (d_v): {(v_slow.grad - v_math.grad).abs().max().item():.15f}")


if __name__ == "__main__":
    test_equivalence()
