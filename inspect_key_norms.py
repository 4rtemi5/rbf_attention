import os
from unittest.mock import patch

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import rbf_attention

# Import from your existing files
from train_rbf_transformer import CausalLM, TrainingConfig, prepare_tiny_stories


def main():
    config = TrainingConfig()

    # 1. Load the Validation Data
    print("Loading dataset...")
    _, val_loader, tokenizer = prepare_tiny_stories(
        config.batch_size,
        config.max_seq_len,
        config.sample_ratio,
        config.validation_ratio,
        config.num_registers,
    )
    vocab_size = len(tokenizer)

    # 2. Initialize Models and Load Weights
    print("Loading models...")
    std_model = CausalLM(
        vocab_size=vocab_size,
        num_layers=config.num_layers,
        d_model=config.emb_dim,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        pos_emb_type=config.pos_emb_type,
        num_registers=config.num_registers,
        attention_type="standard",
    ).to(config.device)
    std_model.load_state_dict(
        torch.load("outputs/standard_weights.pt", weights_only=True)
    )
    std_model.eval()

    rbf_model = CausalLM(
        vocab_size=vocab_size,
        num_layers=config.num_layers,
        d_model=config.emb_dim,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        pos_emb_type=config.pos_emb_type,
        num_registers=config.num_registers,
        attention_type=config.rbf_eval_attention,
    ).to(config.device)
    rbf_model.load_state_dict(torch.load("outputs/rbf_weights.pt", weights_only=True))
    rbf_model.eval()

    # 3. Define Hook Functions to Intercept Keys
    std_k_norms = []
    rbf_k_norms = []

    original_sdpa = F.scaled_dot_product_attention

    def hooked_sdpa(q, k, v, *args, **kwargs):
        # Calculate L2 norm along the head dimension
        norms = k.norm(dim=-1).detach().cpu().flatten()
        std_k_norms.append(norms)
        return original_sdpa(q, k, v, *args, **kwargs)

    original_rbf = rbf_attention.compute_rbf_logits

    def hooked_rbf(q, k):
        # Calculate L2 norm along the head dimension
        norms = k.norm(dim=-1).detach().cpu().flatten()
        rbf_k_norms.append(norms)
        return original_rbf(q, k)

    # 4. Run Evaluation to Collect Distributions
    print("Extracting Key magnitudes over the validation set...")
    num_batches_to_sample = 10  # Limit batches to avoid running out of memory

    with torch.no_grad():
        # Patch the functions temporarily
        with (
            patch(
                "torch.nn.functional.scaled_dot_product_attention",
                side_effect=hooked_sdpa,
            ),
            patch("rbf_attention.compute_rbf_logits", side_effect=hooked_rbf),
        ):
            for i, batch in enumerate(val_loader):
                if i >= num_batches_to_sample:
                    break

                input_ids = batch["input_ids"].to(config.device)

                # Context x (ignore the target shifting since we just want internal representations)
                x = input_ids[:, :-1]

                with torch.amp.autocast(
                    device_type=config.device,
                    dtype=torch.float16,
                    enabled=config.use_amp,
                ):
                    # Forward passes trigger the mocked functions
                    std_model(x)
                    rbf_model(x)

    # Concatenate all collected norms
    std_k_norms_flat = torch.cat(std_k_norms).numpy()
    rbf_k_norms_flat = torch.cat(rbf_k_norms).numpy()

    # 5. Plot the Distributions
    print("Plotting distributions...")
    plt.figure(figsize=(10, 6))

    # We use density=True to normalize the histograms for a fair comparison
    plt.hist(
        std_k_norms_flat,
        bins=100,
        alpha=0.6,
        density=True,
        label="Standard DP Attention",
    )
    plt.hist(
        rbf_k_norms_flat,
        bins=100,
        alpha=0.6,
        density=True,
        label="RBF Attention",
    )

    plt.title("Distribution of Key Magnitudes (||K||_2) Across Validation Set")
    plt.xlabel("L2 Norm Magnitude")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", alpha=0.75)

    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/key_magnitude_distribution.png"
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot successfully saved to {save_path}")


if __name__ == "__main__":
    main()
