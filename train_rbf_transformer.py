import contextlib
import os
import time
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from rbf_attention import CustomCausalAttention


@dataclass
class TrainingConfig:
    """Configuration for the training script."""

    # Model params
    emb_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 512

    # Dataset params
    batch_size: int = 64
    sample_ratio: str = "100%"
    validation_ratio: float = 0.001

    # Training params
    train_rbf = False
    train_standard = False
    epochs: int = 1
    log_steps: int = 100
    eval_steps: int = 500  # Evaluate every N steps
    learning_rate: float = 3e-3
    warmup_steps: int = 1000
    use_amp: bool = torch.cuda.is_available()
    standard_training_attention: str = "standard"
    rbf_training_attention: str = "rbf"

    # use slow variants for evaluation to be able to output attention maps
    standard_eval_attention: str = "standard_slow"
    rbf_eval_attention: str = "rbf_slow"

    # Generation params
    prompt: str = (
        "Once upon a time, a little boy named Tim wanted to play with a little "
    )
    max_gen_length: int = 512

    # Infrastructure
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        return asdict(self)

    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**d)


def compute_rbf_logits(q, k):
    q_sq = q.pow(2).sum(dim=-1, keepdim=True)
    k_sq = k.pow(2).sum(dim=-1).unsqueeze(-2)
    dot_product = q @ k.swapaxes(-2, -1)
    dist_sq = q_sq + k_sq - 2.0 * dot_product
    dist_sq = torch.relu(dist_sq)
    return -dist_sq / (q.size(-1) ** 0.5)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, attention_type):
        super().__init__()
        self.attn = CustomCausalAttention(num_heads, emb_dim, attention_type)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4), nn.GELU(), nn.Linear(emb_dim * 4, emb_dim)
        )
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class CausalLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=256,
        num_layers=4,
        num_heads=4,
        attention_type="standard",
        max_seq_len=1024,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, emb_dim))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(emb_dim, num_heads, attention_type)
                for _ in range(num_layers)
            ]
        )
        self.to_logits = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, : x.size(1), :]
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            all_attn_weights.append(attn_weights)
        return self.to_logits(x), all_attn_weights


def prepare_tiny_stories(config: TrainingConfig):
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "roneneldan/TinyStories", split=f"train[:{config.sample_ratio}]"
    )
    dataset = dataset.train_test_split(test_size=config.validation_ratio, seed=42)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_seq_len + 1,
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    )
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"])

    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=config.batch_size,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, tokenizer


@contextlib.contextmanager
def oom_detection(device):
    try:
        yield
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA OOM detected! Here is the memory state:")
            print(torch.cuda.memory_summary(device=device))
            torch.cuda.empty_cache()
        raise e


def train_variant(
    model_type,
    train_loader,
    val_loader,
    vocab_size,
    pad_token_id,
    config: TrainingConfig,
    save_path: str,
):
    model = CausalLM(
        vocab_size=vocab_size,
        num_layers=config.num_layers,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        attention_type=model_type,
        max_seq_len=config.max_seq_len,
    ).to(config.device)

    if config.device == "cuda":
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    total_steps = config.epochs * len(train_loader)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - config.warmup_steps,
        eta_min=1e-5,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-5, total_iters=config.warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps],
    )
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    print(f"\n--- Training {model_type.upper()} Attention Model ---")

    train_loss_history, val_loss_history, train_step_history, val_step_history = (
        [],
        [],
        [],
        [],
    )
    global_step = 0
    running_train_loss = 0.0

    config_dict = config.to_dict()
    config_dict["model_type"] = model_type
    with wandb.init(project="fun_attention", config=config.to_dict()) as run:
        run.watch(model, criterion=criterion, log="all", log_freq=config.log_steps)

        t0 = time.time()

        try:
            for epoch in range(config.epochs):
                model.train()

                for batch_idx, batch in enumerate(train_loader):
                    input_ids = batch["input_ids"].to(config.device)
                    attention_mask = batch["attention_mask"].to(config.device)

                    x = input_ids[:, :-1]
                    y = input_ids[:, 1:].clone()
                    mask = attention_mask[:, 1:]
                    y[mask == 0] = -100

                    optimizer.zero_grad()
                    with oom_detection(config.device):
                        with torch.amp.autocast(
                            device_type=config.device,
                            dtype=torch.float16,
                            enabled=config.use_amp,
                        ):
                            logits, _ = model(x)

                            loss = criterion(
                                logits.reshape(-1, vocab_size), y.reshape(-1)
                            )

                        scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    loss_value = loss.item()
                    running_train_loss += loss_value
                    global_step += 1

                    train_loss_history.append(loss_value)
                    train_step_history.append(global_step)

                    log_dict = {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "loss": loss_value,
                        "lr": scheduler.get_last_lr()[0],
                        "train_perplexity": torch.exp(torch.tensor(loss_value)).item(),
                    }

                    if global_step % config.log_steps == 0:
                        t = time.time() - t0
                        print(
                            f"Epoch {epoch + 1} | Step {global_step}/{config.epochs * len(train_loader)} | Time: {t:.2f}s | Train Loss: {loss.item():.4f}"
                        )
                        t0 = time.time()

                    if (
                        config.eval_steps is not None
                        and global_step % config.eval_steps == 0
                    ):
                        avg_train_loss = running_train_loss / config.eval_steps

                        model.eval()
                        total_val_loss = 0
                        with torch.no_grad():
                            for val_batch in val_loader:
                                val_input_ids = val_batch["input_ids"].to(config.device)
                                val_attention_mask = val_batch["attention_mask"].to(
                                    config.device
                                )

                                val_x = val_input_ids[:, :-1]
                                val_y = val_input_ids[:, 1:].clone()
                                val_mask = val_attention_mask[:, 1:]
                                val_y[val_mask == 0] = -100

                                val_logits, _ = model(val_x)
                                val_loss = criterion(
                                    val_logits.reshape(-1, vocab_size),
                                    val_y.reshape(-1),
                                )
                                total_val_loss += val_loss.item()

                        t = time.time() - t0
                        avg_val_loss = total_val_loss / len(val_loader)
                        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

                        val_loss_history.append(avg_val_loss)
                        val_step_history.append(global_step)

                        log_dict["val_loss"] = avg_val_loss
                        log_dict["val_perplexity"] = val_perplexity

                        print(
                            f"*** Eval Step {global_step} Summary  | Val-Time: {t:.2f}s | Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_perplexity:.2f} ***"
                        )

                        running_train_loss = 0.0
                        model.train()
                    t0 = time.time()
                    run.log(log_dict)
            print("Training complete.")
        except KeyboardInterrupt:
            print("Training interrupted by user.")

    print("Saving weights...")

    # Unwrap model if compiled and save weights
    unwrapped_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(unwrapped_model.state_dict(), save_path)
    print(f"Saved weights to {save_path}\n\n")

    return (
        unwrapped_model,
        train_loss_history,
        val_loss_history,
        train_step_history,
        val_step_history,
    )


def generate_story(model, tokenizer, prompt, config: TrainingConfig):
    unwrapped_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    unwrapped_model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)

    with torch.no_grad():
        for _ in range(config.max_gen_length):
            context = input_ids[:, -config.max_seq_len :]
            with torch.amp.autocast(
                device_type=config.device, dtype=torch.float16, enabled=config.use_amp
            ):
                logits, _ = unwrapped_model(context)
            next_token_logits = logits[0, -1, :]

            next_token = (
                torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            )
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def visualize_attention_hf(model, tokenizer, prompt, config, save_path=None):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    tokens = [t.replace("Ġ", "") for t in tokenizer.convert_ids_to_tokens(input_ids[0])]

    with torch.no_grad():
        _, attn_weights = model(input_ids)

    unwrapped_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    attention_type = unwrapped_model.layers[0].attn.attention_type

    if attn_weights is None or attn_weights[0] is None:
        print(
            f"Cannot visualize attention for '{attention_type}' model as weights are not returned."
        )
        return

    attn_weights_to_plot = attn_weights[0][0].cpu()
    num_heads = attn_weights_to_plot.shape[0]

    # FIX 1: Add layout="constrained" to handle dynamic sizing for tall labels
    fig, axes = plt.subplots(
        1, num_heads, figsize=(5 * num_heads, 4), layout="constrained"
    )

    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        cax = axes[h].matshow(attn_weights_to_plot[h], cmap="viridis", vmin=0, vmax=1)
        axes[h].set_xticks(range(len(tokens)))
        axes[h].set_yticks(range(len(tokens)))
        axes[h].set_xticklabels(tokens, rotation=90, ha="left")
        axes[h].set_yticklabels(tokens)

        # FIX 2: Add a pad so the subplot titles don't overlap with the vertical x-ticks
        axes[h].set_title(f"Head {h + 1}", pad=10)

        fig.colorbar(cax, ax=axes[h], fraction=0.046, pad=0.04)

    # FIX 3: Remove the hardcoded y=1.05 and let the layout engine place it properly
    plt.suptitle(f"{attention_type.upper()} Attention Weights")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# --- Execution Block ---

config = TrainingConfig()
os.makedirs("outputs", exist_ok=True)

train_loader, val_loader, tokenizer = prepare_tiny_stories(config)
vocab_size = len(tokenizer)

# Training
if config.train_rbf:
    rbf_model_fast, rbf_train_loss, rbf_val_loss, rbf_train_steps, rbf_val_steps = (
        train_variant(
            config.rbf_training_attention,
            train_loader,
            val_loader,
            vocab_size,
            tokenizer.pad_token_id,
            config=config,
            save_path="outputs/rbf_weights.pt",
        )
    )

if config.train_standard:
    std_model_fast, std_train_loss, std_val_loss, std_train_steps, std_val_steps = (
        train_variant(
            config.standard_training_attention,
            train_loader,
            val_loader,
            vocab_size,
            tokenizer.pad_token_id,
            config=config,
            save_path="outputs/standard_weights.pt",
        )
    )

# Evaluation
print("Loading saved weights into SLOW models for evaluation and visualization...")

std_model_slow = CausalLM(
    vocab_size=vocab_size,
    num_layers=config.num_layers,
    emb_dim=config.emb_dim,
    num_heads=config.num_heads,
    attention_type=config.standard_eval_attention,
    max_seq_len=config.max_seq_len,
).to(config.device)
std_model_slow.load_state_dict(
    torch.load("outputs/standard_weights.pt", weights_only=True)
)

rbf_model_slow = CausalLM(
    vocab_size=vocab_size,
    num_layers=config.num_layers,
    emb_dim=config.emb_dim,
    num_heads=config.num_heads,
    attention_type=config.rbf_eval_attention,
    max_seq_len=config.max_seq_len,
).to(config.device)
rbf_model_slow.load_state_dict(torch.load("outputs/rbf_weights.pt", weights_only=True))

# 3. Generate Stories (Using Slow Models)
print(
    "Standard Attention (Slow Eval):",
    generate_story(std_model_slow, tokenizer, config.prompt, config=config),
    end="\n\n",
)
print(
    "RBF Attention (Slow Eval):",
    generate_story(rbf_model_slow, tokenizer, config.prompt, config=config),
    end="\n\n",
)

# 4. Visualize Attention (Using Slow Models)
visualize_attention_hf(
    std_model_slow,
    tokenizer,
    config.prompt,
    config,
    save_path="outputs/standard_attention.png",
)
visualize_attention_hf(
    rbf_model_slow,
    tokenizer,
    config.prompt,
    config,
    save_path="outputs/rbf_attention.png",
)

# 5. Plot Loss
if config.train_rbf and config.train_standard:
    plt.figure(figsize=(10, 6))
    plt.plot(
        std_train_steps,
        std_train_loss,
        label="Standard Train Loss",
        color="blue",
        linestyle="-",
    )
    plt.plot(
        std_val_steps,
        std_val_loss,
        label="Standard Val Loss",
        color="blue",
        linestyle="--",
    )
    plt.plot(
        rbf_train_steps,
        rbf_train_loss,
        label="New&Fun Train Loss",
        color="orange",
        linestyle="-",
    )
    plt.plot(
        rbf_val_steps,
        rbf_val_loss,
        label="New&Fun Val Loss",
        color="orange",
        linestyle="--",
    )
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/loss_plot.png", bbox_inches="tight")
    plt.close()
