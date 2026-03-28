import contextlib
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def setup_logging(rank: int) -> logging.Logger:
    """Configures logging such that only rank 0 logs INFO/DEBUG, others log WARNING+."""
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] [Rank %(rank)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record

    logging.setLogRecordFactory(record_factory)
    return logging.getLogger(__name__)


@dataclass
class Config:
    # data params
    num_classes: int = 10
    image_size: int = 224

    # training params
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_workers: int = 4
    pin_memory: bool = True

    # Distributed & Production params
    seed: int = 42
    use_amp: bool = True
    grad_accum_steps: int = 4
    clip_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 5
    test_interval: int = 1
    checkpoint_dir: str = "checkpoints"


def set_seed(seed: int, rank: int = 0):
    """Sets seed for reproducibility. Offsets the seed by rank for diverse augmentations."""
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> Tuple[torch.device, int, int, int, bool]:
    """Initializes distributed training handling torchrun or SLURM environments."""
    is_distributed = (
        int(os.environ.get("WORLD_SIZE", 0)) > 1
        or int(os.environ.get("SLURM_NTASKS", 0)) > 1
    )

    if is_distributed:
        if "RANK" in os.environ:  # standard torchrun / torch.distributed.launch
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        elif "SLURM_PROCID" in os.environ:  # SLURM fallback
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            local_rank = int(
                os.environ.get("SLURM_LOCALID", rank % (torch.cuda.device_count() or 1))
            )
        else:
            raise RuntimeError("Could not infer distributed environment setup.")

        # Setup device robustly
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            torch.accelerator.set_device_index(local_rank)
            device = torch.device(
                f"{torch.accelerator.current_accelerator().type}:{local_rank}"
            )
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        backend = dist.get_default_backend_for_device(device)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    else:
        rank, local_rank, world_size = 0, 0, 1
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            device = torch.device(torch.accelerator.current_accelerator().type)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, rank, local_rank, world_size, is_distributed


# --- Dummy Implementations (as requested) ---


class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=10, num_classes=10):
        self.inputs = torch.randn(num_samples, input_size)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def build_dataloaders(
    config: Config, is_distributed: bool
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    """Builds dummy train/val datasets and dataloaders with standard distributed samplers."""
    train_dataset = DummyDataset(num_samples=1000, num_classes=config.num_classes)
    val_dataset = DummyDataset(num_samples=200, num_classes=config.num_classes)

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, train_sampler


def build_model(config: Config) -> nn.Module:
    """Returns a dummy model architecture."""
    return nn.Sequential(
        nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, config.num_classes)
    )


def build_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """Returns a dummy optimizer."""
    return optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def build_criterion(config: Config) -> nn.Module:
    """Returns a dummy criterion (loss function)."""
    return nn.CrossEntropyLoss()


# --- Core Mechanics ---


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Any,
    config: Config,
):
    """Saves model checkpoint. Uses DCP for FSDP sharded checkpoints."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}")

    if isinstance(model, FSDP):
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT)
        state_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": FSDP.optim_state_dict(model, optimizer),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "config": config,
        }
        dcp.save(state_dict, checkpoint_id=path)
        logging.getLogger(__name__).info(f"Distributed Checkpoint saved: {path}")
    else:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "config": config,
            }
            path = f"{path}.pt"
            temp_path = f"{path}.tmp"
            torch.save(state, temp_path)
            os.rename(temp_path, path)
            logging.getLogger(__name__).info(f"Checkpoint saved: {path}")


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Any,
    device: torch.device,
    config: Config,
    rank: int,
):
    """Production grade training loop with AMP, Grad Accumulation, and Clipping."""
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    num_samples = torch.tensor(0.0, device=device)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        is_accumulating = (batch_idx + 1) % config.grad_accum_steps != 0 and (
            batch_idx + 1
        ) != len(dataloader)
        sync_context = (
            model.no_sync()
            if is_accumulating and hasattr(model, "no_sync")
            else contextlib.nullcontext()
        )

        with sync_context:
            # Forward pass with AMP
            with torch.amp.autocast(device_type=device.type, enabled=config.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Normalize loss to account for gradient accumulation
                loss = loss / config.grad_accum_steps

            # Backward pass
            scaler.scale(loss).backward()

        # Step when accumulated enough gradients or at the very end of epoch
        if not is_accumulating:
            if config.clip_grad_norm > 0.0:
                # Unscale prior to clipping
                scaler.unscale_(optimizer)
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(config.clip_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.clip_grad_norm
                    )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Track metrics (multiply back by grad_accum_steps for accurate logging)
        loss_val = loss.detach() * config.grad_accum_steps
        total_loss += loss_val * inputs.size(0)
        num_samples += inputs.size(0)

        if rank == 0 and batch_idx % config.log_interval == 0:
            logger = logging.getLogger(__name__)
            logger.info(
                f"Epoch: [{epoch}] Batch: [{batch_idx}/{len(dataloader)}] Loss: {loss_val.item():.4f}"
            )

    return total_loss, num_samples


@torch.no_grad()
def evaluate(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
    is_distributed: bool,
):
    """Distributed evaluation loop gathering accurate metrics globally."""
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    num_samples = torch.tensor(0.0, device=device)

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=config.use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.detach() * inputs.size(0)
        correct += (outputs.argmax(dim=-1) == targets).sum()
        num_samples += inputs.size(0)

    if is_distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)

    avg_loss = (total_loss / num_samples).item()
    accuracy = (correct / num_samples).item() * 100.0

    logger = logging.getLogger(__name__)
    logger.info(
        f"--- Eval Epoch: {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% ---"
    )
    return avg_loss, accuracy


def main():
    config = Config()

    # Setup Distributed and Device
    device, rank, local_rank, world_size, is_distributed = setup_distributed()

    # Setup Logging (Only Rank 0 logs INFO, others log WARNING)
    logger = setup_logging(rank)
    logger.info(
        f"Starting training on {world_size} devices. Rank: {rank}, Device: {device}"
    )

    # Set seeds for reproducibility
    set_seed(config.seed, rank)

    # Build basic components
    train_loader, val_loader, train_sampler = build_dataloaders(config, is_distributed)
    model = build_model(config).to(device)
    criterion = build_criterion(config).to(device)

    # Mixed Precision Scaler
    scaler_device = (
        device.type if device.type in ["cuda", "xpu", "npu", "mps"] else "cuda"
    )
    scaler = torch.amp.GradScaler(device=scaler_device, enabled=config.use_amp)

    # Distributed wrapper (FSDP)
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if device.type == "cpu":
            model = FSDP(model)
        else:
            model = FSDP(model, device_id=local_rank)

    # Build optimizer AFTER model wrapper logic (required if transitioning to FSDP later)
    optimizer = build_optimizer(model, config)

    logger.info("Ready to start training loop.")

    for epoch in range(1, config.epochs + 1):
        if is_distributed and train_sampler is not None:
            # Crucial for deterministic shuffling per-epoch in distributed training
            train_sampler.set_epoch(epoch)

        # Train Epoch
        train_loss, train_samples = train_one_epoch(
            epoch,
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            config,
            rank,
        )

        # Sync metrics across processes
        if is_distributed:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_samples, op=dist.ReduceOp.SUM)

        avg_train_loss = (train_loss / train_samples).item()
        logger.info(f"End of Epoch {epoch} | Average Train Loss: {avg_train_loss:.4f}")

        # Evaluate
        if epoch % config.test_interval == 0:
            evaluate(
                epoch, model, val_loader, criterion, device, config, is_distributed
            )

        # Checkpoint (All ranks participate in collective checkpoint)
        if epoch % config.save_interval == 0:
            save_checkpoint(epoch, model, optimizer, scaler, config)

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
