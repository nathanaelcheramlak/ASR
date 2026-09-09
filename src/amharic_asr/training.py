"""Small FP32 training helpers with explicit state and finite-gradient checks."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC

from .config import TrainingConfig


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch; GPU kernels may still be nondeterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """Choose CUDA when available unless a specific device was configured."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return device


def create_optimizer(
    model: Wav2Vec2ForCTC, config: TrainingConfig
) -> torch.optim.AdamW:
    """Give the CTC head and trainable encoder parameters separate learning rates."""
    head = [parameter for parameter in model.lm_head.parameters() if parameter.requires_grad]
    encoder = [
        parameter for parameter in model.wav2vec2.parameters() if parameter.requires_grad
    ]
    groups = []
    if head:
        groups.append({"params": head, "lr": config.head_learning_rate})
    if encoder:
        groups.append({"params": encoder, "lr": config.encoder_learning_rate})
    if not groups:
        raise ValueError("The model has no trainable parameters.")
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def train_epoch(
    model: Wav2Vec2ForCTC,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    """Train one FP32 epoch without depending on notebook global variables.

    Disabling stochastic layers follows the later controlled experiments:
    eval mode disables dropout and SpecAugment while autograd stays enabled.
    The caller owns model placement, epoch order, and checkpoint policy.
    """
    model.train(not config.disable_stochastic_layers)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_loss = 0.0
    total_norm = 0.0
    steps = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite CTC loss at batch {steps + 1}.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, config.max_gradient_norm, error_if_nonfinite=True
        )
        optimizer.step()
        total_loss += float(loss.detach().item())
        total_norm += float(norm.item())
        steps += 1
    if not steps:
        raise ValueError("Training loader is empty.")
    return {"mean_batch_loss": total_loss / steps, "mean_gradient_norm": total_norm / steps}
