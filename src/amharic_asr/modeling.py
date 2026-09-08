"""XLS-R initialization, selective fine-tuning, and legacy weight restoration."""

from pathlib import Path

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor

from .config import ModelConfig


def configure_trainable_layers(model: Wav2Vec2ForCTC, unfrozen_layers: int) -> None:
    """Freeze the encoder except its final N layers; always train the CTC head."""
    layers = model.wav2vec2.encoder.layers
    if not 0 <= unfrozen_layers <= len(layers):
        raise ValueError(f"unfrozen_layers must be between 0 and {len(layers)}.")
    model.requires_grad_(False)
    # [-0:] selects every layer, so head-only training needs an explicit guard.
    if unfrozen_layers:
        for layer in layers[-unfrozen_layers:]:
            layer.requires_grad_(True)
    model.lm_head.requires_grad_(True)


def build_model(processor: Wav2Vec2Processor, config: ModelConfig) -> Wav2Vec2ForCTC:
    """Load a pretrained encoder or a complete CTC model with compatible labels.

    A base XLS-R encoder receives a fresh CTC head. Existing CTC checkpoints must
    use the same vocabulary size and blank ID; use their saved processor too.
    """
    saved_config = Wav2Vec2Config.from_pretrained(config.name_or_path)
    if any(name.endswith("ForCTC") for name in (saved_config.architectures or [])):
        if (
            saved_config.vocab_size != len(processor.tokenizer)
            or saved_config.pad_token_id != processor.tokenizer.pad_token_id
        ):
            raise ValueError("CTC checkpoint and processor vocabulary/blank ID do not match.")
    model = Wav2Vec2ForCTC.from_pretrained(
        config.name_or_path,
        vocab_size=len(processor.tokenizer),
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=False,
    )
    if model.config.feat_extract_norm != "layer":
        raise ValueError("This attention-mask workflow expects a layer-normalized XLS-R encoder.")
    configure_trainable_layers(model, config.unfrozen_layers)
    if config.gradient_checkpointing and config.unfrozen_layers:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model.float()


def load_adapted_weights(model: Wav2Vec2ForCTC, checkpoint_path: str | Path) -> None:
    """Restore the notebook's partial {head, upper_layers} FP32 checkpoints.

    Start with the same original baseline and processor used by that experiment.
    This restores model weights only, not optimizer state or training progress.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("precision") != "fp32":
        raise ValueError("Expected a legacy checkpoint marked precision='fp32'.")
    weights = checkpoint["upper_layers"]
    layers = model.wav2vec2.encoder.layers
    if not weights or len(weights) > len(layers):
        raise ValueError("Invalid number of upper encoder layers in checkpoint.")
    model.lm_head.load_state_dict(checkpoint["head"], strict=True)
    for layer, state in zip(layers[-len(weights):], weights, strict=True):
        layer.load_state_dict(state, strict=True)
