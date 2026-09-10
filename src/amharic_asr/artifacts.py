"""Export complete inference models and human-readable experiment metadata."""

import json
from dataclasses import asdict
from pathlib import Path

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .config import ExperimentConfig


def export_model(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    destination: str | Path,
    config: ExperimentConfig,
    history: list[dict],
) -> Path:
    """Save to a new directory to avoid replacing previous experiments.

    Exports support inference and weight initialization. Optimizer/RNG state is
    not included; exact continuation uses the original research checkpoints.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(destination))
    processor.save_pretrained(str(destination))
    for filename, payload in (
        ("experiment_config.json", asdict(config)),
        ("history.json", history),
    ):
        (destination / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return destination
