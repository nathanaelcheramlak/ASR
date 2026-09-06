"""Typed configuration with paths independent of the working directory."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    train_manifest: Path
    validation_manifest: Path
    audio_dir: Path
    processor_dir: Path
    run_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "facebook/wav2vec2-xls-r-300m"
    sample_rate: int = 16000
    unfrozen_layers: int = 2
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError("The XLS-R workflow expects 16,000 Hz audio.")
        if self.unfrozen_layers < 0:
            raise ValueError("unfrozen_layers must be nonnegative.")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 3
    batch_size: int = 1
    head_learning_rate: float = 1e-4
    encoder_learning_rate: float = 2e-6
    weight_decay: float = 0.0
    max_gradient_norm: float = 1.0
    device: str = "auto"
    disable_stochastic_layers: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if min(self.head_learning_rate, self.encoder_learning_rate, self.max_gradient_norm) <= 0:
            raise ValueError("Learning rates and max_gradient_norm must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative.")


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path, *, project_root: str | Path) -> ExperimentConfig:
    """Load a TOML file; resolve configured relative paths against project_root."""
    root = Path(project_root).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open("rb") as source:
        values = tomllib.load(source)
    unknown = set(values) - {"paths", "model", "training"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    paths = {
        key: (root / Path(value).expanduser()).resolve()
        for key, value in values["paths"].items()
    }
    return ExperimentConfig(
        paths=PathsConfig(**paths),
        model=ModelConfig(**values.get("model", {})),
        training=TrainingConfig(**values.get("training", {})),
    )
