# Amharic ASR

A modular research project for Amharic automatic speech recognition using
XLS-R and Connectionist Temporal Classification (CTC). The workflow supports
transcript preparation, selective encoder fine-tuning, decoding, and evaluation
with the WaxalNLP `amh_asr` corpus.

## Features

- Unicode-aware normalization for Amharic transcripts.
- Training-only vocabulary construction and tokenizer round-trip validation.
- Local audio loading, dynamic batch padding, and CTC alignment checks.
- Head-only training or fine-tuning of selected encoder layers.
- Character error rate (CER), word error rate (WER), and decoding diagnostics.
- Greedy decoding and CTC prefix beam search.
- Model and processor exports with configuration and metric history.

## Project structure

```text
Amharic-ASR/
|-- configs/
|   `-- default.toml         # Paths, model settings, and hyperparameters
|-- src/amharic_asr/
|   |-- config.py           # Typed configuration and path resolution
|   |-- text.py             # Transcript normalization
|   |-- manifests.py        # JSONL loading and split validation
|   |-- tokenization.py     # Vocabulary and processor creation
|   |-- data.py             # Audio dataset and CTC batch collation
|   |-- modeling.py         # Model initialization and trainable layers
|   |-- training.py         # Optimizer, reproducibility, and training loop
|   |-- evaluation.py       # CER, WER, and prediction diagnostics
|   |-- decoding.py         # CTC prefix beam search
|   `-- artifacts.py        # Model and metadata exports
|-- notebooks/
|   |-- amharic_asr.ipynb   # Main workflow
|   |-- research/          # Experiments organized by topic
|   `-- archive/           # Original notebook and saved outputs
|-- data/
|   |-- raw/               # Source metadata and downloads
|   |-- manifests/         # Training and validation JSONL files
|   `-- audio/             # Local audio recordings
|-- artifacts/             # Generated processors, models, and metrics
|-- docs/                  # Data contract and migration reference
`-- pyproject.toml         # Package metadata and development tools
```

## Requirements

- Python 3.11 or newer.
- Compatible PyTorch and TorchCodec builds for your operating system and hardware.
- FFmpeg compatible with the installed TorchCodec version.
- Local audio recordings and training/validation manifests.

A CUDA-capable GPU accelerates training. The main workflow also supports CPU
execution. Datasets and pretrained model weights are not bundled with the project.

## Installation

From the project root, install the package in your Python environment:

```sh
python -m pip install -e "."
```

For notebook use and development tools:

```sh
python -m pip install -e ".[research,dev]"
```

Dependency versions are declared in [pyproject.toml](pyproject.toml). Select
PyTorch and TorchCodec builds that match your hardware and CUDA runtime.

## Data preparation

Place recordings in `data/audio/` and create separate UTF-8 JSONL manifests for
training and validation. Each line must contain a record with these fields:

```json
{"id":"clip-001","speaker_id":"speaker-001","transcription":"ሰላም","audio_file":"clip-001.mp3"}
```

Audio paths resolve relative to the configured audio directory. Recording IDs
and speakers must be disjoint across training and validation. The dataset
decodes recordings to mono audio at 16 kHz and checks that each label sequence
can be aligned with the available CTC frames.

See [Data format](docs/data-format.md) for normalization rules, optional fields,
and validation behavior.

## Configuration

Edit [configs/default.toml](configs/default.toml) before starting a run:

```toml
[paths]
train_manifest = "data/manifests/train.jsonl"
validation_manifest = "data/manifests/validation.jsonl"
audio_dir = "data/audio"
processor_dir = "artifacts/processor"
run_dir = "artifacts/runs/baseline"
```

Relative data and artifact paths resolve against the project root. The same file
defines the model, sample rate, trainable encoder layers, learning rates, batch
size, epochs, random seed, and execution device.

The default model is `facebook/wav2vec2-xls-r-300m`. Set
`model.unfrozen_layers = 0` for head-only training or a positive value to fine-tune
the final encoder layers alongside the CTC head. Use a distinct `paths.run_dir`
for each experiment.

## Training and evaluation

Open [notebooks/amharic_asr.ipynb](notebooks/amharic_asr.ipynb) and follow the
workflow:

1. Load configuration and validate the manifests and speaker separation.
2. Create a processor from training transcripts or load an existing processor.
3. Initialize the model, datasets, optimizer, and data loaders.
4. Train and evaluate each epoch.
5. Review predictions and exported metrics.

The workflow exports each epoch to a new directory under `paths.run_dir`, including
the model, processor, resolved configuration, and metric history. CER and WER are
stored as fractions and displayed as percentages in the notebook.

When loading an existing model, reuse its original processor to preserve token
IDs. Exports support inference and weight initialization; they do not include
optimizer or random-number-generator state for exact training continuation.

Choose decoding parameters using development data and report final performance
on a separate test split.

## Development

Keep reusable logic in `src/amharic_asr/` and use notebooks for experiment
orchestration and analysis. Local datasets and generated model artifacts are
excluded from version control by `.gitignore`.

Lint and format the package with:

```sh
ruff check src
ruff format src
```

## Research reference

[Research notebooks](notebooks/research/) group the historical experiments by
topic. They retain experiment-specific state and artifact dependencies; the
main notebook provides the entry point for new runs.

The [original notebook](notebooks/archive/AmharicASR.original.ipynb) preserves the
complete research record. See the [migration guide](docs/migration.md) for the
source-cell map and checkpoint compatibility details.
