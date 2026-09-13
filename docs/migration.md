# Notebook migration

The input was `AmharicASR (1).ipynb`, containing 63 code cells, extensive saved
outputs, widget state, and a chronological sequence of experiments. The exact
input is now `notebooks/archive/AmharicASR.original.ipynb`.

## Entry points

- `notebooks/amharic_asr.ipynb`: concise workflow using the new Python package.
- `notebooks/research/`: original source cells grouped by research phase, with
  explanatory headings, cleared outputs, and original cell-index metadata.
- `notebooks/archive/AmharicASR.original.ipynb`: original source and outputs,
  without modifications.

## Original source-cell map

Indices below are zero-based positions in the original JSON `cells` array. Some
original comments use different cell numbers; those comments remain intact.

| Research notebook | Original indices | Contents |
| --- | --- | --- |
| `00_dataset_discovery.ipynb` | 0–10 | Environment, dataset metadata, text normalization, vocabulary, audio probes |
| `01_data_preparation.ipynb` | 11–16 | Drive persistence, speaker-balanced sampling, audio extraction and archives |
| `02_baseline_and_ctc.ipynb` | 17–23 | Dataset/collator, CTC feasibility, baseline recovery, initial diagnostics |
| `03_incremental_training.ipynb` | 24–30 | Incremental training, recovery, checkpointing and evaluation |
| `04_learnability_diagnostics.ipynb` | 31–37 | Failure audits, head memorization, controlled encoder comparison |
| `05_controlled_experiments.ipynb` | 38–46 | Small datasets, balanced training, two-stage comparisons, FP32 repairs |
| `06_scaled_training_and_quality.ipynb` | 47–55 | Scaled pilot, manual reviews, filtering, continuation, label audit |
| `07_two_hour_pilot.ipynb` | 56–62 | Two-hour selection/training, blank bias, beam search, joint pilot, four-recording test |

Every original source cell, including the final empty cell, appears once across
the research notebooks. Their source text has not been rewritten. New Markdown
cells label the phases and original indices. The new notebooks have no saved
outputs, execution counts, or Colab widget state.

## Historical state is explicit

The research notebooks are organized records, not independent pipelines. They
still use globals such as `train_records`, `train_dataset`, `tokenizer`,
`processor`, `DATA_DIR`, cached features, and previous in-memory model branches.
They retain Google Colab imports, `/content` paths, manual review decisions,
hard-coded sample counts, and assertions about previously saved artifacts.

Do not use “Run All” across the research notebooks as a reproduction recipe.
Some cells restore rather than create earlier checkpoints, and some repair
branches are alternatives to preceding experiments. Read the original comments
and select the appropriate branch with its required artifacts. No runner was
added that executes this historical sequence automatically.

## Reusable extractions

| Original implementation | Reusable module |
| --- | --- |
| Cell 7 normalization | `text.py` |
| Cells 8–9 vocabulary and round-trip checks | `tokenization.py` |
| Cells 14, 38–61 manifest/split checks | `manifests.py` |
| Cells 17–18 dataset, dynamic padding, frame checks | `data.py` |
| Cell 19 and later selective fine-tuning | `modeling.py` |
| Later FP32 training, optimizer groups, clipping | `training.py` |
| CER/WER, blank fraction, output-length audits | `evaluation.py` |
| Original cell 59 / commented “Cell 55” prefix beam search | `decoding.py` |
| Model and metadata persistence | `artifacts.py` |

Shared functions now receive their inputs explicitly. Paths and hyperparameters
live in typed TOML-backed configuration. The dataset and vocabulary validate
inputs using exceptions rather than removable runtime assertions. The main
workflow uses an ordinary FP32 epoch loop with distinct head/encoder learning
rates; it does not claim numerical parity with the original experiments.

Compared with the original initial diagnostic, CTC loss uses
`ctc_zero_infinity=False`, and actual decoded lengths are checked. Impossible
alignments therefore fail visibly instead of contributing a zeroed loss.
Evaluation removes padded output frames before decoding batched recordings.
`unfrozen_layers=0` explicitly trains only the head, avoiding Python's `[-0:]`
slice selecting all encoder layers.

## Existing checkpoints

Full Hugging Face checkpoints can be loaded through `model.name_or_path`; use an
absolute local path and set `paths.processor_dir` to the matching saved processor.
The checkpoint's original vocabulary must be reused. `build_model` rejects
CTC checkpoints whose vocabulary size or blank ID differs from the processor.

Later experiments store partial dictionaries with `precision`, `head`, and
`upper_layers`. `modeling.load_adapted_weights` supports that FP32 format after
you construct the model from the same original baseline and matching processor.
It restores weights only. Other historical checkpoint formats, cached-feature
heads, and exact optimizer/RNG continuation remain in the research notebooks.

`artifacts.export_model` writes a complete model, processor, resolved
configuration, and metric history to a new directory. It is an inference/weight
export, not a resumable optimizer checkpoint. Exports are made per epoch in the
new workflow so earlier epochs remain available.

## Scope of verification

Only file/JSON/source preservation checks were performed during organization.
No dependencies were installed, no Python/notebook cells were run, and no model
or dataset was downloaded. The new package and workflow require runtime
validation in your chosen ASR environment before use for training.
