# Local data contract

Training and validation manifests are UTF-8 JSON Lines: one JSON object per line.
Each record must include nonempty string fields `id`, `speaker_id`,
`transcription`, and `audio_file`. Additional research fields are preserved.

```json
{"id":"clip-001","speaker_id":"speaker-001","transcription":"ሰላም","audio_file":"clip-001.mp3","duration_seconds":4.2}
```

`audio_file` resolves relative to the configured `audio_dir`. The reusable
workflow reads existing local files; it does not download the WaxalNLP corpus.
`duration_seconds` is optional for the reusable dataset because it checks the
actual decoded sample count. Many historical experiments require that field.

`read_manifest` rejects duplicate recording IDs, missing required fields, and
transcripts that become empty after normalization. `validate_speaker_separation`
rejects overlap in both IDs and speakers between the supplied splits. Split
construction and quality review remain explicit research decisions.

Normalization retains Ethiopic letters/numerals, lowercase Latin letters, and
ASCII digits. It uses Unicode NFC, removes control/format characters, replaces
punctuation and unsupported scripts with spaces, and collapses whitespace.
This policy comes from the original notebook; it does not equate different
Ethiopic syllables or rewrite spelling.

For a new model, `create_processor` builds a sorted vocabulary from normalized
training text, then appends `|`, `[UNK]`, and `[PAD]`. The padding token is also
the CTC blank. IDs and vocabulary size are derived rather than hard-coded.
Validation text does not expand the vocabulary. The reusable dataset raises on
unknown labels; inspect any validation OOV cases and choose a documented handling
policy before training.

When continuing an existing experiment, reuse its saved tokenizer/processor.
Rebuilding from a smaller manifest can assign different IDs even if the
vocabulary size happens to match. Checkpoint dimensions alone cannot detect that.

The dataset uses TorchCodec to decode mono audio at 16 kHz. A CTC sequence requires
at least `number_of_labels + adjacent_label_repeats` output frames. Output frames
are computed from the model's convolution kernels and strides. Invalid audio,
infeasible labels, and unknown characters raise descriptive errors.

Generated processors and model exports belong under `artifacts/`. The dataset,
manifests, and trained weights were not present in the supplied workspace and
have not been fabricated during this refactor.
