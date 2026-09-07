"""Training-only vocabulary creation and label round-trip checks."""

import json
from collections.abc import Sequence
from pathlib import Path

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from .manifests import Record
from .text import normalize_amharic_ctc


def validate_labels(text: str, tokenizer: Wav2Vec2CTCTokenizer) -> list[int]:
    """Encode a reference without applying CTC collapse to repeated labels."""
    normalized = normalize_amharic_ctc(text)
    if not normalized:
        raise ValueError("Transcript is empty after normalization.")
    labels = tokenizer.encode(normalized, add_special_tokens=False)
    if tokenizer.unk_token_id in labels or tokenizer.pad_token_id in labels:
        raise ValueError(f"Transcript contains unknown or blank labels: {normalized!r}")
    decoded = tokenizer.decode(
        labels, group_tokens=False, clean_up_tokenization_spaces=False
    )
    if decoded != normalized:
        raise ValueError(f"Label round-trip failed: {normalized!r} -> {decoded!r}")
    return labels


def create_processor(
    train_records: Sequence[Record], output_dir: str | Path, *, sample_rate: int = 16000
) -> Wav2Vec2Processor:
    """Create a processor from training text only, in a new output directory.

    Use Wav2Vec2Processor.from_pretrained to reuse an existing vocabulary.
    Never rebuild the vocabulary when loading an existing model checkpoint.
    """
    characters = sorted(
        {
            char
            for record in train_records
            for char in normalize_amharic_ctc(record["transcription"])
        }
        - {" "}
    )
    if not characters:
        raise ValueError("Training transcripts contain no usable characters.")
    vocabulary = {char: index for index, char in enumerate(characters)}
    for token in ("|", "[UNK]", "[PAD]"):
        vocabulary[token] = len(vocabulary)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    vocab_path = destination / "vocab.json"
    vocab_path.write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        bos_token=None,
        eos_token=None,
        do_lower_case=False,
        clean_up_tokenization_spaces=False,
    )
    for record in train_records:
        validate_labels(record["transcription"], tokenizer)
    processor = Wav2Vec2Processor(
        feature_extractor=Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sample_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        ),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(str(destination))
    return processor
