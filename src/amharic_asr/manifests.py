"""UTF-8 manifest handling and explicit data leakage checks."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .text import normalize_amharic_ctc

Record = dict[str, Any]


def read_manifest(path: str | Path) -> list[Record]:
    """Read JSONL and validate the common training/evaluation record schema."""
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object.")
            for field in ("id", "speaker_id", "transcription", "audio_file"):
                if not isinstance(record.get(field), str) or not record[field].strip():
                    raise ValueError(f"{path}:{line_number}: missing or invalid {field}.")
            if record["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate ID {record['id']}.")
            if not normalize_amharic_ctc(record["transcription"]):
                raise ValueError(f"{path}:{line_number}: empty normalized transcript.")
            seen.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no recordings: {path}")
    return records


def write_manifest(records: Iterable[Record], path: str | Path) -> None:
    """Write records as UTF-8 JSONL, preserving Ethiopic characters."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_speaker_separation(train: Sequence[Record], validation: Sequence[Record]) -> None:
    """Reject shared recording IDs or speakers across the two splits."""
    for field in ("id", "speaker_id"):
        overlap = {record[field] for record in train} & {record[field] for record in validation}
        if overlap:
            raise ValueError(f"Training/validation overlap in {field}: {sorted(overlap)[:10]}")
