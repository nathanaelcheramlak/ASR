"""On-demand audio loading, CTC feasibility, and dynamic batch padding."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder
from transformers import Wav2Vec2Processor

from .manifests import Record
from .tokenization import validate_labels


def output_frames(num_samples: int, kernels: Sequence[int], strides: Sequence[int]) -> int:
    """Compute convolution output length using the actual model configuration."""
    if len(kernels) != len(strides):
        raise ValueError("Convolution kernel and stride counts must match.")
    length = num_samples
    for kernel, stride in zip(kernels, strides, strict=True):
        length = (length - kernel) // stride + 1
    return max(0, length)


def required_ctc_frames(labels: Sequence[int]) -> int:
    """Adjacent repeated labels need an intervening CTC blank frame."""
    return len(labels) + sum(left == right for left, right in zip(labels, labels[1:]))


class AmharicASRDataset(Dataset):
    """Decode mono audio at the processor rate and reject infeasible CTC labels."""

    def __init__(
        self,
        records: Sequence[Record],
        audio_dir: str | Path,
        processor: Wav2Vec2Processor,
        *,
        conv_kernel: Sequence[int],
        conv_stride: Sequence[int],
    ) -> None:
        self.records = list(records)
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.conv_kernel = tuple(conv_kernel)
        self.conv_stride = tuple(conv_stride)
        self.sample_rate = processor.feature_extractor.sampling_rate
        self.labels = [
            validate_labels(record["transcription"], processor.tokenizer)
            for record in self.records
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        path = self.audio_dir / record["audio_file"]
        samples = AudioDecoder(
            str(path), sample_rate=self.sample_rate, num_channels=1
        ).get_all_samples()
        if samples.sample_rate != self.sample_rate:
            raise ValueError(f"Unexpected sample rate for {record['id']}.")
        waveform = samples.data[0].cpu().numpy().astype(np.float32)
        if not waveform.size or not np.isfinite(waveform).all():
            raise ValueError(f"Empty or nonfinite audio: {record['id']}")
        labels = self.labels[index]
        frames = output_frames(len(waveform), self.conv_kernel, self.conv_stride)
        required = required_ctc_frames(labels)
        if frames < required:
            raise ValueError(f"Infeasible CTC recording {record['id']}: {frames} < {required}")
        inputs = self.processor.feature_extractor(waveform, sampling_rate=self.sample_rate)
        return {"input_values": inputs.input_values[0], "labels": labels, "id": record["id"]}


@dataclass
class CTCCollator:
    """Pad audio and mask padded target labels with the CTC ignore value, -100."""

    processor: Wav2Vec2Processor

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_values": item["input_values"]} for item in examples],
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in examples],
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels["input_ids"].masked_fill(labels["attention_mask"] != 1, -100)
        return batch
