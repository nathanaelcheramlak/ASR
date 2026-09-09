"""Corpus-level CER/WER evaluation with padding-aware CTC decoding."""

import torch
from jiwer import cer, wer
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .data import output_frames
from .text import normalize_amharic_ctc


def evaluate(
    model: Wav2Vec2ForCTC,
    loader: DataLoader,
    processor: Wav2Vec2Processor,
    device: torch.device,
    *,
    blank_penalty: float = 0.0,
) -> dict:
    """Evaluate greedily, trimming padded audio frames and restoring model mode.

    Report CER/WER as fractions. Tune blank penalties on development data only.
    References decode with group_tokens=False to preserve repeated characters.
    """
    references = []
    predictions = []
    blank_count = 0
    frame_count = 0
    total_loss = 0.0
    batches = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                if (
                    not torch.isfinite(output.loss).item()
                    or not torch.isfinite(output.logits).all().item()
                ):
                    raise RuntimeError("Nonfinite loss or logits during evaluation.")
                scores = output.logits.float().clone()
                scores[:, :, processor.tokenizer.pad_token_id] -= blank_penalty
                predicted_ids = scores.argmax(dim=-1)
                for row, labels in enumerate(batch["labels"]):
                    length = output_frames(
                        int(batch["attention_mask"][row].sum().item()),
                        model.config.conv_kernel,
                        model.config.conv_stride,
                    )
                    ids = predicted_ids[row, :length].cpu().tolist()
                    labels = labels[labels != -100].cpu().tolist()
                    prediction = processor.tokenizer.decode(ids)
                    reference = processor.tokenizer.decode(labels, group_tokens=False)
                    predictions.append(normalize_amharic_ctc(prediction))
                    references.append(normalize_amharic_ctc(reference))
                    blank_count += ids.count(processor.tokenizer.pad_token_id)
                    frame_count += len(ids)
                total_loss += float(output.loss.item())
                batches += 1
    finally:
        model.train(was_training)
    if not references:
        raise ValueError("Evaluation loader is empty.")
    return {
        "cer": float(cer(references, predictions)),
        "wer": float(wer(references, predictions)),
        "mean_batch_loss": total_loss / batches,
        "blank_fraction": blank_count / max(frame_count, 1),
        "length_ratio": sum(len(text.replace(" ", "")) for text in predictions)
        / max(sum(len(text.replace(" ", "")) for text in references), 1),
        "references": references,
        "predictions": predictions,
    }
