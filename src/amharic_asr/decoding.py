"""CTC prefix beam search adapted from the notebook's beam-search diagnostic."""

import math

import torch
from transformers import Wav2Vec2CTCTokenizer

from .text import normalize_amharic_ctc

NEGATIVE_INFINITY = -float("inf")


def _logadd(left: float, right: float) -> float:
    if left == NEGATIVE_INFINITY:
        return right
    if right == NEGATIVE_INFINITY:
        return left
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(min(left, right) - maximum))


def prefix_beam_decode(
    logits: torch.Tensor,
    tokenizer: Wav2Vec2CTCTokenizer,
    *,
    blank_penalty: float = 0.0,
    beam_width: int = 10,
    top_k: int = 12,
) -> str:
    """Decode one [frames, vocabulary] sequence without an external language model.

    Trim padded frames before calling. The search prunes to top_k nonblank
    candidates per frame and keeps beam_width prefixes in log probability space.
    """
    if logits.ndim != 2 or logits.shape[-1] != len(tokenizer):
        raise ValueError("Expected logits with shape [frames, tokenizer vocabulary size].")
    if beam_width < 1 or top_k < 1:
        raise ValueError("beam_width and top_k must be positive.")
    if not torch.isfinite(logits).all().item():
        raise ValueError("Logits must be finite.")
    blank_id = tokenizer.pad_token_id
    scores = logits.detach().float().cpu().clone()
    scores[:, blank_id] -= blank_penalty
    log_probs = torch.log_softmax(scores, dim=-1)
    nonblank = log_probs.clone()
    nonblank[:, blank_id] = NEGATIVE_INFINITY
    values, indices = torch.topk(nonblank, k=min(top_k, scores.shape[-1] - 1), dim=-1)
    beams = {(): (0.0, NEGATIVE_INFINITY)}
    for blank_score, tokens, token_scores in zip(
        log_probs[:, blank_id].tolist(), indices.tolist(), values.tolist(), strict=True
    ):
        next_beams = {}

        def update(prefix: tuple[int, ...], index: int, value: float) -> None:
            if value == NEGATIVE_INFINITY:
                return
            probabilities = next_beams.setdefault(prefix, [NEGATIVE_INFINITY, NEGATIVE_INFINITY])
            probabilities[index] = _logadd(probabilities[index], value)

        for prefix, (blank, nonblank_score) in beams.items():
            total = _logadd(blank, nonblank_score)
            update(prefix, 0, total + blank_score)
            for token, score in zip(tokens, token_scores, strict=True):
                if prefix and token == prefix[-1]:
                    update(prefix, 1, nonblank_score + score)
                    update(prefix + (token,), 1, blank + score)
                else:
                    update(prefix + (token,), 1, total + score)
        best = sorted(
            next_beams.items(), key=lambda item: _logadd(*item[1]), reverse=True
        )[:beam_width]
        beams = {prefix: tuple(probabilities) for prefix, probabilities in best}
    best_prefix = max(beams, key=lambda prefix: _logadd(*beams[prefix]))
    # Prefixes have already undergone CTC collapse; preserve repeated characters.
    return normalize_amharic_ctc(tokenizer.decode(list(best_prefix), group_tokens=False))
