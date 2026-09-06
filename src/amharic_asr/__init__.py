"""Reusable building blocks for Amharic CTC speech recognition.

Importing this package does not load models, access data, or start training.
"""

from .text import normalize_amharic_ctc

__all__ = ["normalize_amharic_ctc"]
