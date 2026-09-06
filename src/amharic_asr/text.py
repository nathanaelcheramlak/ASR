"""Transcript normalization shared by training and evaluation."""

import unicodedata


def normalize_amharic_ctc(text: str) -> str:
    """Preserve the original notebook's NFC and character-filtering policy.

    Ethiopic letters/numerals, lowercase Latin letters, and ASCII digits remain.
    Controls disappear; punctuation and unsupported characters become spaces.
    """
    result = []
    for char in unicodedata.normalize("NFC", text):
        name = unicodedata.name(char, "")
        category = unicodedata.category(char)
        if char.isspace():
            result.append(" ")
        elif name.startswith(
            ("ETHIOPIC SYLLABLE", "ETHIOPIC LETTER", "ETHIOPIC DIGIT", "ETHIOPIC NUMBER")
        ):
            result.append(char)
        elif "LATIN" in name and category.startswith("L"):
            result.append(char.lower())
        elif char in "0123456789":
            result.append(char)
        elif not category.startswith("C"):
            result.append(" ")
    return " ".join("".join(result).split())
