"""Stage 2: clean text while preserving Arabic UTF-8 and source metadata."""

from __future__ import annotations

import importlib
import re
import unicodedata


documents_stage = importlib.import_module("01_documents")

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def preprocess_text(text: str) -> str:
    """Normalize safe presentation details without erasing evidence-bearing words."""

    value = unicodedata.normalize("NFC", str(text or ""))
    value = CONTROL_CHARACTERS.sub(" ", value)
    value = value.translate(ARABIC_DIGITS).translate(PERSIAN_DIGITS)
    value = value.replace("ـ", "")
    return re.sub(r"\s+", " ", value).strip()


def preprocess_documents(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    processed: list[dict[str, str]] = []
    for document in documents:
        row = dict(document)
        row["original_text"] = str(document.get("text", ""))
        row["text"] = preprocess_text(row["original_text"])
        processed.append(row)
    return processed


if __name__ == "__main__":
    rows = preprocess_documents(documents_stage.load_documents())
    print(f"Preprocessed {len(rows)} documents.")
