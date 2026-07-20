"""Stage 4: create deterministic local vector representations."""

from __future__ import annotations

import hashlib
import importlib
import math
import re


chunking_stage = importlib.import_module("03_chunking")
TOKEN_PATTERN = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
EMBEDDING_DIMENSION = 384


def embed_text(text: str, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    """Return a normalized hashing vector that works offline and supports Arabic."""

    vector = [0.0] * dimension
    for token in TOKEN_PATTERN.findall(str(text).lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def vectorize_chunks(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{**chunk, "embedding": embed_text(str(chunk["text"]))} for chunk in chunks]


def embed_query(query: str) -> list[float]:
    return embed_text(query)


if __name__ == "__main__":
    preprocessing = importlib.import_module("02_preprocessing")
    docs = preprocessing.preprocess_documents(preprocessing.documents_stage.load_documents())
    vectors = vectorize_chunks(chunking_stage.chunk_documents(docs))
    print(f"Vectorized {len(vectors)} chunks into {EMBEDDING_DIMENSION} dimensions.")
