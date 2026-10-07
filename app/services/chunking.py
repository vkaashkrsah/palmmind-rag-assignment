import re
from enum import Enum


class ChunkStrategy(str, Enum):
    fixed = "fixed"        # fixed-size character windows with overlap
    sentence = "sentence"  # sentence/paragraph-aware packing


def fixed_size_chunks(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    step = size - overlap
    return [c for i in range(0, len(text), step) if (c := text[i : i + size].strip())]


def sentence_chunks(text: str, max_chars: int = 800) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for part in (re.sub(r"\s+", " ", p).strip() for p in parts):
        if not part:
            continue
        if len(part) > max_chars:  # oversized sentence: fall back to fixed windows
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(fixed_size_chunks(part, max_chars))
        elif len(current) + len(part) + 1 > max_chars:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, strategy: ChunkStrategy) -> list[str]:
    if strategy is ChunkStrategy.sentence:
        return sentence_chunks(text)
    return fixed_size_chunks(text)
