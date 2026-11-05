from typing import List
from src.models import Chunk


def split_section(text: str, init: int, end: int, max_chunk_size: int,
                  overlap: int, file_path: str) -> List[Chunk]:
    """
    Split text[init:end] into chunks of at most max_chunk_size chars.

    Consecutive chunks share `overlap` characters so that content lying
    on a boundary appears whole in at least one chunk. A section that
    already fits produces a single chunk.

    Args:
        text: Full text of the file.
        init: Index where the section starts.
        end: Index where the section ends (exclusive).
        max_chunk_size: Maximum length of each chunk, in characters.
        overlap: Characters shared by consecutive chunks.
        file_path: Path of the file, copied into every chunk.

    Returns:
        The chunks of the section, in order.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    overlap = max(0, min(overlap, max_chunk_size // 2))
    step = max_chunk_size - overlap

    chunks: List[Chunk] = []
    start = init
    while start < end:
        piece_end = min(start + max_chunk_size, end)
        chunks.append(Chunk(chunk_text=text[start:piece_end],
                            first_character_index=start,
                            last_character_index=piece_end,
                            file_path=file_path))
        if piece_end == end:
            break
        start += step
    return chunks
