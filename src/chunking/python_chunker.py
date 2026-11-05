import ast
from typing import List, Tuple
from src.models import Chunk
from src.chunking.markdown_chunker import ranges_to_sections
from src.chunking.chunk_utils import split_section


def line_offsets(text: str) -> List[int]:
    """
    Compute where each line of a text starts.

    offsets[i] = character index where line i+1 (1-indexed) starts.

    Args:
        text: Full text of the file.

    Returns:
        A list with the start index of every line, plus the length of
        the text as the last element.
    """
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def find_top_level_defs(text: str) -> List[Tuple[int, int]]:
    """
    Find the character ranges of the top-level functions and classes.

    The range of a definition starts at its first decorator (if it has
    any) and ends at the last line of its body.

    Args:
        text: Full text of the Python file.

    Returns:
        (start, end) character ranges, in order. Empty if the file has
        a syntax error or no definitions.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    offsets = line_offsets(text)
    defs = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if node.end_lineno is None:
                continue
            if node.decorator_list:
                start_lineno = node.decorator_list[0].lineno
            else:
                start_lineno = node.lineno
            start = offsets[start_lineno - 1]
            end = offsets[node.end_lineno]
            defs.append((start, end))

    return defs


def chunk_python(text: str, max_chunk_size: int,
                 file_path: str, overlap: int = 200
                 ) -> List[Chunk]:
    """Chunk a Python file by top-level functions and classes.
    Args:
        text: Full text of the file.
        max_chunk_size: Maximum length of each chunk, in characters.
        file_path: Path of the file, copied into every chunk.
        overlap: Characters shared by chunks of a split long section.

    Returns:
        The list of chunks of the file.
    """
    chunk_list = []

    defs = find_top_level_defs(text)
    sections = ranges_to_sections(defs, len(text))
    for init, end in sections:
        chunk_list.extend(split_section(text, init, end, max_chunk_size,
                                        overlap, file_path))
    return chunk_list
