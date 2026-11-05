from typing import List, Tuple
from src.models import Chunk
from src.chunking.chunk_utils import split_section
import re


def find_code_blocks(text: str) -> List[Tuple[int, int]]:
    """
    Find the fenced code blocks of a Markdown text.

    The lines that start with ``` are paired two by two: the first
    opens a block and the second closes it.

    Args:
        text: Full text of the file.

    Returns:
        (start, end) character ranges, one per code block, in order.
    """
    fences = list(re.finditer(r'^```.*$', text, re.MULTILINE))
    ranges = []
    for i in range(0, len(fences) - 1, 2):
        ranges.append((fences[i].start(), fences[i + 1].end()))
    return ranges


def find_headers(text: str) -> List[Tuple[int, int]]:
    """
    Find the header lines of a Markdown file.

    Lines that start with '#' inside fenced code blocks are ignored,
    because they are comments and not titles.

    Args:
        text: Full text of the file.

    Returns:
        (start, end) character ranges of each header line, in order.
    """
    headers = []
    code_ranges = find_code_blocks(text)

    for match in re.finditer(r'^#{1,6}\s.*$', text, re.MULTILINE):
        if any(start <= match.start() < end for start, end in code_ranges):
            continue
        headers.append((match.start(), match.end()))
    return headers


def section_titles(text: str, headers: List[Tuple[int, int]],
                   init: int, end: int) -> str:
    """
    Return the titles of the headers that start inside [init, end].

    Args:
        text: Full text of the file.
        headers: (start, end) ranges of every header of the file.
        init: Start of the section.
        end: End of the section.

    Returns:
        The titles joined by spaces, without the leading '#' marks.
    """
    titles = [text[h_start:h_end].lstrip('#').strip()
              for h_start, h_end in headers if init <= h_start < end]
    return " ".join(titles)


def ranges_to_sections(ranges: List[Tuple[int, int]], text_length: int
                       ) -> List[Tuple[int, int]]:
    """
    Turn a list of (start, end) ranges into (start, end) sections
    covering the full text, filling any gap before the first range.

    Each section starts where a range starts and lasts until the start
    of the next one (the last one lasts until the end of the text).

    Args:
        ranges: (start, end) ranges, in order (headers or definitions).
        text_length: Length of the full text.

    Returns:
        Contiguous (start, end) sections that cover the whole text. If
        there are no ranges, a single section with the whole text.
    """
    sections = []

    if not ranges:
        sections.append((0, text_length))
        return sections
    if ranges[0][0] != 0:
        sections.append((0, ranges[0][0]))
    for i, (start, end) in enumerate(ranges):
        if i + 1 < len(ranges):
            next_start = ranges[i + 1][0]
        else:
            next_start = text_length
        sections.append((start, next_start))
    return sections


def merge_small_sections(
        sections: List[Tuple[int, int]], min_size: int, max_chunk_size: int
        ) -> List[Tuple[int, int]]:
    """
    Merge consecutive small sections into bigger ones.

    A section shorter than min_size merges with the next one, as long as the
    result is not longer than max_chunk_size.

    Args:
        sections: Contiguous (start, end) ranges, in order.
        min_size: Sections shorter than this are merged with the next.
        max_chunk_size: Maximum length of a merged section.

    Returns:
        The sections after the merge, still contiguous and in order.
    """
    if not sections:
        return []
    result: List[Tuple[int, int]] = []
    cur_init, cur_end = sections[0]
    for init, end in sections[1:]:
        current = cur_end - cur_init
        next_section = end - init
        if (current < min_size) and (current + next_section <= max_chunk_size):
            cur_end = end
        else:
            result.append((cur_init, cur_end))
            cur_init, cur_end = init, end
    result.append((cur_init, cur_end))
    return result


def chunk_markdown(text: str, max_chunk_size: int, file_path: str,
                   overlap: int = 200) -> List[Chunk]:
    """
    Chunk a Markdown file by header sections.

    Args:
        text: Full text of the file.
        max_chunk_size: Maximum length of each chunk, in characters.
        file_path: Path of the file, copied into every chunk.
        overlap: Characters shared by chunks of a split long section.

    Returns:
        The list of chunks of the file.
    """
    chunk_list = []
    min_size = 300
    headers = find_headers(text)
    sections = ranges_to_sections(headers, len(text))
    sections = merge_small_sections(sections, min_size, max_chunk_size)
    for init, end in sections:
        title = section_titles(text, headers, init, end)
        new_chunks = split_section(text, init, end, max_chunk_size,
                                   overlap, file_path)
        for chunk in new_chunks:
            chunk.context = title
        chunk_list.extend(new_chunks)
    return chunk_list
