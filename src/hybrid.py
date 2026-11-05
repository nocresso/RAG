from typing import Dict, List, Tuple
from src.models import MinimalSource

RRF_CONSTANT = 60  # standard value

SourceKey = Tuple[str, int, int]


def source_key(source: MinimalSource) -> SourceKey:
    """
    Return a hashable key that identifies a chunk.

    Two sources are the same chunk if they have the same file and the
    same character range.

    Args:
        source: The source to identify.

    Returns:
        A tuple (file_path, first_character_index, last_character_index).
    """
    return (source.file_path,
            source.first_character_index,
            source.last_character_index)


def hybrid_retrieval(bm25: List[MinimalSource],
                     semantic: List[MinimalSource],
                     k: int, rrf_constant: int = RRF_CONSTANT
                     ) -> List[MinimalSource]:
    """
    Merge the lexical and the semantic ranking with Reciprocal Rank
    Fusion.

    Each chunk gets weight / (rrf_constant + position) from every
    ranking where it appears (position starts at 1). The lexical
    ranking has weight 2 and the semantic one weight 1, so BM25 counts
    twice as much. The scores are added up and the chunks are sorted
    from highest to lowest.

    Args:
        bm25: Sources found by the lexical index, best first.
        semantic: Sources found by the semantic index, best first.
        k: How many sources to return.
        rrf_constant: Softens the weight of the first positions.

    Returns:
        Up to k sources, best first.
    """
    scores: Dict[SourceKey, float] = {}
    by_key: Dict[SourceKey, MinimalSource] = {}
    for ranking, weight in ((bm25, 3.0), (semantic, 1.0)):
        for position, source in enumerate(ranking, start=1):
            key = source_key(source)
            by_key[key] = source
            scores[key] = (scores.get(key, 0.0)
                           + weight / (rrf_constant + position))
    ordered = sorted(scores, key=lambda key: -scores[key])
    return [by_key[key] for key in ordered[:k]]
