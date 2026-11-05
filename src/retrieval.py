from src.indexing import BM25Index
from src.models import (MinimalSource,
                        RagDataset, MinimalSearchResults,
                        StudentSearchResults)
from src.errors import InvalidQueryError, InvalidArgumentError
from src.hybrid import hybrid_retrieval
from typing import List, TYPE_CHECKING, Optional
from tqdm import tqdm

if TYPE_CHECKING:
    from src.semantic import SemanticIndex

MODES = ("bm25", "semantic", "hybrid")


class Retrieval:
    """
    Load the index saved in index_path.

    Args:
        index_path: Directory where the index was saved.
        mode: "bm25" (lexical), "semantic" (embeddings) or hybrid.
    """
    def __init__(self, index_path: str, mode: str = "bm25") -> None:
        if mode not in MODES:
            raise InvalidArgumentError(
                f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.index_path = index_path
        self.bm25: Optional[BM25Index] = None
        self.semantic: Optional["SemanticIndex"] = None
        if mode in ("bm25", "hybrid"):
            self.bm25 = BM25Index.load(index_path)
        if mode in ("semantic", "hybrid"):
            from src import semantic
            self.semantic = semantic.SemanticIndex.load(index_path)

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """
        Return the k sources that best match the query.

        Args:
            query: The question, as free text.
            k: How many sources to return.

        Returns:
            Up to k sources, best first.
        """
        if k <= 0:
            raise InvalidQueryError(f"k must be positive, got {k}")
        if not query.strip():
            raise InvalidQueryError("the query is empty")
        pool = k * 3 if self.mode == "hybrid" else k
        lexical: List[MinimalSource] = []
        semantic: List[MinimalSource] = []
        if self.bm25 is not None:
            lexical = self.bm25.search(query, pool)
        if self.semantic is not None:
            semantic = self.semantic.search(query, pool)
        if self.mode == "hybrid":
            return hybrid_retrieval(lexical, semantic, k)
        if self.mode == "semantic":
            return semantic
        return lexical

    def search_dataset(self, dataset: RagDataset,
                       k: int) -> StudentSearchResults:
        """
        Search every question of a dataset.
        Questions with an empty text get an empty list of sources.

        Args:
            dataset: Questions to search.
            k: How many sources to return for each question.

        Returns:
            The search results of all the questions.
        """
        if k <= 0:
            raise InvalidQueryError(f"k must be positive, got {k}")
        results: List[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            try:
                sources = self.search(question.question, k)
            except InvalidQueryError:
                sources = []
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources
            ))
        return StudentSearchResults(search_results=results, k=k)
