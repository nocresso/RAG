import re
from typing import List, Dict, Any
from collections import Counter, defaultdict
from src.models import Chunk, MinimalSource
from pathlib import Path
from src.chunking.markdown_chunker import chunk_markdown
from src.chunking.python_chunker import chunk_python
from src.files import read_file
from src.errors import (FileReadError, IndexNotFoundError,
                        InvalidIndexError, InvalidChunkSizeError)
from tqdm import tqdm
import math
import json

INDEX_VERSION = 2
MAX_CHUNK_SIZE_LIMIT = 2000
INDEX_FILENAME = "bm25_index.json"


def split_identifier(word: str) -> List[str]:
    """
    Split an identifier into lowercase parts.

    snake_case is split on underscores and CamelCase on case changes.
    For example "max_num_seqs" gives ["max", "num", "seqs"] and
    "LLMEngine" gives ["llm", "engine"].

    Args:
        word: A single word or identifier, without spaces.

    Returns:
        The lowercase parts. A plain word gives a list with itself.
    """
    word = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', word)
    word = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', word)
    return [part.lower() for part in word.split('_') if part]


def tokenize(text: str) -> List[str]:
    """
    Convert a text into a list of normalized tokens.

    The text is lowercased and split into words. Identifiers with
    underscores or in CamelCase are kept whole and also split into
    their parts, so both "max_num_seqs" and "max", "num", "seqs"
    can be matched.

    Args:
        text: Any text (a chunk or a query).

    Returns:
        The list of tokens, in order of appearance.
    """
    tokens: List[str] = []
    raw_tokens = re.findall(r'\w+', text)
    for word in raw_tokens:
        whole = word.lower()
        tokens.append(whole)
        parts = split_identifier(word)
        if parts != [whole]:
            tokens.extend(parts)
    return tokens


def chunk_files(root: str, max_chunk_size: int) -> List[Chunk]:
    """
    Read every supported file under root and split it into chunks.

    Args:
        root: Directory to scan recursively.
        max_chunk_size: Maximum length of each chunk, in characters.

    Returns:
        The chunks of all the files.
    """
    if not 0 < max_chunk_size <= MAX_CHUNK_SIZE_LIMIT:
        raise InvalidChunkSizeError(
            f"max_chunk_size must be between 1 and "
            f"{MAX_CHUNK_SIZE_LIMIT}, got {max_chunk_size}")
    chunks: List[Chunk] = []
    if not Path(root).is_dir():
        raise FileReadError(f"not a directory: {root}")
    files = sorted(p for p in Path(root).rglob("*") if p.is_file())
    for path in tqdm(files, desc="Indexing files"):
        str_path = path.as_posix()
        rel_path = path.relative_to(root).as_posix()
        if path.suffix == ".py":
            text = read_file(str_path)
            new_chunks = chunk_python(text, max_chunk_size, str_path)
        elif path.suffix in (".md", ".txt"):
            text = read_file(str_path)
            new_chunks = chunk_markdown(text, max_chunk_size, str_path)
        else:
            continue
        for chunk in new_chunks:
            chunk.context = f"{rel_path} {chunk.context}"
        chunks.extend(new_chunks)
    return chunks


class BM25Index:
    """
    Lexical search index that ranks chunks with the BM25 function.

    Everything that depends only on the corpus (term frequencies,
    document frequencies, IDF, chunk lengths and their average) is
    computed once when the index is built. A query only reads it.
    """
    def __init__(self, chunks: List[Chunk], k1: float = 1.5,
                 b: float = 0.9) -> None:
        self.k1 = k1
        self.b = b
        self.sources: List[MinimalSource] = [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            for chunk in chunks
        ]
        self.tf: Dict[str, Dict[int, int]] = {}
        self.chunk_lengths: List[int] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

        self.build_term_frequencies(chunks)
        self._finalize()

    def build_term_frequencies(self, chunks: List[Chunk]) -> None:
        """
        Build the inverted index and the chunk lengths in a single pass.

        Fills self.tf, which maps each token to {chunk_id: times it
        appears in that chunk}, and self.chunk_lengths, the number of
        tokens of each chunk. The indexed text is the chunk context
        followed by its content.

        Args:
            chunks: Chunks to index, in the same order as self.sources.
        """
        tf: Dict[str, Dict[int, int]] = defaultdict(dict)
        chunk_lengths: List[int] = []
        for i, chunk in enumerate(chunks):
            tokens = tokenize(f"{chunk.context} {chunk.chunk_text}")
            chunk_lengths.append(len(tokens))
            chunk_tf = Counter(tokens)
            for word, count in chunk_tf.items():
                tf[word][i] = count
        self.tf = dict(tf)
        self.chunk_lengths = chunk_lengths

    def _finalize(self) -> None:
        """
        Compute what depends only on tf and chunk_lengths: df, idf and
        avgdl. Used after building the index and after loading it.
        """
        self.build_document_frequencies()
        self.compute_idf()
        if self.chunk_lengths:
            self.avgdl = sum(self.chunk_lengths) / len(self.chunk_lengths)

    def build_document_frequencies(self) -> None:
        """
        Fill self.df with the number of chunks that contain each token.

        It is derived from self.tf, so it is recomputed after loading
        and never saved.
        """
        tokens_df: Dict[str, int] = {}
        for word, chunks in self.tf.items():
            tokens_df[word] = len(chunks)
        self.df = tokens_df

    def compute_idf(self) -> None:
        """
        Fill self.idf: how rare (informative) each token is.

        Uses idf = ln((N - df + 0.5) / (df + 0.5) + 1), where N is the
        number of chunks. The "+ 1" keeps the value positive even for
        tokens that appear in more than half of the chunks.
        """
        chunk_nb = len(self.sources)
        idf: Dict[str, float] = {}
        for token, df in self.df.items():
            token_idf = math.log(((chunk_nb - df + 0.5) / (df + 0.5)) + 1)
            idf[token] = token_idf
        self.idf = idf

    def _term_score(self, token: str, chunk_id: int) -> float:
        """
        Return the BM25 contribution of one token to one chunk.

        Args:
            token: A token that appears in the chunk.
            chunk_id: Id of the chunk (its position in chunks).

        Returns:
            The partial score of that token for that chunk.
        """
        freq = self.tf[token][chunk_id]
        length_ratio = self.chunk_lengths[chunk_id] / self.avgdl
        numerator = self.idf[token] * freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * length_ratio)
        return numerator / denominator

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """
        Return the k chunks that best match the query.

        Args:
            query: The question, as free text.
            k: How many chunks to return.

        Returns:
            Up to k chunks, best first. Empty if k <= 0 or nothing
            matches.
        """
        if k <= 0:
            return []
        query_tokens = tokenize(query)
        scores: Dict[int, float] = defaultdict(float)
        for token in query_tokens:
            for chunk_id in self.tf.get(token, {}):
                scores[chunk_id] += self._term_score(token, chunk_id)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [self.sources[chunk_id] for chunk_id, _ in ranked[:k]]

    def save(self, directory: str) -> None:
        """
        Write the index as JSON in directory/INDEX_FILENAME.

        df, idf and avgdl are not saved: load recomputes them. The file
        is written to a temporary name and then renamed, so an
        interrupted save never leaves a broken index.

        Args:
            directory: Folder where the index is saved. It is created
                if it does not exist.
        """
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "version": INDEX_VERSION,
            "k1": self.k1,
            "b": self.b,
            "sources": [source.model_dump() for source in self.sources],
            "chunk_lengths": self.chunk_lengths,
            "postings": {
                token: [[chunk_id, count]
                        for chunk_id, count in counts.items()]
                for token, counts in self.tf.items()
            },
        }
        tmp_path = folder / (INDEX_FILENAME + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        tmp_path.replace(folder / INDEX_FILENAME)

    @classmethod
    def load(cls, directory: str) -> "BM25Index":
        """
        Read an index saved with save().

        Args:
            directory: Folder that contains the index file.

        Returns:
            The loaded index, ready to search.

        Raises:
            IndexNotFoundError: if there is no index in directory.
            InvalidIndexError: if it is corrupted or another version.
        """
        path = Path(directory) / INDEX_FILENAME
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as e:
            raise IndexNotFoundError(
                f"no index found in {directory}: run the index command first"
            ) from e
        except (OSError, ValueError) as e:
            raise InvalidIndexError(
                f"cannot read the index in {directory}: {e}") from e
        if not isinstance(data, dict) or data.get("version") != INDEX_VERSION:
            raise InvalidIndexError(
                f"index in {directory} has a different version: reindex")
        try:
            index = cls([], data["k1"], data["b"])
            index.sources = [MinimalSource(**s) for s in data["sources"]]
            index.chunk_lengths = list(data["chunk_lengths"])
            index.tf = {
                token: {chunk_id: count for chunk_id, count in pairs}
                for token, pairs in data["postings"].items()
            }
        except (KeyError, TypeError, ValueError) as e:
            raise InvalidIndexError(
                f"index in {directory} is corrupted: reindex") from e
        index._finalize()
        return index
