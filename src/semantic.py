import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import List, Any, Dict
from src.models import Chunk, MinimalSource
from src.errors import IndexNotFoundError, InvalidIndexError


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_VERSION = 1
SEMANTIC_FILENAME = "semantic_index.pt"
MAX_TOKENS = 256
BATCH_SIZE = 32


class SemanticIndex:
    """
    Vector index: one normalized embedding per chunk.

    The embeddings are computed once when the index is built. A query
    is embedded and compared with all of them by dot product, which
    equals cosine similarity because every vector has length 1.
    """
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model: Any = AutoModel.from_pretrained(model_name)
        except (OSError, ValueError) as e:
            raise InvalidIndexError(
                f"cannot load the model {model_name}: {e}") from e
        self.model.eval()
        self.model_name = model_name
        self.sources: List[MinimalSource] = []
        self.embeddings: torch.Tensor = torch.empty(0)

    def encode(self, texts: List[str],
               show_progress: bool = False) -> torch.Tensor:
        """
        Turn texts into normalized embeddings.

        Args:
            texts: Texts to embed.
            show_progress: Show a tqdm bar.

        Returns:
            Tensor of shape (len(texts), hidden_size), in the order of
            texts.
        """
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.empty(len(texts), self.model.config.hidden_size)
        starts = range(0, len(texts), BATCH_SIZE)
        with torch.inference_mode():
            for start in tqdm(starts, desc="Embedding",
                              disable=not show_progress):
                ids = order[start:start + BATCH_SIZE]
                batch = self.tokenizer(
                    [texts[i] for i in ids], padding=True,
                    truncation=True, max_length=MAX_TOKENS,
                    return_tensors="pt")
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(
                    hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                out[ids] = torch.nn.functional.normalize(pooled, dim=1)
        return out

    def build(self, chunks: List[Chunk]) -> None:
        """
        Embed every chunk and store the vectors in the index.

        The text embedded for each chunk is its context (relative path
        and section titles) followed by its content, the same text the
        BM25 index uses. Any previous content of the index is replaced.

        Args:
            chunks: Chunks to index. Their order is kept: row i of
                self.embeddings belongs to self.sources[i].
        """
        self.sources = [
            MinimalSource(
                file_path=c.file_path,
                first_character_index=c.first_character_index,
                last_character_index=c.last_character_index,
            )
            for c in chunks
        ]
        texts = [f"{c.context} {c.chunk_text}" for c in chunks]
        self.embeddings = self.encode(texts, show_progress=True)

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """
        Return the k chunks whose meaning is closest to the query.

        The query is embedded and compared with every chunk by dot
        product, which equals cosine similarity because all the
        vectors have length 1.

        Args:
            query: The question, as free text.
            k: How many chunks to return.

        Returns:
            Up to k sources, best first. Empty if k <= 0 or the index
            has no chunks.
        """
        if k <= 0 or not self.sources:
            return []
        query_vector = self.encode([query])[0]
        scores = self.embeddings @ query_vector
        top = torch.topk(scores, min(k, len(self.sources)))
        return [self.sources[i] for i in top.indices.tolist()]

    def save(self, directory: str) -> None:
        """
        Save the index in directory/SEMANTIC_FILENAME.

        The file holds the version, the model name, the sources and the
        embeddings. It is written to a temporary file first and then
        renamed, so an interrupted save never leaves a broken index.

        Args:
            directory: Folder where the index is saved. It is created
                if it does not exist.
        """
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "version": SEMANTIC_VERSION,
            "model_name": self.model_name,
            "sources": [s.model_dump() for s in self.sources],
            "embeddings": self.embeddings,
        }
        tmp_path = folder / (SEMANTIC_FILENAME + ".tmp")
        torch.save(data, tmp_path)
        tmp_path.replace(folder / SEMANTIC_FILENAME)

    @classmethod
    def load(cls, directory: str) -> "SemanticIndex":
        """
        Read an index saved with save().

        Args:
            directory: Folder where the index was saved.

        Returns:
            The index, ready to search.

        Raises:
            IndexNotFoundError: if there is no semantic index.
            InvalidIndexError: if it is corrupted or another version.
        """
        path = Path(directory) / SEMANTIC_FILENAME
        try:
            data = torch.load(path, weights_only=True)
        except FileNotFoundError as e:
            raise IndexNotFoundError(
                f"no semantic index in {directory}: "
                f"run the semantic indexing command first") from e
        except Exception as e:
            raise InvalidIndexError(
                f"cannot read the semantic index: {e}") from e
        if (not isinstance(data, dict)
                or data.get("version") != SEMANTIC_VERSION):
            raise InvalidIndexError("semantic index has another version")
        try:
            index = cls(data["model_name"])
            index.sources = [MinimalSource(**s) for s in data["sources"]]
            index.embeddings = data["embeddings"]
        except (KeyError, TypeError, ValueError) as e:
            raise InvalidIndexError(
                f"semantic index is corrupted: {e}") from e
        return index
