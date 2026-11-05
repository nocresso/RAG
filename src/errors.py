class RagError(Exception):
    """Base class for all expected errors of RAG project."""


class FileReadError(RagError):
    """A file could not be read."""


class InvalidChunkSizeError(RagError):
    """max_chunk_size is outside the allowed range."""


class IndexNotFoundError(RagError):
    """The index has not been built yet"""


class InvalidIndexError(RagError):
    """The index file is corrupted or from an incompatible version."""


class InvalidQueryError(RagError):
    """Invalid or empty query"""


class InvalidDatasetError(RagError):
    """The dataset file is malformed or has an unexpected structure."""


class FileWriteError(RagError):
    """A file could not be written."""


class GenerationError(RagError):
    """The language model could not be loaded or could not answer."""


class InvalidArgumentError(RagError):
    """A command-line argument has the wrong type or value."""
