import threading
import uuid
from typing import Optional, TYPE_CHECKING
from fastapi import FastAPI, HTTPException
from src.errors import InvalidArgumentError, InvalidQueryError, RagError
from src.models import (MinimalAnswer, MinimalSearchResults,
                        QueryRequest)
from src.retrieval import Retrieval

if TYPE_CHECKING:
    from src.generation import Generator


def to_http_error(error: Exception) -> HTTPException:
    """
    Convert an exception of the RAG system into an HTTP error.

    Args:
        error: The exception raised while handling a request.

    Returns:
        The HTTPException to raise: 400 for a wrong query or argument,
        500 for any other error.
    """
    if isinstance(error, (InvalidQueryError, InvalidArgumentError)):
        return HTTPException(status_code=400, detail=str(error))
    if isinstance(error, RagError):
        return HTTPException(status_code=500, detail=str(error))
    return HTTPException(status_code=500, detail="internal server error")


def create_app(index_path: str, mode: str = "bm25",
               with_answer: bool = True) -> FastAPI:
    """
    Build the HTTP API of the RAG system.

    The index and the language model are loaded once, here, and shared
    by all the requests.

    Args:
        index_path: Directory where the index was saved.
        mode: "bm25", "semantic" or "hybrid".
        with_answer: Load the language model so that /answer works.

    Returns:
        The FastAPI application, ready to be served.

    Raises:
        RagError: if the index or the model cannot be loaded.
    """
    retrieval = Retrieval(index_path, mode)
    generator: Optional["Generator"] = None
    if with_answer:
        from src import generation
        generator = generation.Generator()
    lock = threading.Lock()
    app = FastAPI(title="RAG API")

    @app.get("/status")
    def status() -> dict[str, str]:
        """Tell that the server is running."""
        return {"status": "ok"}

    @app.post("/search", response_model=MinimalSearchResults)
    def search(request: QueryRequest) -> MinimalSearchResults:
        """Return the k sources that best match the query."""
        try:
            sources = retrieval.search(request.query, request.k)
        except Exception as e:
            raise to_http_error(e) from e
        return MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=request.query,
            retrieved_sources=sources)

    @app.post("/answer", response_model=MinimalAnswer)
    def answer(request: QueryRequest) -> MinimalAnswer:
        """Return an answer to the query and the sources it is based on."""
        if generator is None:
            raise HTTPException(
                status_code=503,
                detail="the server was started without a language model")
        try:
            sources = retrieval.search(request.query, request.k)
            with lock:
                text = generator.answer(request.query, sources, request.k)
        except Exception as e:
            raise to_http_error(e) from e
        return MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=request.query,
            retrieved_sources=sources,
            answer=text)

    return app
