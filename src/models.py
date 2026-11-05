from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """
    A location in a file of the corpus.

    Attributes:
        file_path: Path of the file, written as in the corpus.
        first_character_index: Index of the first character.
        last_character_index: Index where the span ends (exclusive).
    """
    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)


class UnansweredQuestion(BaseModel):
    """
    A question to be answered by the RAG system.

    Attributes:
        question_id: Unique identifier (a random UUID if not given).
        question: Text of the question.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    A question together with its ground-truth solution.

    Attributes:
        sources: Locations of the corpus that contain the answer.
        answer: Reference answer to the question.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    Represents a dataset of RAG questions.
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    Sources retrieved by the system for a single question.

    Attributes:
        question_id: Identifier of the question.
        question: Text of the question.
        retrieved_sources: Sources found, best first.
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Retrieved sources of a question plus the generated answer.

    Attributes:
        answer: Answer generated from the retrieved sources.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Search results for a whole dataset (output of search_dataset).

    Attributes:
        search_results: One entry per question.
        k: Number of sources requested for each question.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Search results with generated answers for a whole dataset.

    Attributes:
        search_results: One entry per question, with its answer.
        k: Number of sources requested for each question.
    """
    search_results: List[MinimalAnswer]
    k: int


class Chunk(MinimalSource):
    """
    A piece of a file: its location plus its text.

    Attributes:
        chunk_text: Text between the first and the last index.
        context: info used for indexing (section title and path)
    """
    chunk_text: str
    context: str = ""


class QueryRequest(BaseModel):
    """
    A request to the HTTP API: a question and how many sources to use.

    Attributes:
        query: Text of the question.
        k: Number of sources to retrieve (between 1 and 20).
    """
    query: str
    k: int = Field(default=5, ge=1, le=20)
