from pathlib import Path
from src.errors import FileReadError
from src.models import MinimalSource
from pydantic import ValidationError
from src.models import (RagDataset, StudentSearchResults,
                        StudentSearchResultsAndAnswer)
from src.errors import InvalidDatasetError, FileWriteError


def read_file(path: str) -> str:
    """
    Read a UTF-8 text file and return its content.

    Args:
        path: Path of the file to read.

    Returns:
        The content of the file.
    """
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError as e:
        raise FileReadError(f"file not found: {path}.") from e
    except PermissionError as e:
        raise FileReadError(f"permission denied: {path}.") from e
    except OSError as e:
        raise FileReadError(f"file {path} cannot be read.") from e


def write_file(path: Path, content: str) -> None:
    """
    Write content to a UTF-8 text file without ever leaving it half written.

    The text goes first to a temporary file that then replaces the target,
    so a crash while writing keeps the previous version intact.

    Args:
        path: Path of the file to write.
        content: Text to write.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as e:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise FileWriteError(f"cannot write {path}") from e


def read_span(source: MinimalSource) -> str:
    """
    Return the text of the file region described by a source.

    Args:
        source: Location (file and character range) to read.

    Returns:
        The text between the first and the last character index.
    """
    text = read_file(source.file_path)
    return text[source.first_character_index:source.last_character_index]


def load_dataset(path: str) -> RagDataset:
    """
    Read and validate a dataset of questions.

    Args:
        path: Path of the JSON file of the dataset.

    Returns:
        The dataset, validated.
    """
    text = read_file(path)
    try:
        return RagDataset.model_validate_json(text)
    except ValidationError as e:
        raise InvalidDatasetError(
            f"{path} is not a valid dataset ({e.error_count()} errors)") from e


def load_search_results(path: str) -> StudentSearchResults:
    """
    Read and validate the output of search_dataset.

    Args:
        path: Path of the JSON file with the search results.

    Returns:
        The search results, validated.
    """
    text = read_file(path)
    try:
        return StudentSearchResults.model_validate_json(text)
    except ValidationError as e:
        raise InvalidDatasetError(
            f"{path} is not a valid search results file "
            f"({e.error_count()} errors)") from e


def load_answers(path: str) -> StudentSearchResultsAndAnswer:
    """
    Read and validate a file of answers (final or partial).

    Args:
        path: Path of the JSON file with the answers.

    Returns:
        The search results with their answers, validated.
    """
    text = read_file(path)
    try:
        return StudentSearchResultsAndAnswer.model_validate_json(text)
    except ValidationError as e:
        raise InvalidDatasetError(
            f"{path} is not a valid answers file "
            f"({e.error_count()} errors)") from e
