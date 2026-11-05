from src.indexing import chunk_files, BM25Index
from src.errors import RagError, InvalidArgumentError
from src.retrieval import Retrieval
from src.files import (read_span, load_dataset,
                       write_file, load_search_results, load_answers)
from src.models import (MinimalAnswer, StudentSearchResultsAndAnswer,
                        MinimalSearchResults, AnsweredQuestion)
from src.evaluation import compute_recall
from typing import List
import sys
import time
from tqdm import tqdm
from pathlib import Path


def check_int(value: object, name: str) -> int:
    """
    Make sure a command-line argument is an integer.

    Fire turns "5" into an int, but "abc" or "5.5" arrive as other types.

    Args:
        value: The argument as received from the command line.
        name: Name of the argument, used in the error message.

    Returns:
        The same value, as an int.

    Raises:
        InvalidArgumentError: if the value is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentError(
            f"{name} must be an integer, got {value!r}")
    return value


def report_error(message: str) -> None:
    """
    Print an error or a warning on the standard error output.

    Args:
        message: Text to show to the user.
    """
    print(message, file=sys.stderr)


def save_answers(path: Path, answers: List[MinimalAnswer], k: int) -> None:
    """
    Write the answers generated so far as a StudentSearchResultsAndAnswer.

    Args:
        path: File to write.
        answers: Answers generated so far.
        k: Number of sources requested for each question.
    """
    output = StudentSearchResultsAndAnswer(search_results=answers, k=k)
    write_file(path, output.model_dump_json(indent=2))


def load_progress(partial_path: Path, k: int,
                  results: List[MinimalSearchResults]
                  ) -> List[MinimalAnswer]:
    """
    Recover the answers saved by a previous interrupted run.

    The partial file is ignored if it does not exist, is corrupted, was
    made with another k, or belongs to other questions.

    Args:
        partial_path: Partial file written during the previous run.
        k: Number of sources requested in the current run.
        results: Questions of the current run.

    Returns:
        The answers that can be reused (empty if none).
    """
    if not partial_path.exists():
        return []
    try:
        saved = load_answers(str(partial_path))
    except RagError as e:
        report_error(f"Ignoring unusable partial file: {e}")
        return []
    if saved.k != k:
        report_error("Ignoring partial file: it was made with another k")
        return []
    current = {r.question_id: r.question for r in results}
    kept = [a for a in saved.search_results
            if current.get(a.question_id) == a.question]
    if kept:
        print(f"Resuming: {len(kept)} of {len(results)} "
              f"questions already answered")
    return kept


def complete_answers(questions: List[MinimalSearchResults],
                     answers: List[MinimalAnswer],
                     default: str) -> List[MinimalAnswer]:
    """
    Build one answer per question, in the order of the questions.

    Args:
        questions: Questions of the run, in their original order.
        answers: Answers generated successfully.
        default: Answer given to the questions that have none.

    Returns:
        A complete list of answers.
    """
    by_id = {a.question_id: a for a in answers}
    result = []
    for q in questions:
        answer = by_id.get(q.question_id)
        if answer is None:
            answer = MinimalAnswer(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=q.retrieved_sources,
                answer=default)
        result.append(answer)
    return result


class RagCLI:
    """
    Command-line interface of the RAG system.

    Each public method is a command, run with
    `uv run python -m src <command> [options]` through Python Fire.
    """

    def index(self, max_chunk_size: int = 2000,
              root: str = "data/raw/vllm-0.10.1",
              end_path: str = "data/processed") -> None:
        """
        Chunk the corpus and save a BM25 index.

        Args:
            max_chunk_size: Maximum size of a chunk, in characters
                (between 1 and 2000).
            root: Directory with the files to index.
            end_path: Directory where the index is saved.
        """
        try:
            max_chunk_size = check_int(max_chunk_size, "max_chunk_size")
            start = time.perf_counter()
            chunks_list = chunk_files(str(root), max_chunk_size)
            index = BM25Index(chunks_list)
            index.save(str(end_path))
            print(f"Ingestion complete! Indices saved under {end_path}")
            end = time.perf_counter()
            print(f"Time: {end - start}")
        except RagError as e:
            report_error(f"Indexing cannot be completed: {e}")
        except Exception as e:
            report_error(f"Unexpected error while indexing: {e}")

    def search(self, query: str, k: int = 5,
               index_path: str = "data/processed",
               mode: str = "bm25") -> None:
        """
        Print the top-k sources for a single query.

        Args:
            query: The question to search.
            k: How many sources to show.
            index_path: Directory where the index was saved.
            mode: "bm25", "semantic" or "hybrid" (both combined).
        """
        try:
            k = check_int(k, "k")
            retrieval = Retrieval(index_path, mode)
            results = retrieval.search(str(query), k)
            if not results:
                print("No results found.")
                return
            for rank, source in enumerate(results, start=1):
                excerpt = read_span(source)[:200].replace("\n", " ")
                print(f"{rank}. {source.file_path} "
                      f"[{source.first_character_index}:"
                      f"{source.last_character_index}]")
                print(f"   {excerpt}")
        except RagError as e:
            report_error(f"Search cannot be completed: {e}")
        except Exception as e:
            report_error(f"Unexpected error while searching: {e}")

    def search_dataset(self, dataset_path: str, save_directory: str,
                       k: int = 10, index_path: str = "data/processed",
                       mode: str = "bm25"
                       ) -> None:
        """
        Search all the questions of a dataset and save the results.

        The results file has the same name as the dataset file.

        Args:
            dataset_path: JSON file with the questions.
            save_directory: Directory where the results are saved.
            k: How many sources to return for each question.
            index_path: Directory where the index was saved.
            mode: "bm25", "semantic" or "hybrid" (both combined).
        """
        try:
            k = check_int(k, "k")
            dataset = load_dataset(dataset_path)
            retrieval = Retrieval(index_path, mode)
            results = retrieval.search_dataset(dataset, k)
            folder = Path(save_directory)
            folder.mkdir(parents=True, exist_ok=True)
            out_path = folder / Path(dataset_path).name
            write_file(out_path, results.model_dump_json(indent=2))
            print(f"Saved student_search_results to {out_path}")
        except RagError as e:
            report_error(f"Search cannot be completed: {e}")
        except Exception as e:
            report_error(f"Unexpected error while searching: {e}")

    def answer(self, query: str, k: int = 5,
               index_path: str = "data/processed",
               mode: str = "bm25") -> None:
        """
        Answer a single query using the retrieved context.

        Args:
            query: The question to answer.
            k: How many sources to give to the model.
            index_path: Directory where the index was saved.
            mode: "bm25", "semantic" or "hybrid" (both combined).
        """
        try:
            k = check_int(k, "k")
            from src.generation import Generator
            retrieval = Retrieval(index_path, mode)
            sources = retrieval.search(str(query), k)
            print(Generator().answer(str(query), sources, k))
        except RagError as e:
            report_error(f"Answer cannot be completed: {e}")
        except Exception as e:
            report_error(f"Unexpected error while answering: {e}")

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """
        Generate an answer for every question of a search results file.

        Args:
            student_search_results_path: JSON produced by search_dataset.
            save_directory: Directory where the answers are saved.
        """
        try:
            from src.generation import Generator, NO_INFO_ANSWER
            results = load_search_results(student_search_results_path)
            folder = Path(save_directory)
            folder.mkdir(parents=True, exist_ok=True)
            out_path = folder / Path(student_search_results_path).name
            partial_path = out_path.with_suffix(".partial.json")
            generator = Generator()
            answers = load_progress(
                partial_path, results.k, results.search_results)
            done = {a.question_id for a in answers}
            pending = [r for r in results.search_results
                       if r.question_id not in done]
            failed: List[str] = []
            for result in tqdm(pending, desc="Answering",
                               total=len(results.search_results),
                               initial=len(answers)):
                if not result.question.strip():
                    text = NO_INFO_ANSWER
                else:
                    try:
                        text = generator.answer(
                            result.question, result.retrieved_sources, 3)
                    except Exception as e:
                        tqdm.write(f"Question {result.question_id}"
                                   f" failed: {e}", file=sys.stderr)
                        failed.append(result.question_id)
                        continue
                answers.append(MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=text))
                save_answers(partial_path, answers, results.k)
            complete = complete_answers(
                results.search_results, answers, NO_INFO_ANSWER)
            save_answers(out_path, complete, results.k)
            print(f"Saved answered questions to {out_path}")
            if failed:
                report_error(
                    f"{len(failed)} questions failed and got the default "
                    f"answer. Run the same command again to retry them.")
            else:
                partial_path.unlink(missing_ok=True)
        except KeyboardInterrupt:
            report_error(
                "\nInterrupted. Run the same command again to resume.")
        except RagError as e:
            report_error(f"Answer cannot be completed: {e}")
        except Exception as e:
            report_error(f"Unexpected error while answering: {e}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """
        Print the recall@k of a search results file.

        Args:
            student_search_results_path: JSON produced by search_dataset.
            dataset_path: JSON with the ground-truth answered questions.
        """
        try:
            search_results = load_search_results(student_search_results_path)
            dataset = load_dataset(dataset_path)
            results = compute_recall(
                dataset.rag_questions, search_results)
            if not results:
                report_error("Evaluation cannot be completed: the dataset "
                             "has no questions with ground truth.")
                return
            evaluated = sum(isinstance(q, AnsweredQuestion)
                            for q in dataset.rag_questions)
            print("Evaluation results")
            print("===========================")
            print(f"Questions evaluated: {evaluated}")
            for k, recall in results.items():
                print(f"Recall@{k}: {recall:.3f} ({recall:.1%})")
        except RagError as e:
            report_error(f"Evaluation cannot be completed: {e}")
        except Exception as e:
            report_error(f"Unexpected error while evaluating: {e}")

    def index_semantic(self, max_chunk_size: int = 2000,
                       root: str = "data/raw/vllm-0.10.1",
                       end_path: str = "data/processed") -> None:
        """
        Chunk the corpus and save a semantic (embeddings) index.

        It is a separate command from index because embedding every
        chunk is much slower than building the BM25 index.

        Args:
            max_chunk_size: Maximum size of a chunk, in characters
                (between 1 and 2000).
            root: Directory with the files to index.
            end_path: Directory where the index is saved.
        """
        try:
            from src.semantic import SemanticIndex
            max_chunk_size = check_int(max_chunk_size, "max_chunk_size")
            start = time.perf_counter()
            chunks_list = chunk_files(str(root), max_chunk_size)
            index = SemanticIndex()
            index.build(chunks_list)
            index.save(str(end_path))
            print(f"Semantic index saved under {end_path}")
            print(f"Time: {time.perf_counter() - start}")
        except RagError as e:
            report_error(f"Semantic indexing cannot be completed: {e}")
        except Exception as e:
            report_error(f"Unexpected error while indexing: {e}")

    def serve(self, host: str = "127.0.0.1", port: int = 8000,
              index_path: str = "data/processed", mode: str = "bm25",
              with_answer: bool = True) -> None:
        """
        Start the local HTTP API.

        Args:
            host: Address where the server listens.
            port: Port where the server listens.
            index_path: Directory where the index was saved.
            mode: "bm25", "semantic" or "hybrid".
            with_answer: Load the language model so that /answer works.
        """
        try:
            port = check_int(port, "port")
            import uvicorn
            from src.api import create_app
            app = create_app(index_path, mode, with_answer)
            uvicorn.run(app, host=str(host), port=port)
        except KeyboardInterrupt:
            print("\nServer stopped.")
        except RagError as e:
            report_error(f"Server cannot be started: {e}")
        except Exception as e:
            report_error(f"Unexpected error while serving: {e}")
