from typing import Tuple, List, Dict
from src.models import (MinimalSource,
                        AnsweredQuestion, StudentSearchResults,
                        UnansweredQuestion)

DEFAULT_KS = [1, 3, 5, 10]


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Compute the overlap between two character ranges (IoU).

    IoU is the length of the intersection divided by the length of the
    union of the two ranges.

    Args:
        a: First range, as (start, end).
        b: Second range, as (start, end).

    Returns:
        A value between 0.0 (no overlap) and 1.0 (identical ranges).
    """
    a_ini, a_fin = a
    b_ini, b_fin = b
    total = max(a_fin, b_fin) - min(a_ini, b_ini)
    overlap = min(a_fin, b_fin) - max(a_ini, b_ini)
    if overlap <= 0:
        return 0.0
    return overlap / total


def is_found(correct: MinimalSource,
             retrieved: List[MinimalSource]) -> bool:
    """
    Tell whether a correct source was retrieved.

    A correct source is found if one of the retrieved sources is in the
    same file and overlaps its character range with an IoU of at least
    0.05.

    Args:
        correct: The ground-truth source.
        retrieved: Sources returned by the system.

    Returns:
        True if the correct source was found.
    """
    correct_char = (correct.first_character_index,
                    correct.last_character_index)
    for source in retrieved:
        if source.file_path == correct.file_path:
            source_char = (source.first_character_index,
                           source.last_character_index)
            if iou(correct_char, source_char) >= 0.05:
                return True
    return False


def recall_at_k(k: int,
                correct_sources: List[MinimalSource],
                retrieved: List[MinimalSource],
                ) -> float:
    """
    Compute the recall of one question using only the first k results.

    Args:
        k: How many retrieved sources to take into account.
        correct_sources: Ground-truth sources of the question.
        retrieved: Sources returned by the system, best first.

    Returns:
        The share of correct sources that were found (0.0 if the
        question has no correct sources).
    """
    total = len(correct_sources)
    correct = 0
    if total == 0:
        return 0.0
    retrieved = retrieved[:k]
    for source in correct_sources:
        if is_found(source, retrieved):
            correct += 1
    return correct / total


def recall_for_question(question: AnsweredQuestion,
                        retrieved: List[MinimalSource]
                        ) -> Dict[int, float]:
    """
    Compute the recall of one question for every k of DEFAULT_KS.

    Args:
        question: A question with its ground-truth sources.
        retrieved: Sources returned by the system, best first.

    Returns:
        A dictionary {k: recall@k}.
    """
    results: Dict[int, float] = {}
    correct_sources = question.sources
    for k in DEFAULT_KS:
        results[k] = recall_at_k(k, correct_sources, retrieved)
    return results


def compute_recall(
        questions: List[AnsweredQuestion | UnansweredQuestion],
        search_results: StudentSearchResults
                   ) -> Dict[int, float]:
    """
    Compute the average recall@k over all the answered questions.

    Questions without ground truth are ignored. A question that has no
    entry in the search results counts as recall 0.

    Args:
        questions: Questions of the dataset.
        search_results: Results produced by search_dataset.

    Returns:
        A dictionary {k: average recall@k}, or an empty one if there
        are no answered questions.
    """
    answered = [q for q in questions if isinstance(q, AnsweredQuestion)]
    results: Dict[int, float] = {k: 0.0 for k in DEFAULT_KS}
    retrieved_by_id: Dict[str, List[MinimalSource]] = {
        r.question_id: r.retrieved_sources
        for r in search_results.search_results}
    total_questions = len(answered)
    if total_questions == 0:
        return {}
    for q in answered:
        ind_recall = recall_for_question(
            q, retrieved_by_id.get(q.question_id, []))
        for k in DEFAULT_KS:
            results[k] += ind_recall[k]
    for k in DEFAULT_KS:
        results[k] = results[k] / total_questions
    return results
