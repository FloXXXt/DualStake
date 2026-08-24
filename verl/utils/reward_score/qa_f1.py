# F1 score implementation for QA reward, following the same structure and style 
# as qa_em.py.

import re
import string
import random

# -----------------------------------------------------------------------------
# Normalization (identical to qa_em.py)
# -----------------------------------------------------------------------------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# -----------------------------------------------------------------------------
# Token-level F1 score (SQuAD-style)
# -----------------------------------------------------------------------------
def f1_score(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    pred = normalize_answer(prediction)
    pred_tokens = pred.split()

    if len(pred_tokens) == 0:
        return 0.0

    best_f1 = 0.0
    for golden_answer in golden_answers:
        gold = normalize_answer(golden_answer)
        gold_tokens = gold.split()

        if len(gold_tokens) == 0:
            continue

        # Count overlaps (multiset intersection)
        gold_counts = {}
        for t in gold_tokens:
            gold_counts[t] = gold_counts.get(t, 0) + 1

        common = 0
        for t in pred_tokens:
            if gold_counts.get(t, 0) > 0:
                common += 1
                gold_counts[t] -= 1

        if common == 0:
            continue

        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        best_f1 = max(best_f1, f1)

    return best_f1


# -----------------------------------------------------------------------------
# Extract solution (<answer>...</answer>)
# Reuse the same function from qa_em
# -----------------------------------------------------------------------------
from . import qa_em
extract_solution = qa_em.extract_solution


# -----------------------------------------------------------------------------
# Main scoring function: compute F1-only reward
# -----------------------------------------------------------------------------
def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """
    F1-only scoring function.

    Args:
        solution_str: decoded full sequence
        ground_truth: dict, contains `target`
        method: kept for compatibility
        format_score: not used in F1-only setting (kept for compatibility)
        score: max score for fully correct answer (usually 1.0)
    """

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    # No answer → format error → reward = 0
    if answer is None:
        return 0.0

    # Compute token-level F1
    f1 = f1_score(answer, ground_truth['target'])

    # Reward = f1 * score
    return score * f1
