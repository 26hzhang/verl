

from .utils import extract_answer_math
import re

def find_first_letter_lower(s):
    match = re.search(r'[a-zA-Z]', s)
    if match:
        return match.group().lower()
    return None

def compute_score(solution_str, ground_truth) -> float:
    retval = -1
    try:
        answer = extract_answer_math(solution_str)
        pred_label = find_first_letter_lower(answer)
        if ground_truth.lower() == pred_label:
            # correct = True
            retval = 1.
    except Exception as e:
        print(e)
    reward = retval
    acc = retval == 1.0
    return {
        "score": reward,
        "acc": acc,
        # "pred": pred,
    }
