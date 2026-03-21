"""Standalone is_continuous_scoring for subprocess usage.

Canonical implementation lives in RoboPhD.report_generator. This copy
exists because the error_analysis scripts are invoked via subprocess
where the RoboPhD package may not be on sys.path.

Keep in sync with report_generator.is_continuous_scoring.
"""


def is_continuous_scoring(scores_by_question: dict) -> bool:
    """Detect whether scores are continuous (not mostly binary).

    Returns True if scores should use continuous-score report format.
    Binary format is used when >= 80% of scores are exactly 0 or 1,
    AND all scores are in [0, 1].
    """
    all_scores = []
    for qid, agent_scores in scores_by_question.items():
        for agent, score in agent_scores.items():
            all_scores.append(score)

    if not all_scores:
        return False

    # Any score outside [0, 1] → continuous
    if any(s < 0.0 or s > 1.0 for s in all_scores):
        return True

    # Less than 80% binary → continuous
    binary_count = sum(1 for s in all_scores if s == 0.0 or s == 1.0)
    return binary_count / len(all_scores) < 0.8
