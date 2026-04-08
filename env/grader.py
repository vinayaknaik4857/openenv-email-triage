from __future__ import annotations

from env.models import SupportTask, TriageAction


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _sla_score(predicted_hours: int | None, expected_hours: int) -> float:
    if predicted_hours is None:
        return 0.0
    if predicted_hours == expected_hours:
        return 1.0
    if predicted_hours <= expected_hours:
        return 0.8
    ratio = expected_hours / predicted_hours
    return _clamp(ratio)


def grade_classification(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    if action.category is None:
        return 0.0, {"classification": 0.0}
    correct = action.category == task.target.category
    score = 1.0 if correct else 0.0
    wrong_penalty = 0.08 if not correct else 0.0
    final = _clamp(score - wrong_penalty)
    return final, {"classification": final, "wrong_answer_penalty": wrong_penalty}


def grade_priority(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    priority_score = 1.0 if action.priority == task.target.priority else 0.0
    sla_score = _sla_score(action.response_sla_hours, task.target.response_sla_hours)
    blended = (0.65 * priority_score) + (0.35 * sla_score)
    wrong_penalty = 0.06 if priority_score == 0.0 else 0.0
    final = _clamp(blended - wrong_penalty)
    return final, {
        "priority_score": round(priority_score, 4),
        "sla_score": round(sla_score, 4),
        "wrong_answer_penalty": round(wrong_penalty, 4),
    }


def grade_response(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    text = (action.response_draft or "").lower()
    if not text:
        return 0.0, {
            "keyword_coverage": 0.0,
            "polite_tone": 0.0,
            "helpfulness": 0.0,
            "wrong_answer_penalty": 0.0,
        }

    required = [kw.lower() for kw in task.target.required_response_keywords]
    matched = sum(1 for kw in required if kw in text)
    coverage = 0.0 if not required else matched / len(required)

    polite_markers = ["sorry", "thanks", "thank you", "appreciate", "please"]
    helpful_markers = ["investigate", "update", "next", "resolve", "review", "share", "eta", "assist"]
    polite_tone = 1.0 if any(word in text for word in polite_markers) else 0.0
    helpfulness = 1.0 if any(word in text for word in helpful_markers) else 0.0

    raw_score = (0.6 * coverage) + (0.2 * polite_tone) + (0.2 * helpfulness)
    wrong_penalty = 0.04 if coverage < 0.34 else 0.0
    final = _clamp(raw_score - wrong_penalty)
    return final, {
        "keyword_coverage": round(coverage, 4),
        "polite_tone": round(polite_tone, 4),
        "helpfulness": round(helpfulness, 4),
        "wrong_answer_penalty": round(wrong_penalty, 4),
    }
