from __future__ import annotations

from env.models import SupportTask, TriageAction


def _strict(x: float) -> float:
    return max(0.01, min(0.99, float(x)))



def _sla_score(predicted_hours: int | None, expected_hours: int) -> float:
    if predicted_hours is None:
        return _strict(0.05)
    if predicted_hours == expected_hours:
        return _strict(0.95)
    if predicted_hours <= expected_hours:
        return _strict(0.8)
    return _strict(expected_hours / predicted_hours)


def grade_classification(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    if action.category is None:
        s = _strict(0.05)
        return s, {"classification": s}
    correct = action.category == task.target.category
    base = 0.95 if correct else 0.1
    final = _strict(base - (0.05 if not correct else 0.0))
    return final, {"classification": final, "wrong_answer_penalty": _strict(0.05 if not correct else 0.01)}


def grade_priority(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    p = 0.95 if action.priority == task.target.priority else 0.1
    s = _sla_score(action.response_sla_hours, task.target.response_sla_hours)
    final = _strict((0.65 * p) + (0.35 * s))
    return final, {"priority_score": _strict(p), "sla_score": _strict(s), "wrong_answer_penalty": _strict(0.06 if p < 0.2 else 0.01)}


def grade_response(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    text = (action.response_draft or "").lower().strip()
    if not text:
        s = _strict(0.05)
        return s, {"keyword_coverage": s, "polite_tone": s, "helpfulness": s, "wrong_answer_penalty": _strict(0.04)}

    required = [kw.lower() for kw in task.target.required_response_keywords]
    matched = sum(1 for kw in required if kw in text)
    coverage = _strict(matched / len(required)) if required else _strict(0.5)

    polite_markers = ["sorry", "thanks", "thank you", "appreciate", "please"]
    helpful_markers = ["investigate", "update", "next", "resolve", "review", "share", "eta", "assist", "revoke", "audit"]

    polite = _strict(0.95 if any(w in text for w in polite_markers) else 0.1)
    helpful = _strict(0.95 if any(w in text for w in helpful_markers) else 0.1)

    raw = (0.6 * coverage) + (0.2 * polite) + (0.2 * helpful)
    final = _strict(raw - (0.04 if coverage < 0.34 else 0.0))
    return final, {"keyword_coverage": _strict(coverage), "polite_tone": _strict(polite), "helpfulness": _strict(helpful), "wrong_answer_penalty": _strict(0.04 if coverage < 0.34 else 0.01)}
