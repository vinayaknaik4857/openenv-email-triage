from __future__ import annotations

from env.models import PartialTriage, SupportTask, TriageAction


def _clamp(score: float) -> float:
    return round(min(1.0, max(0.0, float(score))), 4)


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lowered = text.lower()
    matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return matched / len(keywords)


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def grade_classification(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    if action.category is None:
        return 0.0, {"classification_accuracy": 0.0}
    exact = 1.0 if action.category == task.target.category else 0.0
    return _clamp(exact), {"classification_accuracy": _clamp(exact)}


def grade_priority(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    priority_score = 1.0 if action.priority == task.target.priority else 0.0

    if action.response_sla_hours is None:
        sla_score = 0.0
    elif action.response_sla_hours == task.target.response_sla_hours:
        sla_score = 1.0
    elif action.response_sla_hours < task.target.response_sla_hours:
        sla_score = 0.8
    else:
        sla_score = max(0.0, task.target.response_sla_hours / action.response_sla_hours)

    final = _clamp((0.7 * priority_score) + (0.3 * sla_score))
    return final, {
        "priority_accuracy": _clamp(priority_score),
        "sla_alignment": _clamp(sla_score),
    }


def grade_response(task: SupportTask, action: TriageAction) -> tuple[float, dict[str, float]]:
    text = (action.response_draft or "").strip()
    if not text:
        return 0.0, {
            "keyword_coverage": 0.0,
            "polite_tone": 0.0,
            "actionability": 0.0,
            "brevity": 0.0,
        }

    coverage = _keyword_coverage(text, task.target.required_response_keywords)
    polite = 1.0 if _contains_any(text, ["sorry", "thanks", "thank you", "appreciate", "please"]) else 0.35
    actionability = 1.0 if _contains_any(
        text,
        ["investigate", "review", "update", "next", "share", "revoke", "resolve", "eta"],
    ) else 0.25
    brevity = 1.0 if 25 <= len(text.split()) <= 90 else 0.6

    final = _clamp((0.5 * coverage) + (0.2 * polite) + (0.2 * actionability) + (0.1 * brevity))
    return final, {
        "keyword_coverage": _clamp(coverage),
        "polite_tone": _clamp(polite),
        "actionability": _clamp(actionability),
        "brevity": _clamp(brevity),
    }


def grade_completion(partial: PartialTriage, task: SupportTask) -> tuple[float, dict[str, float]]:
    checks = {
        "category_present": 1.0 if partial.category is not None else 0.0,
        "priority_present": 1.0 if partial.priority is not None else 0.0,
        "sla_present": 1.0 if partial.response_sla_hours is not None else 0.0,
        "response_present": 1.0 if bool((partial.response_draft or "").strip()) else 0.0,
    }
    complete = all(value == 1.0 for value in checks.values())
    exact_alignment = 1.0 if (
        partial.category == task.target.category
        and partial.priority == task.target.priority
        and partial.response_sla_hours is not None
        and partial.response_sla_hours <= task.target.response_sla_hours
        and _keyword_coverage(partial.response_draft or "", task.target.required_response_keywords) >= 0.67
    ) else 0.0
    score = _clamp((0.6 * (sum(checks.values()) / len(checks))) + (0.4 * exact_alignment))
    return score, {**checks, "exact_alignment": _clamp(exact_alignment), "completion_score": score}


def grade_episode(
    task: SupportTask,
    partial: PartialTriage,
    penalties: int,
    step_count: int,
) -> tuple[float, dict[str, float]]:
    classification = 1.0 if partial.category == task.target.category else 0.0
    priority = 1.0 if partial.priority == task.target.priority else 0.0
    if partial.response_sla_hours is None:
        sla = 0.0
    elif partial.response_sla_hours == task.target.response_sla_hours:
        sla = 1.0
    elif partial.response_sla_hours < task.target.response_sla_hours:
        sla = 0.85
    else:
        sla = max(0.0, task.target.response_sla_hours / partial.response_sla_hours)
    response = grade_response(task, TriageAction(
        task_id=task.task_id,
        email_id=task.email.email_id,
        action_type="draft",
        response_draft=partial.response_draft,
        rationale="episode-grade",
    ))[0]
    efficiency = max(0.0, 1.0 - max(0, step_count - 4) * 0.08)
    penalty_score = max(0.0, 1.0 - (penalties * 0.15))

    score = _clamp(
        (0.2 * classification)
        + (0.2 * priority)
        + (0.15 * sla)
        + (0.3 * response)
        + (0.05 * efficiency)
        + (0.1 * penalty_score)
    )
    return score, {
        "classification": _clamp(classification),
        "priority": _clamp(priority),
        "sla": _clamp(sla),
        "response": _clamp(response),
        "efficiency": _clamp(efficiency),
        "penalty_score": _clamp(penalty_score),
        "episode_score": score,
    }
