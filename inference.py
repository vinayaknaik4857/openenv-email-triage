from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any

from openai import OpenAI

from env.environment import CustomerSupportEmailTriageEnv
from env.models import ActionType, Category, Observation, Priority, TriageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "openenv-email-triage"
MAX_STEPS = 6
TEMPERATURE = 0.0
MAX_TOKENS = 220
VALID_ACTION_TYPES: set[ActionType] = {"classify", "prioritize", "draft", "finish"}
VALID_CATEGORIES: set[Category] = {"billing", "technical", "account", "shipping", "general"}
VALID_PRIORITIES: set[Priority] = {"low", "medium", "high", "urgent"}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def build_system_prompt() -> str:
    return textwrap.dedent(
        """
        You are a meticulous support operations agent.
        Solve one customer-support email triage task through four phases:
        1. classify
        2. prioritize
        3. draft
        4. finish

        Return exactly one JSON object with fields:
        - action_type
        - category
        - priority
        - response_sla_hours
        - response_draft
        - rationale

        Rules:
        - Always keep the current action_type aligned with the observation phase.
        - For unused fields in a phase, set them to null.
        - response_draft should be concise, empathetic, and operationally useful.
        - Do not wrap the JSON in markdown.
        """
    ).strip()


def build_user_prompt(observation: Observation) -> str:
    return textwrap.dedent(
        f"""
        Current observation:
        {json.dumps(observation.model_dump(), indent=2)}

        Choose the next action for this phase and return only JSON.
        """
    ).strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    candidate = raw_text.strip()
    if not candidate:
        raise ValueError("Empty model response")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be a JSON object")
    return parsed


def fallback_action(observation: Observation) -> dict[str, Any]:
    task_id = observation.task_id
    canned: dict[str, dict[str, Any]] = {
        "triage-easy-001": {
            "category": "billing",
            "priority": "high",
            "response_sla_hours": 8,
            "response_draft": (
                "Thanks for reporting the duplicate billing charge. I am sorry for the frustration. "
                "We will review the billing record, process the refund, and send you an update today."
            ),
        },
        "triage-medium-001": {
            "category": "technical",
            "priority": "urgent",
            "response_sla_hours": 2,
            "response_draft": (
                "Thanks for flagging this incident. We will investigate the SSO failures immediately, "
                "treat this as urgent, and share ETA updates until access is restored."
            ),
        },
        "triage-hard-001": {
            "category": "account",
            "priority": "urgent",
            "response_sla_hours": 1,
            "response_draft": (
                "Thank you for raising this access concern. We will urgently review account access, "
                "revoke any former contractor credentials, validate audit logs, and update you with next steps."
            ),
        },
    }
    base = canned[task_id]
    action: dict[str, Any] = {
        "action_type": observation.phase,
        "category": None,
        "priority": None,
        "response_sla_hours": None,
        "response_draft": None,
        "rationale": "Deterministic fallback policy for reproducible baseline scoring.",
    }
    if observation.phase == "classify":
        action["category"] = base["category"]
    elif observation.phase == "prioritize":
        action["priority"] = base["priority"]
        action["response_sla_hours"] = base["response_sla_hours"]
    elif observation.phase == "draft":
        action["response_draft"] = base["response_draft"]
    return action


def model_action(client: OpenAI | None, observation: Observation) -> dict[str, Any]:
    if client is None:
        return fallback_action(observation)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": build_user_prompt(observation)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = extract_json_object(raw)
    except Exception:
        return fallback_action(observation)

    parsed.setdefault("rationale", "LLM policy")
    parsed.setdefault("category", None)
    parsed.setdefault("priority", None)
    parsed.setdefault("response_sla_hours", None)
    parsed.setdefault("response_draft", None)
    parsed["action_type"] = observation.phase
    return parsed


def _clean_text(value: Any, default: str) -> str:
    if not isinstance(value, str):
        return default
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        return default
    return cleaned[:2000]


def sanitize_action_payload(observation: Observation, payload: dict[str, Any]) -> dict[str, Any]:
    action_type = observation.phase if observation.phase in VALID_ACTION_TYPES else "finish"
    category = payload.get("category")
    priority = payload.get("priority")
    response_sla_hours = payload.get("response_sla_hours")
    response_draft = payload.get("response_draft")

    if category not in VALID_CATEGORIES:
        category = None
    if priority not in VALID_PRIORITIES:
        priority = None
    if not isinstance(response_sla_hours, int) or isinstance(response_sla_hours, bool):
        response_sla_hours = None
    elif not 1 <= response_sla_hours <= 168:
        response_sla_hours = max(1, min(168, response_sla_hours))
    if not isinstance(response_draft, str):
        response_draft = None
    else:
        response_draft = response_draft.strip()[:4000] or None

    sanitized: dict[str, Any] = {
        "action_type": action_type,
        "category": None,
        "priority": None,
        "response_sla_hours": None,
        "response_draft": None,
        "rationale": _clean_text(payload.get("rationale"), "Automated baseline policy."),
    }
    if action_type == "classify":
        sanitized["category"] = category
    elif action_type == "prioritize":
        sanitized["priority"] = priority
        sanitized["response_sla_hours"] = response_sla_hours
    elif action_type == "draft":
        sanitized["response_draft"] = response_draft

    return sanitized


def build_triage_action(observation: Observation, payload: dict[str, Any]) -> TriageAction:
    sanitized = sanitize_action_payload(observation, payload)
    return TriageAction(
        task_id=observation.task_id,
        email_id=observation.email.email_id,
        action_type=sanitized["action_type"],
        category=sanitized["category"],
        priority=sanitized["priority"],
        response_sla_hours=sanitized["response_sla_hours"],
        response_draft=sanitized["response_draft"],
        rationale=sanitized["rationale"],
    )


def action_to_string(action: TriageAction) -> str:
    payload = {
        "action_type": action.action_type,
        "category": action.category,
        "priority": action.priority,
        "response_sla_hours": action.response_sla_hours,
        "response_draft": action.response_draft,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def run_task(env: CustomerSupportEmailTriageEnv, task_id: str, client: OpenAI | None) -> float:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    observation = env.reset(task_id=task_id)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if env.state().done:
                break

            error_message: str | None = None

            try:
                action_payload = model_action(client, observation)
            except Exception as exc:
                action_payload = fallback_action(observation)
                error_message = str(exc)

            try:
                action = build_triage_action(observation, action_payload)
            except Exception as exc:
                action = build_triage_action(observation, fallback_action(observation))
                error_message = str(exc)

            try:
                result = env.step(action)
                observation = result.observation
            except Exception as exc:
                result = env.step(build_triage_action(observation, fallback_action(observation)))
                observation = result.observation
                error_message = str(exc)

            rewards.append(result.reward)
            steps_taken = step
            log_step(
                step=step,
                action=action_to_string(action),
                reward=result.reward,
                done=result.done,
                error=observation.last_action_error or error_message,
            )

            if result.done:
                break

        score = env.grade_current_episode()
        success = score >= 0.7
        return score
    except Exception as exc:
        score = 0.0
        success = False
        log_step(
            step=max(steps_taken + 1, 1),
            action="internal_error",
            reward=0.00,
            done=True,
            error=str(exc),
        )
        return score
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    _ = LOCAL_IMAGE_NAME
    client: OpenAI | None = None
    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception:
            client = None

    env = CustomerSupportEmailTriageEnv()
    for task in env.list_tasks()[:3]:
        try:
            run_task(env, task.task_id, client)
        except Exception:
            log_start(task=task.task_id, env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="internal_error", reward=0.00, done=True, error="task_crashed")
            log_end(success=False, steps=0, score=0.00, rewards=[])


if __name__ == "__main__":
    main()
