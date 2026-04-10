from __future__ import annotations

import uuid
from typing import Iterable

from pydantic import ValidationError

from env.grader import grade_classification, grade_completion, grade_episode, grade_priority, grade_response
from env.models import (
    EnvState,
    Observation,
    PartialTriage,
    StepInfo,
    StepResult,
    SupportTask,
    TriageAction,
)
from env.tasks import get_tasks


class CustomerSupportEmailTriageEnv:
    """Deterministic OpenEnv environment for customer support email triage."""

    def __init__(self) -> None:
        self._tasks: list[SupportTask] = get_tasks()
        self._episode_id: str = ""
        self._current_task: SupportTask | None = None
        self._done: bool = True
        self._phase: str = "done"
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._last_step_reward: float = 0.0
        self._partial: PartialTriage = PartialTriage()
        self._penalties: int = 0
        self._last_action_error: str | None = None
        self._reward_breakdown: dict[str, float] = {}
        self._final_score: float = 0.0

    def reset(self, task_id: str | None = None) -> Observation:
        self._episode_id = str(uuid.uuid4())
        self._done = False
        self._phase = "classify"
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._last_step_reward = 0.0
        self._partial = PartialTriage()
        self._penalties = 0
        self._last_action_error = None
        self._reward_breakdown = {}
        self._final_score = 0.0

        if task_id is None:
            self._current_task = self._tasks[0]
        else:
            matches = [task for task in self._tasks if task.task_id == task_id]
            if not matches:
                valid = ", ".join(task.task_id for task in self._tasks)
                raise ValueError(f"Unknown task_id '{task_id}'. Valid IDs: {valid}")
            self._current_task = matches[0]

        return self._build_observation()

    def step(self, action: TriageAction | dict) -> StepResult:
        if self._done or self._current_task is None:
            raise RuntimeError("Environment is done. Call reset() before step().")

        self._step_count += 1
        notes: list[str] = []
        step_reward = 0.0
        reward_components: dict[str, float] = {}
        self._last_action_error = None

        try:
            parsed = action if isinstance(action, TriageAction) else TriageAction.model_validate(action)
        except ValidationError as exc:
            parsed = None
            self._apply_penalty("Invalid action schema.", notes)
            self._last_action_error = str(exc)

        if parsed is not None and not self._validate_identity(parsed):
            self._apply_penalty("Action identity does not match current task/email.", notes)
            self._last_action_error = "task_id/email_id mismatch"
            parsed = None

        if parsed is None:
            reward_components["invalid_action_penalty"] = 0.0
        elif parsed.action_type != self._phase:
            self._apply_penalty(f"Expected action_type={self._phase}.", notes)
            self._last_action_error = f"invalid_action_type:{parsed.action_type}"
            reward_components["out_of_sequence_penalty"] = 0.0
        elif self._phase == "classify":
            phase_score, details = grade_classification(self._current_task, parsed)
            reward_components.update(details)
            step_reward = round(0.25 * phase_score, 4)
            self._partial.category = parsed.category
            self._phase = "prioritize"
        elif self._phase == "prioritize":
            phase_score, details = grade_priority(self._current_task, parsed)
            reward_components.update(details)
            step_reward = round(0.25 * phase_score, 4)
            self._partial.priority = parsed.priority
            self._partial.response_sla_hours = parsed.response_sla_hours
            self._phase = "draft"
        elif self._phase == "draft":
            phase_score, details = grade_response(self._current_task, parsed)
            reward_components.update(details)
            step_reward = round(0.35 * phase_score, 4)
            self._partial.response_draft = parsed.response_draft
            self._phase = "finish"
        elif self._phase == "finish":
            phase_score, details = grade_completion(self._partial, self._current_task)
            reward_components.update(details)
            step_reward = round(0.15 * phase_score, 4)
            self._phase = "done"
            self._done = True

        if self._step_count >= self._current_task.max_steps and not self._done:
            self._apply_penalty("Reached max steps before finish.", notes)
            self._phase = "done"
            self._done = True

        self._last_step_reward = self._normalize_reward(step_reward)
        self._cumulative_reward = self._normalize_reward(self._cumulative_reward + self._last_step_reward)
        self._reward_breakdown = {key: round(value, 4) for key, value in reward_components.items()}
        self._finalize_episode_if_needed()

        info = StepInfo(
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            phase=self._phase,  # type: ignore[arg-type]
            reward_components=self._reward_breakdown,
            penalties=self._penalties,
            notes=notes,
            score=self._final_score,
        )

        return StepResult(
            observation=self._build_observation(),
            reward=self._last_step_reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvState:
        return EnvState(
            episode_id=self._episode_id,
            current_task_id=self._current_task.task_id if self._current_task else None,
            done=self._done,
            phase=self._phase,  # type: ignore[arg-type]
            step_count=self._step_count,
            max_steps=self._current_task.max_steps if self._current_task else 0,
            cumulative_reward=self._cumulative_reward,
            final_score=self._final_score,
            current_task_difficulty=self._current_task.difficulty if self._current_task else None,
            partial=self._partial,
            penalties=self._penalties,
            available_task_ids=[task.task_id for task in self._tasks],
            last_action_error=self._last_action_error,
            reward_breakdown=self._reward_breakdown,
        )

    def list_tasks(self) -> list[SupportTask]:
        return list(self._tasks)

    def grade_current_episode(self) -> float:
        return self._final_score

    def _validate_identity(self, action: TriageAction) -> bool:
        if self._current_task is None:
            return False
        return action.task_id == self._current_task.task_id and action.email_id == self._current_task.email.email_id

    def _apply_penalty(self, reason: str, notes: list[str]) -> None:
        self._penalties += 1
        notes.append(reason)
        self._cumulative_reward = self._normalize_reward(self._cumulative_reward - 0.03)

    def _remaining_steps(self) -> int:
        if self._current_task is None:
            return 0
        return max(0, self._current_task.max_steps - self._step_count)

    def _available_actions(self) -> list[str]:
        return [] if self._done else [self._phase]

    def _hints(self) -> list[str]:
        if self._current_task is None:
            return ["Call reset() to start an episode."]
        if self._phase == "classify":
            return [
                "Choose the single best support queue category.",
                *self._current_task.success_notes[:1],
            ]
        if self._phase == "prioritize":
            return [
                "Set business priority and an SLA in hours.",
                "Shorter SLAs are acceptable for severe incidents; overly slow SLAs lose credit.",
            ]
        if self._phase == "draft":
            return [
                "Draft a professional reply with empathy and next actions.",
                *self._current_task.success_notes[-1:],
            ]
        if self._phase == "finish":
            return ["Finalize once all required fields are filled."]
        return ["Episode complete. Call reset() for the next task."]

    def _build_observation(self) -> Observation:
        if self._current_task is None:
            raise RuntimeError("No current task. Call reset() first.")
        return Observation(
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            objective=self._current_task.objective,
            instructions=self._current_task.instructions,
            phase=self._phase,  # type: ignore[arg-type]
            step_count=self._step_count,
            remaining_steps=self._remaining_steps(),
            email=self._current_task.email,
            partial=self._partial,
            available_actions=self._coerce_action_literals(self._available_actions()),
            hints=self._hints(),
            last_step_reward=self._last_step_reward,
            cumulative_reward=self._cumulative_reward,
            last_action_error=self._last_action_error,
        )

    def _finalize_episode_if_needed(self) -> None:
        if not self._done or self._current_task is None:
            return
        score, details = grade_episode(
            task=self._current_task,
            partial=self._partial,
            penalties=self._penalties,
            step_count=self._step_count,
        )
        self._reward_breakdown = {key: round(value, 4) for key, value in details.items()}
        self._final_score = score
        self._cumulative_reward = self._normalize_reward(max(self._cumulative_reward, score))

    @staticmethod
    def _normalize_reward(value: float) -> float:
        return round(min(1.0, max(0.0, value)), 4)

    @staticmethod
    def _coerce_action_literals(values: Iterable[str]) -> list[str]:
        return [value for value in values]
