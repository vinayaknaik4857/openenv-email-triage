from __future__ import annotations

import uuid
from typing import Iterable

from pydantic import ValidationError

from env.grader import grade_classification, grade_priority, grade_response
from env.models import EnvState, Observation, PartialTriage, StepInfo, StepResult, SupportTask, TriageAction
from env.tasks import get_tasks


class CustomerSupportEmailTriageEnv:
    def __init__(self) -> None:
        self._tasks: list[SupportTask] = get_tasks()
        self._episode_id: str = ""
        self._current_task: SupportTask | None = None
        self._done: bool = True
        self._phase: str = "done"
        self._step_count: int = 0
        self._cumulative_reward: float = 0.01
        self._partial: PartialTriage = PartialTriage()
        self._penalties: int = 0

    def reset(self, task_id: str | None = None) -> Observation:
        self._episode_id = str(uuid.uuid4())
        self._done = False
        self._phase = "classify"
        self._step_count = 0
        self._cumulative_reward = 0.01
        self._partial = PartialTriage()
        self._penalties = 0

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

        notes: list[str] = []
        reward_components: dict[str, float] = {}
        step_reward = 0.0

        try:
            parsed = action if isinstance(action, TriageAction) else TriageAction.model_validate(action)
        except ValidationError:
            parsed = None
            self._apply_penalty(notes, "Invalid action schema.")

        if parsed is not None and not self._validate_identity(parsed):
            parsed = None
            self._apply_penalty(notes, "Action identity does not match current task/email.")

        self._step_count += 1

        if parsed is None:
            reward_components["invalid_action"] = 0.01
        elif parsed.action_type != self._phase:
            self._apply_penalty(notes, "Invalid action for current phase.")
        elif self._phase == "classify":
            phase_score, details = grade_classification(self._current_task, parsed)
            reward_components.update(details)
            step_reward += 0.35 * phase_score
            self._partial.category = parsed.category
            self._phase = "prioritize"
        elif self._phase == "prioritize":
            phase_score, details = grade_priority(self._current_task, parsed)
            reward_components.update(details)
            step_reward += 0.35 * phase_score
            self._partial.priority = parsed.priority
            self._partial.response_sla_hours = parsed.response_sla_hours
            self._phase = "draft"
        elif self._phase == "draft":
            phase_score, details = grade_response(self._current_task, parsed)
            reward_components.update(details)
            step_reward += 0.2 * phase_score
            self._partial.response_draft = parsed.response_draft
            self._phase = "finish"
        elif self._phase == "finish":
            completed = self._is_task_completed()
            reward_components["completion"] = 0.99 if completed else 0.01
            step_reward += 0.09 if completed else 0.0
            if not completed:
                self._apply_penalty(notes, "Finished without completing all required fields.")
            self._phase = "done"
            self._done = True

        if self._step_count >= self._current_task.max_steps and not self._done:
            self._apply_penalty(notes, "Reached max steps before finish.")
            self._done = True
            self._phase = "done"

        self._cumulative_reward = max(0.01, min(0.99, self._cumulative_reward + max(0.0, step_reward)))

        info = StepInfo(
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            phase=self._phase,
            reward_components={k: round(v, 4) for k, v in reward_components.items()},
            penalties=self._penalties,
            notes=notes,
        ).model_dump()

        return StepResult(
            observation=self._build_observation(),
            reward=max(0.01, min(0.99, round(self._cumulative_reward, 4))),
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
            cumulative_reward=max(0.01, min(0.99, round(self._cumulative_reward, 4))),
            partial=self._partial,
            penalties=self._penalties,
            available_task_ids=[task.task_id for task in self._tasks],
        )

    def list_tasks(self) -> list[SupportTask]:
        return self._tasks

    def _validate_identity(self, action: TriageAction) -> bool:
        if self._current_task is None:
            raise RuntimeError("No active task.")
        if action.task_id != self._current_task.task_id:
            return False
        if action.email_id != self._current_task.email.email_id:
            return False
        return True

    def _apply_penalty(self, notes: list[str], reason: str) -> None:
        self._penalties += 1
        notes.append(reason)
        self._cumulative_reward = max(0.01, self._cumulative_reward - 0.03)

    def _is_task_completed(self) -> bool:
        return all(
            (
                self._partial.category is not None,
                self._partial.priority is not None,
                self._partial.response_sla_hours is not None,
                bool(self._partial.response_draft and self._partial.response_draft.strip()),
            )
        )

    def _remaining_steps(self) -> int:
        if self._current_task is None:
            return 0
        return max(0, self._current_task.max_steps - self._step_count)

    def _available_actions(self) -> list[str]:
        if self._done:
            return []
        return [self._phase]

    def _hints(self) -> list[str]:
        if self._phase == "classify":
            return ["Set only action_type=classify and category."]
        if self._phase == "prioritize":
            return ["Set action_type=prioritize with priority and response_sla_hours."]
        if self._phase == "draft":
            return ["Set action_type=draft with response_draft message."]
        if self._phase == "finish":
            return ["Set action_type=finish to end the episode."]
        return ["Episode complete. Call reset()."]

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
        )

    @staticmethod
    def _coerce_action_literals(values: Iterable[str]) -> list[str]:
        return [value for value in values]
