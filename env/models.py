from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Difficulty = Literal["easy", "medium", "hard"]
CustomerTier = Literal["standard", "premium", "enterprise"]
Category = Literal["billing", "technical", "account", "shipping", "general"]
Priority = Literal["low", "medium", "high", "urgent"]
ActionType = Literal["classify", "prioritize", "draft", "finish"]
Phase = Literal["classify", "prioritize", "draft", "finish", "done"]


class EmailItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email_id: str
    sender: str
    subject: str
    body: str
    customer_tier: CustomerTier
    created_at: str


class TriageTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    response_sla_hours: int = Field(ge=1, le=168)
    required_response_keywords: list[str] = Field(default_factory=list)


class SupportTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    objective: str
    instructions: str
    email: EmailItem
    target: TriageTarget
    max_steps: int = Field(ge=4, le=12, default=6)


class TriageAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    email_id: str
    action_type: ActionType
    category: Category | None = None
    priority: Priority | None = None
    response_sla_hours: int | None = Field(default=None, ge=1, le=168)
    response_draft: str | None = Field(default=None, max_length=4000)
    rationale: str = Field(min_length=1, max_length=2000)


class PartialTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category | None = None
    priority: Priority | None = None
    response_sla_hours: int | None = Field(default=None, ge=1, le=168)
    response_draft: str | None = None


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    objective: str
    instructions: str
    phase: Phase
    step_count: int = Field(ge=0)
    remaining_steps: int = Field(ge=0)
    email: EmailItem
    partial: PartialTriage
    available_actions: list[ActionType]
    hints: list[str]


class EnvState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    current_task_id: str | None
    done: bool
    phase: Phase
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=0)
    cumulative_reward: float = Field(ge=0.0, le=1.0)
    partial: PartialTriage
    penalties: int = Field(ge=0)
    available_task_ids: list[str]


class StepInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Difficulty
    phase: Phase
    reward_components: dict[str, float]
    penalties: int = Field(ge=0)
    notes: list[str] = Field(default_factory=list)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict[str, Any]
