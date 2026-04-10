---
title: OpenEnv Email Triage
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# OpenEnv Email Triage

`openenv-email-triage` is a real-world OpenEnv benchmark for customer support operations. The agent receives an inbound support email and must complete the same workflow a support lead or triage analyst performs in production: route the email to the right queue, assign business priority, set an SLA, draft a customer-facing reply, and finalize the ticket without wasting steps.

This is useful for training and evaluating agents on operational judgment instead of synthetic puzzle solving. The environment rewards correct routing, urgency calibration, concise response drafting, and efficient completion while penalizing invalid or looping behavior.

## What makes this real-world

Support and operations teams do this work every day:

- Billing teams handle duplicate charges and refund requests.
- Reliability teams escalate customer-facing outages with tight SLAs.
- Security and platform teams triage access-governance incidents before audits.

The benchmark captures that workflow with deterministic tasks, typed actions and observations, and reproducible graders.

## OpenEnv interface

The environment implements the standard OpenEnv methods:

- `reset(task_id: str | None = None) -> Observation`
- `step(action: TriageAction | dict) -> StepResult`
- `state() -> EnvState`

Typed models live in [env/models.py](env/models.py), the environment logic lives in [env/environment.py](env/environment.py), and metadata lives in [openenv.yaml](openenv.yaml).

## Action space

`TriageAction`

- `task_id: str`
- `email_id: str`
- `action_type: "classify" | "prioritize" | "draft" | "finish"`
- `category: "billing" | "technical" | "account" | "shipping" | "general" | null`
- `priority: "low" | "medium" | "high" | "urgent" | null`
- `response_sla_hours: int | null`
- `response_draft: str | null`
- `rationale: str`

The environment is phase-based. Each step exposes exactly one valid next action type, which keeps trajectories comparable across models and makes reward shaping meaningful.

## Observation space

`Observation`

- `task_id`, `difficulty`, `objective`, `instructions`
- `phase`, `step_count`, `remaining_steps`
- `email` with sender, subject, body, customer tier, and timestamp
- `partial` with the work completed so far
- `available_actions`, `hints`
- `last_step_reward`, `cumulative_reward`, `last_action_error`

## Tasks

Three deterministic tasks are included, with increasing difficulty:

1. `triage-easy-001`
   Duplicate billing charge for a standard customer requesting a refund.
2. `triage-medium-001`
   Premium customer SSO outage affecting multiple users and requiring an urgent ETA-oriented response.
3. `triage-hard-001`
   Enterprise access-governance escalation involving former contractors and an impending compliance audit.

Task definitions are in [env/tasks.py](env/tasks.py).

## Reward design and graders

The reward function is dense across the full trajectory instead of only at the terminal state:

- Classification contributes up to `0.25`
- Priority and SLA contribute up to `0.25`
- Draft quality contributes up to `0.35`
- Completion contributes up to `0.15`

Partial progress signals:

- Correct category gets immediate credit.
- Priority and SLA receive graded partial credit.
- Draft scoring rewards keyword coverage, tone, actionability, and reasonable length.
- Invalid actions and out-of-order steps apply penalties and reduce the final episode score.

Task graders are deterministic and produce normalized scores in `[0.0, 1.0]`. Grading logic lives in [env/grader.py](env/grader.py).

## Baseline inference

The required inference script is [inference.py](inference.py). It:

- Uses the OpenAI Python client for all LLM calls
- Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, and `LOCAL_IMAGE_NAME`
- Falls back to a deterministic policy when `HF_TOKEN` is missing, which makes local verification reproducible
- Emits the mandatory structured stdout lines:
  - `[START] task=<task_name> env=<benchmark> model=<model_name>`
  - `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
  - `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

### Reproducible baseline scores

The fallback baseline is deterministic and should consistently achieve:

- `triage-easy-001`: `1.000`
- `triage-medium-001`: `0.988`
- `triage-hard-001`: `1.000`
- Average: `0.996`

Those scores are intentionally reproducible for validation. Frontier models can still be compared by disabling fallback mode and supplying `HF_TOKEN`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_or_router_token"
export LOCAL_IMAGE_NAME="openenv-email-triage"
```

Run the baseline:

```bash
python inference.py
```

Start the HTTP server:

```bash
python server.py
```

Available endpoints:

- `GET /`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`

## Docker

Build:

```bash
docker build -t openenv-email-triage .
```

Run the Space-compatible server:

```bash
docker run --rm -p 7860:7860 openenv-email-triage
```

Then verify:

```bash
curl http://localhost:7860/
curl -X POST http://localhost:7860/reset
```

## Hugging Face Spaces

This repo is container-ready for a Docker Space:

- `sdk: docker`
- `app_port: 7860`
- health endpoint at `/`

Add the `openenv` tag in the Hugging Face Space settings and set `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` as Space secrets or variables.

## Suggested validation checklist

Before submitting:

1. Run `python -m compileall .`
2. Run `python inference.py`
3. Run `docker build -t openenv-email-triage .`
4. Run `docker run --rm -p 7860:7860 openenv-email-triage`
5. Run `openenv validate`

## Repository structure

- [env/environment.py](env/environment.py): main environment implementation
- [env/models.py](env/models.py): typed OpenEnv models
- [env/tasks.py](env/tasks.py): deterministic tasks
- [env/grader.py](env/grader.py): phase and episode graders
- [server.py](server.py): local entrypoint
- [server/app.py](server/app.py): Space app
- [inference.py](inference.py): baseline runner
