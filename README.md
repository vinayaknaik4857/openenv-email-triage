# OpenEnv: Customer Support Email Triage

Real-world OpenEnv environment where an agent triages customer support emails by:
1. classifying issue category,
2. setting priority + response SLA,
3. drafting a customer-facing response,
4. finalizing the triage workflow.

## Why this is real-world

Support operations teams perform this process daily for billing issues, outages, and security/account incidents. The environment includes deterministic graders and shaped reward signals for partial progress.

## Files

- `env/models.py`: typed Pydantic models (`Observation`, `TriageAction`, `EnvState`, `StepResult`)
- `env/tasks.py`: exactly 3 tasks (`easy`, `medium`, `hard`)
- `env/grader.py`: deterministic grading with response quality checks
- `env/environment.py`: OpenEnv API (`reset`, `step`, `state`)
- `openenv.yaml`: environment metadata/spec manifest
- `inference.py`: baseline runner using OpenAI client + strict logs
- `Dockerfile`: containerized run (defaults to `python inference.py`)
- `server.py`: optional HTTP wrapper (`/reset`, `/step`, `/state`) for validator probes

## Action Space

`TriageAction`:
- `task_id: str`
- `email_id: str`
- `action_type: classify | prioritize | draft | finish`
- `category: billing | technical | account | shipping | general` (used in `classify`)
- `priority: low | medium | high | urgent` (used in `prioritize`)
- `response_sla_hours: int` (used in `prioritize`)
- `response_draft: str` (used in `draft`)
- `rationale: str`

## Observation Space

`Observation`:
- `task_id`, `difficulty`, `objective`, `instructions`
- `phase`, `step_count`, `remaining_steps`
- `email` (sender/subject/body/tier/timestamp)
- `partial` (what the agent has already filled)
- `available_actions`, `hints`

## Tasks

1. `triage-easy-001` (`easy`): duplicate billing charge + refund request  
2. `triage-medium-001` (`medium`): multi-user SSO outage incident  
3. `triage-hard-001` (`hard`): enterprise access revocation + audit readiness

## Reward Design

- Total score is clamped in `[0.0, 1.0]`.
- Partial progress is rewarded across phases:
  - classification
  - priority/SLA
  - response quality
  - completion
- Response quality checks include:
  - required keyword coverage
  - polite tone
  - helpfulness
- Wrong answers and invalid actions get small penalties.

## Inference Contract

`inference.py`:
- Uses OpenAI client
- Reads env vars: `API_BASE_URL`, `MODEL_NAME`, `OPENAI_API_KEY`, `HF_TOKEN`
- Strictly parses JSON model output
- Enforces output keys: `category`, `priority`, `response`
- Retries on invalid parse
- Prints logs exactly:
  - `[START]`
  - `[STEP] Task X Score: Y`
  - `[END]`
- Prints final `Average Score`

## Setup (Local)

```bash
pip install -r requirements.txt
```

Set env vars:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export OPENAI_API_KEY="your_key"
export HF_TOKEN="your_hf_token"
```

Run baseline:

```bash
python inference.py
```

No quota? Use offline deterministic mode:

```bash
MOCK_MODE=true python inference.py
```

## Docker

Build:

```bash
docker build -t openenv-email-triage .
```

Run default command (`inference.py`):

```bash
docker run --rm openenv-email-triage
```

Run with real API:

```bash
docker run --rm \
  -e MOCK_MODE=false \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4.1-mini" \
  -e OPENAI_API_KEY="your_key" \
  -e HF_TOKEN="your_hf_token" \
  openenv-email-triage
```

## Optional HF Space HTTP Validation

If your validator requires HTTP ping/reset endpoints, run:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

Endpoints:
- `GET /`
- `POST /reset`
- `POST /step`
- `GET /state`

## Step-by-step Submission Checklist

1. Install dependencies.
2. Run `python -m compileall .`
3. Run `MOCK_MODE=true python inference.py` and verify logs format.
4. Run `python inference.py` with real API vars (if quota available).
5. Run `docker build -t openenv-email-triage .`
6. Run `docker run --rm openenv-email-triage`
7. (If needed by validator) start `server.py` with `uvicorn` and verify `/` and `/reset`.
