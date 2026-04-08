from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException

from env.environment import CustomerSupportEmailTriageEnv
from env.models import TriageAction

app = FastAPI(title="OpenEnv Customer Support Email Triage")
env = CustomerSupportEmailTriageEnv()


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "openenv-email-triage"}


@app.post("/reset")
def reset(task_id: str | None = None) -> dict:
    try:
        obs = env.reset(task_id=task_id)
        return {"observation": obs.model_dump()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(action: dict) -> dict:
    try:
        parsed = TriageAction.model_validate(action)
        result = env.step(parsed)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict:
    return {"state": env.state().model_dump()}


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
