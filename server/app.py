"""
FastAPI application wrapping the SupportTriageEnv.

Endpoints:
  POST /reset              → ResetResult
  POST /step               → StepResult
  GET  /state              → EnvState
  GET  /health             → {"status": "ok"}
  GET  /tasks              → list of task names

The /reset endpoint accepts an empty body {} to satisfy the validation script.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .env import SupportTriageEnv, TASKS
from .models import (
    EnvState,
    ResetResult,
    StepResult,
    TriageAction,
    TriageObservation,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Support Triage OpenEnv",
    description=(
        "Multi-step enterprise customer support triage environment. "
        "An AI agent reads tickets, queries a simulated backend, "
        "and routes or escalates each case. Implements the full OpenEnv spec."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server process (stateful, single-session)
_env = SupportTriageEnv()

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


class StepRequest(BaseModel):
    action: TriageAction


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": "support-triage-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks() -> dict:
    return {
        "tasks": list(TASKS.keys()),
        "descriptions": {k: v["ticket_text"][:80] + "..." for k, v in TASKS.items()},
    }


@app.post("/reset", response_model=ResetResult)
def reset(req: ResetRequest = None) -> ResetResult:
    """
    Reset the environment and return the initial observation.
    Accepts an empty body {} for compatibility with the validation script.
    """
    task_name = "easy"
    if req is not None and req.task:
        task_name = req.task

    try:
        obs: TriageObservation = _env.reset(task_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return ResetResult(observation=obs)


@app.post("/step", response_model=StepResult)
def step(req: StepRequest) -> StepResult:
    """Advance the environment by one step with the given action."""
    try:
        result: StepResult = _env.step(req.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result


@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    """Return the full internal environment state."""
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
