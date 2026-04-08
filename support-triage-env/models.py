"""
Pydantic models for the Support Triage OpenEnv environment.
All models use strict types to prevent LLM hallucinations.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations — strict allowed values
# ---------------------------------------------------------------------------

class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SECURITY = "security"
    REFUNDS = "refunds"
    GENERAL = "general"


class ActionType(str, Enum):
    LOOKUP_LOGS = "lookup_logs"
    CHECK_BILLING = "check_billing"
    CHECK_ACCOUNT = "check_account"
    ROUTE = "route"
    ESCALATE = "escalate"
    RESOLVE = "resolve"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Action — what the agent sends each step
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    The action an agent takes at each step of the episode.

    action_type: One of the allowed ActionType values.
      - lookup_logs    : Query the system log for this ticket.
      - check_billing  : Query the billing record for this account.
      - check_account  : Query account / authentication history.
      - route          : Route ticket to a department (requires department).
      - escalate       : Escalate to security / management (requires department).
      - resolve        : Mark ticket resolved with a short resolution note.

    department: Required when action_type is 'route' or 'escalate'.
    note: Optional free-text note (max 200 chars, logged but not scored directly).
    """

    action_type: ActionType = Field(..., description="The type of action to perform.")
    department: Optional[Department] = Field(
        None,
        description="Target department — required for 'route' and 'escalate' actions.",
    )
    note: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional free-text note or reasoning (not scored directly).",
    )

    @field_validator("department", mode="before")
    @classmethod
    def _coerce_department(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v.lower()
        return v


# ---------------------------------------------------------------------------
# Observation — what the environment returns after each step
# ---------------------------------------------------------------------------

class TriageObservation(BaseModel):
    """
    What the agent sees after each step.

    ticket_id      : Unique ticket identifier.
    ticket_text    : The raw customer support message.
    step           : Current step number (1-indexed).
    max_steps      : Maximum steps allowed for this episode.
    query_result   : Result of the last lookup/check action (empty string if none).
    last_reward    : Reward received for the last action.
    cumulative_reward : Total reward accumulated so far.
    done           : Whether the episode has ended.
    message        : Human-readable status message from the environment.
    available_actions : List of action_type strings the agent may use next.
    """

    ticket_id: str
    ticket_text: str
    step: int
    max_steps: int
    query_result: str = ""
    last_reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    message: str = ""
    available_actions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step result — wrapper returned by the environment's step() function
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: TriageObservation
    reward: float
    done: bool
    info: Dict[str, object] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reset result
# ---------------------------------------------------------------------------

class ResetResult(BaseModel):
    observation: TriageObservation


# ---------------------------------------------------------------------------
# State — full internal state snapshot
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    task_name: str
    ticket_id: str
    ticket_text: str
    step: int
    max_steps: int
    cumulative_reward: float
    done: bool
    actions_taken: List[str] = Field(default_factory=list)
    query_results: Dict[str, str] = Field(default_factory=dict)
