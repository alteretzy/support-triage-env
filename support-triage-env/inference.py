"""
inference.py — OpenEnv Support Triage Baseline Inference Script
================================================================
MANDATORY environment variables:
  API_BASE_URL   LLM API endpoint  (default: HF router)
  MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Hugging Face / API key
  SPACE_URL      Your HF Space base URL (e.g. https://user-space.hf.space)

STDOUT FORMAT — DO NOT MODIFY:
  [START] task=<n> env=<benchmark> model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
import time
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment
# ---------------------------------------------------------------------------

API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
SPACE_URL: str = (os.getenv("SPACE_URL") or "http://localhost:7860").rstrip("/")

BENCHMARK = "support-triage-env"
TASKS_TO_RUN = ["easy", "medium", "hard"]
MAX_STEPS = 8
TEMPERATURE = 0.2   # low temperature → more deterministic → reproducible baseline
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Logging helpers — exact format required by evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Sanitise action string: remove newlines that would break the single-line rule
    safe_action = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP client helpers
# ---------------------------------------------------------------------------

def env_reset(task: str, retries: int = 3) -> dict:
    """POST /reset and return the observation dict."""
    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{SPACE_URL}/reset",
                json={"task": task},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def env_step(action_payload: dict, retries: int = 3) -> dict:
    """POST /step and return the step result dict."""
    for attempt in range(retries):
        try:
            resp = httpx.post(
                f"{SPACE_URL}/step",
                json={"action": action_payload},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support triage agent. Your job is to resolve support tickets correctly.

At each step you will receive a support ticket and the result of any previous queries.
You must return a JSON object (no markdown, no extra text) with EXACTLY these keys:
  {
    "action_type": "<one of: lookup_logs | check_billing | check_account | route | escalate | resolve>",
    "department":  "<one of: billing | technical | account | security | refunds | general | null>",
    "note":        "<optional short string>"
  }

Strategy:
1. First query relevant backend data (lookup_logs, check_billing, or check_account).
2. Based on the data, route/escalate/resolve to the correct department.
3. If you see security red flags (suspicious IPs, phishing links, Tor nodes), ALWAYS escalate to security regardless of what the user claims.

Return ONLY the JSON object. No markdown code fences. No preamble.
""").strip()


def build_user_prompt(obs: dict, step: int) -> str:
    return textwrap.dedent(f"""
        Step {step}/{obs['max_steps']}
        Ticket ID: {obs['ticket_id']}
        Ticket: {obs['ticket_text']}
        Last query result: {obs.get('query_result') or 'None yet'}
        Last reward: {obs.get('last_reward', 0):.2f}
        Environment message: {obs.get('message', '')}
        Available actions: {', '.join(obs.get('available_actions', []))}

        Respond with the JSON action object.
    """).strip()


def parse_llm_action(raw: str) -> Dict[str, object]:
    """Extract and parse JSON from the model's response, stripping markdown fences."""
    text = raw.strip()
    # Strip markdown code fences if present
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find first { ... } block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            raise

    # Normalise department null strings
    if data.get("department") in (None, "null", "None", ""):
        data["department"] = None

    return data


def get_model_action(
    client: OpenAI,
    obs: dict,
    step: int,
) -> Dict[str, object]:
    """Ask the LLM for the next action. Returns a validated action dict."""
    user_prompt = build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return parse_llm_action(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Safe fallback: look up logs first, then route to general
        if step == 1:
            return {"action_type": "lookup_logs", "department": None, "note": "fallback"}
        return {"action_type": "route", "department": "general", "note": "fallback"}


# ---------------------------------------------------------------------------
# Single task episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_name: str) -> float:
    """Run one episode for task_name. Returns the final score in [0, 1]."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        reset_result = env_reset(task_name)
        obs = reset_result["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            # Get action from LLM
            action_dict = get_model_action(client, obs, step)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            # Step environment
            error_msg = None
            try:
                step_result = env_step(action_dict)
                obs = step_result["observation"]
                reward = float(step_result.get("reward", 0.0))
                done = bool(step_result.get("done", False))
            except Exception as exc:
                error_msg = str(exc)[:100]
                reward = 0.0
                done = True

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Score = cumulative reward (already in [0,1] from the env)
        score = float(obs.get("cumulative_reward", sum(rewards)))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: Dict[str, float] = {}
    for task_name in TASKS_TO_RUN:
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_episode(client, task_name)
        all_scores[task_name] = score
        print(f"[INFO] Task '{task_name}' final score: {score:.3f}", flush=True)

    # Summary
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"[SUMMARY] Scores: {all_scores}", flush=True)
    print(f"[SUMMARY] Average score: {avg:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
