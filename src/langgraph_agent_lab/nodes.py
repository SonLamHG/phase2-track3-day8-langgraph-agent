"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

from .state import AgentState, ApprovalDecision, Route, make_event

RISKY_KEYWORDS = {
    "refund", "refunds", "refunded", "refunding",
    "delete", "deletes", "deleted", "deleting", "deletion",
    "send", "sends", "sending", "sent",
    "cancel", "cancels", "cancelled", "canceled", "cancellation",
    "remove", "removes", "removed", "removing", "removal",
    "revoke", "revokes", "revoked", "revoking",
    "wipe", "wipes", "wiped", "wiping",
    "purge", "purges", "purged",
    "terminate", "terminates", "terminated",
    "disable", "disables", "disabled",
}
TOOL_KEYWORDS = {
    "status", "statuses",
    "order", "orders",
    "lookup", "lookups",
    "check", "checks", "checking",
    "track", "tracks", "tracking", "tracked",
    "find", "finds", "found",
    "search", "searches", "searching",
    "query", "queries",
    "fetch", "fetches", "retrieve",
}
ERROR_KEYWORDS = {
    "timeout", "timeouts", "timed",
    "fail", "fails", "failed", "failing", "failure", "failures",
    "error", "errors", "errored",
    "crash", "crashes", "crashed", "crashing",
    "unavailable",
    "broken",
    "unreachable",
}
VAGUE_PRONOUNS = {"it", "this", "that", "them", "those", "these"}
_PUNCT = "?!.,;:'\"()[]{}"


def _tokenize(query: str) -> set[str]:
    """Lowercase + strip surrounding punctuation; return whole-word set."""
    return {w.strip(_PUNCT) for w in query.lower().split() if w.strip(_PUNCT)}


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    TODO(student): add normalization, PII checks, and metadata extraction.
    """
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route by whole-word keyword matching.

    Priority: risky > tool > missing_info > error > simple.
    Word-boundary matching avoids false positives like "sender" matching "send"
    or "checkpoint" matching "check".
    """
    words = _tokenize(state.get("query", ""))
    route = Route.SIMPLE
    risk_level = "low"
    if words & RISKY_KEYWORDS:
        route = Route.RISKY
        risk_level = "high"
    elif words & TOOL_KEYWORDS:
        route = Route.TOOL
    elif len(words) < 6 and (words & VAGUE_PRONOUNS):
        route = Route.MISSING_INFO
    elif words & ERROR_KEYWORDS:
        route = Route.ERROR
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"route={route.value}")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    TODO(student): generate a specific clarification question from state.
    """
    question = "Can you provide the order id or the missing context?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    TODO(student): implement idempotent tool execution and structured tool results.
    """
    attempt = int(state.get("attempt", 0))
    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={state.get('scenario_id', 'unknown')}"
    else:
        result = f"mock-tool-result for scenario={state.get('scenario_id', 'unknown')}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval.

    TODO(student): create a proposed action with evidence and risk justification.
    """
    return {
        "proposed_action": "prepare refund or external action; approval required",
        "events": [make_event("risky_action", "pending_approval", "approval required")],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.

    TODO(student): implement reject/edit decisions and timeout escalation.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        decision = ApprovalDecision(approved=True, comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision.

    TODO(student): implement bounded retry, exponential backoff metadata, and fallback route.
    """
    attempt = int(state.get("attempt", 0)) + 1
    errors = [f"transient failure attempt={attempt}"]
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [make_event("retry", "completed", "retry attempt recorded", attempt=attempt)],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response.

    TODO(student): ground the answer in tool_results and approval where relevant.
    """
    if state.get("tool_results"):
        answer = f"I found: {state['tool_results'][-1]}"
    else:
        answer = "This is a safe mock answer. Replace with your agent response."
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.

    TODO(student): replace heuristic with LLM-as-judge or structured validation.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    if "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "tool result indicates failure, retry needed")],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    TODO(student): persist to dead-letter queue, alert on-call, or create support ticket.
    """
    return {
        "final_answer": "Request could not be completed after maximum retry attempts. Logged for manual review.",
        "events": [make_event("dead_letter", "completed", f"max retries exceeded, attempt={state.get('attempt', 0)}")],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
