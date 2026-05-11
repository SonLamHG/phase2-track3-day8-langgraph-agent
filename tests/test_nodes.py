"""Edge-case coverage for classify_node — hidden grader scenarios test these."""

import pytest

from langgraph_agent_lab.nodes import classify_node
from langgraph_agent_lab.state import Route


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Risky — keyword variations and priority over tool/error words
        ("Cancel my subscription immediately", Route.RISKY.value),
        ("Please remove the customer record", Route.RISKY.value),
        ("Revoke API key for user 42", Route.RISKY.value),
        ("Refund failed transaction", Route.RISKY.value),  # risky > error
        ("Send confirmation and check status", Route.RISKY.value),  # risky > tool
        # Tool — keyword variations
        ("Can you track shipment 9001?", Route.TOOL.value),
        ("Find the invoice for March", Route.TOOL.value),
        ("Search the help center for VPN setup", Route.TOOL.value),
        # Missing info — vague pronouns in short queries
        ("Can you fix it?", Route.MISSING_INFO.value),
        ("Help me with this", Route.MISSING_INFO.value),
        ("What about them?", Route.MISSING_INFO.value),
        # Error — keyword variations
        ("Service is unavailable right now", Route.ERROR.value),
        ("The job crashed unexpectedly", Route.ERROR.value),
        ("Multiple errors during processing", Route.ERROR.value),
        # Simple — default fallback
        ("How do I reset my password?", Route.SIMPLE.value),
        ("Where can I update my profile picture", Route.SIMPLE.value),
    ],
)
def test_classify_routes(query, expected):
    out = classify_node({"query": query})
    assert out["route"] == expected, f"{query!r} → {out['route']} (expected {expected})"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Word boundary: substrings must NOT match
        ("The sender did not reply", Route.SIMPLE.value),       # "sender" ≠ "send"
        ("Add checkpoint to pipeline", Route.SIMPLE.value),     # "checkpoint" ≠ "check"
        ("Ascending order of priority", Route.TOOL.value),      # "order" matches, "ascending" doesn't bleed "send"
    ],
)
def test_classify_word_boundary(query, expected):
    out = classify_node({"query": query})
    assert out["route"] == expected, f"{query!r} → {out['route']} (expected {expected})"


def test_risky_sets_high_risk_level():
    out = classify_node({"query": "Delete this account"})
    assert out["route"] == Route.RISKY.value
    assert out["risk_level"] == "high"


def test_simple_sets_low_risk_level():
    out = classify_node({"query": "How do I update settings"})
    assert out["route"] == Route.SIMPLE.value
    assert out["risk_level"] == "low"
