"""Streamlit HITL approval UI for the support-ticket agent.

Run:
    pip install -e ".[ui]"
    streamlit run src/langgraph_agent_lab/ui.py

The page picks a scenario, runs the graph through the SQLite checkpointer, and —
when the risky path triggers ``interrupt()`` inside ``approval_node`` — exposes
approve/reject buttons to resume the thread with a structured decision.
"""

from __future__ import annotations

import os

# Force real interrupt() *before* importing the graph wiring so approval_node
# takes the LANGGRAPH_INTERRUPT branch.
os.environ.setdefault("LANGGRAPH_INTERRUPT", "true")

import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402
from langgraph.types import Command  # noqa: E402

from langgraph_agent_lab.graph import build_graph  # noqa: E402
from langgraph_agent_lab.persistence import build_checkpointer  # noqa: E402
from langgraph_agent_lab.scenarios import load_scenarios  # noqa: E402
from langgraph_agent_lab.state import Scenario, initial_state  # noqa: E402

UI_DB_PATH = "outputs/ui_checkpoints.db"
SCENARIOS_PATH = "data/sample/scenarios.jsonl"


@st.cache_resource
def get_graph_and_db() -> tuple[Any, str]:
    Path(UI_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    cp = build_checkpointer("sqlite", UI_DB_PATH)
    return build_graph(checkpointer=cp), UI_DB_PATH


@st.cache_data
def get_scenarios() -> list[Scenario]:
    return load_scenarios(SCENARIOS_PATH)


def _pending_interrupt(graph: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    snap = graph.get_state(config)
    if not snap:
        return None
    for task in snap.tasks:
        for itr in task.interrupts:
            return dict(itr.value) if isinstance(itr.value, dict) else {"value": itr.value}
    return None


def _format_scenario(s: Scenario) -> str:
    flag = " ⚠️ HITL" if s.requires_approval else ""
    return f"{s.id} · {s.expected_route.value}{flag}"


def _render_sidebar(scenarios: list[Scenario], db_path: str) -> Scenario:
    st.sidebar.header("Scenario")
    labels = [_format_scenario(s) for s in scenarios]
    idx = st.sidebar.radio(
        "Pick one:",
        options=list(range(len(scenarios))),
        format_func=lambda i: labels[i],
    )
    selected = scenarios[idx]

    st.sidebar.divider()
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset session", use_container_width=True,
                          help="Clear thread state in the browser only"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    if st.sidebar.button("🧹 Wipe SQLite DB", use_container_width=True,
                          help=f"Delete {db_path} and rebuild the graph"):
        Path(db_path).unlink(missing_ok=True)
        st.cache_resource.clear()
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(f"Checkpoints: `{db_path}`")
    st.sidebar.caption(f"LANGGRAPH_INTERRUPT=`{os.getenv('LANGGRAPH_INTERRUPT')}`")
    return selected


def _render_query_panel(selected: Scenario, graph: Any) -> None:
    st.subheader("Query")
    st.code(selected.query, language="text")
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected route", selected.expected_route.value)
    m2.metric("Approval required", "yes" if selected.requires_approval else "no")
    m3.metric("Max attempts", selected.max_attempts)

    if st.button("▶ Run scenario", type="primary", use_container_width=True):
        state = initial_state(selected)
        # Unique thread per click so each Run starts a clean graph instance.
        thread_id = f"thread-{selected.id}-ui-{uuid.uuid4().hex[:8]}"
        state["thread_id"] = thread_id
        config = {"configurable": {"thread_id": thread_id}}
        st.session_state.config = config
        st.session_state.scenario_id = selected.id
        with st.spinner("Running graph..."):
            try:
                result = graph.invoke(state, config=config)
                st.session_state.last_result = result
            except Exception as exc:
                st.error(f"Graph error: {exc}")
                st.session_state.last_result = None
        st.rerun()


def _render_run_state(graph: Any) -> None:
    st.subheader("Run state")
    config = st.session_state.get("config")
    if not config:
        st.info("Run a scenario from the left panel to begin.")
        return

    interrupt_payload = _pending_interrupt(graph, config)
    if interrupt_payload:
        st.warning("⏸  Graph paused — awaiting approval")
        st.markdown("**Proposed action**")
        st.code(str(interrupt_payload.get("proposed_action", "")), language="text")
        st.markdown(f"**Risk level:** `{interrupt_payload.get('risk_level', 'unknown')}`")
        comment = st.text_input("Reviewer comment (optional)", key="comment")
        approve_col, reject_col = st.columns(2)
        decision: dict[str, Any] | None = None
        if approve_col.button("✅ Approve", type="primary", use_container_width=True):
            decision = {"approved": True, "reviewer": "ui", "comment": comment or "approved via UI"}
        if reject_col.button("❌ Reject", use_container_width=True):
            decision = {"approved": False, "reviewer": "ui", "comment": comment or "rejected via UI"}
        if decision is not None:
            with st.spinner("Resuming graph..."):
                try:
                    result = graph.invoke(Command(resume=decision), config=config)
                    st.session_state.last_result = result
                except Exception as exc:
                    st.error(f"Resume error: {exc}")
            st.rerun()
        return

    result = st.session_state.get("last_result")
    if not result:
        st.info("Graph state is empty.")
        return
    st.success("✅ Graph completed")
    st.write(f"**Route:** `{result.get('route', '?')}`")
    st.markdown("**Final output**")
    st.code(result.get("final_answer") or result.get("pending_question") or "(none)", language="text")
    if result.get("approval"):
        st.markdown("**Approval recorded**")
        st.json(result["approval"])
    if result.get("errors"):
        st.markdown("**Errors during run**")
        for err in result["errors"]:
            st.text(f"• {err}")


def _render_timeline_and_history(graph: Any) -> None:
    config = st.session_state.get("config")
    if not config:
        return
    st.divider()
    timeline_col, history_col = st.columns([3, 2])

    with timeline_col:
        st.subheader("Event timeline")
        events = (st.session_state.get("last_result") or {}).get("events", [])
        if not events:
            st.caption("No events recorded yet.")
        for i, ev in enumerate(events, 1):
            st.markdown(
                f"`{i:02d}` **{ev.get('node', '?')}** · _{ev.get('event_type', '?')}_ — "
                f"{ev.get('message', '')}"
            )

    with history_col:
        st.subheader("Checkpoint history")
        try:
            history = list(graph.get_state_history(config))
        except Exception as exc:
            st.caption(f"(history unavailable: {exc})")
            return
        st.caption(f"{len(history)} checkpoints persisted in SQLite")
        for i, snap in enumerate(history[:10]):
            step = snap.metadata.get("step", "?") if snap.metadata else "?"
            next_node = ", ".join(snap.next) if snap.next else "—"
            st.text(f"#{i:02d}  step={step}  next={next_node}")
        if len(history) > 10:
            st.caption(f"… and {len(history) - 10} earlier checkpoints")


def main() -> None:
    st.set_page_config(
        page_title="LangGraph Lab — HITL",
        page_icon="🤖",
        layout="wide",
    )
    st.title("Support-Ticket Agent — HITL Approval Demo")
    st.caption(
        "Pick a scenario from the sidebar, run the graph, and approve or reject "
        "when the risky path interrupts. SQLite persistence means refreshing the "
        "browser does not lose state — checkpoints survive."
    )

    graph, db_path = get_graph_and_db()
    scenarios = get_scenarios()
    selected = _render_sidebar(scenarios, db_path)

    left, right = st.columns([1, 1])
    with left:
        _render_query_panel(selected, graph)
    with right:
        _render_run_state(graph)

    _render_timeline_and_history(graph)


main()
