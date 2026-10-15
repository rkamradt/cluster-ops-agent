"""Tests for graph routing logic: loop cap, validate/retry, diagnose trigger.

These tests use lightweight fakes — no Ollama, no Kubernetes required.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from app.graph import (
    AgentState,
    _UNHEALTHY_KEYWORDS,
    _route_from_agent,
    _route_from_tools,
    _route_from_validate,
)
from app.config import MAX_STEPS


# ── route_from_agent ──────────────────────────────────────────────────────────


def _agent_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [AIMessage(content="hello")],
        "step_count": 0,
        "retry_count": 0,
        "validation_status": "",
    }
    base.update(overrides)
    return base


class TestRouteFromAgent:
    def test_no_tool_calls_goes_to_end(self):
        state = _agent_state(messages=[AIMessage(content="all good")])
        assert _route_from_agent(state) == END

    def test_tool_calls_go_to_validate(self):
        msg = AIMessage(content="", tool_calls=[{"name": "list_pods", "args": {"namespace": "home"}, "id": "1"}])
        state = _agent_state(messages=[msg])
        assert _route_from_agent(state) == "validate"

    def test_step_cap_overrides_tool_calls(self):
        msg = AIMessage(content="", tool_calls=[{"name": "list_pods", "args": {}, "id": "1"}])
        state = _agent_state(messages=[msg], step_count=MAX_STEPS)
        assert _route_from_agent(state) == END

    def test_step_cap_exact_boundary(self):
        state = _agent_state(step_count=MAX_STEPS - 1)
        # No tool calls → still END, but cap not triggered
        assert _route_from_agent(state) == END

    def test_step_zero_with_tool_calls(self):
        msg = AIMessage(content="", tool_calls=[{"name": "list_pods", "args": {"namespace": "home"}, "id": "1"}])
        state = _agent_state(messages=[msg], step_count=0)
        assert _route_from_agent(state) == "validate"


# ── route_from_validate ───────────────────────────────────────────────────────


class TestRouteFromValidate:
    def test_passed_goes_to_tools(self):
        state = _agent_state(validation_status="passed")
        assert _route_from_validate(state) == "tools"

    def test_retry_goes_to_agent(self):
        state = _agent_state(validation_status="retry")
        assert _route_from_validate(state) == "agent"

    def test_failed_goes_to_end(self):
        state = _agent_state(validation_status="failed")
        assert _route_from_validate(state) == END

    def test_empty_status_defaults_to_tools(self):
        state = _agent_state(validation_status="")
        assert _route_from_validate(state) == "tools"


# ── route_from_tools ──────────────────────────────────────────────────────────


def _state_with_tool_msg(content: str) -> AgentState:
    return _agent_state(
        messages=[
            HumanMessage(content="what's wrong?"),
            ToolMessage(content=content, tool_call_id="t1"),
        ]
    )


class TestRouteFromTools:
    def test_healthy_pod_goes_to_agent(self):
        state = _state_with_tool_msg('[{"name":"pod-1","phase":"Running"}]')
        assert _route_from_tools(state) == "agent"

    @pytest.mark.parametrize("keyword", list(_UNHEALTHY_KEYWORDS))
    def test_unhealthy_keyword_triggers_diagnose(self, keyword: str):
        state = _state_with_tool_msg(f'[{{"name":"pod-1","state":"{keyword}"}}]')
        assert _route_from_tools(state) == "diagnose"

    def test_case_insensitive_detection(self):
        state = _state_with_tool_msg('reason: CrashLoopBackOff, restartCount: 5')
        assert _route_from_tools(state) == "diagnose"

    def test_no_tool_message_goes_to_agent(self):
        # If no ToolMessage in history, default to agent
        state = _agent_state(messages=[HumanMessage(content="hello")])
        assert _route_from_tools(state) == "agent"

    def test_only_last_tool_message_checked(self):
        # First tool msg is unhealthy, second (most recent) is healthy
        state = _agent_state(
            messages=[
                ToolMessage(content="crashloopbackoff detected", tool_call_id="t1"),
                HumanMessage(content="ok"),
                ToolMessage(content='[{"phase":"Running"}]', tool_call_id="t2"),
            ]
        )
        # Most recent ToolMessage is healthy → should go to agent
        assert _route_from_tools(state) == "agent"


# ── validate_node integration ─────────────────────────────────────────────────


@pytest.mark.asyncio
class TestValidateNode:
    async def test_valid_call_passes(self):
        from app.nodes import validate_node

        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "list_pods", "args": {"namespace": "home"}, "id": "1"}
            ],
        )
        state = _agent_state(messages=[msg], retry_count=0)
        result = await validate_node(state)
        assert result["validation_status"] == "passed"

    async def test_missing_namespace_triggers_retry(self):
        from app.nodes import validate_node

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "list_pods", "args": {}, "id": "1"}],
        )
        state = _agent_state(messages=[msg], retry_count=0)
        result = await validate_node(state)
        assert result["validation_status"] == "retry"
        assert result["retry_count"] == 1
        # A correction HumanMessage should be added
        assert any(
            isinstance(m, HumanMessage) for m in result.get("messages", [])
        )

    async def test_second_failure_marks_failed(self):
        from app.nodes import validate_node

        msg = AIMessage(
            content="",
            tool_calls=[{"name": "list_pods", "args": {}, "id": "1"}],
        )
        state = _agent_state(messages=[msg], retry_count=1)
        result = await validate_node(state)
        assert result["validation_status"] == "failed"
        # A failure AIMessage should be added
        assert any(
            isinstance(m, AIMessage) for m in result.get("messages", [])
        )

    async def test_disallowed_namespace_triggers_retry(self):
        from app.nodes import validate_node

        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_pods",
                    "args": {"namespace": "kube-system"},
                    "id": "1",
                }
            ],
        )
        state = _agent_state(messages=[msg], retry_count=0)
        result = await validate_node(state)
        assert result["validation_status"] == "retry"

    async def test_no_tool_calls_passes_through(self):
        from app.nodes import validate_node

        state = _agent_state(messages=[AIMessage(content="done")])
        result = await validate_node(state)
        assert result["validation_status"] == "passed"
