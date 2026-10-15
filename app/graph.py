"""LangGraph StateGraph definition for the Cluster Ops Agent.

Graph layout:
    START
      │
      ▼
   [agent]  ←──────────────────────────────────────┐
      │                                             │
      ├─ tool calls? ──► [validate] ──► [tools] ────┘
      │                      │
      │                 retry? → back to agent
      │                 failed? → END
      │
      ├─ step cap reached? ──────────────────────── END
      │
      └─ no more tools ──────────────────────────── END
                                    ↑
                    [tools] ─► unhealthy? ─► [diagnose] ─► END
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from app.config import MAX_STEPS
from app.nodes import agent_node, diagnose_node, validate_node
from app.tools import ALL_TOOLS

# Keywords in tool output that flag a pod/app as unhealthy
_UNHEALTHY_KEYWORDS = frozenset(
    {
        "crashloopbackoff",
        "oomkilled",
        "degraded",
        "imagepullbackoff",
        "errimagepull",
        "backoff",
        "failed",
        "unhealthy",
        "crashloop",
    }
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int
    retry_count: int
    validation_status: str  # "passed" | "retry" | "failed" | ""


# ── Routing functions ────────────────────────────────────────────────────────


def _route_from_agent(state: AgentState) -> str:
    """After agent: check step cap, then route to validate or END."""
    if state.get("step_count", 0) >= MAX_STEPS:
        return END
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "validate"
    return END


def _route_from_validate(state: AgentState) -> str:
    """After validate: passed → tools; retry → agent; failed → END."""
    status = state.get("validation_status", "passed")
    if status == "retry":
        return "agent"
    if status == "failed":
        return END
    return "tools"


def _route_from_tools(state: AgentState) -> str:
    """After tools: if the freshest tool message shows unhealthy signals → diagnose; else → agent."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            content = str(msg.content).lower()
            if any(kw in content for kw in _UNHEALTHY_KEYWORDS):
                return "diagnose"
            break
    return "agent"


# ── Graph builder ────────────────────────────────────────────────────────────


def build_graph(checkpointer=None):
    """Compile and return the LangGraph StateGraph.

    Pass a checkpointer (e.g. AsyncSqliteSaver) for session persistence.
    Pass None for stateless use (CLI / tests).
    """
    g = StateGraph(AgentState)

    g.add_node("agent", agent_node)
    g.add_node("validate", validate_node)
    g.add_node("tools", ToolNode(ALL_TOOLS))
    g.add_node("diagnose", diagnose_node)

    g.add_edge(START, "agent")

    g.add_conditional_edges(
        "agent",
        _route_from_agent,
        {"validate": "validate", END: END},
    )
    g.add_conditional_edges(
        "validate",
        _route_from_validate,
        {"tools": "tools", "agent": "agent", END: END},
    )
    g.add_conditional_edges(
        "tools",
        _route_from_tools,
        {"diagnose": "diagnose", "agent": "agent"},
    )
    g.add_edge("diagnose", END)

    return g.compile(checkpointer=checkpointer)
