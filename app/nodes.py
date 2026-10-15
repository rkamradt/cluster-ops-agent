"""LangGraph node functions: agent, validate, diagnose."""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

from app.config import (
    MAX_STEPS,
    NAMESPACE_ALLOWLIST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
)
from app.tools import ALL_TOOLS

# ── System prompts ───────────────────────────────────────────────────────────

AGENT_SYSTEM = f"""\
You are a read-only Kubernetes cluster operations assistant for a personal K3s cluster.
Answer questions about cluster health by calling the available tools.

Rules:
- You may only inspect these namespaces: {', '.join(NAMESPACE_ALLOWLIST)}
- Never attempt to modify, delete, or exec into anything — you have no write access
- When you have enough information to answer, stop calling tools and respond clearly
- If asked to do something outside your scope, say so and explain what you CAN tell them

Available tools: list_pods, get_pod_status, recent_events, pod_logs, argocd_app_status
"""

DIAGNOSE_PROMPT = """\
Based on the tool results already collected in this conversation, an unhealthy pod or \
application has been detected. Produce a concise structured diagnosis:

**Likely Cause**: one sentence identifying the root cause
**Evidence**: bullet points citing specific tool output (error messages, restart counts, events)
**Suggested Next Check**: what a human operator should look at or try next

Be specific. Do not call any more tools — work only from what has already been collected.
"""

# ── LLM singletons ──────────────────────────────────────────────────────────

_llm_with_tools: ChatOllama | None = None
_llm_plain: ChatOllama | None = None


def _get_llm_with_tools() -> ChatOllama:
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = ChatOllama(
            base_url=OLLAMA_URL,
            model=OLLAMA_MODEL,
            timeout=OLLAMA_TIMEOUT,
        ).bind_tools(ALL_TOOLS)
    return _llm_with_tools


def _get_llm_plain() -> ChatOllama:
    global _llm_plain
    if _llm_plain is None:
        _llm_plain = ChatOllama(
            base_url=OLLAMA_URL,
            model=OLLAMA_MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
    return _llm_plain


# ── Node functions ───────────────────────────────────────────────────────────

async def agent_node(state: dict) -> dict:
    """Main agent: invoke Llama 3.2 with tools bound."""
    messages = state["messages"]

    # Inject system prompt on the first turn
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=AGENT_SYSTEM)] + list(messages)

    step_count = state.get("step_count", 0) + 1

    llm = _get_llm_with_tools()
    response = await llm.ainvoke(messages)

    return {"messages": [response], "step_count": step_count}


async def validate_node(state: dict) -> dict:
    """Validate tool call arguments from the agent; retry once on bad args."""
    messages = state["messages"]
    last = messages[-1]
    retry_count = state.get("retry_count", 0)

    if not isinstance(last, AIMessage) or not last.tool_calls:
        # Nothing to validate — pass through
        return {"validation_status": "passed"}

    issues: list[str] = []
    for tc in last.tool_calls:
        name = tc["name"]
        args = tc.get("args", {})

        # Tools that require namespace
        if name in ("list_pods", "recent_events", "get_pod_status", "pod_logs"):
            ns = args.get("namespace", "")
            if not ns:
                issues.append(f"'{name}' is missing the required 'namespace' argument.")
            elif ns not in NAMESPACE_ALLOWLIST:
                issues.append(
                    f"'{name}' uses namespace '{ns}' which is not in the allowlist "
                    f"({', '.join(NAMESPACE_ALLOWLIST)})."
                )

        # Tools that require pod name
        if name in ("get_pod_status", "pod_logs"):
            if not args.get("pod"):
                issues.append(f"'{name}' is missing the required 'pod' argument.")

    if not issues:
        return {"validation_status": "passed"}

    if retry_count < 1:
        correction = (
            "Your tool call had issues — please correct them and try again:\n"
            + "\n".join(f"- {i}" for i in issues)
        )
        return {
            "messages": [HumanMessage(content=correction)],
            "retry_count": retry_count + 1,
            "validation_status": "retry",
        }

    # Second failure: give up gracefully
    failure_msg = (
        "I was unable to form a valid tool call after retrying. "
        "Here are the issues I encountered:\n"
        + "\n".join(f"- {i}" for i in issues)
        + "\n\nPlease rephrase your question and try again."
    )
    return {
        "messages": [AIMessage(content=failure_msg)],
        "validation_status": "failed",
    }


async def diagnose_node(state: dict) -> dict:
    """Focused diagnosis node: synthesise collected evidence into a structured report."""
    messages = state["messages"]
    llm = _get_llm_plain()
    diagnosis_messages = list(messages) + [HumanMessage(content=DIAGNOSE_PROMPT)]
    response = await llm.ainvoke(diagnosis_messages)
    return {"messages": [response]}
