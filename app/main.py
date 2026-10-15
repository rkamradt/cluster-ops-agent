"""FastAPI application: SSE streaming endpoint, static UI, rate limiting."""
from __future__ import annotations  # noqa: F401 (needed for str | None on Python 3.9)

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import DB_PATH, MAX_QUESTION_LEN, RATE_LIMIT_MAX, RATE_LIMIT_WINDOW_SEC
from app.graph import AgentState, build_graph

# ── Rate limiter ─────────────────────────────────────────────────────────────

# Respect Cloudflare's real-IP header; fall back to direct connection IP
def _get_real_ip(request: Request) -> str:
    return (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or get_remote_address(request)
    )


limiter = Limiter(key_func=_get_real_ip)
_RATE_STR = f"{RATE_LIMIT_MAX}/{RATE_LIMIT_WINDOW_SEC}second"

# ── Application state ─────────────────────────────────────────────────────────

_graph = None
_mermaid: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _mermaid
    # Build the graph with the SQLite checkpointer
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        _graph = build_graph(checkpointer)
        _mermaid = _graph.get_graph().draw_mermaid()
        yield
    _graph = None


app = FastAPI(title="Cluster Ops Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/graph.mmd", response_class=HTMLResponse)
async def graph_mermaid():
    return HTMLResponse(content=_mermaid, media_type="text/plain")


@app.get("/ask")
@limiter.limit(_RATE_STR)
async def ask(request: Request, question: str, thread_id: str | None = None):
    """SSE endpoint: streams node-by-node progress then the final answer."""
    if len(question) > MAX_QUESTION_LEN:
        return StreamingResponse(
            _error_stream(f"Question too long (max {MAX_QUESTION_LEN} chars)."),
            media_type="text/event-stream",
        )

    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "step_count": 0,
        "retry_count": 0,
        "validation_status": "",
    }

    return StreamingResponse(
        _stream_graph(initial_state, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-Id": tid,
        },
    )


# ── SSE helpers ───────────────────────────────────────────────────────────────

_NODE_LABELS = {
    "agent": "thinking…",
    "validate": "validating tool call…",
    "tools": "querying cluster…",
    "diagnose": "diagnosing…",
}


async def _stream_graph(initial_state: AgentState, config: dict):
    """Yield SSE events for each node update, then a final 'answer' event."""
    final_content = ""
    try:
        async for event in _graph.astream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node_name, node_output in event.items():
                label = _NODE_LABELS.get(node_name, node_name)
                yield _sse("progress", {"node": node_name, "label": label})

                # Capture the last AI message as the answer
                for msg in reversed(node_output.get("messages", [])):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_content = msg.content
                        break

        yield _sse("answer", {"content": final_content})
        yield _sse("done", {})

    except Exception as exc:  # noqa: BLE001
        yield _sse("error", {"message": str(exc)})


async def _error_stream(message: str):
    yield _sse("error", {"message": message})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Static files (index.html) ─────────────────────────────────────────────────

_PUBLIC = Path(__file__).parent.parent / "public"
if _PUBLIC.exists():
    app.mount("/", StaticFiles(directory=str(_PUBLIC), html=True), name="static")
