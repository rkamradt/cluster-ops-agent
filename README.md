# Cluster Ops Agent

A read-only AI agent that answers questions about a live K3s Kubernetes cluster.
Ask it "are there any crashlooping pods?" or "what's the ArgoCD sync status of the home app?" and it calls real cluster APIs, reasons over the results, and streams an answer back.

**Live at [agent.kamradtfamily.net](https://agent.kamradtfamily.net)**

---

## What it does

Type a question. The agent inspects pods, events, logs, and ArgoCD application status — then answers in plain English. If it finds an unhealthy pod it automatically switches to a focused diagnosis mode that identifies the likely cause, cites the evidence from the logs, and suggests what to check next.

Canned questions on the page get you started immediately:

- *List all pods in the home namespace*
- *Are there any unhealthy pods?*
- *What is the ArgoCD sync status of the home app?*
- *Show me recent events in k8s-dashboard*
- *Is anything crashlooping or OOMKilled right now?*

---

## How it works

### The graph

Built with [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` — a directed graph where each node is a Python function and edges are routing decisions made at runtime.

```
START
  │
  ▼
[agent] ──── has tool calls? ────► [validate] ──── args OK? ────► [tools]
  ▲                                     │                              │
  │                                 retry once                         │
  │                                 on bad args                        │
  └──────────────────────────────────────────────────────── no issues ─┘
  │
  └─── step cap (6) reached, or done ────► END
  
[tools] ──── unhealthy signal? ────► [diagnose] ────► END
         └── healthy ──────────────► [agent]
```

| Node | Role |
|---|---|
| **agent** | Llama 3.2 with tools bound; decides what to call next |
| **validate** | Checks tool call arguments; retries once with a corrective message if malformed |
| **tools** | `ToolNode` executing the five read-only K8s tools |
| **diagnose** | Focused LLM call (no tools) that produces a structured "likely cause / evidence / next check" report when an unhealthy pod or app is detected |

State is a typed `TypedDict` with an `add_messages` reducer. The graph is compiled with an `AsyncSqliteSaver` checkpointer so each browser session (`thread_id`) has persistent conversation history.

### The tools

All tools enforce a namespace allowlist before making any API call. All output is redacted before it reaches the LLM or the page.

| Tool | What it reads |
|---|---|
| `list_pods(namespace)` | Pod names, phases, container statuses, restart counts |
| `get_pod_status(namespace, pod)` | Full pod status including conditions and last state |
| `recent_events(namespace, limit)` | Kubernetes events sorted newest-first |
| `pod_logs(namespace, pod, tail_lines)` | Last N lines of container logs (max 200) |
| `argocd_app_status(app)` | Sync/health status read directly from the `applications.argoproj.io` CRD |

### The streaming UI

The FastAPI `/ask` endpoint is a [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) stream. Each LangGraph node emits a `progress` event ("thinking…", "querying cluster…", "diagnosing…") before its result arrives, so the slow CPU model feels intentional rather than broken. The final answer arrives as an `answer` event.

---

## Design trade-offs

**Small CPU-only model (Llama 3.2 3B Q4)**
The cluster runs on a single old HP server with no GPU. Llama 3.2 3B fits in RAM and produces usable tool calls, but it's slow (~30–90s per question) and occasionally generates malformed JSON arguments. The `validate` node exists specifically to catch and recover from this — it retries once with a corrective message before failing gracefully. The SSE progress stream makes the wait feel like work, not a hang.

**Read-only tools and namespace allowlist**
The ServiceAccount has `get/list/watch` on pods, pod logs, events, and ArgoCD Applications only. No secrets, configmaps, exec, or writes — anything not listed is denied by RBAC. On top of that, an application-layer allowlist in `config.py` prevents the LLM from querying namespaces outside the approved set even if the SA could technically reach them.

**Loop cap (6 steps)**
A small model can get confused and call the same tool repeatedly. The graph tracks `step_count` and forces an exit at 6 steps regardless of whether the model thinks it's done. In practice well-formed questions resolve in 2–3 steps.

**Redaction**
Seven regex rules strip IPv4 addresses, JWTs, connection strings, credential key-value pairs, internal FQDNs (`*.svc.cluster.local`), email addresses, and long hex strings from all tool output before it reaches the LLM context or the browser. The redactor has 19 unit tests covering each pattern.

**SQLite checkpointer over MongoDB**
The cluster runs MongoDB, but `langgraph-checkpoint-mongodb` adds a network dependency: if MongoDB is down, the agent is down. SQLite on a small PVC (`/data/checkpoints.db`) is self-contained. For a single-replica showcase on a single-node cluster there is no meaningful downside.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Agent framework | LangGraph 1.2 (`StateGraph`, `ToolNode`, `AsyncSqliteSaver`) |
| LLM | Llama 3.2 3B Q4 via Ollama (`langchain-ollama`) |
| API | FastAPI + uvicorn, Server-Sent Events |
| K8s client | `httpx` reading the in-cluster ServiceAccount token |
| Persistence | SQLite (`langgraph-checkpoint-sqlite`) on a 100Mi PVC |
| Container | `python:3.12-slim`, port 8000 |
| Deploy | GitHub Actions → ghcr.io → Helm → ArgoCD → K3s |
| Ingress | Cloudflare Tunnel (no Ingress resource in cluster) |
| Tests | pytest, pytest-asyncio — 60 tests, no LLM or cluster required |

---

## Running locally

```bash
# Prerequisites: kubectl access to the cluster

# Terminal 1 — K8s API proxy (no auth headers needed)
kubectl proxy --port=8001

# Terminal 2 — Ollama tunnel
kubectl port-forward -n llm svc/ollama 11434:11434

# Terminal 3 — the agent
git clone https://github.com/rkamradt/cluster-ops-agent
cd cluster-ops-agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export K8S_PROXY_URL=http://localhost:8001
export OLLAMA_URL=http://localhost:11434

# CLI
python ask.py "are there any crashlooping pods?"

# Web UI
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Tests

```bash
pytest tests/ -v
# 60 passed — no cluster or LLM needed
```

---

## Cluster deployment

Follows the standard [`rkamradt-helm-charts`](https://github.com/rkamradt/rkamradt-helm-charts) pattern. See [DEPLOYING.md](https://github.com/rkamradt/misc-spine/blob/main/DEPLOYING.md) in `misc-spine` for the full pipeline.

Key additions over a standard service:

- `templates/clusterrole.yaml` — minimal RBAC (pods, logs, events, ArgoCD CRD; no secrets or writes)
- `templates/pvc.yaml` — 100Mi `local-path` PVC for the SQLite checkpointer
- `startupProbe` with 150s window — Python + LangGraph imports are slow on a constrained CPU node
