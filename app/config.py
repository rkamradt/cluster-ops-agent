"""Central configuration — all tunables come from env vars with safe defaults."""
from __future__ import annotations

import os

# Namespaces the agent is allowed to query (application-layer allowlist)
NAMESPACE_ALLOWLIST: list[str] = os.environ.get(
    "NAMESPACE_ALLOWLIST",
    "home,k8s-dashboard,gol,portfolio,llm,cluster-ops-agent",
).split(",")

# Ollama settings
OLLAMA_URL: str = os.environ.get(
    "OLLAMA_URL", "http://ollama.llm.svc.cluster.local:11434"
)
OLLAMA_MODEL: str = os.environ.get(
    "OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M"
)
OLLAMA_TIMEOUT: float = float(os.environ.get("OLLAMA_TIMEOUT", "120"))

# Graph limits
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "6"))

# SQLite checkpointer path (PVC mount in cluster)
DB_PATH: str = os.environ.get("DB_PATH", "/data/checkpoints.db")

# Rate limiting
RATE_LIMIT_MAX: int = int(os.environ.get("RATE_LIMIT_MAX", "20"))
RATE_LIMIT_WINDOW_SEC: int = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "3600"))

# Max question length (characters)
MAX_QUESTION_LEN: int = int(os.environ.get("MAX_QUESTION_LEN", "500"))

# Kubernetes API — auto-detected: in-cluster SA token path, or kubectl proxy URL for local dev
K8S_API_URL: str = os.environ.get("K8S_API_URL", "https://kubernetes.default.svc")
K8S_TOKEN_PATH: str = os.environ.get(
    "K8S_TOKEN_PATH",
    "/var/run/secrets/kubernetes.io/serviceaccount/token",
)
K8S_CA_PATH: str = os.environ.get(
    "K8S_CA_PATH",
    "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
)
# For local dev: `kubectl proxy --port=8001` then set K8S_PROXY_URL=http://localhost:8001
K8S_PROXY_URL: str | None = os.environ.get("K8S_PROXY_URL")
