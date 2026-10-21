"""Read-only Kubernetes tools exposed to the LangGraph agent.

Every tool enforces the namespace allowlist before making any API call.
All output is passed through the redactor before being returned.
"""
from __future__ import annotations

import re

from langchain_core.tools import tool

from app.config import ARGOCD_APP_ALLOWLIST, NAMESPACE_ALLOWLIST
from app.k8s import get_k8s_client
from app.redactor import redact, redact_obj

# Kubernetes name grammar: lowercase alphanumeric, hyphens and dots only.
# RFC 1123 label: max 63 chars; subdomain: max 253.  We use 253 for both.
_K8S_NAME = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")


def _assert_name(value: str, kind: str) -> None:
    """Raise ValueError if *value* is not a valid, safe Kubernetes name."""
    if not _K8S_NAME.fullmatch(value or "") or ".." in value:
        raise ValueError(f"Invalid {kind} name: {value!r}")


def _assert_namespace(namespace: str) -> None:
    if namespace not in NAMESPACE_ALLOWLIST:
        raise ValueError(
            f"Namespace '{namespace}' is not in the allowlist. "
            f"Allowed: {', '.join(NAMESPACE_ALLOWLIST)}"
        )


@tool
async def list_pods(namespace: str) -> list[dict]:
    """List all pods in *namespace* with their phase and container statuses."""
    _assert_namespace(namespace)
    client = get_k8s_client()
    data = await client.get_json(f"/api/v1/namespaces/{namespace}/pods")
    result = []
    for item in data.get("items", []):
        meta = item.get("metadata", {})
        status = item.get("status", {})
        result.append(
            redact_obj(
                {
                    "name": meta.get("name"),
                    "namespace": namespace,
                    "phase": status.get("phase", "Unknown"),
                    "conditions": [
                        {"type": c["type"], "status": c["status"]}
                        for c in status.get("conditions", [])
                    ],
                    "containerStatuses": [
                        {
                            "name": cs["name"],
                            "ready": cs["ready"],
                            "restartCount": cs["restartCount"],
                            "state": cs.get("state", {}),
                            "lastState": cs.get("lastState", {}),
                        }
                        for cs in status.get("containerStatuses", [])
                    ],
                }
            )
        )
    return result


@tool
async def get_pod_status(namespace: str, pod: str) -> dict:
    """Get detailed status for a specific pod, including conditions and container states."""
    _assert_namespace(namespace)
    _assert_name(pod, "pod")
    client = get_k8s_client()
    data = await client.get_json(f"/api/v1/namespaces/{namespace}/pods/{pod}")
    status = data.get("status", {})
    return redact_obj(
        {
            "name": pod,
            "namespace": namespace,
            "phase": status.get("phase", "Unknown"),
            "conditions": status.get("conditions", []),
            "containerStatuses": status.get("containerStatuses", []),
            "startTime": status.get("startTime"),
            "message": status.get("message"),
            "reason": status.get("reason"),
        }
    )


@tool
async def recent_events(namespace: str, limit: int = 20) -> list[dict]:
    """Return the most recent Kubernetes events in *namespace* (max 50)."""
    _assert_namespace(namespace)
    limit = min(limit, 50)
    client = get_k8s_client()
    data = await client.get_json(f"/api/v1/namespaces/{namespace}/events")
    items = data.get("items", [])
    # Sort newest-first
    items.sort(
        key=lambda e: (
            e.get("lastTimestamp") or e.get("eventTime") or ""
        ),
        reverse=True,
    )
    events = []
    for item in items[:limit]:
        involved = item.get("involvedObject", {})
        events.append(
            redact_obj(
                {
                    "type": item.get("type"),
                    "reason": item.get("reason"),
                    "message": item.get("message", ""),
                    "regarding": {
                        "kind": involved.get("kind"),
                        "name": involved.get("name"),
                    },
                    "count": item.get("count"),
                    "lastTimestamp": item.get("lastTimestamp")
                    or item.get("eventTime"),
                }
            )
        )
    return events


@tool
async def pod_logs(namespace: str, pod: str, tail_lines: int = 100) -> str:
    """Fetch the last *tail_lines* lines of logs from a pod (max 200). Output is redacted."""
    _assert_namespace(namespace)
    _assert_name(pod, "pod")
    tail_lines = min(tail_lines, 200)
    client = get_k8s_client()
    raw = await client.get_text(
        f"/api/v1/namespaces/{namespace}/pods/{pod}/log?tailLines={tail_lines}"
    )
    return redact(raw)


@tool
async def argocd_app_status(app: str) -> dict:
    """Get the sync and health status of an ArgoCD Application by name (reads the CRD directly)."""
    _assert_name(app, "app")
    if app not in ARGOCD_APP_ALLOWLIST:
        raise ValueError(
            f"ArgoCD app '{app}' is not in the allowlist. "
            f"Allowed: {', '.join(ARGOCD_APP_ALLOWLIST)}"
        )
    client = get_k8s_client()
    data = await client.get_json(
        f"/apis/argoproj.io/v1alpha1/namespaces/argocd/applications/{app}"
    )
    status = data.get("status", {})
    return redact_obj(
        {
            "name": app,
            "sync": status.get("sync", {}).get("status"),
            "health": status.get("health", {}).get("status"),
            "healthMessage": status.get("health", {}).get("message"),
            "operationState": (status.get("operationState") or {}).get("phase"),
            "conditions": status.get("conditions", []),
            "resources": [
                {
                    "kind": r.get("kind"),
                    "name": r.get("name"),
                    "namespace": r.get("namespace"),
                    "health": (r.get("health") or {}).get("status"),
                    "status": r.get("status"),
                }
                for r in status.get("resources", [])[:20]
            ],
        }
    )


# Exported list for use in ToolNode and system prompt
ALL_TOOLS = [list_pods, get_pod_status, recent_events, pod_logs, argocd_app_status]
