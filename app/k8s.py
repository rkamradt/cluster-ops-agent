"""Minimal async Kubernetes API client.

Supports two modes:
  - In-cluster: reads the mounted ServiceAccount token and CA cert.
  - Local dev:  uses kubectl proxy (`kubectl proxy --port=8001`) via K8S_PROXY_URL.
"""
from __future__ import annotations

import os
import httpx

from app.config import K8S_API_URL, K8S_TOKEN_PATH, K8S_CA_PATH, K8S_PROXY_URL


def _build_client() -> httpx.AsyncClient:
    if K8S_PROXY_URL:
        # Local dev: kubectl proxy handles auth
        return httpx.AsyncClient(
            base_url=K8S_PROXY_URL,
            timeout=30.0,
        )
    if os.path.exists(K8S_TOKEN_PATH):
        # In-cluster: ServiceAccount token + cluster CA
        with open(K8S_TOKEN_PATH) as fh:
            token = fh.read().strip()
        return httpx.AsyncClient(
            base_url=K8S_API_URL,
            headers={"Authorization": f"Bearer {token}"},
            verify=K8S_CA_PATH,
            timeout=30.0,
        )
    raise RuntimeError(
        "No Kubernetes credentials found. "
        "Either run inside a cluster with a ServiceAccount, "
        "or set K8S_PROXY_URL=http://localhost:8001 (after `kubectl proxy --port=8001`)."
    )


class K8sClient:
    """Thin async wrapper around the Kubernetes REST API."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient = _build_client()

    async def get_json(self, path: str) -> dict:
        """GET *path* and return parsed JSON."""
        resp = await self._client.get(path)
        resp.raise_for_status()
        return resp.json()

    async def get_text(self, path: str) -> str:
        """GET *path* and return raw text (used for pod logs)."""
        resp = await self._client.get(path)
        resp.raise_for_status()
        return resp.text

    async def close(self) -> None:
        await self._client.aclose()


# Module-level singleton — created lazily so import doesn't fail without credentials
_client: K8sClient | None = None


def get_k8s_client() -> K8sClient:
    global _client
    if _client is None:
        _client = K8sClient()
    return _client
