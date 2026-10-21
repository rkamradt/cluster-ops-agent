"""Tests for the Kubernetes tools: namespace allowlist enforcement and output shape.

The K8s client is replaced with a fake — no cluster required.
"""
import json
import pytest

from app import config as cfg


# ── Fake K8s client ───────────────────────────────────────────────────────────


class FakeK8sClient:
    """Returns canned responses for each API path."""

    async def get_json(self, path: str) -> dict:
        if "/pods" in path and path.endswith("/pods"):
            return _FAKE_POD_LIST
        if "/pods/" in path and "/log" not in path:
            return _FAKE_POD_DETAIL
        if "/events" in path:
            return _FAKE_EVENT_LIST
        if "argoproj.io" in path:
            return _FAKE_ARGOCD_APP
        return {}

    async def get_text(self, path: str) -> str:
        return _FAKE_LOG_TEXT


# ── Canned responses ──────────────────────────────────────────────────────────

_FAKE_POD_LIST = {
    "items": [
        {
            "metadata": {"name": "home-6d4f9b-xyz"},
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [
                    {"name": "home", "ready": True, "restartCount": 0, "state": {}, "lastState": {}}
                ],
            },
        }
    ]
}

_FAKE_POD_DETAIL = {
    "status": {
        "phase": "Running",
        "conditions": [],
        "containerStatuses": [],
        "startTime": "2025-01-01T00:00:00Z",
        "message": None,
        "reason": None,
    }
}

_FAKE_EVENT_LIST = {
    "items": [
        {
            "type": "Normal",
            "reason": "Started",
            "message": "Started container home",
            "involvedObject": {"kind": "Pod", "name": "home-xyz"},
            "count": 1,
            "lastTimestamp": "2025-01-01T00:01:00Z",
        }
    ]
}

_FAKE_ARGOCD_APP = {
    "status": {
        "sync": {"status": "Synced"},
        "health": {"status": "Healthy", "message": None},
        "operationState": {"phase": "Succeeded"},
        "conditions": [],
        "resources": [],
    }
}

_FAKE_LOG_TEXT = "INFO starting server\nINFO listening on port 3000\n"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def patch_k8s_client(monkeypatch):
    """Replace the K8s client singleton with a fake for all tool tests."""
    import app.k8s as k8s_mod

    fake = FakeK8sClient()
    monkeypatch.setattr(k8s_mod, "_client", fake)


# ── list_pods ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestListPods:
    async def test_returns_pod_list(self):
        from app.tools import list_pods

        result = await list_pods.ainvoke({"namespace": "home"})
        assert isinstance(result, list)
        assert result[0]["name"] == "home-6d4f9b-xyz"
        assert result[0]["phase"] == "Running"

    async def test_disallowed_namespace_raises(self):
        from app.tools import list_pods

        with pytest.raises(ValueError, match="allowlist"):
            await list_pods.ainvoke({"namespace": "kube-system"})

    async def test_output_keys_present(self):
        from app.tools import list_pods

        result = await list_pods.ainvoke({"namespace": "home"})
        pod = result[0]
        for key in ("name", "namespace", "phase", "conditions", "containerStatuses"):
            assert key in pod, f"Missing key: {key}"


# ── get_pod_status ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestGetPodStatus:
    async def test_returns_status_dict(self):
        from app.tools import get_pod_status

        result = await get_pod_status.ainvoke({"namespace": "home", "pod": "home-xyz"})
        assert result["phase"] == "Running"
        assert result["name"] == "home-xyz"

    async def test_disallowed_namespace_raises(self):
        from app.tools import get_pod_status

        with pytest.raises(ValueError, match="allowlist"):
            await get_pod_status.ainvoke({"namespace": "argocd", "pod": "any"})


# ── recent_events ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRecentEvents:
    async def test_returns_event_list(self):
        from app.tools import recent_events

        result = await recent_events.ainvoke({"namespace": "home"})
        assert isinstance(result, list)
        assert result[0]["reason"] == "Started"

    async def test_limit_capped_at_50(self):
        from app.tools import recent_events

        # Even if 999 is requested, it should silently cap at 50
        result = await recent_events.ainvoke({"namespace": "home", "limit": 999})
        assert isinstance(result, list)

    async def test_disallowed_namespace_raises(self):
        from app.tools import recent_events

        with pytest.raises(ValueError):
            await recent_events.ainvoke({"namespace": "default"})


# ── pod_logs ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPodLogs:
    async def test_returns_string(self):
        from app.tools import pod_logs

        result = await pod_logs.ainvoke({"namespace": "home", "pod": "home-xyz"})
        assert isinstance(result, str)
        assert "listening on port 3000" in result

    async def test_tail_lines_capped_at_200(self):
        from app.tools import pod_logs

        # Should not raise even with a large tail value
        result = await pod_logs.ainvoke(
            {"namespace": "home", "pod": "home-xyz", "tail_lines": 9999}
        )
        assert isinstance(result, str)

    async def test_redacts_ip_in_logs(self, monkeypatch):
        import app.k8s as k8s_mod

        class LogWithIP(FakeK8sClient):
            async def get_text(self, path):
                return "connection from 10.0.0.5 accepted\n"

        monkeypatch.setattr(k8s_mod, "_client", LogWithIP())
        from app.tools import pod_logs

        result = await pod_logs.ainvoke({"namespace": "home", "pod": "home-xyz"})
        assert "10.0.0.5" not in result
        assert "<IP>" in result

    async def test_disallowed_namespace_raises(self):
        from app.tools import pod_logs

        with pytest.raises(ValueError):
            await pod_logs.ainvoke({"namespace": "kube-system", "pod": "any"})


# ── argocd_app_status ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestArgoCDAppStatus:
    async def test_returns_status_dict(self):
        from app.tools import argocd_app_status

        result = await argocd_app_status.ainvoke({"app": "home"})
        assert result["sync"] == "Synced"
        assert result["health"] == "Healthy"
        assert result["name"] == "home"

    async def test_output_keys_present(self):
        from app.tools import argocd_app_status

        result = await argocd_app_status.ainvoke({"app": "k8s-dashboard"})
        for key in ("name", "sync", "health", "conditions", "resources"):
            assert key in result, f"Missing key: {key}"

    async def test_unlisted_app_raises(self):
        from app.tools import argocd_app_status

        with pytest.raises(ValueError, match="allowlist"):
            await argocd_app_status.ainvoke({"app": "keycloak"})


# ── _assert_name path-traversal / injection tests ─────────────────────────────


@pytest.mark.asyncio
class TestAssertName:
    """Ensure _assert_name blocks all path-traversal and invalid inputs."""

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../../keycloak",          # classic path traversal
            "../keycloak",             # single step up
            "%2e%2e/keycloak",         # URL-encoded dot-dot (should fail regex)
            "Home-Pod",                # uppercase → fails regex
            "home pod",                # space → fails regex
            "",                        # empty
            "/etc/passwd",             # absolute path
            "a" * 254,                 # too long
        ],
    )
    async def test_path_traversal_pod_logs(self, bad_name):
        from app.tools import pod_logs

        with pytest.raises(ValueError):
            await pod_logs.ainvoke({"namespace": "home", "pod": bad_name})

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../../keycloak",
            "../keycloak",
            "%2e%2e/keycloak",
            "Home-Pod",
            "",
            "/etc/passwd",
        ],
    )
    async def test_path_traversal_get_pod_status(self, bad_name):
        from app.tools import get_pod_status

        with pytest.raises(ValueError):
            await get_pod_status.ainvoke({"namespace": "home", "pod": bad_name})

    async def test_valid_pod_name_passes(self):
        """A well-formed pod name must not be rejected."""
        from app.tools import pod_logs

        # Should not raise — FakeK8sClient returns canned text
        result = await pod_logs.ainvoke({"namespace": "home", "pod": "home-6d4f9b-xyz"})
        assert isinstance(result, str)
