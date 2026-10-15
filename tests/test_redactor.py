"""Unit tests for the redactor module.

These run without any Kubernetes or LLM dependencies.
"""
import pytest
from app.redactor import redact, redact_obj


class TestRedactIPv4:
    def test_simple_ip(self):
        assert redact("connected from 192.168.1.42") == "connected from <IP>"

    def test_multiple_ips(self):
        result = redact("src=10.0.0.1 dst=172.16.5.2")
        assert "<IP>" in result
        assert "10.0.0.1" not in result
        assert "172.16.5.2" not in result

    def test_ip_in_url_already_redacted(self):
        # Connection strings are redacted first; residual IPs then cleaned up
        result = redact("mongodb://10.0.0.1:27017/mydb")
        assert "10.0.0.1" not in result
        assert "27017" not in result


class TestRedactJWT:
    def test_jwt_three_parts(self):
        jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        assert redact(jwt) == "<JWT>"

    def test_jwt_embedded_in_log(self):
        line = f"Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.sig123abc"
        result = redact(line)
        assert "eyJ" not in result


class TestRedactConnectionStrings:
    def test_mongodb(self):
        result = redact("MONGO_URI=mongodb://user:pass@host:27017/db")
        assert "mongodb://" not in result
        assert "<CONNECTION_STRING>" in result

    def test_postgres(self):
        result = redact("postgres://admin:secret@db.internal:5432/prod")
        assert "admin" not in result
        assert "secret" not in result

    def test_redis(self):
        result = redact("redis://localhost:6379")
        assert "redis://" not in result


class TestRedactCredentialKeyValues:
    def test_token_equals(self):
        result = redact("token=abc123xyz")
        assert "abc123xyz" not in result
        assert "REDACTED" in result

    def test_password_colon(self):
        result = redact("password: hunter2")
        assert "hunter2" not in result

    def test_secret_mixed_case(self):
        result = redact("SECRET=myTopSecret")
        assert "myTopSecret" not in result


class TestRedactInternalHosts:
    def test_svc_cluster_local(self):
        result = redact("connecting to ollama.llm.svc.cluster.local:11434")
        assert "ollama.llm.svc.cluster.local" not in result
        assert "<INTERNAL_HOST>" in result

    def test_other_svc(self):
        result = redact("host: mongodb.rkamradt-platform.svc.cluster.local")
        assert "mongodb.rkamradt-platform.svc.cluster.local" not in result


class TestRedactEmail:
    def test_plain_email(self):
        result = redact("admin@example.com")
        assert "admin@example.com" not in result
        assert "<EMAIL>" in result


class TestRedactHex:
    def test_long_hex_token(self):
        result = redact("sha256=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4")
        assert "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" not in result

    def test_short_hex_preserved(self):
        # Short hex (< 20 chars) like git SHAs in names should not be redacted
        result = redact("pod sha-1a2b3c4 is healthy")
        # "1a2b3c4" is 7 chars — below threshold, preserved
        assert "1a2b3c4" in result


class TestRedactObj:
    def test_dict_values(self):
        obj = {"url": "mongodb://user:pass@host/db", "count": 5}
        result = redact_obj(obj)
        assert "mongodb://" not in str(result)
        assert result["count"] == 5

    def test_nested_list(self):
        obj = [{"ip": "192.168.1.1"}, {"safe": "value"}]
        result = redact_obj(obj)
        assert "192.168.1.1" not in str(result)
        assert result[1]["safe"] == "value"

    def test_non_string_passthrough(self):
        assert redact_obj(42) == 42
        assert redact_obj(None) is None
        assert redact_obj(True) is True
