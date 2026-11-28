"""可观测性层回归测试：请求 ID、指标标签、健康检查分层、日志脱敏。

这些用例锁的都是「踩过或差点踩」的坑：
1. `/health` 曾走 `get_vector_store()` 触发 BGE 模型加载（实测 28s），
   而 compose healthcheck `timeout: 5s` → 容器永远 unhealthy 并反复重启。
   这里用「get_vector_store 一被调用就抛异常」来锁死「轻量检查不许碰向量库」。
2. 指标 path 标签若直接用 URL，session_id 会撑爆 Prometheus 时间序列基数。
3. 全局异常处理曾把 `str(exc)` 原样返回前端（泄露路径/SQL/上游报错）。
"""

import json
import logging
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api
from utils.metrics import UNMATCHED_PATH, normalize_path, resolve_path_label
from utils.observability import (
    JsonFormatter,
    get_request_id,
    new_request_id,
    normalize_request_id,
    request_id_var,
    scrub,
)

# ═══════════════════════════════════════════
#  请求 ID
# ═══════════════════════════════════════════


class TestRequestId:
    def test_generated_id_is_short_hex(self):
        rid = new_request_id()
        assert len(rid) == 12
        assert all(c in "0123456789abcdef" for c in rid)

    def test_accepts_gateway_supplied_id(self):
        assert normalize_request_id("trace-abc_123.4") == "trace-abc_123.4"

    def test_rejects_log_injection(self):
        """换行/控制符必须被丢弃，否则可以伪造日志行。"""
        rid = normalize_request_id("abc\n2026-01-01 [ERROR] FAKE")
        assert "\n" not in rid
        assert len(rid) == 12

    def test_rejects_oversized_id(self):
        assert len(normalize_request_id("x" * 500)) == 12

    def test_echoed_in_response_and_propagated(self, client):
        resp = client.get("/health/live", headers={"X-Request-ID": "trace-abc123"})
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"] == "trace-abc123"

    def test_generated_when_header_absent(self, client):
        resp = client.get("/health/live")
        assert len(resp.headers["X-Request-ID"]) == 12

    def test_log_record_carries_request_id(self, caplog):
        token = request_id_var.set("abc123def456")
        try:
            with caplog.at_level(logging.INFO, logger="obs.test"):
                logging.getLogger("obs.test").info("hello")
        finally:
            request_id_var.reset(token)
        assert caplog.records[-1].request_id == "abc123def456"

    def test_default_request_id_is_dash(self):
        assert get_request_id() == "-"


# ═══════════════════════════════════════════
#  脱敏
# ═══════════════════════════════════════════


class TestScrub:
    @pytest.mark.parametrize(
        "raw, leaked",
        [
            ("key=sk-abcdefghijklmnop123456", "sk-abcdefghijklmnop123456"),
            ("Authorization: Bearer abcdefghijklmnopqrst", "abcdefghijklmnopqrst"),
            ("api_key=SECRET_VALUE_123", "SECRET_VALUE_123"),
            ("password: hunter2hunter2", "hunter2hunter2"),
            ("https://alice:p@ssw0rd@example.com/path", "p@ssw0rd"),
            (
                "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
                "eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            ),
        ],
    )
    def test_secrets_are_masked(self, raw, leaked):
        assert leaked not in scrub(raw)
        assert "***" in scrub(raw)

    def test_plain_text_untouched(self):
        text = "GET /health/live → 200 (2ms)"
        assert scrub(text) == text

    def test_non_string_passthrough(self):
        assert scrub(None) is None  # type: ignore[arg-type]


# ═══════════════════════════════════════════
#  指标与路径归一化
# ═══════════════════════════════════════════


class _FakeRoute:
    def __init__(self, path: str):
        self.path = path


class _FakeRequest:
    def __init__(self, scope: dict, method: str = "GET"):
        self.scope = scope
        self.method = method


class TestMetricsLabels:
    def test_prefers_route_template(self):
        """有路由模板时用模板，session_id 不会进 label。"""
        req = _FakeRequest({"route": _FakeRoute("/sessions/{session_id}/history")})
        assert resolve_path_label(req) == "/sessions/{session_id}/history"

    def test_unmatched_paths_are_collapsed(self):
        """无路由（404/扫描器路径）统一归档，防止基数爆炸。"""
        assert resolve_path_label(_FakeRequest({})) == UNMATCHED_PATH

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("/sessions/123/history", "/sessions/{id}/history"),
            ("/users/550e8400-e29b-41d4-a716-446655440000", "/users/{id}"),
            ("/a/deadbeefcafe1234/b", "/a/{id}/b"),
            ("/health", "/health"),
        ],
    )
    def test_normalize_path(self, raw, expected):
        assert normalize_path(raw) == expected

    def test_metrics_endpoint_exposes_http_counter(self, client):
        client.get("/health/live")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert 'http_requests_total{method="GET",path="/health/live"' in resp.text
        # 抓取端点自身不计入指标，否则每次 scrape 都污染 RED 统计
        assert 'path="/metrics"' not in resp.text

    def test_metrics_does_not_leak_user_data(self, client):
        body = client.get("/metrics").text
        # path 标签是路由模板（含字面量 {session_id}）而非真实 ID；
        # 这里断言不存在任何凭证/鉴权信息
        assert "Authorization" not in body
        assert "Bearer " not in body
        assert "sk-" not in body


# ═══════════════════════════════════════════
#  健康检查分层
# ═══════════════════════════════════════════


class TestHealthSplit:
    def test_liveness_never_touches_vector_store(self, client, monkeypatch):
        """存活探针连向量库都不该 import —— 这里让 get_vector_store 直接爆炸。"""

        def boom(name: str):
            raise AssertionError("/health/live 不应该触碰向量库")

        monkeypatch.setattr("knowledge.vector_store.get_vector_store", boom)
        started = time.perf_counter()
        resp = client.get("/health/live")
        elapsed = time.perf_counter() - started

        assert resp.status_code == 200
        assert resp.json() == {"status": "alive"}
        assert elapsed < 1.0, f"存活探针耗时 {elapsed:.2f}s，绝不能加载模型"

    def test_readiness_is_light_by_default(self, client, monkeypatch):
        """默认就绪检查只读文件，不加载 embedding 模型（原实现 28s）。"""

        def boom(name: str):
            raise AssertionError("/health 默认不应打开 collection")

        monkeypatch.setattr("knowledge.vector_store.get_vector_store", boom)
        started = time.perf_counter()
        body = client.get("/health").json()
        elapsed = time.perf_counter() - started

        assert body["status"] == "healthy"
        assert body["checks"]["database"] is True
        assert body["checks"]["chroma"] is True
        assert elapsed < 2.0, f"就绪检查耗时 {elapsed:.2f}s，疑似又加载了模型"

    def test_deep_check_opens_collection(self, client, monkeypatch):
        from fakes import FakeVectorStore

        called = {}

        def fake_get(name: str):
            called["name"] = name
            return FakeVectorStore(7)

        monkeypatch.setattr("knowledge.vector_store.get_vector_store", fake_get)
        body = client.get("/health?deep=true").json()
        assert called["name"] == "obsidian"
        assert body["checks"]["chroma"] is True

    def test_readiness_reports_missing_chroma_file(self, client, monkeypatch):
        body = api._check_chroma_light()
        assert body is True or isinstance(body, str)  # 有库时 True，无库时错误描述


# ═══════════════════════════════════════════
#  错误响应不泄露内部信息
# ═══════════════════════════════════════════


class TestErrorLeak:
    def _mini_app(self) -> FastAPI:
        """独立 app：把生产环境的全局异常处理器挂到会抛异常的路由上。"""
        app = FastAPI()
        app.add_exception_handler(Exception, api.global_exception_handler)

        @app.get("/boom")
        async def boom():
            raise RuntimeError("secret-detail-xyz at C:/private/path.py")

        return app

    def test_exception_detail_not_returned_to_client(self):
        client = TestClient(self._mini_app(), raise_server_exceptions=False)
        resp = client.get("/boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        serialized = json.dumps(body, ensure_ascii=False)
        assert "secret-detail-xyz" not in serialized
        assert "private/path.py" not in serialized
        # 给用户的排障锚点仍然保留
        assert "request_id" in body["error"]["detail"]


# ═══════════════════════════════════════════
#  结构化日志
# ═══════════════════════════════════════════


class TestJsonLogFormatter:
    def test_outputs_parseable_json_with_request_id(self):
        record = logging.LogRecord(
            name="obs.demo", level=logging.INFO, pathname=__file__, lineno=1,
            msg="查询完成: %s", args=("ok",), exc_info=None,
        )
        record.request_id = "rid-123"
        payload = json.loads(JsonFormatter().format(record))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "obs.demo"
        assert payload["message"] == "查询完成: ok"
        assert payload["request_id"] == "rid-123"

    def test_json_log_is_scrubbed(self):
        record = logging.LogRecord(
            name="obs.demo", level=logging.INFO, pathname=__file__, lineno=1,
            msg="api_key=SECRET123", args=(), exc_info=None,
        )
        record.request_id = "-"
        assert "SECRET123" not in JsonFormatter().format(record)
