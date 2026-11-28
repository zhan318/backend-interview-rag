"""Prometheus 指标注册表（RED 方法 + 关键链路埋点）。

指标设计原则
============
- **RED**：Rate（请求数）、Errors（按 status 分）、Duration（耗时直方图）。
- **低基数**：HTTP 的 path 标签必须用 **路由模板**（`/sessions/{session_id}`），
  不能直接用 URL —— 否则每个 session_id 生成一条时间序列，几千用户就把
  Prometheus 打爆（cardinality explosion）。
- **分层定位**：HTTP 一层看「哪个接口慢」，LLM / 检索 / 工具各自一层看「慢在哪」。

注意：uvicorn 单进程部署时这是准确的。若将来改 `--workers N`，
需要设置 `PROMETHEUS_MULTIPROC_DIR` 并换 MultiProcessCollector，否则各 worker
的计数互相覆盖（本项目的 uvicorn 启动参数未开多 worker）。
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from utils.logger import get_logger
from utils.observability import get_request_id

logger = get_logger(__name__)

# 秒级耗时桶：从 5ms（纯内存）到 120s（LLM 长尾），覆盖本项目的全部链路
_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)

# ═══════════════════════════════════════════
#  HTTP 层（RED）
# ═══════════════════════════════════════════

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "HTTP 请求总数",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求耗时",
    ["method", "path"],
    buckets=_DURATION_BUCKETS,
)

HTTP_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "当前正在处理的 HTTP 请求数",
    ["method"],
)

# ═══════════════════════════════════════════
#  LLM 层
# ═══════════════════════════════════════════

LLM_CALLS_TOTAL = Counter(
    "llm_calls_total",
    "LLM 调用次数",
    ["provider", "model", "status"],  # status: ok / error
)

LLM_CALL_DURATION = Histogram(
    "llm_call_duration_seconds",
    "LLM 调用耗时（流式为整个流消费完的总时长）",
    ["provider", "stream"],
    buckets=_DURATION_BUCKETS,
)

# ═══════════════════════════════════════════
#  检索层（dense / BM25 / RRF / 精排）
# ═══════════════════════════════════════════

RETRIEVAL_STAGE_DURATION = Histogram(
    "retrieval_stage_duration_seconds",
    "检索各阶段耗时",
    ["stage"],  # dense / bm25 / fuse / rerank
    buckets=_DURATION_BUCKETS,
)

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "RAG 检索结果分布",
    ["result"],  # hit / empty（empty = 精排阈值过滤后无相关内容）
)

RERANK_DEGRADED_TOTAL = Counter(
    "rerank_degraded_total",
    "精排降级次数（过载保护）",
    ["reason"],  # busy / timeout / error
)

# ═══════════════════════════════════════════
#  Agent 工具层
# ═══════════════════════════════════════════

AGENT_TOOL_CALLS_TOTAL = Counter(
    "agent_tool_calls_total",
    "Agent 工具调用次数",
    ["tool", "status"],  # ok / error / unknown
)

AGENT_TOOL_DURATION = Histogram(
    "agent_tool_duration_seconds",
    "Agent 工具执行耗时",
    ["tool"],
    buckets=_DURATION_BUCKETS,
)

# ═══════════════════════════════════════════
#  路径归一化
# ═══════════════════════════════════════════

# 兜底规则：纯数字 / 长 hex / UUID 段落 → {id}
_DYNAMIC_SEGMENT = re.compile(
    r"/(?:\d+|[0-9a-fA-F]{8,}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F-]{12,})(?=/|$)"
)

# 未匹配到路由的请求统一归档，避免扫描器路径（/wp-admin.php 等）制造海量 label
UNMATCHED_PATH = "__unmatched__"


def normalize_path(path: str) -> str:
    """把 URL 里的动态段替换成 {id}（无路由模板时的兜底）。"""
    return _DYNAMIC_SEGMENT.sub("/{id}", path or "/")


def resolve_path_label(request) -> str:
    """取指标用的 path 标签：优先用 FastAPI 路由模板。

    `request.scope["route"]` 只在路由匹配完成后才有值，所以必须在
    `call_next()` 之后调用。未匹配（404）时统一记 `__unmatched__`。
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if template:
        return template
    return UNMATCHED_PATH


def observe_http(request, status_code: int, duration_s: float) -> str:
    """记录一次 HTTP 请求的 RED 指标，返回用于日志的 path 标签。"""
    path = resolve_path_label(request)
    method = request.method
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status_code)).inc()
    HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration_s)
    return path


@contextmanager
def track_duration(histogram: Histogram, **labels) -> Iterator[None]:
    """给任意代码段计时并写入直方图（异常也记录，便于发现慢失败）。"""
    started = time.perf_counter()
    try:
        yield
    finally:
        histogram.labels(**labels).observe(time.perf_counter() - started)


def render_metrics() -> tuple[bytes, str]:
    """返回 (Prometheus 文本, Content-Type)。"""
    return generate_latest(), CONTENT_TYPE_LATEST


def describe_metric_health() -> dict:
    """给 /health 用的轻量指标快照（不触发任何模型加载）。"""
    return {"request_id": get_request_id()}
