"""可靠性与过载保护回归测试（2026-11-27 压测驱动的两个 P0 修复）。

背景（`performance-test/reports/report-2026-11-12.md`）：

1. **LLM 层**：openai SDK 默认 read timeout=600s、max_retries=2。单 key 触发 429 后
   一个请求最坏拖十几分钟，重试把线程池占满 → 真打模式阶梯并发全部 0 完成。
2. **检索层**：Cross-Encoder 精排是 CPU 密集（单次 20 候选 × 500 字 ≈ 6.4s），
   `asyncio.to_thread` 默认线程池上限约 32，10 并发饱和、30 并发卡死。

这里固化四条“契约”，防止修复被后续重构悄悄退回去：
- 配置契约：客户端必须显式带 timeout / max_retries；
- 行为契约：并发超限 → 立即降级，不排队、不抛异常；
- 可用性契约：降级/超时后仍返回 top_k 条（宁可略差，不可卡死）；
- 语义契约：429/超时翻译成 503/504，不把 SDK 原文抛给前端。
"""

import threading
import time

import httpx
import pytest
from langchain_core.documents import Document
from openai import APITimeoutError, RateLimitError

from chatbot.chatbot import DeepSeekAdapter, OpenAICompatibleAdapter
from config import LLM_MAX_RETRIES, LLM_TIMEOUT
from knowledge import hybrid, pipeline, reranker


def _docs(n: int = 5) -> list[Document]:
    return [Document(page_content=f"文档{i}", metadata={"rel_path": f"note{i}.md"}) for i in range(n)]


# ═══════════════════════════════════════════
#  P0-1：LLM 客户端超时与重试
# ═══════════════════════════════════════════

def test_llm_clients_explicitly_set_timeout_and_retries():
    """两个适配器都必须显式覆盖 SDK 默认值（600s / 2 次）。

    这是「配置契约」：只要有人把参数删掉，这条用例立刻红。
    """
    ds = DeepSeekAdapter(api_key="sk-test", base_url="https://example.invalid", model="m")
    compat = OpenAICompatibleAdapter(
        model_name="m", api_key="sk-test", base_url="https://example.invalid"
    )

    for adapter in (ds, compat):
        assert adapter.client.timeout == LLM_TIMEOUT
        assert adapter.client.max_retries == LLM_MAX_RETRIES
        # 异步通道同样要带上，否则 arun 链路又退回默认值
        assert adapter.aclient.timeout == LLM_TIMEOUT
        assert adapter.aclient.max_retries == LLM_MAX_RETRIES


def test_llm_defaults_are_tighter_than_sdk():
    """守住“比 SDK 默认更严”这个意图，防止有人把值调回 600s 还觉得没问题。"""
    from openai._constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT

    # DEFAULT_TIMEOUT 是 httpx.Timeout 对象（read=600），不是数字
    assert float(DEFAULT_TIMEOUT.read) > LLM_TIMEOUT
    assert LLM_MAX_RETRIES <= DEFAULT_MAX_RETRIES


# ═══════════════════════════════════════════
#  P0-2：精排过载保护
# ═══════════════════════════════════════════

def test_rerank_degrades_immediately_when_concurrency_full(monkeypatch):
    """并发已满时必须“立即”降级：不排队、不调用精排、不抛异常。"""
    docs = _docs()
    sem = threading.BoundedSemaphore(1)
    assert sem.acquire(blocking=False)  # 占住唯一许可，模拟已有精排在跑
    monkeypatch.setattr(reranker, "_sem", sem)

    called = []
    monkeypatch.setattr(reranker, "rerank", lambda *a, **k: called.append(1) or [])
    reranker.reset_rerank_stats()

    started = time.perf_counter()
    out = reranker.rerank_guarded("问题", docs, top_n=2)
    elapsed = time.perf_counter() - started

    assert out == docs[:2], "降级结果应为 RRF 融合顺序的前 top_n 条"
    assert called == [], "并发已满时不应再调用精排（否则就是排队，正是要消灭的行为）"
    assert elapsed < 0.05, f"降级必须是快速失败，实际耗时 {elapsed:.3f}s"
    assert reranker.get_rerank_stats()["degraded_busy"] == 1


def test_rerank_degrades_on_timeout(monkeypatch):
    """超过等待上限时本请求降级，且降级结果依然可用（不为空、不抛错）。"""

    def slow_rerank(*args, **kwargs):
        time.sleep(0.2)
        return _docs(1)

    docs = _docs()
    monkeypatch.setattr(reranker, "rerank", slow_rerank)
    monkeypatch.setattr(reranker.config, "RERANK_TIMEOUT", 0.02)
    reranker.reset_rerank_stats()

    out = reranker.rerank_guarded("问题", docs, top_n=3)

    assert out == docs[:3]
    assert reranker.get_rerank_stats()["degraded_timeout"] == 1
    # 等后台任务跑完归还许可（Python 无法中断线程，只能等它自然结束）
    time.sleep(0.3)
    assert reranker._sem.acquire(blocking=False) is True
    reranker._sem.release()


def test_rerank_normal_path_still_reranks(monkeypatch):
    """正常路径不能被保护逻辑吃掉：精排结果必须生效，且许可要归还。"""
    docs = _docs()
    monkeypatch.setattr(reranker, "rerank", lambda q, d, top_n, threshold: list(reversed(d))[:top_n])
    reranker.reset_rerank_stats()

    out = reranker.rerank_guarded("问题", docs, top_n=2)

    assert out == list(reversed(docs))[:2]
    stats = reranker.get_rerank_stats()
    assert stats["ok"] == 1 and stats["degraded_total"] == 0
    # 许可已归还，否则连续请求会被自己饿死
    assert reranker._sem.acquire(blocking=False) is True
    reranker._sem.release()


def test_retrieve_obsidian_uses_guarded_rerank(monkeypatch):
    """生产检索链路必须走带保护的入口。

    防的是什么：有人图省事把 pipeline 改回直接调 `rerank()`，
    单测全绿、压测再次卡死。这条用例把“接线”本身也变成契约。
    """
    docs = _docs()
    monkeypatch.setattr(pipeline.config, "RERANK_ENABLED", True)
    monkeypatch.setattr(pipeline, "get_vector_store", lambda name: _FakeStore(docs))
    monkeypatch.setattr(hybrid, "get_bm25_retriever", lambda store: _FakeStore(docs))
    monkeypatch.setattr(hybrid, "fuse_documents", lambda dense, bm25, top_n: docs)

    called = {}

    def fake_guarded(query, candidates, top_n, threshold):
        called["args"] = (query, len(candidates), top_n, threshold)
        return candidates[:top_n]

    monkeypatch.setattr(reranker, "rerank_guarded", fake_guarded)
    dense, bm25, merged = pipeline.retrieve_obsidian("问题", k=2)

    assert called, "retrieve_obsidian 必须调用 rerank_guarded"
    assert called["args"][2] == 2
    assert merged == docs[:2]
    assert len(dense) == len(docs) and len(bm25) == len(docs)


# ═══════════════════════════════════════════
#  语义契约：429 / 超时 → 503 / 504
# ═══════════════════════════════════════════

class _RateLimitedAgent:
    async def arun(self, *args, **kwargs):
        raise _rate_limit_error()


class _TimeoutAgent:
    async def arun(self, *args, **kwargs):
        raise _timeout_error()


def _rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("Error code: 429 - rate limit exceeded", response=response, body=None)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://example.invalid/v1/chat"))


@pytest.mark.parametrize(
    ("agent", "status", "keyword"),
    [
        (_RateLimitedAgent(), 503, "稍后重试"),
        (_TimeoutAgent(), 504, "超时"),
    ],
)
def test_chat_maps_llm_errors_to_retryable_status(client, user, monkeypatch, agent, status, keyword):
    """限流与超时必须是可重试语义，且不把 SDK 原始报错透给前端。"""
    import api

    monkeypatch.setattr(api, "get_agent", lambda session_id: agent)
    monkeypatch.setattr("chatbot.chatbot.chat", lambda *a, **k: None)

    resp = client.post("/chat", json={"message": "在吗"}, headers=user["headers"])

    assert resp.status_code == status
    message = resp.json()["error"]["message"]
    assert keyword in message
    # 不能泄漏 SDK 原文（含厂商域名 / Error code / 堆栈）
    assert "Error code" not in message
    assert "example.invalid" not in message


class _FakeStore:
    """最小向量库替身：只提供 retrieve_obsidian 用到的 search()"""

    def __init__(self, docs):
        self._docs = docs

    def search(self, query, k):
        return self._docs[:k]
