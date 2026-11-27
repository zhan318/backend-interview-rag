"""Cross-Encoder 精排：对 RRF 融合后的候选集做相关性打分，取最终 top-k。

设计要点：
- **懒加载**：首次调用才加载模型，不拖慢服务启动；
- **优雅降级**：模型缺失/加载失败时退化为「按原顺序截断」，检索功能不受影响；
- **可选阈值**：低于阈值的文档被丢弃，可用于「知识库无相关内容」的判定。

为什么需要它（本项目实测结论）：
RRF 只依据排名位置融合，看不懂内容。实测中「每日疑难总结」这类流水笔记因技术术语
密集、BM25 词面匹配分高，把真正回答问题的专题笔记挤出了 top-3。Cross-Encoder 直接
读 query 与文档内容算相关性，能纠正这类「词面匹配 ≠ 真正相关」的误判。
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from functools import lru_cache

from utils.logger import get_logger

from .config import config

logger = get_logger(__name__)

RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_MAX_LEN = 512

# ═══════════════════════════════════════════
#  过载保护（2026-11 压测驱动）
#  精排是 CPU 密集：单次 20 候选 × 500 字实测约 6.4s，是纯检索(15ms)的 400+ 倍。
#  默认 asyncio 线程池上限约 32，10 并发就把 CPU 打满、30 并发直接卡死。
#  策略：并发上限 + 等待上限，超限立即降级为 RRF 融合顺序（可用性优先）。
# ═══════════════════════════════════════════
_MAX_CONCURRENCY = max(1, config.RERANK_MAX_CONCURRENCY)
_sem = threading.BoundedSemaphore(_MAX_CONCURRENCY)
_pool = ThreadPoolExecutor(max_workers=_MAX_CONCURRENCY, thread_name_prefix="rerank")

_stats_lock = threading.Lock()
_stats = {
    "calls": 0,
    "ok": 0,
    "degraded_busy": 0,
    "degraded_timeout": 0,
    "degraded_error": 0,
    "total_latency_ms": 0.0,
}


def _record(kind: str, latency_s: float = 0.0) -> None:
    with _stats_lock:
        _stats["calls"] += 1
        _stats[kind] += 1
        _stats["total_latency_ms"] += latency_s * 1000


def get_rerank_stats() -> dict:
    """返回精排运行统计（供 /health 或运维排查；测试也用它断言降级是否发生）"""
    with _stats_lock:
        snap = dict(_stats)
    ok = snap["ok"] or 1
    snap["avg_latency_ms"] = round(snap["total_latency_ms"] / ok, 1)
    snap["degraded_total"] = (
        snap["degraded_busy"] + snap["degraded_timeout"] + snap["degraded_error"]
    )
    snap["max_concurrency"] = _MAX_CONCURRENCY
    return snap


def reset_rerank_stats() -> None:
    """清空统计（测试用；不重置信号量，避免掩盖并发泄漏）"""
    with _stats_lock:
        for k in _stats:
            _stats[k] = 0 if k != "total_latency_ms" else 0.0



@lru_cache(maxsize=1)
def _get_model():
    """懒加载 CrossEncoder（进程内只加载一次）"""
    from sentence_transformers import CrossEncoder

    logger.info("加载精排模型: %s", RERANK_MODEL)
    return CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LEN)


def rerank(query: str, docs: list, top_n: int = 3, threshold: float | None = None) -> list:
    """对候选文档精排，返回 top_n 条。

    Args:
        query: 用户查询
        docs: RRF 融合后的候选 Document 列表（建议 20~30 条）
        top_n: 最终返回条数（送 LLM 的上下文数量）
        threshold: 可选分数阈值，低于它的文档被丢弃

    Returns:
        精排后的 Document 列表；模型不可用时返回按原顺序的前 top_n 条。
    """
    if not docs:
        return []
    if len(docs) <= 1:
        return docs[:top_n]

    try:
        model = _get_model()
        scores = model.predict([(query, d.page_content) for d in docs])
    except Exception as e:
        # 模型未安装 / 下载失败 / 显存不足等：降级为原顺序，保证检索可用
        logger.warning("精排模型不可用，降级为原顺序截断: %s", e)
        return docs[:top_n]

    ranked = sorted(zip(docs, scores, strict=True), key=lambda x: -float(x[1]))
    if threshold is not None:
        before = len(ranked)
        ranked = [(d, s) for d, s in ranked if float(s) >= threshold]
        if len(ranked) < before:
            logger.debug(
                "精排阈值 %.3f 过滤掉 %d/%d 条候选", threshold, before - len(ranked), before
            )
        if not ranked:
            logger.info("精排后无候选达到阈值 %.3f，判定为「知识库无相关内容」", threshold)
    return [d for d, _ in ranked[:top_n]]


def _rerank_and_release(query: str, docs: list, top_n: int, threshold: float | None) -> list:
    """在工作线程里执行精排，并保证无论成功失败都归还并发许可。"""
    try:
        return rerank(query, docs, top_n=top_n, threshold=threshold)
    finally:
        _sem.release()


def rerank_guarded(query: str, docs: list, top_n: int = 3,
                   threshold: float | None = None) -> list:
    """带过载保护的精排入口（生产链路统一走这里）。

    三种降级条件，任一命中都返回「RRF 融合顺序」的前 top_n 条：
    1. 并发已满（当前有 RERANK_MAX_CONCURRENCY 个精排在跑）→ 立即降级，不排队；
    2. 等待超过 RERANK_TIMEOUT 秒 → 本请求降级（后台任务仍会跑完，受并发上限约束）；
    3. 任务提交失败 → 降级。

    为什么不排队：CPU 密集任务排队只会把请求延迟越堆越高，
    最终表现为「30 并发全部超时」。过载时快速给一个略差但可用的结果，比卡死好。
    """
    if not docs:
        return []
    if len(docs) <= 1:
        return docs[:top_n]

    if not _sem.acquire(blocking=False):
        _record("degraded_busy")
        logger.warning(
            "精排并发已达上限 %d，降级为 RRF 融合顺序（返回 %d 条）", _MAX_CONCURRENCY, min(top_n, len(docs))
        )
        return docs[:top_n]

    started = time.perf_counter()
    try:
        future = _pool.submit(_rerank_and_release, query, docs, top_n, threshold)
    except Exception as e:
        _sem.release()
        _record("degraded_error")
        logger.warning("精排任务提交失败，降级为 RRF 融合顺序: %s", e)
        return docs[:top_n]

    try:
        ranked = future.result(timeout=config.RERANK_TIMEOUT)
    except FutureTimeoutError:
        _record("degraded_timeout")
        logger.warning(
            "精排超时（>%.1fs），本请求降级为 RRF 融合顺序", config.RERANK_TIMEOUT
        )
        return docs[:top_n]
    except Exception as e:
        _record("degraded_error")
        logger.warning("精排执行异常，降级为 RRF 融合顺序: %s", e)
        return docs[:top_n]

    _record("ok", time.perf_counter() - started)
    return ranked
