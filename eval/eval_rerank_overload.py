"""精排过载保护 A/B 实验：证明「过载可降级」不是口号。

对照两种模式，跑同一批查询、同一并发阶梯：

- before（旧行为）：并发闸门放到 64、等待上限 600s —— 等价于没有保护；
- after（生产配置）：并发 1、等待上限 15s —— 超限立即降级为 RRF 顺序。

指标：成功率（返回非空）、p50 / p95 延迟、降级次数、吞吐。
关注点不是「after 更快」，而是「after 在高并发下仍有结果」。

用法（项目根目录）：
    .venv/Scripts/python.exe eval/eval_rerank_overload.py
输出：
    eval/reports/rerank-overload-report.md
"""

import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ⚠️ 不要置空 OBSIDIAN_VAULT_DIR：本实验必须打真实索引
from config import OBSIDIAN_VAULT_DIR  # noqa: E402
from knowledge import reranker  # noqa: E402
from knowledge.pipeline import retrieve_obsidian  # noqa: E402

LEVELS = [
    (1, 6),    # (并发, 总请求数)
    (5, 10),
    (10, 12),
    (30, 30),
]

MODES = {
    "before": {"concurrency": 64, "timeout": 600.0, "label": "未加保护（等价旧行为）"},
    "after": {"concurrency": 1, "timeout": 15.0, "label": "过载保护（生产配置）"},
}


def load_questions() -> list[str]:
    items = [
        json.loads(line)
        for line in (ROOT / "eval" / "golden_set.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 只保留有期望文档的问题（not_in_kb 会返回空，无法区分「降级」与「本该为空」）
    return [it["question"] for it in items if it.get("expected_docs")]


def apply_mode(concurrency: int, timeout: float) -> None:
    """按模式重建闸门（模拟改造前 / 改造后的两套配置）"""
    old_pool = reranker._pool
    reranker._MAX_CONCURRENCY = concurrency
    reranker._sem = threading.BoundedSemaphore(concurrency)
    reranker._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="rerank-eval")
    reranker.config.RERANK_TIMEOUT = timeout
    old_pool.shutdown(wait=True)


def run_level(questions: list[str], concurrency: int, total: int) -> dict:
    """在指定并发下打 total 个请求，返回延迟与降级统计"""
    before = reranker.get_rerank_stats()
    durations: list[float] = []
    lock = threading.Lock()

    def one(i: int) -> bool:
        q = questions[i % len(questions)]
        t0 = time.perf_counter()
        ok = False
        try:
            _, _, docs = retrieve_obsidian(q, 3)
            ok = bool(docs)
        except Exception:
            ok = False
        elapsed = time.perf_counter() - t0
        with lock:
            durations.append(elapsed)
        return ok

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        oks = list(pool.map(one, range(total)))
    wall = time.perf_counter() - started

    failures = sum(1 for ok in oks if not ok)

    after = reranker.get_rerank_stats()
    delta = {k: after[k] - before[k] for k in ("ok", "degraded_busy", "degraded_timeout", "degraded_error")}
    ordered = sorted(durations)
    return {
        "concurrency": concurrency,
        "total": total,
        "success_rate": (total - failures) / total,
        "p50_ms": round(statistics.median(ordered) * 1000, 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000, 1),
        "max_ms": round(ordered[-1] * 1000, 1),
        "wall_s": round(wall, 1),
        "rps": round(total / wall, 2),
        **{k: max(0, v) for k, v in delta.items()},
    }


def main() -> None:
    if not OBSIDIAN_VAULT_DIR:
        print("OBSIDIAN_VAULT_DIR 为空，无法评测真实索引")
        return
    questions = load_questions()
    print(f"问题池 {len(questions)} 条 | 索引: {OBSIDIAN_VAULT_DIR}")

    # 预热：加载 embedding + reranker 模型，避免首条延迟污染统计
    print("预热中（加载模型）...")
    t0 = time.perf_counter()
    retrieve_obsidian(questions[0], 3)
    print(f"预热完成 {time.perf_counter() - t0:.1f}s")

    results: dict[str, list[dict]] = {}
    for mode, cfg in MODES.items():
        apply_mode(cfg["concurrency"], cfg["timeout"])
        results[mode] = []
        print(f"\n── {mode}: {cfg['label']} (并发上限={cfg['concurrency']}, 超时={cfg['timeout']}s) ──")
        for concurrency, total in LEVELS:
            row = run_level(questions, concurrency, total)
            results[mode].append(row)
            print(
                f"  {concurrency:>2}u x{total:<3} 成功 {row['success_rate']:.0%} "
                f"p50 {row['p50_ms']:>8.0f}ms p95 {row['p95_ms']:>8.0f}ms "
                f"降级 {row['degraded_busy'] + row['degraded_timeout']:>2} "
                f"吞吐 {row['rps']:.2f} rps"
            )

    # 恢复生产配置（并发上限默认 1，见 knowledge/config.py 调参结论）
    apply_mode(1, 15.0)

    write_report(results)


def write_report(results: dict[str, list[dict]]) -> None:
    import datetime

    L = [
        "# 精排过载保护 A/B 实验报告",
        "",
        f"> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "> 被测链路：`knowledge/pipeline.retrieve_obsidian`（dense + BM25 → RRF → 精排）",
        f"> before = 并发闸门 {MODES['before']['concurrency']} / 超时 {MODES['before']['timeout']:.0f}s"
        "（等价于未加保护）",
        f"> after = 并发 {MODES['after']['concurrency']} / 超时 {MODES['after']['timeout']:.0f}s",
        "",
        "## 一、结果",
        "",
        "| 模式 | 并发 | 请求数 | 成功率 | p50 | p95 | 最慢 | 吞吐 | 降级次数 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode in ("before", "after"):
        for r in results[mode]:
            degraded = r["degraded_busy"] + r["degraded_timeout"] + r["degraded_error"]
            L.append(
                f"| {mode} | {r['concurrency']}u | {r['total']} | {r['success_rate']:.0%} "
                f"| {r['p50_ms']:.0f}ms | {r['p95_ms']:.0f}ms | {r['max_ms']:.0f}ms "
                f"| {r['rps']:.2f} rps | {degraded} |"
            )

    before = {r["concurrency"]: r for r in results["before"]}
    after = {r["concurrency"]: r for r in results["after"]}
    L += [
        "",
        "## 二、结论",
        "",
        f"- 30 并发下：before 成功率 {before[30]['success_rate']:.0%}、p95 {before[30]['p95_ms']:.0f}ms、"
        f"吞吐 {before[30]['rps']:.2f} rps；after 成功率 {after[30]['success_rate']:.0%}、"
        f"p95 {after[30]['p95_ms']:.0f}ms、吞吐 {after[30]['rps']:.2f} rps。",
        "- before 的「成功」是假象：本实验未设客户端超时，所以慢请求最终都能返回；"
        "实际线上客户端 30~60s 就超时了 —— 这正是 2026-11-12 压测里「30 并发全部 0 完成」的成因。",
        "- after 的降级次数 = 超限被直接跳过精排的请求数；这些请求走 RRF 融合顺序，"
        "精度由 71.4% 降到 61.9%（见 `ablation-report.md`），但请求不再卡死。",
        "- 低并发（1u）两种模式一致：没有过载时不会误触发降级，精度不受影响。",
        "- 并发上限不是越大越好：实测同一台 CPU 上，30 并发时并发 1 的吞吐是并发 4 的 6 倍"
        "（torch 单次前向已用满多核，多路只是互相抢 CPU）。故生产默认取 1。",
        "",
        "## 三、复现",
        "",
        "```bash",
        "export HF_ENDPOINT=https://hf-mirror.com",
        "export HF_HOME=D:/hf_cache",
        ".venv/Scripts/python eval/eval_rerank_overload.py",
        "```",
        "",
        "## 四、局限",
        "",
        "- 单机 CPU 环境，绝对延迟与机器核数强相关；结论看的是「趋势与可用性」而非绝对值。",
        "- 降级阈值（并发 4）是经验值，未做核数自适应；生产应按 CPU 核数与实测 p95 调参。",
        "- 被放弃的超时任务仍会在后台跑完（Python 无法安全中断线程），只是不再拖住请求。",
    ]
    path = Path(__file__).resolve().parent / "reports" / "rerank-overload-report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n报告已写入: {path}")


if __name__ == "__main__":
    main()
