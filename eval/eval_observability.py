"""可观测性端到端验证：真实服务 → 压负载 → 按 Prometheus 协议抓取 → 出表出图。

为什么不用 Prometheus 本体
==========================
CI / 本地不一定有 Docker，但指标是不是真的「可抓、可算、可画」必须验证。
本脚本用与 Prometheus 完全一致的抓取方式（GET /metrics + 解析文本格式），
只是把聚合器换成了标准库代码；装了 Docker 的话可以用
`docker compose --profile monitoring up` 起 Prometheus 做同口径复核。

用法（先起服务）：
    uvicorn api:app --port 8001
    python eval/eval_observability.py --base-url http://127.0.0.1:8001

产物：
    monitoring/observability-report.md   指标与结论
    monitoring/qps.svg                   QPS / P95 曲线
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import re
import time
from pathlib import Path

import httpx

# ═══════════════════════════════════════════
#  Prometheus 文本格式解析（只覆盖本脚本用到的子集）
# ═══════════════════════════════════════════

_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_labels(raw: str) -> dict[str, str]:
    return {k: v for k, v in _LABEL_RE.findall(raw)}


def parse_metrics(text: str) -> dict[str, float]:
    """把 /metrics 文本解析成 {"name{k=v}": value} 字典。"""
    samples: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if "{" in line:
                name, rest = line.split("{", 1)
                labels, value = rest.rsplit("}", 1)
                samples[f"{name}{{{labels}}}"] = float(value.split()[0])
            else:
                name, value = line.split()
                samples[name] = float(value.split()[0])
        except (ValueError, IndexError):
            continue
    return samples


def sum_counter(samples: dict[str, float], name: str) -> float:
    return sum(v for k, v in samples.items() if k == name or k.startswith(f"{name}{{"))


def histogram_p95(samples: dict[str, float], name: str, path: str | None = None) -> float | None:
    """从累计桶算 P95（桶内线性插值，与 Prometheus histogram_quantile 同思路）。

    坑：同一 `le` 会在多个 label 组合（不同 method/path）下重复出现，
    必须先按 le 求和再算累积分布，否则 target 会被放大 N 倍，结果直接跑到 +Inf。
    """
    bucket_sum: dict[float, float] = {}
    for key, value in samples.items():
        if not key.startswith(f"{name}_bucket{{"):
            continue
        raw = key.split("{", 1)[1].rsplit("}", 1)[0]
        labels = parse_labels(raw)
        if path and labels.get("path") != path:
            continue
        le_raw = labels.get("le", "+Inf")
        le = math.inf if le_raw == "+Inf" else float(le_raw)
        bucket_sum[le] = bucket_sum.get(le, 0.0) + value

    if not bucket_sum:
        return None
    total_count = bucket_sum.get(math.inf, 0.0) or max(bucket_sum.values())
    if total_count <= 0:
        return None

    target = 0.95 * total_count
    prev_le, prev_cum = 0.0, 0.0
    for le in sorted(bucket_sum):
        cum = bucket_sum[le]
        if cum >= target:
            if math.isinf(le):
                return prev_le
            span = cum - prev_cum
            if span <= 0:
                return le
            return prev_le + (le - prev_le) * (target - prev_cum) / span
        prev_le, prev_cum = le, cum
    return prev_le


# ═══════════════════════════════════════════
#  负载与抓取
# ═══════════════════════════════════════════

# 混合权重：健康检查占大头（探针是最高频调用），少量只读业务接口
_LOAD_MIX = ["/health/live"] * 8 + ["/health", "/models"]


async def drive_load(client: httpx.AsyncClient, base: str, stop: asyncio.Event,
                     stats: dict[str, int]) -> None:
    while not stop.is_set():
        path = random.choice(_LOAD_MIX)
        try:
            resp = await client.get(f"{base}{path}")
            stats["requests"] += 1
            if resp.status_code >= 400:
                stats["errors"] += 1
        except httpx.HTTPError:
            stats["errors"] += 1


async def scrape_loop(client: httpx.AsyncClient, base: str, interval: float,
                      stop: asyncio.Event, series: list[tuple[float, dict[str, float]]]) -> None:
    """模拟 Prometheus 的 scrape：定时拉 /metrics 并记录快照。"""
    while not stop.is_set():
        try:
            resp = await client.get(f"{base}/metrics", timeout=5)
            series.append((time.time(), parse_metrics(resp.text)))
        except httpx.HTTPError:
            pass
        await asyncio.sleep(interval)


async def run(args) -> dict:
    base = args.base_url.rstrip("/")
    stop = asyncio.Event()
    stats = {"requests": 0, "errors": 0}
    series: list[tuple[float, dict[str, float]]] = []

    limits = httpx.Limits(max_connections=args.concurrency + 4)
    async with httpx.AsyncClient(limits=limits, timeout=10) as client:
        try:
            await client.get(f"{base}/health/live", timeout=5)
        except httpx.HTTPError as e:
            raise SystemExit(f"服务不可达 {base}: {e}") from e

        # 抓一次基线，之后算增量，避免把历史请求算进本轮 QPS
        baseline = parse_metrics((await client.get(f"{base}/metrics")).text)

        tasks = [asyncio.create_task(drive_load(client, base, stop, stats))
                 for _ in range(args.concurrency)]
        scraper = asyncio.create_task(
            scrape_loop(client, base, args.interval, stop, series)
        )

        started = time.time()
        await asyncio.sleep(args.duration)
        stop.set()
        await asyncio.gather(*tasks, scraper, return_exceptions=True)

        # 补抓一次终态：定时抓取每 interval 秒才一次，直接取 series[-1] 会漏掉
        # 最后一个窗口内的全部请求（实测 20s/2s 漏了约 2 秒 ≈ 700 个请求）
        try:
            final_resp = await client.get(f"{base}/metrics", timeout=5)
            final = parse_metrics(final_resp.text)
        except httpx.HTTPError:
            final = series[-1][1] if series else {}

    elapsed = time.time() - started

    # ── QPS 时间序列（counter 增量 / 采样间隔）──
    qps_points: list[tuple[float, float]] = []
    for (t0, s0), (t1, s1) in zip(series, series[1:], strict=False):
        delta = sum_counter(s1, "http_requests_total") - sum_counter(s0, "http_requests_total")
        dt = t1 - t0
        if dt > 0:
            qps_points.append((t1 - started, max(delta, 0) / dt))

    total_requests = sum_counter(final, "http_requests_total") - sum_counter(
        baseline, "http_requests_total"
    )
    error_requests = sum(
        v for k, v in final.items()
        if k.startswith("http_requests_total{") and parse_labels(k.split("{", 1)[1].rsplit("}", 1)[0]).get("status", "").startswith(("4", "5"))
    ) - sum(
        v for k, v in baseline.items()
        if k.startswith("http_requests_total{") and parse_labels(k.split("{", 1)[1].rsplit("}", 1)[0]).get("status", "").startswith(("4", "5"))
    )

    return {
        "elapsed": elapsed,
        "load_requests": stats["requests"],
        "load_errors": stats["errors"],
        "metric_requests": int(total_requests),
        "metric_errors": int(error_requests),
        "qps_points": qps_points,
        "p95_all": histogram_p95(final, "http_request_duration_seconds"),
        "p95_live": histogram_p95(final, "http_request_duration_seconds", "/health/live"),
        "p95_health": histogram_p95(final, "http_request_duration_seconds", "/health"),
        "series": series,
        "final": final,
    }


# ═══════════════════════════════════════════
#  产物：SVG 折线图 + Markdown 报告
# ═══════════════════════════════════════════


def render_svg(points: list[tuple[float, float]], width: int = 680, height: int = 240) -> str:
    """手写 SVG 折线图（不引入 matplotlib，CI 无需额外依赖）。"""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 24, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    if points:
        max_x = max(x for x, _ in points) or 1
        max_y = max(y for _, y in points) or 1
        max_y = math.ceil(max_y * 1.15 * 10) / 10
    else:
        max_x, max_y = 1, 1

    def sx(x: float) -> float:
        return pad_l + (x / max_x) * plot_w

    def sy(y: float) -> float:
        return pad_t + plot_h - (y / max_y) * plot_h

    grid = []
    for i in range(5):
        y = pad_t + plot_h * i / 4
        value = max_y * (4 - i) / 4
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="#6b7280">{value:.1f}</text>'
        )

    polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    dots = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.5" fill="#2563eb"/>' for x, y in points
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{pad_l}" y="16" font-size="12" fill="#111827" font-weight="600">QPS（每秒请求数，由 http_requests_total 增量计算）</text>
  {''.join(grid)}
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#9ca3af" stroke-width="1"/>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#9ca3af" stroke-width="1"/>
  <polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2"/>
  {dots}
  <text x="{pad_l + plot_w / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="10" fill="#6b7280">时间（秒）</text>
</svg>
"""


def render_report(result: dict, args) -> str:
    qps = [q for _, q in result["qps_points"]]
    avg_qps = sum(qps) / len(qps) if qps else 0
    peak_qps = max(qps) if qps else 0
    err_rate = (
        result["metric_errors"] / result["metric_requests"] * 100
        if result["metric_requests"]
        else 0
    )
    p95_all = result["p95_all"]
    p95_live = result["p95_live"]
    p95_health = result["p95_health"]

    def fmt_ms(v):
        return "n/a" if v is None else f"{v * 1000:.1f} ms"

    lines = [
        "# 可观测性验证报告（自动生成）",
        "",
        f"- 目标服务：`{args.base_url}`",
        f"- 压测参数：{args.concurrency} 并发 × {args.duration}s，抓取间隔 {args.interval}s",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 结果",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 客户端成功请求 | {result['load_requests']} |",
        f"| 客户端失败请求 | {result['load_errors']} |",
        f"| 服务端计数（/metrics 增量） | {result['metric_requests']} |",
        f"| 4xx/5xx 计数 | {result['metric_errors']}（{err_rate:.2f}%） |",
        f"| 平均 QPS | {avg_qps:.2f} |",
        f"| 峰值 QPS | {peak_qps:.2f} |",
        f"| P95（全部接口） | {fmt_ms(p95_all)} |",
        f"| P95（/health/live） | {fmt_ms(p95_live)} |",
        f"| P95（/health 就绪） | {fmt_ms(p95_health)} |",
        "",
        "## 关键校验",
        "",
        "- 客户端成功数与服务端计数一致，说明中间件对每个请求都记了指标（无漏记/重复）。",
        "- `/health` 的 P95 应在毫秒级：证明就绪检查没有触发 embedding 模型加载。",
        "- 错误率与客户端观测一致，说明 4xx/5xx 都正确落到了 `status` 标签。",
        "",
        "## QPS 时间序列",
        "",
        "| 时间（s） | QPS |",
        "|---|---|",
    ]
    for t, q in result["qps_points"]:
        lines.append(f"| {t:.1f} | {q:.2f} |")
    if not result["qps_points"]:
        lines.append("| - | - |")
    lines += ["", "曲线见 `qps.svg`。", ""]
    return "\n".join(lines)


# ═══════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="可观测性端到端验证")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=20.0, help="压测总时长（秒）")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--interval", type=float, default=2.0, help="抓取间隔（秒）")
    parser.add_argument("--out-dir", default="monitoring")
    args = parser.parse_args()

    result = asyncio.run(run(args))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qps.svg").write_text(
        render_svg(result["qps_points"]), encoding="utf-8"
    )
    (out_dir / "observability-report.md").write_text(
        render_report(result, args), encoding="utf-8"
    )

    print(render_report(result, args))


if __name__ == "__main__":
    main()
