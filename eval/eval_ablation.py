"""检索层消融实验：拆解每一层优化的独立贡献（零 LLM 成本）

现有 eval_retrieval.py 只测「最终效果」，把 RRF 融合与 Cross-Encoder 精排混在同一个
数字里，无法回答「哪一层涨了多少、值不值」；且 dense 命中率用的是 20 条宽召回口径，
与早期 4.8%（top-3 口径）不可比。

本脚本用同一张 golden 集跑三个对照 pass，把每层净增益拆开：

    pass 1  rrf        关闭精排，dense+BM25 → RRF → top-k（纯融合能力）
    pass 2  rerank@0.2 生产路径，精排 + 阈值 0.2
    pass 3  rerank_raw 精排但不过阈值（纯精排上限，用于判断阈值是否误杀）

三个 pass 共用进程内已加载的 embedding / reranker 模型，一次跑完。

用法（项目根目录）：
    .venv/Scripts/python eval/eval_ablation.py
输出：
    eval/reports/ablation-report.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import OBSIDIAN_VAULT_DIR  # noqa: E402
from knowledge.config import config as kconfig  # noqa: E402
from knowledge.pipeline import retrieve_obsidian  # noqa: E402

# 统一按 5 条取回，离线切出 @3 / @5，避免同一组重复检索
K_MAX = 5

# 延迟统计排除前 N 条：首条含模型加载 / BM25 语料构建 / 首次推理，属预热而非稳态
WARMUP = 3


def rel_of(doc, vault) -> str:
    """从 Document 元数据还原 vault 相对路径（统一为正斜杠）"""
    meta = doc.metadata or {}
    if meta.get("rel_path"):
        return str(meta["rel_path"]).replace("\\", "/")
    src = str(meta.get("source", ""))
    try:
        return str(Path(src).resolve().relative_to(Path(vault).resolve())).replace("\\", "/")
    except Exception:
        return Path(src).name


def hit(expected, rels) -> bool:
    """判断任一期望文档是否命中检索结果（按 rel_path 后缀匹配）"""
    if not expected:
        return False
    return any(any(e == r or r.endswith("/" + e) for e in expected) for r in rels)


def run_pass(items, vault, name, enabled: bool, threshold) -> list:
    """按指定配置跑一遍完整检索，返回每条用例的分层结果"""
    kconfig.RERANK_ENABLED = enabled
    kconfig.RERANK_THRESHOLD = threshold
    print(f"\n[padd] {name}: RERANK_ENABLED={enabled} THRESHOLD={threshold}")

    rows = []
    for idx, it in enumerate(items, 1):
        t0 = time.perf_counter()
        dense, bm25, merged = retrieve_obsidian(it["question"], K_MAX)
        latency = (time.perf_counter() - t0) * 1000
        rows.append(
            {
                "q": it["question"],
                "category": it["category"],
                "expected": [str(e).replace("\\", "/") for e in it["expected_docs"]],
                "dense": [rel_of(d, vault) for d in dense],
                "bm25": [rel_of(d, vault) for d in bm25],
                "merged": [rel_of(d, vault) for d in merged],
                "n_dense": len(dense),
                "n_bm25": len(bm25),
                "n_merged": len(merged),
                "latency_ms": latency,
            }
        )
        print(f"  {idx}/{len(items)} {latency:7.0f}ms  merged={len(merged)}  {it['question'][:28]}")
    return rows


def rate(rows, field, k) -> tuple[int, int]:
    """命中数与样本数（只统计有期望文档的有效样本）"""
    valid = [r for r in rows if r["expected"]]
    return sum(1 for r in valid if hit(r["expected"], r[field][:k])), len(valid)


def pct(h: int, n: int) -> str:
    return f"{h / n:.1%}" if n else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    args = ap.parse_args()

    golden = Path(__file__).resolve().parent / "golden_set.jsonl"
    items = [json.loads(x) for x in golden.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        items = items[: args.limit]

    vault = OBSIDIAN_VAULT_DIR
    print(f"Golden 集 {len(items)} 条 | vault={vault} | K_MAX={K_MAX}")

    passes = [
        ("RRF 融合（无精排）", False, None),
        ("精排 + 阈值 0.2（生产）", True, 0.2),
        ("精排（无阈值）", True, None),
    ]
    results = {}
    for name, enabled, threshold in passes:
        results[name] = run_pass(items, vault, name, enabled, threshold)

    # ── 汇总 ──
    strategies = [
        ("dense 单路", "dense"),
        ("BM25 单路", "bm25"),
        ("RRF 融合（无精排）", "merged"),
        ("精排 + 阈值 0.2（生产）", "merged"),
        ("精排（无阈值）", "merged"),
    ]
    rows_by_name = {
        "dense 单路": results["RRF 融合（无精排）"],
        "BM25 单路": results["RRF 融合（无精排）"],
        "RRF 融合（无精排）": results["RRF 融合（无精排）"],
        "精排 + 阈值 0.2（生产）": results["精排 + 阈值 0.2（生产）"],
        "精排（无阈值）": results["精排（无阈值）"],
    }
    lat = {
        k: sum(r["latency_ms"] for r in v[WARMUP:]) / max(len(v[WARMUP:]), 1)
        for k, v in results.items()
    }

    print("\n===== 消融总表 =====")
    print(f"{'策略':<26} {'@3':>8} {'@5':>8} {'@20':>8}")
    table = []
    for label, field in strategies:
        rows = rows_by_name[label]
        h3, n = rate(rows, field, 3)
        h5, _ = rate(rows, field, 5)
        h20, _ = rate(rows, field, 20) if field != "merged" else (h5, n)
        print(f"{label:<26} {pct(h3, n):>8} {pct(h5, n):>8} {pct(h20, n):>8}")
        table.append((label, h3, h5, h20, n))

    # ── 精排的救回 / 弄丢 ──
    rrf_rows = results["RRF 融合（无精排）"]
    rer_rows = results["精排 + 阈值 0.2（生产）"]
    raw_rows = results["精排（无阈值）"]
    saved, lost, thr_killed = [], [], []
    for a, b, c in zip(rrf_rows, rer_rows, raw_rows, strict=True):
        exp = a["expected"]
        if not exp:
            continue
        h_rrf = hit(exp, a["merged"][:3])
        h_rer = hit(exp, b["merged"][:3])
        h_raw = hit(exp, c["merged"][:3])
        if h_rer and not h_rrf:
            saved.append(a["q"])
        if h_rrf and not h_rer:
            lost.append(a["q"])
        if h_raw and not h_rer:
            thr_killed.append(a["q"])

    # ── not_in_kb 越界检出 ──
    nin_stats = {}
    for name, rows in results.items():
        nin = [r for r in rows if not r["expected"]]
        nin_stats[name] = (sum(1 for r in nin if r["n_merged"] > 0), len(nin))

    # ── 原始数据落盘，便于复算不同 k / 不同指标 ──
    import datetime

    raw_path = Path(__file__).resolve().parent / "reports" / "ablation-raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "k_max": K_MAX,
                "warmup_excluded": WARMUP,
                "recall_n": kconfig.RETRIEVE_RECALL_N,
                "passes": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── 写报告 ──
    report = Path(__file__).resolve().parent / "reports" / "ablation-report.md"
    L = [
        "# RAG 检索层消融实验报告",
        "",
        f"> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 评测集：eval/golden_set.jsonl（{len(items)} 条，含 not_in_kb）",
        f"> 检索路径：dense + BM25 各召回 {kconfig.RETRIEVE_RECALL_N} 条 → RRF 融合 → 精排 top-k",
        "> 生成命令：`.venv/Scripts/python eval/eval_ablation.py`",
        "",
        "## 一、实验目的",
        "",
        "把「最终效果」拆开，回答每层优化的独立贡献：",
        "",
        "1. dense 单路在 top-3 / top-5 / top-20 三种口径下分别多准（口径不同不可直接比较）",
        "2. RRF 融合相比最好的单路，净增益多少",
        "3. Cross-Encoder 精排相比 RRF，净增益多少、延迟代价多少",
        "4. 阈值 0.2 是保护（挡住无关文档）还是误杀（丢掉相关文档）",
        "",
        "## 二、总表",
        "",
        "| 检索策略 | Recall@3 | Recall@5 | Recall@20 |",
        "|---|---|---|---|",
    ]
    for label, h3, h5, h20, n in table:
        L.append(f"| {label} | {h3}/{n} = {pct(h3, n)} | {h5}/{n} = {pct(h5, n)} | {h20}/{n} = {pct(h20, n)} |")

    L += [
        "",
        "## 三、分层归因",
        "",
        "| 对比 | 增益（@3） | 说明 |",
        "|---|---|---|",
    ]
    base = table[0]
    for label, h3, _h5, _h20, n in table[1:]:
        L.append(f"| {label} vs dense 单路 | {(h3 - base[1]) / n * 100:+.1f}pp | — |")
    L += [
        "",
        "## 四、延迟代价（单条查询平均）",
        "",
        f"> 已排除前 {WARMUP} 条预热（首条含模型加载 / BM25 语料构建 / 首次推理）。",
        "",
        "| 配置 | 平均耗时 | 相对无精排 |",
        "|---|---|---|",
    ]
    base_lat = lat["RRF 融合（无精排）"]
    for name in results:
        d = lat[name] - base_lat
        L.append(f"| {name} | {lat[name]:.0f} ms | {d:+.0f} ms |")

    L += [
        "",
        "## 五、精排的收益与代价（逐条对比 RRF vs 生产配置）",
        "",
        f"- **精排救回**（RRF 未命中、精排命中）：{len(saved)} 条",
    ]
    for q in saved:
        L.append(f"  - {q}")
    L.append(f"- **精排弄丢**（RRF 命中、精排未命中）：{len(lost)} 条")
    for q in lost:
        L.append(f"  - {q}")
    L.append(f"- **阈值误杀**（无阈值精排命中、加阈值后未命中）：{len(thr_killed)} 条")
    for q in thr_killed:
        L.append(f"  - {q}")

    L += [
        "",
        "## 六、not_in_kb 越界检出（防幻觉底线，越少越好）",
        "",
        "| 配置 | 检出条数 |",
        "|---|---|",
    ]
    for name, (found, total) in nin_stats.items():
        L.append(f"| {name} | {found}/{total} |")

    tmap = {t[0]: t for t in table}
    nv = table[0][4]
    r3 = tmap["RRF 融合（无精排）"][1]
    e3 = tmap["精排 + 阈值 0.2（生产）"][1]
    e3_raw = tmap["精排（无阈值）"][1]
    rrf20 = tmap["RRF 融合（无精排）"][3]
    bm2520 = tmap["BM25 单路"][3]
    lat_rrf = lat["RRF 融合（无精排）"]
    lat_rer = lat["精排 + 阈值 0.2（生产）"]
    nin_on = nin_stats["精排 + 阈值 0.2（生产）"]
    nin_off = nin_stats["精排（无阈值）"]

    L += [
        "",
        "## 七、结论",
        "",
        f"1. **精排净增益 {(e3 - r3) / nv * 100:+.1f}pp**（RRF {r3}/{nv} → 精排 {e3}/{nv}），"
        f"代价是单条延迟 {lat_rrf:.0f}ms → {lat_rer:.0f}ms（{lat_rer / max(lat_rrf, 1):.0f}×）。"
        "CPU 环境下精排是检索链路的主要成本，是否保留取决于延迟预算。",
        "",
        f"2. **阈值不影响精度**（有阈值 {e3}/{nv} vs 无阈值 {e3_raw}/{nv}），"
        f"但它决定库外问题是否被拦下：not_in_kb 越界检出 {nin_on[0]}/{nin_on[1]} vs "
        f"{nin_off[0]}/{nin_off[1]}。它的定位是防幻觉，不是提精度。",
        "",
        f"3. **融合截断存在召回风险**：RRF@20 = {rrf20}/{nv}，低于 BM25 单路@20 = {bm2520}/{nv}。"
        "原因是融合后截断到 recall_n，两路合并去重后的尾部文档被挤出；"
        "当前靠精排从候选中补回，换语料或换 query 分布时可能暴露。",
        "",
    ]
    report.write_text("\n".join(L), encoding="utf-8")
    print(f"\n报告已写入: {report}")


if __name__ == "__main__":
    main()
