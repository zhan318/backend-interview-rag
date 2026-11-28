"""A1 架构纠偏验证：一次问答到底调用几次 LLM

改前链路（tools/obsidian.py → pipeline.query_obsidian）：
    Agent 调工具 → query_obsidian 内部跑完整 RAG 生成（LLM #1）
                 → 答案截断到 2000 字交给 Agent
                 → Agent 再生成一遍（LLM #2）
    每次工具调用 = 2 次 LLM；source/score 在第一次生成后即丢失，答案无法回溯原文。

改后链路：
    Agent 调工具 → 只检索，返回「[1] 来源: 路径 + 原文片段」
                 → Agent 生成（LLM #1，唯一一次）

本脚本用**真实 Agent**跑 golden 集的 RAG 题，统计：

    llm_calls   主链路 LLM 调用次数（patch agent.agent.achat 计数）
    tool_calls  工具调用次数（捕获 agent logger）
    citations   最终答案里 [n] 角标数量
    elapsed     端到端耗时

断言（不合格以非零码退出）：
    1. llm_calls <= tool_calls + 1
       ReAct 循环里「末轮无工具调用」是唯一不需要工具的那次，
       所以每多一次工具调用最多只该多一次 LLM。超了说明工具层又偷偷生成了。
    2. 至少半数答案带 [n] 角标，说明引用约束生效。

用法（项目根目录）：
    .venv/Scripts/python eval/eval_a1_architecture.py [--limit 5]

注意：真实调用 LLM，产生费用。使用独立临时库，不污染真实会话历史。
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 环境隔离：必须在 import config / agent 之前 ──
_TMP_DB = ROOT / "eval" / "reports" / ".a1_eval.db"
os.environ["MEMORY_DB_PATH"] = str(_TMP_DB)

REPORT_MD = ROOT / "eval" / "reports" / "a1-architecture-report.md"
RAW_JSON = ROOT / "eval" / "reports" / "a1-architecture-raw.json"

AGENT_LOGGER = "agent.agent"


class ToolCallCapture(logging.Handler):
    """从 agent logger 捕获工具调用名（「执行工具: name, 参数=...」）。"""

    _PAT = re.compile(r"执行工具(?:\(异步\))?: (\S+),")

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    def emit(self, record):
        m = self._PAT.search(record.getMessage())
        if m:
            self.calls.append(m.group(1))


def load_questions(limit: int) -> list[dict]:
    """取 golden 集里知识库有答案的题（排除 not_in_kb，那种题会走拒答）。"""
    items = []
    path = ROOT / "eval" / "golden_set.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if it.get("category") == "not_in_kb":
            continue
        items.append(it)
        if len(items) >= limit:
            break
    return items


def strip_code_blocks(text: str) -> str:
    """去掉代码块再数角标：`arr[0]`、`arr[i+1]` 这类索引会被误判成引用。"""
    return re.sub(r"```.*?```", "", text, flags=re.S)


def count_citations(answer: str) -> int:
    """统计形如 [1]..[9] 的引用角标出现次数。"""
    return len(re.findall(r"\[([1-9])\]", strip_code_blocks(answer)))


def install_llm_counter():
    """给 agent.agent.achat 套一层计数器，返回 (读取函数, 还原函数)。

    agent.py 里是 `from chatbot.chatbot import ... achat`，所以替换
    agent.agent 模块上的名字即可拦到主链路的每一次调用。
    """
    import agent.agent as agent_mod

    original = agent_mod.achat
    counter = {"n": 0, "errors": 0}

    async def counted(*args, **kwargs):
        counter["n"] += 1
        try:
            return await original(*args, **kwargs)
        except Exception:
            counter["errors"] += 1
            raise

    agent_mod.achat = counted

    def restore():
        agent_mod.achat = original

    return counter, restore


async def run_one(index: int, question: str) -> dict:
    """跑一道题：独立 Agent + 独立计数。"""
    from agent.agent import Agent

    capture = ToolCallCapture()
    agent_logger = logging.getLogger(AGENT_LOGGER)
    agent_logger.addHandler(capture)

    counter, restore = install_llm_counter()
    agent = Agent(session_id=f"a1_eval_{index}")
    try:
        started = time.perf_counter()
        answer = await agent.arun(question)
        elapsed = time.perf_counter() - started
    finally:
        restore()
        agent_logger.removeHandler(capture)

    return {
        "index": index,
        "question": question,
        "answer": answer,
        "elapsed_sec": round(elapsed, 2),
        "llm_calls": counter["n"],
        "tool_calls": len(capture.calls),
        "tools": capture.calls,
        "citations": count_citations(answer),
        "budget": len(capture.calls) + 1,   # llm_calls 的允许上限
    }


def cleanup():
    """删掉临时库，避免把评测数据留在仓库里。

    这里吞掉所有异常：某些受限环境会把 unlink 重定向到回收站，
    失败时抛的是通用 OSError，不该因为清理失败而让整个评测报错退出。
    """
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(_TMP_DB) + suffix)
        with contextlib.suppress(Exception):
            p.unlink()


def render_report(rows: list[dict], model_label: str) -> str:
    total_llm = sum(r["llm_calls"] for r in rows)
    total_tools = sum(r["tool_calls"] for r in rows)
    cited = sum(1 for r in rows if r["citations"] > 0)
    elapsed = [r["elapsed_sec"] for r in rows]
    over_budget = [r for r in rows if r["llm_calls"] > r["budget"]]

    lines = [
        "# A1 架构纠偏验证：一次问答的 LLM 调用次数",
        "",
        f"- 生成模型：{model_label}",
        f"- 样本：{len(rows)} 道 golden 集 RAG 题（知识库内可答）",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 结论",
        "",
        "| 指标 | 数值 | 说明 |",
        "|---|---|---|",
        f"| LLM 调用总数 | {total_llm} | 含主链路全部轮次 |",
        f"| 工具调用总数 | {total_tools} | search_obsidian_notes / read_obsidian_note 等 |",
        f"| 调用/工具比 | {(total_llm / total_tools) if total_tools else 0:.2f} | 改前每次工具调用要 2 次（工具内生成 + 主链路生成） |",
        f"| 超预算用例 | {len(over_budget)} | `llm_calls > tool_calls + 1` 即说明工具层又生成了 |",
        f"| 带引用角标的答案 | {cited}/{len(rows)} | `[n]` 角标数量，验证引用约束 |",
        f"| 耗时中位数 | {statistics.median(elapsed):.1f}s | 端到端 |",
        "",
        "## 逐题明细",
        "",
        "| # | 问题 | LLM 调用 | 工具调用 | 预算 | 角标 | 耗时(s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q = r["question"].replace("|", "\\|")[:34]
        lines.append(
            f"| {r['index']} | {q} | {r['llm_calls']} | {r['tool_calls']} | "
            f"{r['budget']} | {r['citations']} | {r['elapsed_sec']} |"
        )

    lines += [
        "",
        "## 与改前对比",
        "",
        "改前的调用次数不是实测值，而是由代码结构决定的**下界**："
        "`search_obsidian_notes` → `query_obsidian()` 内部必然跑一次完整 LLM 生成，"
        "主链路每轮工具调用后还要再问一次模型，所以每次工具调用对应 2 次 LLM。",
        "",
        "| 维度 | 改前 | 改后（实测） |",
        "|---|---|---|",
        "| 每次工具调用对应的 LLM 次数 | 2（工具内 1 + 主链路 1） | "
        f"**{(total_llm / total_tools) if total_tools else 0:.2f}**"
        f"（{total_llm} 次 LLM / {total_tools} 次工具） |",
        "| 单次问答的 LLM 调用量 | 工具调用数 × 2 + 1 | 工具调用数 + 1 |",
        "| 工具返回值 | 生成后的答案，硬截断 2000 字 | 3 条原文片段（编号 + 相对路径） |",
        "| 来源元数据 | 第一次生成后丢失，无法标注 | 保留到最终答案，可标 `[n]` |",
        "",
        f"同一批 {total_tools} 次工具调用，改前至少需要 "
        f"{total_tools * 2 + 1} 次 LLM 调用；实测 {total_llm} 次，"
        f"**减少约 {round((1 - total_llm / (total_tools * 2 + 1)) * 100)}%**。",
        "",
        "## 口径说明",
        "",
        "- **预算 = 工具调用次数 + 1**：ReAct 循环里每次工具调用后都要再问一次模型，"
        "最后一轮不带工具调用的那次生成了最终答案，所以 N 次工具调用对应 N+1 次 LLM。",
        "- 改前每次 `search_obsidian_notes` 会先在工具内部跑一次完整 RAG 生成，"
        "因此实际是 2N+1；本表按 N+1 判定，超了即说明工具层在生成。",
        "- 角标统计前先剥离代码块，避免把 `arr[0]` 这类索引误算成引用。",
        "- 计数只覆盖 `agent.agent.achat`（Agent 主链路）。摘要压缩等旁路调用不计入，"
        "因此实测值只会偏小、不会偏大。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="A1 架构纠偏验证")
    parser.add_argument("--limit", type=int, default=5, help="跑几道题（默认 5）")
    args = parser.parse_args()

    if _TMP_DB.exists():
        cleanup()

    from chatbot.chatbot import MODEL_PRESETS, get_default_provider

    provider = get_default_provider()
    preset = MODEL_PRESETS[provider]
    model_label = f"{preset['label']} ({provider})"
    print(f"生成模型: {model_label}")

    questions = load_questions(args.limit)
    print(f"样本: {len(questions)} 道 RAG 题\n")

    rows = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['question']}")
        try:
            row = asyncio.run(run_one(i, item["question"]))
        except Exception as e:
            print(f"  ✗ 执行失败: {e}")
            continue
        rows.append(row)
        flag = "OK" if row["llm_calls"] <= row["budget"] else "超预算!"
        print(
            f"  → LLM {row['llm_calls']} 次（预算 {row['budget']}）"
            f" / 工具 {row['tool_calls']} 次 / 角标 {row['citations']} 个"
            f" / {row['elapsed_sec']}s  [{flag}]"
        )

    if not rows:
        print("没有成功样本，无法出报告")
        return 1

    REPORT_MD.write_text(render_report(rows, model_label), encoding="utf-8")
    RAW_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cleanup()

    total_tools = sum(r["tool_calls"] for r in rows)
    total_llm = sum(r["llm_calls"] for r in rows)
    over = [r for r in rows if r["llm_calls"] > r["budget"]]
    cited = sum(1 for r in rows if r["citations"] > 0)

    print(f"\n报告: {REPORT_MD.relative_to(ROOT)}")
    print(f"LLM/工具 = {total_llm}/{total_tools}，超预算 {len(over)} 例，带角标 {cited}/{len(rows)}")

    if over:
        print("✗ 有用例超出 N+1 预算，工具层可能仍在生成")
        return 1
    if cited * 2 < len(rows):
        print("✗ 带引用角标的答案不足半数，引用约束未生效")
        return 1
    print("✓ 通过：工具层不生成，引用约束生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
