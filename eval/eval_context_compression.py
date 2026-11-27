"""上下文压缩效果量化实验（滚动摘要机制的收益与代价）

验证 memory/summarizer.py 的滚动摘要是否真的控制了长会话的上下文规模，
以及压缩过程有没有丢掉关键信息。

与 agent/agent.py:240 `_load_context` 完全一致的口径：

    压缩后每轮携带 = system prompt + 会话摘要 + 最近 5 条原始消息
    无压缩对照     = system prompt + 全部历史消息

计数：tiktoken `cl100k_base`。它与 DeepSeek / GLM 实际 tokenizer 不完全一致，
      绝对数字会有偏差，但用于「压缩比」这一相对指标是可靠的。

用法（项目根目录）：
    .venv/Scripts/python eval/eval_context_compression.py
输出：
    eval/reports/context-compression-report.md
"""

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ⚠️ 必须在 import memory 之前注入临时库，避免写真实 chat_history.db
TMP_DB = Path(tempfile.gettempdir()) / "ctx_compress_eval.db"
for suffix in ("", "-wal", "-shm"):
    p = Path(str(TMP_DB) + suffix)
    if p.exists():
        p.unlink()
os.environ["MEMORY_DB_PATH"] = str(TMP_DB)
os.environ["OBSIDIAN_VAULT_DIR"] = ""

import tiktoken  # noqa: E402

from memory.memory import Memory  # noqa: E402
from memory.summarizer import (  # noqa: E402
    COMPRESSION_THRESHOLD,
    KEEP_RECENT,
    compress_if_needed,
)

ENC = tiktoken.get_encoding("cl100k_base")
SESSION = "ctx-eval-session"

# 埋入对话的关键事实：(名称, 摘要中应出现的任一关键词)
KEY_FACTS = [
    ("技术栈", ["FastAPI"]),
    ("embedding 选型", ["bge-small-zh", "bge"]),
    ("检索指标", ["4.8", "57.1"]),
    ("学习偏好", ["原理"]),
    ("待办事项", ["healthcheck", "Docker"]),
]

# 前 5 轮手写：自然埋入关键事实
HANDWRITTEN = [
    (
        "我在用 FastAPI + ChromaDB 做后端面经 RAG 项目，现在检索效果不太行，你觉得先从哪查？",
        "先分层定位，别急着换模型。第一步看召回：把 dense 单路和 BM25 单路分开评测，"
        "各取 top-20 算命中率。如果 dense 明显低于 BM25，说明语义检索没起作用，"
        "大概率是 embedding 模型不适合中文。第二步才看融合和精排。"
        "你现在的 embedding 用的是 Chroma 自带的 all-MiniLM-L6-v2 吗？那个是英文模型，"
        "中文 query 基本失效，实测 dense 命中率只有 4.8%，全靠 BM25 字面匹配撑着。",
    ),
    (
        "对，就是默认那个。那我是不是应该换成中文模型？",
        "应该换，但要清楚代价。换成 BAAI/bge-small-zh-v1.5 之后，维度从 384 变 512，"
        "所有 collection 必须全量重建，不能增量兼容。另外 query 侧要加 BGE 官方的检索指令前缀，"
        "向量要做归一化，否则相似度分布会偏。换完之后 dense 同口径命中率能到 57.1% 左右，"
        "相当于从几乎不可用变成主力召回通道。这是一个明确的决策点。",
    ),
    (
        "明白了，就决定换 bge-small-zh-v1.5。换完 dense 同口径命中率从 4.8% 提到 57.1%，"
        "但融合之后反而降了，这是为什么？",
        "这是典型的 RRF 缺陷暴露。RRF 只看排名位置、不看内容相关性，"
        "BM25 因为术语字面重合度高，把一些流水笔记顶到前面，把 dense 真正相关的"
        "专题笔记挤出 top-k。解决办法是加 Cross-Encoder 精排：宽召回 20 条 → RRF 融合 → "
        "bge-reranker-base 逐条打分 → 取 top-3。精排能读懂语义，纠正词面匹配的误判，"
        "实测纯 RRF 融合只有 61.9%，精排后能到 71.4%。",
    ),
    (
        "精排模型跑在 CPU 上会不会很慢？另外我学东西喜欢先看原理再看代码，"
        "你以后讲的时候按这个顺序来。",
        "CPU 上 bge-reranker-base 对 20 条候选打分实测约 6.4 秒，单看一次不算致命，"
        "但它比纯检索的 15 毫秒慢几百倍，并发高时会成为瓶颈，因为它是 CPU 密集的，"
        "会把 asyncio 默认线程池占满。建议给精排单独限流，超限就跳过精排直接用 RRF 结果，"
        "用精度换可用性。你的学习偏好我记下了：先原理、后代码。",
    ),
    (
        "还有个事，下周三之前我要把 Docker 的 healthcheck 补上，这个怎么设计比较好？",
        "healthcheck 要区分「进程活着」和「依赖可用」。建议两档：/health 只做进程自检，"
        "毫秒级返回，给容器编排用；/health/deep 检查向量库、数据库、模型是否就绪，"
        "超时设短一点。注意别在 /health 里触发 embedding 模型加载——那会让健康检查"
        "从毫秒变成几十秒，反而拖垮探活。下周三前的这个待办优先级不低。",
    ),
]

# 后续 15 轮：模板生成，模拟真实的长对话（保证总消息数远超压缩阈值）
FILLER_TOPICS = [
    ("RRF 的 k 值怎么选", "k=60 是论文默认值，作用是把排名差异压平，避免头部过度集中。"),
    ("混合检索权重怎么定", "RRF 不需要权重，它只看排名；如果要用权重，得先做分数归一化。"),
    ("切片大小怎么定", "500 字 + 50 重叠是中文笔记的稳妥起点，标题要注入到每个 chunk 头部。"),
    ("增量索引怎么做", "用 mtime 加切片数双重校验，只比 mtime 会漏掉切片丢失的情况。"),
    ("怎么防幻觉", "检索为空时明确说没找到，不要硬塞不相关上下文给模型。"),
    ("评测集怎么建", "golden set 要覆盖 direct / cross / reasoning 三类，还要有库外问题。"),
    ("怎么测路由准确率", "白盒判分，用日志捕获实际工具调用序列，再和预期对比。"),
    ("并发上不去怎么办", "先看是 CPU 密集还是 IO 等待，精排属于前者，LLM 调用属于后者。"),
    ("超时怎么配", "LLM 客户端必须显式设超时和重试次数，SDK 默认值不适合生产。"),
    ("限流怎么处理", "429 要快速失败并返回 503，不要无限重试把线程池打满。"),
    ("数据库锁怎么优化", "开 WAL 模式，写操作批量提交，减少写放大。"),
    ("日志怎么打", "结构化日志加 trace_id，一次请求的检索、精排、生成耗时要能串起来。"),
    ("怎么做优雅降级", "每个外部依赖都要有降级分支，模型不可用时退回规则逻辑。"),
    ("测试怎么分层", "单元测试测逻辑，接口测试测契约，端到端测路由，效果评测测准确率。"),
    ("简历上怎么写", "每个技术点按「问题 → 方案 → 量化结果」写，没有数字的优化等于没做。"),
]
FILLER_ANSWER = (
    "这个问题要分两层看。{point}"
    "工程上还要注意可观测性：把耗时、命中率、降级次数打成指标，"
    "否则出了问题只能靠猜。落地时先用最小改动验证假设，再决定要不要引入更重的方案，"
    "避免为了架构好看而增加维护成本。"
)


def build_dialog():
    """构造 20 轮对话（40 条消息）"""
    pairs = list(HANDWRITTEN)
    for topic, point in FILLER_TOPICS:
        pairs.append((f"接着问一个：{topic}？", FILLER_ANSWER.format(point=point)))
    return pairs


def count_tokens(messages) -> int:
    """按 chat 格式估算 token（每条消息加 role 开销）"""
    total = 0
    for m in messages:
        total += len(ENC.encode(m["content"])) + 4
    return total


def build_system_prompt() -> str:
    """取真实 system prompt 的规模（此处用等长占位，避免 import agent 的副作用）"""
    return (
        "你是一个个人知识库助手。遇到涉及个人笔记、学习记录、工作内容的问题，"
        "必须先调用 search_obsidian_notes 检索，再基于检索结果回答；"
        "检索为空时明确说明未找到，不要编造。回答使用简体中文，条理清晰。"
    ) * 3


def fact_hit(summary: str, keywords) -> bool:
    s = summary.lower()
    return any(k.lower() in s for k in keywords)


def main():
    pairs = build_dialog()
    # 不调 create_session：它要求 users 表存在对应用户（外键约束），
    # 而 conversations 表本身没有外键，直接按 session_id 写入即可。
    mem = Memory(db_path=str(TMP_DB))
    sid = SESSION

    # 逐条写入（含工具调用噪声，模拟 ReAct 中间结果）
    for i, (q, a) in enumerate(pairs):
        mem.save_message(sid, "user", q)
        if i % 4 == 0:
            mem.save_message(
                sid, "assistant",
                f'[工具调用] search_obsidian_notes(query="{q[:20]}") → 命中 3 条笔记',
            )
        mem.save_message(sid, "assistant", a)

    total_msgs = mem.count_messages(sid)
    print(f"会话消息数: {total_msgs} | 压缩阈值: {COMPRESSION_THRESHOLD} | 保留: {KEEP_RECENT}")

    # ── 压缩前口径：system + 全部历史 ──
    system_prompt = build_system_prompt()
    history_all = mem.load_history(sid, limit=10_000)
    before_msgs = [{"role": "system", "content": system_prompt}] + history_all
    before_tokens = count_tokens(before_msgs)

    # ── 触发压缩（真调一次 LLM）──
    t0 = time.perf_counter()
    compressed = compress_if_needed(sid, mem, provider=os.getenv("LLM_PROVIDER", "glm"))
    cost_s = time.perf_counter() - t0
    print(f"压缩触发: {compressed} | 耗时 {cost_s:.1f}s")
    if not compressed:
        print("⚠️ 压缩未发生（LLM 调用失败或消息数不足），后续数字不可用")
        return

    summary = (mem.get_session_summary(sid) or {}).get("summary", "")
    print(f"摘要长度: {len(summary)} 字")

    # ── 压缩后口径：system + 摘要 + 最近 5 条 ──
    recent = mem.load_history(sid, limit=KEEP_RECENT)
    after_msgs = [{"role": "system", "content": system_prompt}]
    after_msgs.append({"role": "system", "content": "【以下是你们之前对话的摘要】\n" + summary})
    after_msgs.extend(recent)
    after_tokens = count_tokens(after_msgs)

    # 只算对话部分的对比（剔除固定 system prompt，口径更干净）
    dialog_before = count_tokens(history_all)
    dialog_after = count_tokens(
        [{"role": "system", "content": summary}] + recent
    )

    print("\n===== 结果 =====")
    print(f"压缩前（system+全history）: {before_tokens} tokens")
    print(f"压缩后（system+摘要+{KEEP_RECENT}条）: {after_tokens} tokens")
    print(f"仅对话部分: {dialog_before} → {dialog_after} tokens")
    print(f"压缩比: {dialog_after / dialog_before:.1%}")
    print(f"节省: {1 - dialog_after / dialog_before:.1%}")

    # ── 关键信息保留 ──
    kept = [(name, fact_hit(summary, kws)) for name, kws in KEY_FACTS]
    n_kept = sum(1 for _, ok in kept if ok)
    print(f"\n关键事实保留: {n_kept}/{len(KEY_FACTS)}")
    for name, ok in kept:
        print(f"  {'✅' if ok else '❌'} {name}")

    # ── 写报告 ──
    import datetime

    report = Path(__file__).resolve().parent / "reports" / "context-compression-report.md"
    L = [
        "# 上下文压缩效果量化报告",
        "",
        f"> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "> 实现：`memory/summarizer.py`（滚动摘要）",
        f"> 触发阈值 {COMPRESSION_THRESHOLD} 条，压缩后保留最近 {KEEP_RECENT} 条原始消息",
        "> 计数：tiktoken `cl100k_base`（绝对 token 数与国产模型 tokenizer 有偏差，压缩比可靠）",
        "",
        "## 一、实验设置",
        "",
        f"- 构造 {len(pairs)} 轮对话，落库 {total_msgs} 条消息（含工具调用中间结果噪声）",
        "- 对话中埋入 5 个关键事实，用于检验压缩后的信息保留",
        "- 真实调用一次 LLM 生成摘要（非 mock）",
        "",
        "## 二、上下文规模",
        "",
        "| 口径 | Token 数 | 说明 |",
        "|---|---|---|",
        f"| 压缩前（system + 全部历史） | {before_tokens:,} | 没有压缩机制时的每轮开销 |",
        f"| 压缩后（system + 摘要 + {KEEP_RECENT} 条） | {after_tokens:,} | 当前实现的实际开销 |",
        f"| 仅对话部分（压缩前） | {dialog_before:,} | 剔除固定 system prompt |",
        f"| 仅对话部分（压缩后） | {dialog_after:,} | 同上 |",
        "",
        f"**对话部分压缩比：{dialog_after / dialog_before:.1%}（节省 {1 - dialog_after / dialog_before:.1%}）**",
        "",
        "## 三、信息保留",
        "",
        f"摘要长度：{len(summary)} 字 ｜ 关键事实保留：**{n_kept}/{len(KEY_FACTS)}**",
        "",
        "| 关键事实 | 是否保留 |",
        "|---|---|",
    ]
    for name, ok in kept:
        L.append(f"| {name} | {'✅' if ok else '❌'} |")
    L += [
        "",
        "## 四、摘要原文",
        "",
        "```text",
        summary,
        "```",
        "",
        "## 五、结论",
        "",
        "- 压缩机制把每轮上下文的对话部分从 "
        f"{dialog_before:,} tokens 降到 {dialog_after:,} tokens，"
        f"降幅 {1 - dialog_after / dialog_before:.1%}；",
        f"- 关键事实保留 {n_kept}/{len(KEY_FACTS)}，"
        + ("信息未出现明显丢失。" if n_kept == len(KEY_FACTS) else "存在信息丢失，需调整摘要 prompt 或提高字数上限。"),
        f"- 单次压缩耗时 {cost_s:.1f}s，发生在会话跨过 {COMPRESSION_THRESHOLD} 条时，只做一次、不阻塞主流程（失败静默降级）。",
    ]
    report.write_text("\n".join(L), encoding="utf-8")
    print(f"\n报告已写入: {report}")


if __name__ == "__main__":
    main()
