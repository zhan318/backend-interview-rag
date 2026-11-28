"""检索结果 → 结构化片段（Passage）

设计动机：工具层此前调用 `query_obsidian()` / `query()`，而这两个函数内部
已经完整跑过一次「检索 → 拼 Prompt → LLM 生成」。于是 Agent 拿到的是「二手答案」：

  1. 元数据（source / score）在第一次生成后就不存在了，答案无法回溯原文；
  2. 一次问答触发两次 LLM 调用，延迟与成本翻倍；
  3. 第一次生成的偏差会被第二次当作事实继续放大；
  4. 结果还要按字符数硬截断，可能砍在半句上。

本模块只做一件事：把 Document 列表转成「编号 + 来源 + 原文」的结构化片段，
交给主链路的 Agent 一次性生成。生成只发生一次，引用由检索层的元数据保证。

注意：`score` / `match` 保留在 Passage 里供日志与调试使用，但**不进入给 LLM 的文本**。
理由是阈值已在检索层过滤，能进来的候选都已过线；把分数交给模型只会引出
「相关度只有 0.6 所以不确定」这类无意义表述，且白白消耗 token。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# 单条片段最多保留的字符数：一条超长笔记不应挤占全部上下文预算
MAX_PASSAGE_CHARS = 800
# 一次返回的最大片段数（防御性上限，正常由 RETRIEVE_TOP_K 控制）
MAX_PASSAGES = 5


@dataclass(frozen=True)
class Passage:
    """一条可引用的检索片段。"""

    index: int          # 从 1 开始的引用编号，对应答案里的 [n]
    content: str        # 原文片段（已按 MAX_PASSAGE_CHARS 截断）
    source: str         # 人类可读来源：Obsidian 为 vault 相对路径，知识库为文件名
    match: str = "vector"   # 命中方式：dense / bm25 / both / vector（仅调试用）
    metadata: dict = field(default_factory=dict, compare=False)


def _display_source(doc) -> str:
    """把 Document.metadata 转成「能定位到笔记、又不泄露本机目录」的来源串。

    Obsidian 索引时写入了 rel_path（vault 内相对路径），优先用它 —— 既能让
    Agent 拿去调 read_obsidian_note 精读，也不会把 D:\\agent\\... 暴露给模型。
    """
    meta = getattr(doc, "metadata", None) or {}
    rel = meta.get("rel_path")
    if rel:
        return str(rel).replace("\\", "/")
    src = meta.get("source") or ""
    if not src:
        return "未知来源"
    # 没有相对路径时只留文件名，避免绝对路径外泄
    return Path(src).name or str(src)


def to_passages(
    docs,
    dense_docs=None,
    bm25_docs=None,
    *,
    max_chars: int = MAX_PASSAGE_CHARS,
    limit: int = MAX_PASSAGES,
) -> list[Passage]:
    """把 Document 列表转成 Passage 列表。

    Args:
        docs: 最终要交给 LLM 的文档（已经过精排/阈值过滤）
        dense_docs: dense 召回结果，用于标注命中方式（可选）
        bm25_docs: BM25 召回结果，用于标注命中方式（可选）
        max_chars: 单条片段字符上限
        limit: 最多返回几条
    """
    if not docs:
        return []

    dense_keys = {d.page_content for d in dense_docs} if dense_docs is not None else set()
    bm25_keys = {d.page_content for d in bm25_docs} if bm25_docs is not None else set()

    passages: list[Passage] = []
    for doc in docs:
        if len(passages) >= limit:
            break
        text = (doc.page_content or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"

        if not dense_keys and not bm25_keys:
            match = "vector"
        elif doc.page_content in dense_keys and doc.page_content in bm25_keys:
            match = "both"
        elif doc.page_content in dense_keys:
            match = "dense"
        else:
            match = "bm25"

        passages.append(
            Passage(
                index=len(passages) + 1,   # 跳过空文档后编号仍然连续
                content=text,
                source=_display_source(doc),
                match=match,
                metadata=dict(getattr(doc, "metadata", None) or {}),
            )
        )
    return passages


def format_passages(passages: list[Passage]) -> str:
    """渲染成给 LLM 读的文本：编号 + 来源 + 原文。

    刻意不输出分数与命中方式 —— 见模块文档。
    """
    if not passages:
        return ""
    parts = []
    for p in passages:
        parts.append(f"[{p.index}] 来源: {p.source}\n{p.content}")
    return "\n\n".join(parts)


def passages_to_sources(passages: list[Passage]) -> list[dict]:
    """给 HTTP 层/前端用的精简来源列表（不含正文，避免响应体过大）。"""
    return [
        {"index": p.index, "source": p.source, "match": p.match}
        for p in passages
    ]
