"""检索片段层（Passage）回归测试

核心契约：检索工具只返回「编号 + 来源 + 原文」，**不触发任何 LLM 生成**。

立此测试的背景（A1 架构纠偏）：
改前 `search_obsidian_notes` → `query_obsidian()`，而后者的内部已经跑完一次
「检索 → 拼 Prompt → LLM 生成」，返回值再被截断到 2000 字交给 Agent，Agent 再生成一遍。
后果四条：单次问答两次 LLM 调用；source/score 在第一次生成后即丢失，答案无法回溯原文；
第一次的偏差被第二次当事实放大；截断可能砍在半句上。

改后工具只做检索，生成只在主链路发生一次。以下用例锁死这个契约，
防止后续有人「顺手」把生成塞回工具层。
"""

import inspect

from langchain_core.documents import Document

from knowledge.passages import (
    MAX_PASSAGE_CHARS,
    Passage,
    format_passages,
    passages_to_sources,
    to_passages,
)


def _doc(content: str, **meta) -> Document:
    return Document(page_content=content, metadata=meta)


class TestToPassages:
    """Document → Passage 的转换规则"""

    def test_skips_empty_documents_and_keeps_index_contiguous(self):
        docs = [
            _doc("第一段正文", rel_path="a.md"),
            _doc("   \n\t  ", rel_path="empty.md"),
            _doc("第三段正文", rel_path="c.md"),
        ]
        passages = to_passages(docs)
        assert [p.index for p in passages] == [1, 2], "跳过空文档后编号必须仍然连续"
        assert [p.content for p in passages] == ["第一段正文", "第三段正文"]

    def test_prefers_rel_path_as_source(self):
        doc = _doc("正文", rel_path="AI学习\\算法\\快排.md", source=r"D:\agent\obsidian\知识库\AI学习\算法\快排.md")
        assert to_passages([doc])[0].source == "AI学习/算法/快排.md", "反斜杠应统一为正斜杠"

    def test_falls_back_to_filename_without_leaking_abs_path(self):
        doc = _doc("正文", source=r"D:\agent\obsidian\知识库\杂项.md")
        source = to_passages([doc])[0].source
        assert source == "杂项.md"
        assert "D:" not in source, "没有 rel_path 时也不能把本机盘符路径交给模型"

    def test_unknown_source(self):
        assert to_passages([_doc("正文")])[0].source == "未知来源"

    def test_truncates_long_content(self):
        passages = to_passages([_doc("x" * 5000, rel_path="long.md")])
        content = passages[0].content
        assert len(content) == MAX_PASSAGE_CHARS + 1, "应为上限 + 省略号"
        assert content.endswith("…")

    def test_respects_limit(self):
        docs = [_doc(f"正文{i}", rel_path=f"{i}.md") for i in range(10)]
        assert len(to_passages(docs, limit=3)) == 3

    def test_marks_match_channel(self):
        both = _doc("两路都命中", rel_path="a.md")
        only_dense = _doc("只有 dense", rel_path="b.md")
        only_bm25 = _doc("只有 bm25", rel_path="c.md")
        passages = to_passages(
            [both, only_dense, only_bm25],
            dense_docs=[both, only_dense],
            bm25_docs=[both, only_bm25],
        )
        assert [p.match for p in passages] == ["both", "dense", "bm25"]

    def test_match_is_vector_when_no_channel_given(self):
        assert to_passages([_doc("正文", rel_path="a.md")])[0].match == "vector"

    def test_empty_input(self):
        assert to_passages([]) == []
        assert to_passages(None) == []


class TestFormat:
    """给 LLM 的文本形态"""

    def test_shape(self):
        text = format_passages(
            [
                Passage(index=1, content="第一段", source="AI学习/快排.md"),
                Passage(index=2, content="第二段", source="杂项.md"),
            ]
        )
        assert text == "[1] 来源: AI学习/快排.md\n第一段\n\n[2] 来源: 杂项.md\n第二段"

    def test_does_not_leak_score_or_channel(self):
        """分数与命中方式只用于日志/调试，不进上下文——避免模型用分数做无意义推理。"""
        text = format_passages(
            [Passage(index=1, content="正文", source="a.md", match="both")]
        )
        assert "score" not in text.lower()
        assert "both" not in text
        assert "相关度" not in text

    def test_empty(self):
        assert format_passages([]) == ""

    def test_sources_export(self):
        sources = passages_to_sources(
            [Passage(index=1, content="正文很长的内容", source="a.md", match="dense")]
        )
        assert sources == [{"index": 1, "source": "a.md", "match": "dense"}]
        assert "正文" not in str(sources), "来源列表不应携带正文"


class _FakeStore:
    """最小向量库替身：只实现检索路径用到的 count / search。"""

    def __init__(self, docs=None):
        self._docs = docs or []
        self.search_calls: list[dict] = []

    def count(self) -> int:
        return len(self._docs) if self._docs else 1   # 默认非空，避免触发自动索引

    def search(self, query: str, k: int | None = None):
        self.search_calls.append({"query": query, "k": k})
        return self._docs[: (k or 3)]


def _called_names(func) -> set[str]:
    """收集函数体里被调用的函数名（看 AST 的 Call 节点，不看注释/文档字符串）。

    用文本匹配会误伤 docstring 里的说明文字，所以这里解析语法树：
    只关心「真的调用了」，不关心「提到了」。
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


class TestObsidianToolContract:
    """search_obsidian_notes 只做检索，不做生成"""

    def test_source_code_never_calls_the_generator(self):
        """静态断言：工具实现里不得调用 query_obsidian（那条路径内部会调 LLM）。"""
        from tools import obsidian

        forbidden = {"query_obsidian", "query_obsidian_async"}
        for func in (obsidian.search_obsidian_notes, obsidian.search_obsidian_notes_async):
            hit = _called_names(func) & forbidden
            assert not hit, f"{func.__name__} 不得调用 {hit}（会多一次 LLM 生成）"

    def test_returns_numbered_passages(self, monkeypatch):
        from tools import obsidian

        fake = [
            Passage(index=1, content="快速排序以右端为 pivot", source="AI学习/算法/快排.md"),
            Passage(index=2, content="三路划分实现", source="AI学习/算法/快排-优化.md"),
        ]
        monkeypatch.setattr(
            "knowledge.pipeline.retrieve_obsidian_passages", lambda q, k=None: fake
        )

        out = obsidian.search_obsidian_notes("快排怎么写")
        assert out.startswith("[1] 来源: AI学习/算法/快排.md")
        assert "[2] 来源: AI学习/算法/快排-优化.md" in out
        assert "快速排序以右端为 pivot" in out

    def test_marks_empty_result_explicitly(self, monkeypatch):
        from tools import obsidian

        monkeypatch.setattr(
            "knowledge.pipeline.retrieve_obsidian_passages", lambda q, k=None: []
        )
        out = obsidian.search_obsidian_notes("库里完全没有的主题")
        assert out == obsidian.EMPTY_HINT
        assert "未找到" in out
        assert "不要编造" in out

    def test_async_variant_returns_same_shape(self, monkeypatch):
        import asyncio

        from tools import obsidian

        fake = [Passage(index=1, content="异步命中的正文", source="a.md")]
        monkeypatch.setattr(
            "knowledge.pipeline.retrieve_obsidian_passages", lambda q, k=None: fake
        )
        out = asyncio.run(obsidian.search_obsidian_notes_async("问题"))
        assert out.startswith("[1] 来源: a.md")


class TestKnowledgeToolContract:
    """knowledge_search 同样只做检索"""

    def test_source_code_never_calls_the_generator(self):
        from tools import rag_tool

        forbidden = {"query", "query_async", "rag_query", "rag_query_async"}
        for func in (rag_tool.knowledge_search, rag_tool.knowledge_search_async):
            hit = _called_names(func) & forbidden
            assert not hit, f"{func.__name__} 不得调用 {hit}（会多一次 LLM 生成）"

    def test_returns_numbered_passages(self, monkeypatch):
        from tools import rag_tool

        fake = [Passage(index=1, content="上传文档的正文", source="readme.md")]
        monkeypatch.setattr(
            "knowledge.pipeline.retrieve_knowledge_passages", lambda q, k=None: fake
        )
        out = rag_tool.knowledge_search("问点什么")
        assert out == "[1] 来源: readme.md\n上传文档的正文"

    def test_marks_empty_result_explicitly(self, monkeypatch):
        from tools import rag_tool

        monkeypatch.setattr(
            "knowledge.pipeline.retrieve_knowledge_passages", lambda q, k=None: []
        )
        assert rag_tool.knowledge_search("x") == rag_tool.EMPTY_HINT

    def test_async_variant_returns_same_shape(self, monkeypatch):
        import asyncio

        from tools import rag_tool

        fake = [Passage(index=1, content="异步正文", source="b.md")]
        monkeypatch.setattr(
            "knowledge.pipeline.retrieve_knowledge_passages", lambda q, k=None: fake
        )
        out = asyncio.run(rag_tool.knowledge_search_async("问题"))
        assert out.startswith("[1] 来源: b.md")


class TestRetrievalWiring:
    """检索入口的接线：空库短路，且不触碰生成链路"""

    def test_knowledge_passages_empty_store_returns_empty_list(self, monkeypatch):
        from knowledge import pipeline as pl

        monkeypatch.setattr(pl, "get_vector_store", lambda *a, **kw: _FakeStore([]))
        monkeypatch.setattr(pl, "index_documents", lambda *a, **kw: 0)
        assert pl.retrieve_knowledge_passages("问题") == []

    def test_knowledge_passages_maps_search_results(self, monkeypatch):
        from knowledge import pipeline as pl

        docs = [_doc("知识库正文", source=r"D:\proj\knowledge\docs\a.md")]
        store = _FakeStore(docs)
        monkeypatch.setattr(pl, "get_vector_store", lambda *a, **kw: store)

        passages = pl.retrieve_knowledge_passages("问题", k=2)
        assert len(passages) == 1
        assert passages[0].content == "知识库正文"
        assert passages[0].source == "a.md"
        assert store.search_calls == [{"query": "问题", "k": 2}]

    def test_obsidian_passages_empty_store_returns_empty_list(self, monkeypatch):
        from knowledge import pipeline as pl

        class _Empty(_FakeStore):
            def count(self):
                return 0

        monkeypatch.setattr(pl, "get_vector_store", lambda *a, **kw: _Empty())
        assert pl.retrieve_obsidian_passages("问题") == []
