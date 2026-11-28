"""RAG 知识库检索工具 —— 接入 Agent 的 tool_registry

提供同步（knowledge_search）与异步（knowledge_search_async）两个版本。

只返回「编号 + 来源 + 原文片段」，不返回生成后的答案：生成统一由主链路做一次，
这样来源元数据不会在中途丢失，答案才能回溯到原文档。
（此前这里调用 query()，内部已跑过一次完整生成，单次问答会触发两次 LLM 调用。）
"""

from tools import registry
from utils.logger import get_logger

logger = get_logger(__name__)

EMPTY_HINT = (
    "个人知识库中未找到与该问题相关的内容。"
    "请如实告知用户，不要编造，也不要改用通用常识替代知识库内容。"
)

_SEARCH_DESC = (
    "在个人知识库（用户上传的 PDF/TXT/MD 文档）中检索原文片段。"
    "返回形如「[1] 来源: 文件名 + 正文」的片段列表，不是答案——你需要自己从中归纳；"
    "引用某条片段时在句末标注对应编号，如 [1]。"
    "当用户询问知识库中的内容（项目文档、公司流程、学习资料等）时调用。"
    "普通闲聊、查天气、做计算请使用其他对应工具。"
)


@registry.register(description=_SEARCH_DESC)
def knowledge_search(query: str) -> str:
    """从知识库中检索相关片段（同步版）。

    Args:
        query: 用户的问题或查询关键词
    """
    logger.info("RAG 工具被调用: %s", query)
    from knowledge.passages import format_passages
    from knowledge.pipeline import retrieve_knowledge_passages

    passages = retrieve_knowledge_passages(query)
    if not passages:
        logger.info("知识库检索无命中: %s", query)
        return EMPTY_HINT
    logger.info("知识库检索命中 %d 条片段", len(passages))
    return format_passages(passages)


@registry.register_async(description=_SEARCH_DESC)
async def knowledge_search_async(query: str) -> str:
    """从知识库中检索相关片段（异步版，Agent.arun 使用）。

    Args:
        query: 用户的问题或查询关键词
    """
    import asyncio

    logger.info("RAG 工具被调用(异步): %s", query)
    from knowledge.passages import format_passages
    from knowledge.pipeline import retrieve_knowledge_passages

    passages = await asyncio.to_thread(retrieve_knowledge_passages, query)
    if not passages:
        logger.info("知识库检索无命中(异步): %s", query)
        return EMPTY_HINT
    logger.info("知识库检索命中 %d 条片段(异步)", len(passages))
    return format_passages(passages)
