"""Obsidian 知识库工具 —— 检索、读取与写入用户个人笔记（遵守排除清单）"""

from pathlib import Path

from tools import registry
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_NOTE_LENGTH = 4000

# 检索无结果时的统一话术：明确抑制「用通用常识补答个人事务」的倾向
EMPTY_HINT = (
    "你的 Obsidian 笔记库中未找到与该问题相关的内容。"
    "请如实告知用户，不要编造，也不要用通用常识替代笔记内容。"
)

_SEARCH_DESC = (
    "在用户的 Obsidian 个人知识库中检索相关笔记的原文片段。"
    "返回形如「[1] 来源: 路径 + 正文」的片段列表，不是答案——你需要自己从中归纳；"
    "引用某条片段时在句末标注对应编号，如 [1]。"
    "当用户询问自己的笔记、个人知识、或 vault 中的主题时使用。"
    "它与 knowledge_search 是两套独立的知识库，不要混用。"
)


def _resolve_note(note: str) -> Path | None:
    """将笔记名/路径解析为 vault 内的绝对路径。

    解析顺序：
      1. 绝对路径且文件存在
      2. 相对 vault 根的路径存在
      3. 按文件名（stem）在 vault 内递归查找 .md
    """
    from config import OBSIDIAN_VAULT_DIR

    vault = OBSIDIAN_VAULT_DIR
    if not vault:
        return None
    vault_path = Path(vault).resolve()
    p = Path(note)

    # 1. 绝对路径
    if p.is_absolute() and p.exists():
        return p
    # 2. 相对 vault 的路径
    rp = vault_path / note
    if rp.exists():
        return rp
    # 3. 按文件名查找
    target = p.stem if p.suffix else note
    matches = [
        f for f in vault_path.rglob("*.md")
        if f.stem == target and not f.name.startswith(".")
    ]
    if matches:
        return matches[0]
    return None


@registry.register(description=_SEARCH_DESC)
def search_obsidian_notes(query: str) -> str:
    """在 Obsidian 知识库中检索相关片段（同步版）。

    只做检索，不做生成：返回带编号的原文片段，由主链路统一生成一次答案。
    （此前这里调用 query_obsidian()，内部已跑过一次完整生成，导致单次问答
    两次 LLM 调用、来源元数据丢失、答案无法回溯原文。）

    Args:
        query: 用户的问题或查询关键词
    """
    logger.info("Obsidian 检索工具被调用: %s", query)
    from knowledge.passages import format_passages
    from knowledge.pipeline import retrieve_obsidian_passages

    passages = retrieve_obsidian_passages(query)
    if not passages:
        logger.info("Obsidian 检索无命中: %s", query)
        return EMPTY_HINT
    logger.info("Obsidian 检索命中 %d 条片段", len(passages))
    return format_passages(passages)


@registry.register_async(description=_SEARCH_DESC)
async def search_obsidian_notes_async(query: str) -> str:
    """在 Obsidian 知识库中检索相关片段（异步版，Agent.arun 使用）。

    检索是 CPU 密集的同步实现（含 Cross-Encoder 精排），放入线程池避免阻塞事件循环。

    Args:
        query: 用户的问题或查询关键词
    """
    import asyncio

    logger.info("Obsidian 检索工具被调用(异步): %s", query)
    from knowledge.passages import format_passages
    from knowledge.pipeline import retrieve_obsidian_passages

    passages = await asyncio.to_thread(retrieve_obsidian_passages, query)
    if not passages:
        logger.info("Obsidian 检索无命中(异步): %s", query)
        return EMPTY_HINT
    logger.info("Obsidian 检索命中 %d 条片段(异步)", len(passages))
    return format_passages(passages)


@registry.register(
    description="读取某篇具体 Obsidian 笔记的原文内容。"
                "当用户明确提到某篇笔记名称（如「python之路」「prompt」）"
                "并想查看其具体内容时调用。"
                "注意：被排除清单保护的隐私文件（如存放 API Key 的文件）无法读取。"
)
def read_obsidian_note(note: str) -> str:
    """读取指定 Obsidian 笔记的原文。

    Args:
        note: 笔记名称或路径，例如 "python之路" 或 "外语学习/单词表.md"
    """
    logger.info("Obsidian 读取工具被调用: %s", note)
    from config import OBSIDIAN_VAULT_DIR, OBSIDIAN_WRITE_BLOCK
    from knowledge.loader import is_path_excluded

    vault = OBSIDIAN_VAULT_DIR
    if not vault:
        return "Obsidian vault 未配置（请设置 OBSIDIAN_VAULT_DIR）。"

    candidate = _resolve_note(note)
    if candidate is None:
        return f"未找到笔记: {note}（检查名称或路径是否正确）"

    vault_path = Path(vault).resolve()
    try:
        rel = str(candidate.resolve().relative_to(vault_path))
    except Exception:
        rel = candidate.name
    if is_path_excluded(rel, OBSIDIAN_WRITE_BLOCK):
        return "该文件被排除清单保护，无法读取（可能含敏感信息）。"

    try:
        content = candidate.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"读取笔记失败: {e}"
    if len(content) > MAX_NOTE_LENGTH:
        content = content[:MAX_NOTE_LENGTH] + "\n\n[笔记过长已截断]"
    return f"# {candidate.name}\n\n{content}"


def save_note_to_vault(note: str, content: str) -> dict:
    """把笔记写入 Obsidian vault（供 HTTP 接口使用，如 /notes/save）。

    说明：Agent 不再自动写盘——写笔记统一走前端「存为笔记」按钮调用本函数，
    从源头杜绝 Agent 误判导致的未确认写盘。

    安全检查：路径必须在 vault 内，受排除清单保护。

    Returns:
        {"ok": bool, "message": str, "path": str}  # path 为 vault 内相对路径（失败时为空）
    """
    from config import OBSIDIAN_VAULT_DIR, OBSIDIAN_WRITE_BLOCK
    from knowledge.loader import is_path_excluded

    vault = OBSIDIAN_VAULT_DIR
    if not vault:
        return {"ok": False, "message": "Obsidian vault 未配置（请设置 OBSIDIAN_VAULT_DIR）。", "path": ""}

    vault_path = Path(vault).resolve()
    # 自动补 .md
    p = vault_path / note
    if p.suffix != ".md":
        p = p.with_suffix(".md")

    # 安全检查：必须在 vault 内
    try:
        p = p.resolve()
        p.relative_to(vault_path)
    except ValueError:
        return {"ok": False, "message": f"禁止写入 vault 外的路径: {note}", "path": ""}
    except Exception as e:
        return {"ok": False, "message": f"路径解析失败: {e}", "path": ""}

    # 排除清单检查
    try:
        rel = str(p.relative_to(vault_path))
    except Exception:
        rel = p.name
    if is_path_excluded(rel, OBSIDIAN_WRITE_BLOCK):
        return {"ok": False, "message": "该文件被排除清单保护，无法写入（可能含敏感路径）。", "path": ""}

    # 写入
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content, encoding="utf-8")
        action = "更新" if existed else "创建"
        rel_path = str(p.relative_to(vault_path))
        return {
            "ok": True,
            "message": f"已{action} Obsidian 笔记: {p.name}",
            "path": rel_path,
        }
    except Exception as e:
        return {"ok": False, "message": f"写入笔记失败: {e}", "path": ""}


