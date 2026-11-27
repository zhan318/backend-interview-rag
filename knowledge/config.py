"""知识库配置：切片策略、检索参数、模型选择"""

import os


class KnowledgeConfig:
    """所有知识库配置项，可通过环境变量覆盖"""

    # ── 文档目录 ──
    DOCS_DIR: str = os.getenv(
        "KNOWLEDGE_DOCS_DIR",
        os.path.join(os.path.dirname(__file__), "docs"),
    )

    # ── 向量数据库持久化目录 ──
    VECTOR_DB_DIR: str = os.getenv(
        "KNOWLEDGE_VECTOR_DB_DIR",
        os.path.join(os.path.dirname(__file__), "chroma_db"),
    )

    # ── 文本切片 ──
    CHUNK_SIZE: int = int(os.getenv("KNOWLEDGE_CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "50"))

    # ── 检索 ──
    # 最终送 LLM 的上下文条数
    RETRIEVE_TOP_K: int = int(os.getenv("KNOWLEDGE_RETRIEVE_TOP_K", "3"))
    # 召回池大小：dense / BM25 各自召回条数，经 RRF 融合后交精排筛选。
    # 需大于 RETRIEVE_TOP_K，才有「宽召回 + 严精排」两阶段检索的意义。
    RETRIEVE_RECALL_N: int = int(os.getenv("KNOWLEDGE_RETRIEVE_RECALL_N", "20"))
    # 精排开关：模型不可用时可一键回退（关闭后行为与改造前一致）
    RERANK_ENABLED: bool = os.getenv("KNOWLEDGE_RERANK_ENABLED", "true").lower() != "false"
    # 精排分数阈值：低于此值的候选被判为「知识库无相关内容」并丢弃。
    # 默认 0.2 依据 eval/golden_set.jsonl 实测校准（21 条正样本 + 4 条 not_in_kb）：
    #   正样本 reranker top1 最低 0.2795，负样本最高 0.0932，间隔 0.186。
    # ⚠️ 换语料 / 换精排模型 / 扩充评测集后必须重新校准，不要直接沿用。
    # ⚠️ RERANK_ENABLED=false 或精排模型加载失败时不做此判定（降级模式的固有代价）。
    RERANK_THRESHOLD: float = float(os.getenv("KNOWLEDGE_RERANK_THRESHOLD", "0.2"))

    # ── 精排过载保护（企业级必备：过载可降级，而不是把线程池/CPU 打满卡死）──
    # 同时执行的精排任务上限。精排是 CPU 密集（单次 20 候选 × 500 字 ≈ 6.4s），
    # 超限时直接跳过精排、改用 RRF 融合顺序 —— 用少量精度换整体可用性。
    # ⚠️ 默认 1：CPU 环境下 torch 单次前向已用满多核，并发反而互相抢 CPU。
    #    实测（eval/reports/rerank-overload-report.md，30 并发）：
    #      并发 1 → p50 0.6s、吞吐 2.40 rps；并发 4 → p50 15.6s、吞吐 0.40 rps。
    #    有 GPU 或核数很多的部署可调高，但必须用同一脚本重新压测后再定值。
    RERANK_MAX_CONCURRENCY: int = int(os.getenv("KNOWLEDGE_RERANK_MAX_CONCURRENCY", "1"))
    # 单次精排的等待上限（秒）。超时则本请求降级为 RRF 顺序，避免请求被无限拖住。
    # 注意：Python 无法安全中断运行中的线程，被放弃的任务仍会跑完（受并发上限约束）。
    RERANK_TIMEOUT: float = float(os.getenv("KNOWLEDGE_RERANK_TIMEOUT", "15"))

    # ── Embedding 模型 ──
    # 白名单（非法值启动即报错，见 embeddings.validate_embedding_config）：
    #   "sentence-transformers" —— 本地中文模型（默认推荐，BAAI/bge-small-zh-v1.5，512 维）
    #   "default"               —— ChromaDB 内置 all-MiniLM-L6-v2（英文模型，中文检索几乎失效）
    # 注意：deepseek 官方没有 embedding 接口，不能用于向量化。
    # 默认用本地中文模型：all-MiniLM-L6-v2 对中文 query 几乎失效（dense 命中率实测 4.8%）
    EMBEDDING_MODEL: str = os.getenv("KNOWLEDGE_EMBEDDING", "sentence-transformers")
    # sentence-transformers 具体模型名（BGE 中文小模型，512 维，CPU 可跑）
    EMBEDDING_ST_MODEL: str = os.getenv(
        "KNOWLEDGE_EMBEDDING_ST_MODEL", "BAAI/bge-small-zh-v1.5"
    )

    # ── 向量数据库 ──
    # 可选: "chroma" | "faiss" (预留)
    VECTOR_STORE_TYPE: str = os.getenv("KNOWLEDGE_VECTOR_STORE", "chroma")

    # ── Collection 名称 ──
    COLLECTION_NAME: str = "knowledge_base"


# 全局单例
config = KnowledgeConfig()
