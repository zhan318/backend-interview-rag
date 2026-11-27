"""pytest 共享配置：环境隔离 + 临时数据库 + HTTP 测试客户端。

隔离策略（三层，缺一层就会互相污染）：
1. **数据库**：`MEMORY_DB_PATH` 指向一次性临时目录。必须在 import api / auth
   之前设置 —— 这两个模块在导入时就会 `Memory()` 建库，晚了就写到真实库上了。
2. **外部服务**：`OBSIDIAN_VAULT_DIR` 置空，避免启动索引与看板扫描真实笔记库。
3. **进程内状态**：限流器、Agent 池是模块级内存结构，每个测试前必须清空。
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path（无论从哪个目录运行 pytest 都能 import 到项目模块）
ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
for _path in (str(ROOT), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ── 必须在 import api / auth 之前执行（两者模块级都会创建 Memory 实例）──
_TEST_TMP = tempfile.mkdtemp(prefix="growth_companion_test_")
os.environ["MEMORY_DB_PATH"] = str(Path(_TEST_TMP) / "chat_history.db")
os.environ["OBSIDIAN_VAULT_DIR"] = ""

from memory.memory import Memory  # noqa: E402

# 删表顺序：先删引用方，再删被引用方（foreign_keys=ON 下顺序错了会报错）
_CLEANUP_ORDER = (
    "mbti_results",
    "user_llm_keys",
    "user_sessions",
    "auth_tokens",
    "conversations",
    "session_summaries",
    "users",
)


@pytest.fixture
def memory(tmp_path):
    """返回一个使用临时数据库的 Memory 实例（测试间互不污染）。

    预置测试用户（id=1）：启用外键约束（foreign_keys=ON）后，
    mbti_results/user_sessions 等表引用 users.id，插入前必须存在对应用户。
    """
    db = tmp_path / "test_history.db"
    m = Memory(db_path=str(db))
    with m._connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (id, username, email) VALUES (1, 'test', 'test@test.local')"
        )
        # 预置默认会话：register_user 自增创建的用户 id>=2，
        # user 1 需自带 session 才满足 get_stats 的 session_count>=1 断言
        conn.execute(
            "INSERT OR IGNORE INTO user_sessions (user_id, session_id) VALUES (1, 'u1_default')"
        )
        conn.commit()
    return m


@pytest.fixture
def client():
    """FastAPI 测试客户端 + 干净状态。

    不进入 lifespan（不用 `with TestClient(...)`）：启动钩子会做 Obsidian 全量
    索引、连 MCP server，对接口测试是纯噪音。
    """
    from fastapi.testclient import TestClient

    import api

    # 限流器与 Agent 池是跨测试共享的模块级状态，不清会串味
    api._rate_limit_store.clear()
    api._agent_pool.clear()

    # 清空业务表：让每个测试从确定状态开始，避免用户名/会话冲突
    with api.memory._connect() as conn:
        for table in _CLEANUP_ORDER:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()

    return TestClient(api.app)


@pytest.fixture
def register_user(client):
    """注册一个用户并返回其认证信息；可多次调用创建多个用户（越权测试用）。

    注意：/auth/register 有 5 次/分钟的敏感接口限流，单个测试内不要注册超过 5 个。
    """

    def _register(username: str = "tester", password: str = "pass123456") -> dict:
        resp = client.post(
            "/auth/register",
            json={
                "username": username,
                "email": f"{username}@test.local",
                "password": password,
            },
        )
        assert resp.status_code == 200, f"注册失败: {resp.status_code} {resp.text}"
        data = resp.json()
        return {
            "id": data["user_id"],
            "token": data["token"],
            "headers": {"Authorization": f"Bearer {data['token']}"},
        }

    return _register


@pytest.fixture
def user(register_user):
    """默认已登录用户（多数接口测试只需要一个身份）"""
    return register_user("alice")
