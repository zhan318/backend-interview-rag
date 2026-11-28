"""接口层测试：FastAPI 路由的行为契约。

与 `tests/` 下其他文件的区别：那些是**模块级单测**（直接调函数），本文件是
**接口级测试**（穿过 HTTP 栈），因此能覆盖路由装饰器、依赖注入、Pydantic 校验、
中间件（限流 / 统一异常处理）这些"只有走完请求链路才会触发"的逻辑。

覆盖六类关注点：
1. 正常路径 —— 不只断言 200，还验证响应体关键字段与副作用（写库 / 写文件）
2. 参数校验 —— 非法输入返回 4xx 且错误可读，而不是 500
3. 认证 —— 无 token / 坏 token / 登出后 token 失效
4. 越权 —— A 用户不能读写 B 用户的会话（IDOR 类缺陷）
5. 依赖故障 —— Agent / 向量库挂掉时降级或 500 兜底，不泄漏内部堆栈
6. 限流 —— 敏感接口 5 次/分钟后返回 429

外部依赖一律使用 `tests/fakes.py` 的替身：不联网、不加载模型。
"""

from fakes import FakeAgent, FakeVectorStore, fake_llm_chat

import api

# ═══════════════════════════════════════════
#  认证
# ═══════════════════════════════════════════


class TestAuth:
    def test_register_returns_token_and_user_id(self, client):
        resp = client.post(
            "/auth/register",
            json={"username": "newbie", "email": "newbie@test.local", "password": "pass123456"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["token"]
        assert body["user_id"] > 0

    def test_register_rejects_duplicate_username(self, client, user):
        resp = client.post(
            "/auth/register",
            json={"username": "alice", "email": "another@test.local", "password": "pass123456"},
        )
        assert resp.status_code == 400
        assert "用户名" in resp.json()["error"]["message"]

    def test_register_rejects_too_short_password(self, client):
        resp = client.post(
            "/auth/register",
            json={"username": "shorty", "email": "shorty@test.local", "password": "123"},
        )
        assert resp.status_code == 400
        assert "6 位" in resp.json()["error"]["message"]

    def test_login_accepts_username_or_email(self, client, user):
        by_name = client.post(
            "/auth/login", json={"username": "alice", "password": "pass123456"}
        )
        assert by_name.status_code == 200
        by_email = client.post(
            "/auth/login", json={"username": "alice@test.local", "password": "pass123456"}
        )
        assert by_email.status_code == 200

    def test_login_rejects_wrong_password(self, client, user):
        resp = client.post("/auth/login", json={"username": "alice", "password": "wrong-pass"})
        # 认证失败统一 401（而非 400），与"参数不合法"区分开
        assert resp.status_code == 401

    def test_protected_endpoint_requires_token(self, client):
        assert client.get("/user/profile").status_code == 401

    def test_protected_endpoint_rejects_invalid_token(self, client):
        resp = client.get("/user/profile", headers={"Authorization": "Bearer not-a-real-token"})
        assert resp.status_code == 401

    def test_logout_invalidates_token(self, client, user):
        assert client.post("/auth/logout", headers=user["headers"]).status_code == 200
        # 同一 token 再访问受保护接口应被拒
        assert client.get("/user/profile", headers=user["headers"]).status_code == 401

    def test_sensitive_endpoint_rate_limited_after_five_calls(self, client):
        codes = []
        for i in range(6):
            resp = client.post(
                "/auth/register",
                json={
                    "username": f"burst{i}",
                    "email": f"burst{i}@test.local",
                    "password": "pass123456",
                },
            )
            codes.append(resp.status_code)
        assert codes[:5] == [200] * 5
        assert codes[5] == 429
        assert "频繁" in resp.json()["error"]["message"]


# ═══════════════════════════════════════════
#  用户中心
# ═══════════════════════════════════════════


class TestUserProfile:
    def test_get_profile_returns_current_user(self, client, user):
        resp = client.get("/user/profile", headers=user["headers"])
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "alice"
        assert body["id"] == user["id"]

    def test_update_display_name_persists(self, client, user):
        resp = client.put(
            "/user/profile", json={"display_name": "面试选手"}, headers=user["headers"]
        )
        assert resp.status_code == 200
        assert client.get("/user/profile", headers=user["headers"]).json()["display_name"] == "面试选手"

    def test_update_display_name_rejects_blank(self, client, user):
        resp = client.put("/user/profile", json={"display_name": "   "}, headers=user["headers"])
        assert resp.status_code == 400

    def test_change_password_requires_correct_old_password(self, client, user):
        resp = client.put(
            "/user/password",
            json={"old_password": "wrong-old", "new_password": "newpass123"},
            headers=user["headers"],
        )
        assert resp.status_code == 400

    def test_change_password_allows_login_with_new_password(self, client, user):
        resp = client.put(
            "/user/password",
            json={"old_password": "pass123456", "new_password": "newpass123"},
            headers=user["headers"],
        )
        assert resp.status_code == 200
        relogin = client.post("/auth/login", json={"username": "alice", "password": "newpass123"})
        assert relogin.status_code == 200

    def test_change_password_rejects_short_new_password(self, client, user):
        resp = client.put(
            "/user/password",
            json={"old_password": "pass123456", "new_password": "123"},
            headers=user["headers"],
        )
        assert resp.status_code == 400


# ═══════════════════════════════════════════
#  会话管理（含越权）
# ═══════════════════════════════════════════


class TestSessions:
    def test_session_crud_flow(self, client, user):
        created = client.post("/user/sessions", headers=user["headers"])
        assert created.status_code == 200
        sid = created.json()["session_id"]

        listed = client.get("/user/sessions", headers=user["headers"]).json()["sessions"]
        assert any(s["session_id"] == sid for s in listed)

        renamed = client.patch(
            f"/user/sessions/{sid}", json={"label": "RRF 调试"}, headers=user["headers"]
        )
        assert renamed.status_code == 200
        assert renamed.json()["label"] == "RRF 调试"

        assert client.delete(f"/user/sessions/{sid}", headers=user["headers"]).status_code == 200
        assert (
            client.get(f"/user/sessions/{sid}/messages", headers=user["headers"]).status_code == 404
        )

    def test_rename_session_rejects_blank_label(self, client, user):
        sid = client.post("/user/sessions", headers=user["headers"]).json()["session_id"]
        resp = client.patch(f"/user/sessions/{sid}", json={"label": "  "}, headers=user["headers"])
        assert resp.status_code == 400

    def test_messages_return_persisted_history(self, client, user):
        sid = client.post("/user/sessions", headers=user["headers"]).json()["session_id"]
        api.memory.save_message(sid, "user", "第一个问题")
        api.memory.save_message(sid, "assistant", "第一个回答")

        body = client.get(f"/user/sessions/{sid}/messages", headers=user["headers"]).json()
        assert [m["content"] for m in body["messages"]] == ["第一个问题", "第一个回答"]

    def test_cannot_read_another_users_session(self, client, register_user):
        alice = register_user("alice")
        bob = register_user("bob")
        sid = client.post("/user/sessions", headers=alice["headers"]).json()["session_id"]

        resp = client.get(f"/user/sessions/{sid}/messages", headers=bob["headers"])
        assert resp.status_code == 404

    def test_cannot_delete_another_users_session(self, client, register_user):
        alice = register_user("alice")
        bob = register_user("bob")
        sid = client.post("/user/sessions", headers=alice["headers"]).json()["session_id"]

        assert client.delete(f"/user/sessions/{sid}", headers=bob["headers"]).status_code == 404
        # 确认 alice 的会话仍在
        listed = client.get("/user/sessions", headers=alice["headers"]).json()["sessions"]
        assert any(s["session_id"] == sid for s in listed)


# ═══════════════════════════════════════════
#  聊天（Agent 全部替换为替身）
# ═══════════════════════════════════════════


class TestChat:
    def test_chat_returns_agent_reply_and_passes_context(self, client, user, monkeypatch):
        fake = FakeAgent(reply="收到")
        monkeypatch.setattr(api, "get_agent", lambda session_id: fake)
        # 智能标题是 fire-and-forget 副作用，不属于接口契约，这里关掉以免产生游离 task
        monkeypatch.setattr(api, "_is_default_label", lambda label: False)
        monkeypatch.setattr("chatbot.chatbot.chat", fake_llm_chat)

        resp = client.post(
            "/chat",
            json={"message": "在吗", "session_id": "s-chat", "model_provider": "deepseek"},
            headers=user["headers"],
        )
        assert resp.status_code == 200
        assert resp.json()["reply"] == "收到:在吗"
        assert resp.json()["session_id"] == "s-chat"
        # 路由必须把当前用户身份透传给 Agent（多用户隔离的前提）
        assert fake.calls[0]["user_id"] == user["id"]

    def test_chat_wraps_agent_failure_as_500(self, client, user, monkeypatch):
        fake = FakeAgent()
        fake.raise_error = True
        monkeypatch.setattr(api, "get_agent", lambda session_id: fake)
        monkeypatch.setattr("chatbot.chatbot.chat", fake_llm_chat)

        resp = client.post("/chat", json={"message": "在吗"}, headers=user["headers"])
        assert resp.status_code == 500
        assert "Agent 处理失败" in resp.json()["error"]["message"]

    def test_chat_stream_emits_sse_events(self, client, user, monkeypatch):
        fake = FakeAgent(reply="流")
        monkeypatch.setattr(api, "get_agent", lambda session_id: fake)

        resp = client.post("/chat/stream", json={"message": "讲一下"}, headers=user["headers"])
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "data: " in resp.text
        assert '"type": "token"' in resp.text
        assert '"type": "done"' in resp.text


# ═══════════════════════════════════════════
#  知识库
# ═══════════════════════════════════════════


class TestKnowledge:
    def test_query_returns_pipeline_answer(self, client, monkeypatch):
        monkeypatch.setattr(api, "rag_query", lambda question: f"基于笔记回答:{question}")
        resp = client.post("/knowledge/query", json={"question": "RRF 怎么融合"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "基于笔记回答:RRF 怎么融合"

    def test_index_returns_chunk_count(self, client, monkeypatch):
        monkeypatch.setattr(api, "index_documents", lambda: 42)
        resp = client.post("/knowledge/index")
        assert resp.status_code == 200
        assert resp.json()["indexed_chunks"] == 42

    def test_upload_rejects_unsupported_extension(self, client):
        resp = client.post("/knowledge/upload", files={"file": ("evil.exe", b"MZ")})
        assert resp.status_code == 400
        assert "不支持的文件格式" in resp.json()["error"]["message"]

    def test_upload_writes_file_to_upload_dir(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
        resp = client.post("/knowledge/upload", files={"file": ("notes.md", b"# RRF")})
        assert resp.status_code == 200
        assert resp.json()["filename"] == "notes.md"
        assert (tmp_path / "notes.md").read_bytes() == b"# RRF"


# ═══════════════════════════════════════════
#  MBTI
# ═══════════════════════════════════════════


class TestMBTI:
    def test_questions_returns_full_bank(self, client, user):
        body = client.get("/mbti/questions", headers=user["headers"]).json()
        assert body["total"] == 32
        assert len(body["questions"]) == 32

    def test_analyze_rejects_incomplete_answers(self, client, user):
        resp = client.post(
            "/mbti/analyze",
            json={"ideal_role": "后端工程师", "answers": [0] * 10},
            headers=user["headers"],
        )
        assert resp.status_code == 400
        assert "全部" in resp.json()["error"]["message"]

    def test_analyze_rejects_blank_ideal_role(self, client, user):
        resp = client.post(
            "/mbti/analyze",
            json={"ideal_role": "  ", "answers": [0] * 32},
            headers=user["headers"],
        )
        assert resp.status_code == 400

    def test_analyze_returns_type_and_archives_history(self, client, user, monkeypatch):
        async def fake_analyze(mbti: str, role: str) -> str:
            return f"{mbti} 适配 {role}"

        monkeypatch.setattr("mbti.analyze_fit_async", fake_analyze)
        resp = client.post(
            "/mbti/analyze",
            json={"ideal_role": "后端工程师", "answers": [0] * 32},
            headers=user["headers"],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["mbti"]) == 4
        assert body["analysis"] == f"{body['mbti']} 适配 后端工程师"

        # 副作用：结果应落库，供历史对比
        history = client.get("/mbti/history", headers=user["headers"]).json()
        assert history["total"] == 1


# ═══════════════════════════════════════════
#  笔记与面试归档（写盘依赖用替身）
# ═══════════════════════════════════════════


class TestNotes:
    def test_save_rejects_empty_content(self, client, user):
        resp = client.post("/notes/save", json={"content": "   "}, headers=user["headers"])
        assert resp.status_code == 400

    def test_save_routes_note_to_workbench_folder(self, client, user, monkeypatch):
        captured = {}

        def fake_save(note: str, content: str) -> dict:
            captured["note"] = note
            captured["content"] = content
            return {"ok": True, "path": f"/vault/{note}.md"}

        monkeypatch.setattr("tools.obsidian.save_note_to_vault", fake_save)
        resp = client.post(
            "/notes/save",
            json={"title": "RRF 笔记", "content": "正文"},
            headers=user["headers"],
        )
        assert resp.status_code == 200
        assert captured["note"] == "工作台笔记/RRF 笔记"
        assert captured["content"] == "正文"

    def test_save_sanitizes_illegal_filename_chars(self, client, user, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "tools.obsidian.save_note_to_vault",
            lambda note, content: captured.update(note=note) or {"ok": True, "path": "x"},
        )
        client.post(
            "/notes/save",
            json={"title": 'a/b:c*d?"e"', "content": "正文"},
            headers=user["headers"],
        )
        # 目录前缀「工作台笔记/」是合法的，只需检查标题部分不含路径分隔符与
        # Windows 非法字符，否则会写到预期目录之外
        title_part = captured["note"].split("/", 1)[1]
        for illegal in ("/", ":", "*", "?", '"'):
            assert illegal not in title_part

    def test_save_session_requires_ownership(self, client, register_user):
        alice = register_user("alice")
        bob = register_user("bob")
        sid = client.post("/user/sessions", headers=alice["headers"]).json()["session_id"]

        resp = client.post(
            "/notes/save-session", json={"session_id": sid}, headers=bob["headers"]
        )
        assert resp.status_code == 404


class TestInterview:
    def test_save_rejects_foreign_session(self, client, register_user):
        alice = register_user("alice")
        bob = register_user("bob")
        sid = client.post("/user/sessions", headers=alice["headers"]).json()["session_id"]

        resp = client.post("/interview/save", json={"session_id": sid}, headers=bob["headers"])
        assert resp.status_code == 403

    def test_finish_rejects_foreign_session(self, client, register_user):
        alice = register_user("alice")
        bob = register_user("bob")
        sid = client.post("/user/sessions", headers=alice["headers"]).json()["session_id"]

        resp = client.post("/interview/finish", json={"session_id": sid}, headers=bob["headers"])
        assert resp.status_code == 403


# ═══════════════════════════════════════════
#  健康检查与首页（向量库依赖用替身）
# ═══════════════════════════════════════════


class TestHealth:
    def test_healthy_when_database_and_chroma_ok(self, client, monkeypatch):
        monkeypatch.setattr(
            "knowledge.vector_store.get_vector_store", lambda name: FakeVectorStore(3)
        )
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["checks"] == {"database": True, "chroma": True}

    def test_degraded_when_chroma_unavailable(self, client, monkeypatch):
        """深度检查（?deep=true）才真正打开 collection；默认轻量检查不碰向量库。"""

        def boom(name: str):
            raise RuntimeError("chroma 连接失败")

        monkeypatch.setattr("knowledge.vector_store.get_vector_store", boom)
        body = client.get("/health?deep=true").json()
        assert body["status"] == "degraded"
        assert body["checks"]["database"] is True
        assert "chroma 连接失败" in body["checks"]["chroma"]

    def test_root_reports_active_status(self, client, monkeypatch):
        monkeypatch.setattr(
            "knowledge.pipeline.get_vault_metadata", lambda: {"configured": False}
        )
        body = client.get("/").json()
        assert body["status"] == "active"
        assert body["obsidian"]["configured"] is False


# ═══════════════════════════════════════════
#  用户级 LLM 密钥（脱敏是安全契约）
# ═══════════════════════════════════════════


class TestLLMKeys:
    def test_save_list_and_delete_key(self, client, user):
        saved = client.put(
            "/user/llm-keys",
            json={"provider": "deepseek", "api_key": "sk-secret-123456"},
            headers=user["headers"],
        )
        assert saved.status_code == 200

        body = client.get("/user/llm-keys", headers=user["headers"]).json()
        assert body["configured_providers"] == ["deepseek"]
        # 安全契约：列表接口只能返回脱敏值，明文 key 不得出网
        assert body["keys"][0]["api_key_masked"] == "****3456"
        assert "sk-secret-123456" not in str(body)

        assert (
            client.delete("/user/llm-keys/deepseek", headers=user["headers"]).status_code == 200
        )
        assert (
            client.delete("/user/llm-keys/deepseek", headers=user["headers"]).status_code == 404
        )

    def test_save_rejects_unknown_provider(self, client, user):
        resp = client.put(
            "/user/llm-keys",
            json={"provider": "unknown-vendor", "api_key": "sk-x"},
            headers=user["headers"],
        )
        assert resp.status_code == 400

    def test_models_lists_catalog_and_default(self, client, user):
        body = client.get("/models", headers=user["headers"]).json()
        assert body["models"]
        assert body["default_provider"]
        assert body["model_catalog"]


# ═══════════════════════════════════════════
#  看板与 OAuth 未配置分支
# ═══════════════════════════════════════════


class TestDashboard:
    def test_stats_aggregate_current_user(self, client, user, monkeypatch):
        monkeypatch.setattr(
            "knowledge.pipeline.get_vault_metadata", lambda: {"configured": False}
        )
        body = client.get("/dashboard/stats", headers=user["headers"]).json()
        # 注册时会自动创建默认会话
        assert body["session_count"] >= 1
        assert body["mbti_count"] == 0
        assert body["notes_count"] == 0


class TestGithubOAuth:
    def test_login_returns_501_when_not_configured(self, client, monkeypatch):
        monkeypatch.setattr("config.GITHUB_CLIENT_ID", "")
        resp = client.get("/auth/github/login")
        assert resp.status_code == 501
        assert "未配置" in resp.json()["error"]["message"]

    def test_callback_requires_code(self, client):
        resp = client.get("/auth/github/callback")
        assert resp.status_code == 400
        assert "授权码" in resp.json()["error"]["message"]
