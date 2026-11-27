"""测试替身（Test Doubles）：把外部依赖替换成可控的假实现。

设计原则：
1. **不联网、不加载模型** —— 测试必须在 1 秒内完成，不能因为调 LLM 或加载
   sentence-transformers 而变成分钟级。
2. **接口最小化** —— 只实现被测代码真正会用到的那几个属性/方法，不追求完全
   复刻第三方库（过度抽象反而难维护）。
3. **可断言** —— 记录调用参数，让测试能验证"路由是否把参数正确传下去了"。

目前覆盖三类外部依赖：
- LLM 客户端（openai 风格 completion）
- ReAct Agent（/chat 与 /chat/stream 的核心依赖）
- 向量库（chroma collection，/health 与 /obsidian/status 会调用）
"""

from types import SimpleNamespace


class FakeLLMResponse:
    """模拟 openai 的 ChatCompletion 返回结构。

    被测代码（如 api._auto_title_session）只访问
    ``resp.choices[0].message.content``，所以这里只构造这条链路。
    """

    def __init__(self, content: str = "", tool_calls: list | None = None):
        message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
        self.choices = [SimpleNamespace(message=message)]


def fake_llm_chat(messages, tools=None, tool_choice=None, **kwargs):
    """替代 chatbot.chatbot.chat：永不联网，固定返回一句可识别的内容。"""
    return FakeLLMResponse("测试生成的标题")


class FakeAgent:
    """替代真实 ReAct Agent。

    真实 Agent 会调用 LLM 并执行多轮工具，测试中完全不可控。这里只保留
    /chat、/chat/stream 两个路由需要的入口，并记录调用参数供断言。
    """

    def __init__(self, session_id: str = "fake_session", reply: str = "假回复"):
        self.session_id = session_id
        self.reply = reply
        self.calls: list[dict] = []
        # 置为 True 可模拟 Agent 内部异常（验证路由的 500 兜底）
        self.raise_error = False

    async def arun(self, message: str, **kwargs) -> str:
        self.calls.append({"message": message, **kwargs})
        if self.raise_error:
            raise RuntimeError("模拟 Agent 故障")
        return f"{self.reply}:{message}"

    async def arun_stream(self, message: str, **kwargs):
        self.calls.append({"message": message, **kwargs})
        if self.raise_error:
            raise RuntimeError("模拟流式故障")
        text = f"{self.reply}:{message}"
        for ch in text:
            yield {"type": "token", "content": ch}
        yield {"type": "done", "content": text}


class FakeVectorStore:
    """替代 chroma collection：只实现 count()，不打开任何持久化目录。"""

    def __init__(self, count: int = 0):
        self._count = count

    def count(self) -> int:
        return self._count
