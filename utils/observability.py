"""可观测性基础设施：请求追踪 ID + 日志上下文 + 敏感信息脱敏。

为什么需要
==========
1. **日志会「缠绕」**：并发下没有 request_id，一次请求的日志和别人的混在一起，
   线上排查只能靠时间戳猜。这里用 ContextVar 把 ID 绑到协程上下文，
   再通过 LogRecordFactory 自动写进每条日志，业务代码零侵入。
2. **日志会泄密**：异常栈、LLM 返回、用户输入里可能夹带 API Key / JWT / 密码。
   统一在 formatter 出口做正则脱敏，避免「日志里有 key」这类事故。

设计取舍
========
- 不引入 structlog / loguru：现有代码全部是标准库 logging，换框架要改 7000 行。
- 不用 `Logger.addFilter`：filter 只在 logger 自身 handle 时执行，
  子 logger propagate 到 root handler 时会绕过它。LogRecordFactory 是唯一
  对「所有 record 都生效」的钩子。
"""

import json
import logging
import re
import uuid
from contextvars import ContextVar

# ═══════════════════════════════════════════
#  请求上下文
# ═══════════════════════════════════════════

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# 只接受安全字符，防止把换行/控制符塞进日志（日志注入）
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id() -> str:
    """生成 12 位十六进制请求 ID（够用且不刷屏）。"""
    return uuid.uuid4().hex[:12]


def normalize_request_id(raw: str | None) -> str:
    """校验上游传入的 X-Request-ID；不合法就丢弃并重新生成。

    为什么要校验：`X-Request-ID: abc\\nFAKE LOG LINE` 可以把伪造日志写进来，
    也能撑爆日志字段长度。
    """
    if raw:
        candidate = raw.strip()
        if _VALID_REQUEST_ID.match(candidate):
            return candidate
    return new_request_id()


def get_request_id() -> str:
    """当前上下文的请求 ID；不在请求内时返回 "-"。"""
    return request_id_var.get() or "-"


# ═══════════════════════════════════════════
#  敏感信息脱敏
# ═══════════════════════════════════════════

_SCRUB_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # OpenAI / DeepSeek / Moonshot 风格：sk-xxxxxxxx
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"), "sk-***"),
    # Authorization: Bearer xxx
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{10,}"), r"\1***"),
    # key=value / key: value 形式（api_key / token / password / secret / authorization）
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|access[_-]?token|token|password|passwd|secret|authorization)"
            r"\s*[=:]\s*)[^\s,;&'\"}{]+"
        ),
        r"\1***",
    ),
    # JWT：eyJhbGciOi... 三段式
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "eyJ***"),
    # URL 里的凭证：https://user:pass@host
    (re.compile(r"(?i)(://)[^/\s:@]+:[^/\s@]+@"), r"\1***:***@"),
)


def scrub(text: str) -> str:
    """对日志文本做脱敏。输入非字符串时原样返回（格式化由调用方负责）。"""
    if not isinstance(text, str) or not text:
        return text
    for pattern, repl in _SCRUB_RULES:
        text = pattern.sub(repl, text)
    return text


# ═══════════════════════════════════════════
#  LogRecord 工厂：给每条日志注入 request_id
# ═══════════════════════════════════════════

_ORIGINAL_FACTORY = logging.getLogRecordFactory()
_FACTORY_INSTALLED = False


def _record_factory(*args, **kwargs):
    record = _ORIGINAL_FACTORY(*args, **kwargs)
    record.request_id = request_id_var.get() or "-"
    return record


def install_log_context() -> None:
    """把 request_id 注入所有 LogRecord（幂等，可重复调用）。"""
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return
    logging.setLogRecordFactory(_record_factory)
    _FACTORY_INSTALLED = True


# ═══════════════════════════════════════════
#  日志格式化
# ═══════════════════════════════════════════


class ScrubbingFormatter(logging.Formatter):
    """在输出前统一脱敏的 Formatter。

    放在 formatter 而不是 logger 层：日志文本最终一定经过 formatter，
    包括 exc_info 展开的堆栈，这一层兜底最彻底。
    """

    def format(self, record: logging.LogRecord) -> str:
        return scrub(super().format(record))


class JsonFormatter(logging.Formatter):
    """单行 JSON 日志，供 Loki / ELK / 云日志采集（LOG_FORMAT=json 启用）。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # 业务侧通过 logger.info("...", extra={"k": v}) 附加的结构化字段
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_KEYS or key.startswith("_"):
                continue
            if key in payload:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        return scrub(json.dumps(payload, ensure_ascii=False))


_RESERVED_RECORD_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime", "request_id",
})
