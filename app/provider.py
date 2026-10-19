"""LLM provider adapters. Real calls go through the OpenAI-compatible chat API
(Groq / OpenAI / Gemini); FakeProvider drives controlled tests."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .config import PROVIDERS, Settings
from .errors import AgentError


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None


class Provider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[dict], tools: list[dict], tool_choice: str) -> ChatResult: ...


_DAILY_LIMIT = re.compile(r"per day|TPD|RPD", re.I)


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        if settings.provider not in PROVIDERS:
            raise ValueError(f"unsupported provider {settings.provider!r}")
        key_env, base_url = PROVIDERS[settings.provider]
        api_key = os.getenv(key_env)
        if not api_key:
            raise RuntimeError(f"{key_env} is not set")
        self.name = settings.provider
        self.model = settings.model
        self.settings = settings
        # SDK retries off: timeouts and rate limits are handled explicitly below.
        self._client = OpenAI(api_key=api_key, base_url=base_url,
                              timeout=settings.request_timeout_s, max_retries=0)

    def chat(self, messages: list[dict], tools: list[dict], tool_choice: str) -> ChatResult:
        import openai

        kwargs: dict[str, Any] = dict(
            model=self.model, messages=messages, temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
        )
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = self.settings.extra.get("reasoning_effort", "low")
        for attempt in range(self.settings.rate_limit_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                break
            except openai.APITimeoutError:
                raise AgentError("provider_timeout",
                                 f"provider request exceeded {self.settings.request_timeout_s}s") from None
            except openai.RateLimitError as exc:
                msg = str(exc)
                if _DAILY_LIMIT.search(msg) or attempt == self.settings.rate_limit_retries:
                    raise AgentError("provider_rate_limited", _sanitize(msg)) from None
                wait = _retry_after(exc) or min(60.0, 5.0 * 2 ** attempt)
                time.sleep(min(wait, 90.0))
            except openai.APIStatusError as exc:
                if exc.status_code == 400 and "tool_use_failed" in str(exc):
                    # Provider-side rejection of a call to an undeclared tool.
                    raise AgentError("illegal_tool", f"provider rejected tool call: {_sanitize(str(exc))[:200]}") from None
                raise AgentError("provider_error", f"HTTP {exc.status_code}: {_sanitize(str(exc))}") from None
            except openai.APIError as exc:
                raise AgentError("provider_error", _sanitize(str(exc))) from None
        choice = resp.choices[0]
        msg = choice.message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments or "")
                 for tc in (msg.tool_calls or [])]
        usage = {}
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens,
                     "completion_tokens": resp.usage.completion_tokens}
        return ChatResult(content=msg.content, tool_calls=calls, usage=usage,
                          finish_reason=choice.finish_reason)


def _retry_after(exc: Exception) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(value) + 0.5 if value else None
    except Exception:
        return None


def _sanitize(text: str) -> str:
    text = re.sub(r"(sk|gsk|AIza)[-_A-Za-z0-9]{10,}", "[redacted]", text)
    return text[:400]


class FakeProvider:
    """Scripted provider for tests. `script` is a list of ChatResult / Exception, or a
    callable(messages, tools, tool_choice) -> ChatResult. Every payload is recorded."""

    name = "fake"
    model = "fake-model"

    def __init__(self, script: list | Callable):
        self.script = script
        self.calls: list[dict] = []

    def chat(self, messages, tools, tool_choice):
        import copy

        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "tool_choice": tool_choice})
        if callable(self.script):
            item = self.script(messages, tools, tool_choice)
        else:
            item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def build_provider(settings: Settings) -> Provider:
    return OpenAICompatibleProvider(settings)
