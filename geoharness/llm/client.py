"""Normalised LLM client supporting OpenAI, Anthropic, and DeepSeek backends.

Usage::

    client = create_client()             # auto-detect from env vars
    client = create_client("deepseek")   # force DeepSeek
    response = client.chat(messages, tools=my_tools)

Proxy: set ``GEHARNESS_PROXY`` in ``.env`` or environment, e.g.
``GEHARNESS_PROXY=http://127.0.0.1:7897``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx


# ── .env loading ───────────────────────────────────────────────────────────────


def _load_dotenv() -> None:
    """Load .env from the package directory if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value


_load_dotenv()


# ── Proxy ──────────────────────────────────────────────────────────────────────


def _proxy_url() -> str | None:
    """Return proxy URL from env, checking GEHARNESS_PROXY first."""
    for var in ("GEHARNESS_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(var)
        if value:
            return value
    return None


def _http_client() -> httpx.Client | None:
    """Build an httpx.Client with proxy if configured, else None."""
    proxy = _proxy_url()
    if proxy:
        return httpx.Client(proxy=proxy, timeout=120.0)
    return httpx.Client(timeout=120.0)


# ── Normalised response types ──────────────────────────────────────────────────


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ── Client protocol ────────────────────────────────────────────────────────────


class LLMClient(Protocol):
    """Protocol for LLM backends."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        ...


# ── OpenAI backend ─────────────────────────────────────────────────────────────


class OpenAIClient:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not set")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        from openai import OpenAI  # type: ignore[import-untyped]

        client = OpenAI(api_key=self._api_key, http_client=_http_client())
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        completion = client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
                )

        return LLMResponse(content=message.content, tool_calls=tool_calls)


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
        )
    return openai_tools


# ── Anthropic backend ──────────────────────────────────────────────────────────


class AnthropicClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(
            api_key=self._api_key,
            http_client=_http_client(),
        )

        system_prompt, user_messages = _extract_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": user_messages,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        response = client.messages.create(**kwargs)

        tool_calls: list[ToolCall] = []
        text_content: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    )
                )

        return LLMResponse(
            content="\n".join(text_content) if text_content else None,
            tool_calls=tool_calls,
        )


def _extract_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system message and convert to Anthropic message format."""
    system_prompts: list[str] = []
    user_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            system_prompts.append(content)
        elif role == "user":
            user_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            converted = _convert_assistant_message(msg)
            if converted:
                user_messages.append(converted)
        elif role == "tool":
            user_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content,
                        }
                    ],
                }
            )

    system = "\n\n".join(system_prompts) if system_prompts else None
    return system, user_messages


def _convert_assistant_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an assistant message with potential tool calls to Anthropic format."""
    tool_calls: list[dict[str, Any]] = msg.get("tool_calls", [])
    content = msg.get("content") or ""

    if not tool_calls:
        return {"role": "assistant", "content": content}

    blocks: list[dict[str, Any]] = []
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in tool_calls:
        function = tc.get("function", {})
        blocks.append(
            {
                "type": "tool_use",
                "id": tc["id"],
                "name": function.get("name", ""),
                "input": json.loads(function.get("arguments", "{}"))
                if isinstance(function.get("arguments"), str)
                else function.get("arguments", {}),
            }
        )
    return {"role": "assistant", "content": blocks}


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anthropic_tools: list[dict[str, Any]] = []
    for tool in tools:
        anthropic_tools.append(
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
        )
    return anthropic_tools


# ── DeepSeek backend ────────────────────────────────────────────────────────────


class DeepSeekClient:
    """OpenAI-compatible client pointed at api.deepseek.com.

    Uses the standard ``openai`` SDK with a custom base URL.
    """

    DEEPSEEK_BASE = "https://api.deepseek.com"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. "
                "Place it in geoharness/.env as DEEPSEEK_API_KEY=sk-..."
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        from openai import OpenAI  # type: ignore[import-untyped]

        client = OpenAI(
            api_key=self._api_key,
            base_url=self.DEEPSEEK_BASE,
            http_client=_http_client(),
        )
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        completion = client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
                )

        return LLMResponse(content=message.content, tool_calls=tool_calls)


# ── Factory ────────────────────────────────────────────────────────────────────


def create_client(
    backend: Literal["auto", "openai", "anthropic", "deepseek"] = "auto",
    model: str | None = None,
) -> LLMClient:
    """Create an LLM client, auto-detecting the backend from environment variables.

    Parameters
    ----------
    backend : str
        ``"auto"`` to detect from ``DEEPSEEK_API_KEY`` / ``ANTHROPIC_API_KEY`` /
        ``OPENAI_API_KEY`` env vars, or ``"deepseek"`` / ``"openai"`` /
        ``"anthropic"`` to force a specific backend.
    model : str | None
        Model name override. Defaults per backend: ``deepseek-chat``,
        ``gpt-4o``, ``claude-sonnet-4-6``.
    """
    if backend == "auto":
        if os.environ.get("DEEPSEEK_API_KEY"):
            backend = "deepseek"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            backend = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            backend = "openai"
        else:
            raise ValueError(
                "No LLM API key found. "
                "Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            )

    if backend == "deepseek":
        return DeepSeekClient(model=model or "deepseek-chat")
    elif backend == "openai":
        return OpenAIClient(model=model or "gpt-4o")
    elif backend == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-4-6")
    else:
        raise ValueError(f"Unknown backend: {backend}")
