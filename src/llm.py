from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


DEFAULT_OPENAI_COMPATIBLE_MODEL = "Qwen/Qwen3.5-397B-A17B"
_TOKEN_ESTIMATE_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    token_usage_source: str = "unknown"


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...

    def complete_with_usage(self, messages: list[dict[str, str]]) -> LLMCompletion:
        ...


class MockLLMClient:
    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.complete_with_usage(messages).content

    def complete_with_usage(self, messages: list[dict[str, str]]) -> LLMCompletion:
        self.calls.append(messages)
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = '{"selected_ids": [], "final_answer": "Mock response."}'
        return LLMCompletion(
            content=content,
            input_tokens=estimate_message_tokens(messages),
            output_tokens=estimate_text_tokens(content),
            elapsed_ms=0,
            token_usage_source="estimated",
        )


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_OPENAI_COMPATIBLE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if not self.base_url:
            raise ValueError("llm.base_url is required in the config file for openai-compatible LLM mode")
        if not self.api_key:
            raise ValueError("llm.api_key is required in the config file for openai-compatible LLM mode")

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.complete_with_usage(messages).content

    def complete_with_usage(self, messages: list[dict[str, str]]) -> LLMCompletion:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ValueError(f"LLM request failed: {exc}") from exc

        try:
            content = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected OpenAI-compatible response: {data}") from exc
        input_tokens, output_tokens, source = _extract_usage(data, messages, content)
        return LLMCompletion(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            token_usage_source=source,
        )


def estimate_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_text_tokens(message.get("role", "") + "\n" + message.get("content", "")) for message in messages)


def estimate_text_tokens(text: str) -> int:
    return len(_TOKEN_ESTIMATE_RE.findall(text))


def _extract_usage(
    data: dict,
    messages: list[dict[str, str]],
    content: str,
) -> tuple[int, int, str]:
    usage = data.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            return prompt_tokens, completion_tokens, "provider"
    return estimate_message_tokens(messages), estimate_text_tokens(content), "estimated"
