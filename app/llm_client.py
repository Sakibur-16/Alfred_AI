"""
Provider-agnostic LLM abstraction.

Swap providers via LLM_PROVIDER in .env. Each provider implements the same
tiny interface: `.complete(system, user) -> str`. Add a new provider by
adding one class + one branch in `get_llm_client`.
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings

logger = logging.getLogger("alfred.llm")


class LLMError(Exception):
    """Raised when the LLM provider fails or returns unusable content."""


class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text completion from the model."""

    async def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call the model and parse its output as JSON, tolerating stray markdown fences."""
        raw = await self.complete(system_prompt, user_prompt)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON output: %s | raw=%r", exc, raw[:500])
            raise LLMError(f"Model did not return valid JSON: {exc}") from exc


class AnthropicClient(BaseLLMClient):
    def __init__(self, settings: Settings):
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return "".join(parts)
        except Exception as exc:  # noqa: BLE001 - normalize all provider errors
            logger.error("Anthropic call failed: %s", exc)
            raise LLMError(str(exc)) from exc


class OpenAIClient(BaseLLMClient):
    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI

        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._temperature = settings.llm_temperature

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.error("OpenAI call failed: %s", exc)
            raise LLMError(str(exc)) from exc


_client_singleton: Optional[BaseLLMClient] = None


def get_llm_client() -> BaseLLMClient:
    global _client_singleton
    if _client_singleton is not None:
        return _client_singleton

    settings = get_settings()
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        _client_singleton = AnthropicClient(settings)
    elif provider == "openai":
        _client_singleton = OpenAIClient(settings)
    else:
        raise LLMError(f"Unsupported LLM_PROVIDER: {provider}")
    return _client_singleton


def reset_llm_client_for_tests() -> None:
    global _client_singleton
    _client_singleton = None
