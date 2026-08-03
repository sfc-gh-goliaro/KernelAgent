# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic provider implementation."""

import os

from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None


# effort -> thinking.budget_tokens for legacy Claude (Sonnet 4.5, Haiku 4.5,
# older Claude 4). Modern (4.6+) uses output_config.effort instead.
_BUDGET = {"low": 2_000, "medium": 6_000, "high": 12_000, "xhigh": 24_000, "max": 48_000}


def _classify(model_name: str) -> str:
    """Return 'effort' | 'effort_beta' | 'legacy'."""
    m = model_name.lower().replace(".", "-")
    if "opus-4-5" in m:
        return "effort_beta"
    modern = ("opus-5", "sonnet-5", "haiku-5", "opus-4-6", "opus-4-7", "opus-4-8",
              "sonnet-4-6", "fable-5", "mythos-5")
    if any(t in m for t in modern):
        return "effort"
    legacy = ("sonnet-4-5", "haiku-4-5",
              "opus-4-0", "opus-4-1", "opus-4-2", "opus-4-3", "opus-4-4",
              "sonnet-4-0", "sonnet-4-1", "sonnet-4-2", "sonnet-4-3", "sonnet-4-4",
              "haiku-4-0", "haiku-4-1", "haiku-4-2", "haiku-4-3", "haiku-4-4",
              "claude-3")
    if any(t in m for t in legacy):
        return "legacy"
    return "effort"


def _reasoning_kwargs(model_name: str) -> tuple[dict, dict, dict]:
    """Return (create_kwargs, extra_body, extra_headers) for reasoning.
    All empty if KERNELAGENT_REASONING_EFFORT is unset."""
    effort = (os.environ.get("KERNELAGENT_REASONING_EFFORT") or "").strip().lower()
    if not effort:
        return {}, {}, {}
    mode = _classify(model_name)
    if mode == "effort":
        return {}, {"output_config": {"effort": effort}}, {}
    if mode == "effort_beta":
        eff = effort if effort in ("low", "medium", "high") else "high"
        return {}, {"output_config": {"effort": eff}}, {"anthropic-beta": "effort-2025-11-24"}
    # legacy
    budget = _BUDGET.get(effort, _BUDGET["high"])
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}, {}, {}


def _extract_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    sys_chunks, body = [], []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str) and c:
                sys_chunks.append(c)
        else:
            body.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return ("\n\n".join(sys_chunks) if sys_chunks else None), body


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    def __init__(self):
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        if ANTHROPIC_AVAILABLE and api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            base_url = os.environ.get("ANTHROPIC_BASE_URL")
            self.client = Anthropic(api_key=api_key, base_url=base_url) if base_url \
                          else Anthropic(api_key=api_key)

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("Anthropic client not available")

        system, body = _extract_system(messages)
        if not body:
            body = [{"role": "user", "content": messages[-1]["content"] if messages else ""}]

        max_tokens = min(kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name))
        create_kwargs, extra_body, extra_headers = _reasoning_kwargs(model_name)
        # Legacy thinking requires max_tokens > budget_tokens.
        thinking = create_kwargs.get("thinking")
        if thinking:
            max_tokens = max(max_tokens, thinking["budget_tokens"] + 4096)

        call_kwargs: dict = dict(
            model=model_name,
            max_tokens=max_tokens,
            messages=body,
            **create_kwargs,
        )
        # Extended thinking disallows non-default temperature.
        if not thinking and "output_config" not in extra_body:
            call_kwargs["temperature"] = kwargs.get("temperature", 0.7)
        if system:
            call_kwargs["system"] = system
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        if extra_headers:
            call_kwargs["extra_headers"] = extra_headers

        response = self.client.messages.create(**call_kwargs)

        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        return LLMResponse(content=text, model=model_name, provider=self.name)

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        return [
            self.get_response(
                model_name,
                messages,
                temperature=kwargs.get("temperature", 0.7) + i * 0.1,
            )
            for i in range(n)
        ]

    def is_available(self) -> bool:
        return ANTHROPIC_AVAILABLE and self.client is not None

    @property
    def name(self) -> str:
        return "anthropic"
