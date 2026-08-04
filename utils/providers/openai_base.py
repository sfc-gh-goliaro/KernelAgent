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

"""Base provider for OpenAI-compatible APIs."""

import os
from typing import Any
import logging
from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment

logger = logging.getLogger(__name__)

# Snowflake Cortex / snowhouse client identity (mirrors kernelguy's
# X-SNOWFLAKE-APPLICATION + User-Agent contract).
_SNOWFLAKE_APP = "kernelagent"
_SNOWHOUSE_MAX_RETRIES = 12
_SNOWHOUSE_TIMEOUT_S = 600.0
_SNOWHOUSE_EMPTY_RETRIES = 3


def _model_slug(model_name: str) -> str:
    """Normalize vendor-prefixed Cortex names (e.g. openai-gpt-5.5 → gpt-5.5)."""
    m = (model_name or "").strip().lower()
    for prefix in ("openai-", "azure-", "snowflake-"):
        if m.startswith(prefix):
            m = m[len(prefix) :]
    return m


def _is_reasoning_model(model_name: str) -> bool:
    m = _model_slug(model_name)
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _is_gpt5_family(model_name: str) -> bool:
    return "gpt-5" in _model_slug(model_name)


def _reasoning_effort(kwargs: dict[str, Any] | None = None) -> str | None:
    """User-selected reasoning effort, or None to let the server default.

    Prefers KERNELAGENT_REASONING_EFFORT; falls back to high when the caller
    passes high_reasoning_effort=True (worker/agent convention).
    """
    v = os.environ.get("KERNELAGENT_REASONING_EFFORT")
    if v and v.strip():
        return v.strip()
    if kwargs and kwargs.get("high_reasoning_effort"):
        return "high"
    return None


def _extract_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    sys_chunks, body = [], []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str) and c:
                sys_chunks.append(c)
        else:
            body.append(m)
    return ("\n\n".join(sys_chunks) if sys_chunks else None), body


def _to_responses_input(body_messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for m in body_messages:
        role = m["role"]
        text = m.get("content", "") or ""
        block_type = "output_text" if role == "assistant" else "input_text"
        out.append({"role": role, "content": [{"type": block_type, "text": text}]})
    return out


def _responses_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for c in getattr(item, "content", None) or []:
            ctype = getattr(c, "type", None)
            if ctype in ("output_text", "text"):
                t = getattr(c, "text", None) or ""
                if t:
                    parts.append(t)
    return "".join(parts)


def _message_text_from_item(item: Any) -> str:
    if item is None or getattr(item, "type", None) != "message":
        return ""
    parts: list[str] = []
    for c in getattr(item, "content", None) or []:
        if getattr(c, "type", None) in ("output_text", "text"):
            t = getattr(c, "text", None) or ""
            if t:
                parts.append(t)
    return "".join(parts)


def _collect_streamed_response(stream: Any) -> tuple[str, Any | None]:
    """Consume a Responses SSE stream; return (text, final Response | None).

    Collects ``response.output_text.delta`` and falls back to message text on
    ``response.output_item.done`` / ``response.completed`` / ``response.incomplete``
    (kernelguy also reconstructs text from output_item.done when deltas are absent).
    """
    chunks: list[str] = []
    final = None
    for event in stream:
        etype = getattr(event, "type", None)
        if etype == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if delta:
                chunks.append(delta)
        elif etype == "response.output_item.done":
            # Prefer deltas when present; otherwise take full message text.
            item_text = _message_text_from_item(getattr(event, "item", None))
            if item_text and not chunks:
                chunks.append(item_text)
            elif item_text and chunks and item_text.startswith("".join(chunks)):
                # Suffix only if final item is a superset of streamed deltas.
                suffix = item_text[len("".join(chunks)) :]
                if suffix:
                    chunks.append(suffix)
        elif etype in ("response.completed", "response.incomplete"):
            final = getattr(event, "response", None) or final
        elif etype == "response.failed":
            resp = getattr(event, "response", None)
            err = getattr(resp, "error", None) if resp is not None else None
            msg = getattr(err, "message", None) if err is not None else None
            raise RuntimeError(msg or "Responses stream failed")
        elif etype == "error":
            msg = getattr(event, "message", None) or str(event)
            raise RuntimeError(f"Responses stream error: {msg}")
    text = "".join(chunks)
    if not text and final is not None:
        text = _responses_output_text(final)
    return text, final


def snowflake_application_headers() -> dict[str, str]:
    """Headers Snowflake Cortex expects for app identification / routing."""
    return {
        "X-SNOWFLAKE-APPLICATION": _SNOWFLAKE_APP,
        "User-Agent": _SNOWFLAKE_APP,
    }


try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAICompatibleProvider(BaseProvider):
    """Base provider for OpenAI-compatible APIs."""

    def __init__(self, api_key_env: str, base_url: str | None = None):
        self.api_key_env = api_key_env
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._snowhouse = bool(self.base_url and "snowhouse" in self.base_url)
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        """Initialize OpenAI-compatible client."""
        if not OPENAI_AVAILABLE:
            return

        api_key = self._get_api_key(self.api_key_env)
        if api_key:
            # Configure proxy using centralized utility function
            self._original_proxy_env = configure_proxy_environment()

            client_kwargs: dict[str, Any] = {"api_key": api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self._snowhouse:
                # Match kernelguy: identify as a Cortex app and retry 5xx hard.
                client_kwargs["default_headers"] = snowflake_application_headers()
                client_kwargs["max_retries"] = _SNOWHOUSE_MAX_RETRIES
                client_kwargs["timeout"] = _SNOWHOUSE_TIMEOUT_S
            self.client = OpenAI(**client_kwargs)

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Get single response."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        if self._snowhouse:
            return self._get_snowhouse_response(model_name, messages, **kwargs)

        api_params = self._build_api_params(model_name, messages, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logger.info(
            "OpenAI chat response (single): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_name,
            provider=self.name,
            usage=response.usage.dict()
            if hasattr(response, "usage") and response.usage
            else None,
        )

    def _get_snowhouse_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Snowflake Cortex Responses call (streaming, kernelguy-style)."""
        last_meta: dict[str, Any] = {}
        for attempt in range(1, _SNOWHOUSE_EMPTY_RETRIES + 1):
            params = self._build_responses_params(model_name, messages, **kwargs)
            # On empty-content retries, drop reasoning summary pressure a bit by
            # requesting more output budget if the caller asked for a small cap.
            if attempt > 1:
                cur = int(params.get("max_output_tokens") or 8192)
                params["max_output_tokens"] = max(cur, min(cur * 2, 32000))
                logger.warning(
                    "Snowhouse empty/incomplete content retry %s/%s "
                    "(max_output_tokens=%s)",
                    attempt,
                    _SNOWHOUSE_EMPTY_RETRIES,
                    params["max_output_tokens"],
                )

            stream = self.client.responses.create(**params)
            text, final = _collect_streamed_response(stream)
            usage = None
            status = None
            incomplete = None
            if final is not None:
                status = getattr(final, "status", None)
                incomplete = getattr(final, "incomplete_details", None)
                if getattr(final, "usage", None) is not None:
                    usage = final.usage.model_dump()

            last_meta = {
                "status": status,
                "incomplete_details": incomplete,
                "usage": usage,
                "text_len": len(text or ""),
            }

            if text and text.strip():
                if status == "incomplete":
                    logger.warning(
                        "Snowhouse response incomplete but has content "
                        "(details=%s usage=%s)",
                        incomplete,
                        usage,
                    )
                return LLMResponse(
                    content=text,
                    model=model_name,
                    provider=self.name,
                    usage=usage,
                )

            logger.warning(
                "Snowhouse returned empty content (attempt %s/%s): status=%s "
                "incomplete=%s usage=%s",
                attempt,
                _SNOWHOUSE_EMPTY_RETRIES,
                status,
                incomplete,
                usage,
            )

        raise RuntimeError(
            "Snowhouse Responses returned empty content after "
            f"{_SNOWHOUSE_EMPTY_RETRIES} attempts: {last_meta}"
        )

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        """Get multiple responses using n parameter."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        if self._snowhouse:
            # Responses API doesn't support n; loop with temperature bump.
            out = []
            base_t = float(kwargs.get("temperature", 0.7))
            for i in range(max(1, n)):
                kw = dict(kwargs)
                kw["temperature"] = base_t + (0.1 * i if n > 1 else 0.0)
                out.append(self.get_response(model_name, messages, **kw))
            return out

        api_params = self._build_api_params(model_name, messages, n=n, **kwargs)
        response = self.client.chat.completions.create(**api_params)
        logger.info(
            "OpenAI chat response (multi): %s",
            getattr(response, "model_dump", lambda: str(response))(),
        )

        return [
            LLMResponse(
                content=choice.message.content,
                model=model_name,
                provider=self.name,
                usage=response.usage.dict()
                if hasattr(response, "usage") and response.usage
                else None,
            )
            for choice in response.choices
        ]

    def _build_responses_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        system, body = _extract_system(messages)
        # Reasoning models need headroom beyond the visible answer; Cortex
        # model ids are often vendor-prefixed (openai-gpt-5.5), so use the
        # provider limit helper which understands those names.
        requested = kwargs.get("max_tokens", 24000)
        max_out = min(int(requested), self.get_max_tokens_limit(model_name))
        params: dict[str, Any] = {
            "model": model_name,
            "input": _to_responses_input(body),
            "max_output_tokens": max_out,
            # Stream like kernelguy — more reliable on snowhouse than a single
            # buffered non-stream response for long GPT-5.x generations.
            "stream": True,
            "store": False,
            "metadata": {"client": _SNOWFLAKE_APP},
            # Snowflake-specific app tag (also sent as HTTP header).
            "extra_body": {"client_metadata": {"client": _SNOWFLAKE_APP}},
        }
        if system:
            params["instructions"] = system
        effort = _reasoning_effort(kwargs)
        if effort:
            params["reasoning"] = {"effort": effort, "summary": "auto"}
        return params

    def _build_api_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        """Build API parameters for OpenAI-compatible call."""
        params = {
            "model": model_name,
            "messages": messages,
        }

        # GPT-5 and o-series models pin their own sampling behaviour
        if not (_is_gpt5_family(model_name) or _model_slug(model_name).startswith("o")):
            params["temperature"] = kwargs.get("temperature", 0.7)

        # Use max_completion_tokens for newer models like GPT-5, fallback to max_tokens
        max_tokens_value = min(
            kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
        )
        if _is_gpt5_family(model_name) or _model_slug(model_name).startswith("o"):
            params["max_completion_tokens"] = max_tokens_value
        else:
            params["max_tokens"] = max_tokens_value

        # Add n parameter if specified
        if "n" in kwargs:
            params["n"] = kwargs["n"]

        # Only send reasoning_effort when the user explicitly opts in.
        effort = _reasoning_effort(kwargs)
        if effort and _is_reasoning_model(model_name):
            params["reasoning_effort"] = effort

        return params

    def is_available(self) -> bool:
        """Check if provider is available."""
        return OPENAI_AVAILABLE and self.client is not None

    def supports_multiple_completions(self) -> bool:
        """OpenAI-compatible APIs support native multiple completions."""
        return True
