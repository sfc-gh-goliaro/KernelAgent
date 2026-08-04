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

"""OpenAI provider implementation."""

from .openai_base import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI API provider."""

    def __init__(self):
        super().__init__(api_key_env="OPENAI_API_KEY")

    @property
    def name(self) -> str:
        return "openai"

    def get_max_tokens_limit(self, model_name: str) -> int:
        """Get max tokens limit for OpenAI / Cortex-prefixed models."""
        from .openai_base import _is_gpt5_family, _model_slug

        slug = _model_slug(model_name)
        if _is_gpt5_family(model_name) or slug.startswith(("gpt-4", "o3", "o1", "o4")):
            return 32000
        elif slug.startswith("gpt-3.5"):
            return 16000
        else:
            return 8192
