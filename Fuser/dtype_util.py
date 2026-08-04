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

"""Canonical activation/weight dtype names for Fuser + KernelAgent pipelines."""

from __future__ import annotations

import re

# CLI / env / extractor aliases → canonical prompt/API names used throughout Fuser.
DTYPE_ALIASES: dict[str, str] = {
    "fp32": "float32",
    "float32": "float32",
    "torch.float32": "float32",
    "torch.float": "float32",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "torch.half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "torch.bfloat16": "bfloat16",
}

CANONICAL_DTYPES = ("float32", "float16", "bfloat16")


def normalize_dtype_name(value: str | None, *, default: str = "bfloat16") -> str:
    """Normalize a user/pipeline/extractor dtype string to float32|float16|bfloat16.

    Accepts bare names (``bf16``, ``float32``), ``torch.*`` forms from the
    subgraph extractor (``torch.float32``), and common aliases.
    """
    if value is None or str(value).strip() == "":
        value = default
    key = str(value).strip().lower()
    # Tolerate accidental wrappers: "<class 'torch.float32'>", "dtype=torch.bfloat16"
    key = key.replace("<class '", "").replace("'>", "")
    if "torch." in key and key not in DTYPE_ALIASES:
        m = re.search(r"torch\.(bfloat16|float16|float32|float|half)\b", key)
        if m:
            key = f"torch.{m.group(1)}"
    if key not in DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported dtype {value!r}. Expected one of: "
            f"{', '.join(sorted(set(DTYPE_ALIASES) | set(CANONICAL_DTYPES)))}"
        )
    return DTYPE_ALIASES[key]


def dtype_guidance_block(dtype: str) -> str:
    """Prompt text steering kernels toward the pipeline activation/weight dtype."""
    dtype = normalize_dtype_name(dtype)
    return (
        f"Target activation/weight dtype: {dtype}.\n"
        f"- kernel_function inputs, weights, biases, and outputs must use {dtype}.\n"
        f"- Do NOT hard-require torch.float32 (or any other dtype) via asserts unless "
        f"the target dtype is float32.\n"
        f"- tl.float32 / float32 accumulation inside Triton is fine and expected; that "
        f"does not mean host tensors must be float32.\n"
        f"- Prefer `out = torch.empty(..., dtype=x.dtype)` (or explicit torch.{dtype})."
    )


def stamp_subgraph_dtypes(
    items: list[dict], default_dtype: str, *, force_pipeline_dtype: bool = True
) -> list[dict]:
    """Apply pipeline dtype to subgraphs.

    By default, overwrite extractor dtypes (often ``torch.float32`` from the
    reference problem) with the pipeline PRECISION so generation matches KBV eval.
    Set ``force_pipeline_dtype=False`` to only fill missing fields.
    """
    default_dtype = normalize_dtype_name(default_dtype)
    for item in items:
        raw = item.get("dtype")
        if force_pipeline_dtype or not raw:
            item["dtype"] = default_dtype
        else:
            item["dtype"] = normalize_dtype_name(raw, default=default_dtype)
    return items


def infer_torch_dtype_name_from_source(source: str) -> str | None:
    """
    Infer host/input torch dtype contract from kernel source.

    Only matches explicit ``torch.<dtype>`` tokens so Triton accumulators like
    ``tl.float32`` do not force float32 verification.
    Priority when multiple appear: bfloat16 > float16 > float32.
    """
    if not source:
        return None
    # Strip tl.* / language dtype mentions by only accepting torch.* forms.
    has_bf16 = re.search(r"\btorch\.bfloat16\b", source) is not None
    has_fp16 = (
        re.search(r"\btorch\.float16\b", source) is not None
        or re.search(r"\btorch\.half\b", source) is not None
    )
    has_fp32 = re.search(r"\btorch\.float32\b", source) is not None
    if has_bf16:
        return "bfloat16"
    if has_fp16:
        return "float16"
    if has_fp32:
        return "float32"
    return None
