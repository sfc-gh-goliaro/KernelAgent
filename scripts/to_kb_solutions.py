# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert KernelAgent outputs into kernel_bench_verified solution files.

Assumes each problem was run with run_kernelagent_on_kbv.sh, which cd's
into a per-problem directory so KernelAgent's native outputs are co-located:

    <ka_out>/level_{L}/{problem_name}/
        triton_kernel_logs/session_<ts>_<us>/final_kernel.py   (KA route)
        .fuse/<run_id>/compose_out/composed_kernel.py          (Fuser route)
        .fuse/<run_id>/compose_out/composition_summary.json    ({"success": ...})

where {problem_name} is the KernelBench filename without .py (e.g. 100_HingeLoss).
The leading integer of the name is the problem_id used in the output filename.

Selection per problem:
  1. If any composition_summary.json says "success": true, use composed_kernel.py.
  2. Else use the newest session_*/final_kernel.py (KernelAgent only writes
     final_kernel.py on success, so mere presence is the success signal).
  3. Else skip - no winning kernel.

Each imported kernel gets a tiny ModelNew shim appended so it plugs into
kernel_bench_verified's eval_from_generations.py, which expects a ModelNew
class alongside the original Model.
"""

import argparse
import json
import os
import re
from pathlib import Path


SHIM = """
import torch.nn as _nn

class ModelNew(_nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._ref = Model(*args, **kwargs)

    def forward(self, *inputs):
        return kernel_function(
            *inputs, *self._ref.parameters(), *self._ref.buffers()
        )
"""

DIR_RE = re.compile(r"^(\d+)_")


def find_winning_kernel(problem_dir: Path) -> Path | None:
    for summary in problem_dir.glob(".fuse/*/compose_out/composition_summary.json"):
        try:
            ok = json.loads(summary.read_text(encoding="utf-8")).get("success")
        except (json.JSONDecodeError, OSError):
            continue
        if ok:
            composed = summary.with_name("composed_kernel.py")
            if composed.is_file() and composed.stat().st_size > 0:
                return composed
    finals = sorted(
        problem_dir.glob("triton_kernel_logs/session_*/final_kernel.py"),
        key=lambda p: p.stat().st_mtime,
    )
    return finals[-1] if finals else None


def main():
    ka_root = Path(__file__).resolve().parent.parent
    kbv = Path(os.environ.get("KBV_DIR", str(ka_root.parent / "kernel_bench_verified")))
    p = argparse.ArgumentParser()
    p.add_argument("--ka_out", required=True,
                   help="Root of KernelAgent runs (e.g. ../kernelagent-runs)")
    p.add_argument("--run_name", required=True,
                   help="Destination run name under --runs_dir")
    p.add_argument("--level", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--runs_dir", default=str(kbv / "runs"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--pids", default="",
                   help="Optional comma-separated problem ids to import (default: all)")
    args = p.parse_args()

    src = Path(args.ka_out) / f"level_{args.level}"
    dst = Path(args.runs_dir) / args.run_name
    dst.mkdir(parents=True, exist_ok=True)

    allow = None
    if args.pids.strip():
        allow = {int(x) for x in args.pids.replace(",", " ").split() if x.strip()}

    n_ok = n_missing = n_skip = 0
    for problem_dir in sorted(src.iterdir()):
        m = DIR_RE.match(problem_dir.name)
        if not m or not problem_dir.is_dir():
            continue
        pid = int(m.group(1))
        if allow is not None and pid not in allow:
            continue
        winner = find_winning_kernel(problem_dir)
        if winner is None:
            print(f"[skip] {problem_dir.name}: no winning kernel")
            n_missing += 1
            continue
        out_file = dst / f"level_{args.level}_problem_{pid}_sample_0_kernel.py"
        if out_file.exists() and not args.overwrite:
            print(f"[skip-exists] {out_file.name}")
            n_skip += 1
            continue
        out_file.write_text(winner.read_text(encoding="utf-8").rstrip() + "\n" + SHIM)
        print(f"[write] {out_file.name} <- {winner.relative_to(problem_dir)}")
        n_ok += 1

    print(f"\nDone. wrote={n_ok} skipped_existing={n_skip} missing={n_missing}")
    print(f"Destination: {dst}")


if __name__ == "__main__":
    main()
