#!/usr/bin/env bash
# Full KernelAgent → kernel_bench_verified pipeline:
#   1) Run Fuser.auto_agent (one problem per GPU, refill when free)
#   2) Import winning kernels into runs/{RUN_NAME}/
#   3) Generate baseline times if missing for the selected problems
#   4) Evaluate with eval_from_generations.py
#   5) Generate leaderboard.html
#
# Each problem runs in its own working directory:
#     $OUT_DIR/level_$LEVEL/${problem_name}/
#         triton_kernel_logs/session_<ts>_<us>/final_kernel.py    (KA route)
#         .fuse/<run_id>/compose_out/composed_kernel.py           (Fuser route)

set -euo pipefail

# ---- user knobs (override via env) ----
LEVEL=${LEVEL:-1}
MAX_PROBLEMS=${MAX_PROBLEMS:-5}
PRECISION=${PRECISION:-bf16}
COMPILE=${COMPILE:-default}                    # torch.compile mode, or eager/"" for eager
GPUS=${GPUS:-}                                 # comma/space GPU ids; empty = all
OUT_DIR=${OUT_DIR:-}                           # empty = ../kernelagent-runs next to KernelAgent
KBV_DIR=${KBV_DIR:-}                           # empty = ../kernel_bench_verified
KA_MODEL=${KA_MODEL:-openai-gpt-5.5}
EXTRACT_MODEL=${EXTRACT_MODEL:-openai-gpt-5.5}
DISPATCH_MODEL=${DISPATCH_MODEL:-openai-gpt-5.5}
COMPOSE_MODEL=${COMPOSE_MODEL:-openai-gpt-5.5}
LEADERBOARD_OUT=${LEADERBOARD_OUT:-}           # empty = $KBV_DIR/leaderboard.html
export OPENAI_API_KEY="$(cat ~/.snowhouse-pat)"

# ---- derived ----
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KERNELAGENT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KBV_DIR=${KBV_DIR:-$(cd "$KERNELAGENT_DIR/../kernel_bench_verified" && pwd)}
PROBLEMS_ROOT=$KBV_DIR/KernelBench
OUT_DIR=${OUT_DIR:-$(cd "$KERNELAGENT_DIR/.." && pwd)/kernelagent-runs}
LEADERBOARD_OUT=${LEADERBOARD_OUT:-$KBV_DIR/leaderboard.html}
RUN_NAME=KernelAgent_level${LEVEL}_test

if [[ -z "${COMPILE}" || "${COMPILE}" == "eager" ]]; then
    COMPILE_MODE=eager
    BASELINE_KIND=eager
else
    COMPILE_MODE=$COMPILE
    BASELINE_KIND=compiled
fi
BASELINE=baseline_torch_${BASELINE_KIND}_${PRECISION}

arch_from_compute_cap() {
    case "$1" in
        10.*|11.*) echo Blackwell ;;
        9.*)       echo Hopper ;;
        8.9)       echo Ada ;;
        8.*)       echo Ampere ;;
        7.5)       echo Turing ;;
        7.*)       echo Volta ;;
        6.*)       echo Pascal ;;
        5.*)       echo Maxwell ;;
        *)         echo "" ;;
    esac
}

HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | awk '{print $NF}')
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
GPU_ARCH=$(arch_from_compute_cap "$cap")
if [[ -z "$GPU_ARCH" ]]; then
    echo "Unknown compute capability '$cap'" >&2
    exit 1
fi

if [[ -n "$GPUS" ]]; then
    # shellcheck disable=SC2206
    GPU_IDS=(${GPUS//,/ })
else
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
N=${#GPU_IDS[@]}
if (( N == 0 )); then
    echo "No GPUs available" >&2
    exit 1
fi
echo "[gpus] using ${GPU_IDS[*]} ($N parallel); hardware=$HARDWARE arch=$GPU_ARCH baseline=$BASELINE compile=$COMPILE_MODE"

export ANTHROPIC_API_KEY="$OPENAI_API_KEY"   # SDK reads this, not ANTHROPIC_AUTH_TOKEN
export OPENAI_BASE_URL=https://snowhouse.snowflakecomputing.com/api/v2/cortex/v1
export ANTHROPIC_BASE_URL=https://snowhouse.snowflakecomputing.com/api/v2/cortex/anthropic
export PYTHONPATH="$KERNELAGENT_DIR:${PYTHONPATH:-}"
export KBV_DIR

# ---------------------------------------------------------------------------
# 1) Generate kernels
# ---------------------------------------------------------------------------
pids=()
gpus_in_use=()
selected_problems=()

wait_for_slot() {
    while (( ${#pids[@]} >= N )); do
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                wait "${pids[$i]}" || true
                unset "pids[$i]" "gpus_in_use[$i]"
                pids=("${pids[@]}")
                gpus_in_use=("${gpus_in_use[@]}")
                return
            fi
        done
        sleep 1
    done
}

free_gpu() {
    local used=" ${gpus_in_use[*]} "
    for g in "${GPU_IDS[@]}"; do
        if [[ "$used" != *" $g "* ]]; then
            echo "$g"
            return
        fi
    done
}

count=0
for problem in "$PROBLEMS_ROOT/level$LEVEL"/*.py; do
    if (( count >= MAX_PROBLEMS )); then
        break
    fi
    count=$((count + 1))
    name=${problem##*/}; name=${name%.py}
    selected_problems+=("${name}.py")
    dst="$OUT_DIR/level_$LEVEL/${name}"
    mkdir -p "$dst"

    if [[ -d "$dst/triton_kernel_logs" || -d "$dst/.fuse" ]]; then
        echo "[skip] $name (already has outputs)"
        continue
    fi

    wait_for_slot
    gpu=$(free_gpu)
    echo "[run] $name -> $dst (gpu $gpu)"
    (
        cd "$dst"
        CUDA_VISIBLE_DEVICES=$gpu python -m Fuser.auto_agent \
            --problem "$problem" \
            --ka-model "$KA_MODEL" \
            --extract-model "$EXTRACT_MODEL" \
            --dispatch-model "$DISPATCH_MODEL" \
            --compose-model "$COMPOSE_MODEL" \
            --verify \
            > run.log 2>&1
    ) &
    pids+=($!)
    gpus_in_use+=("$gpu")
done

for pid in "${pids[@]+"${pids[@]}"}"; do
    wait "$pid" || true
done
echo "[done] generation finished ($count problems selected)"

# ---------------------------------------------------------------------------
# 2) Import winning kernels into kbv runs/
# ---------------------------------------------------------------------------
echo "[import] -> $KBV_DIR/runs/$RUN_NAME"
python "$SCRIPT_DIR/to_kb_solutions.py" \
    --ka_out "$OUT_DIR" \
    --run_name "$RUN_NAME" \
    --level "$LEVEL" \
    --runs_dir "$KBV_DIR/runs" \
    --overwrite

# ---------------------------------------------------------------------------
# 3) Baseline times (generate if missing for selected problems)
# ---------------------------------------------------------------------------
BASELINE_PATH="$KBV_DIR/results/timing/$HARDWARE/${BASELINE}.json"
need_baseline=0
if [[ ! -f "$BASELINE_PATH" ]]; then
    need_baseline=1
elif (( ${#selected_problems[@]} > 0 )); then
    if ! python - "$BASELINE_PATH" "$LEVEL" "${selected_problems[@]}" <<'PY'
import json, sys
path, level, *problems = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
level_data = data.get(f"level{level}", {})
for p in problems:
    v = level_data.get(p)
    if not isinstance(v, dict) or v.get("mean") is None:
        sys.exit(1)
sys.exit(0)
PY
    then
        need_baseline=1
    fi
fi

if (( need_baseline )); then
    echo "[baseline] generating $BASELINE_PATH"
    mkdir -p "$(dirname "$BASELINE_PATH")"
    overwrite_flag=False
    [[ -f "$BASELINE_PATH" ]] && overwrite_flag=True
    (
        cd "$KBV_DIR"
        python scripts/generate_baseline_time.py \
            hardware="$HARDWARE" \
            filename="$BASELINE" \
            precision="$PRECISION" \
            compile="$COMPILE_MODE" \
            "levels=[$LEVEL]" \
            overwrite="$overwrite_flag" \
            device_index="${GPU_IDS[0]}"
    )
else
    echo "[baseline] ok $BASELINE_PATH"
fi

# ---------------------------------------------------------------------------
# 4) Evaluate
# ---------------------------------------------------------------------------
echo "[eval] run_name=$RUN_NAME on $N GPUs"
(
    cd "$KBV_DIR"
    python scripts/eval_from_generations.py \
        run_name="$RUN_NAME" \
        dataset_src=local \
        level="$LEVEL" \
        num_samples=1 \
        eval_mode=local \
        "gpu_arch=['$GPU_ARCH']" \
        num_gpu_devices="$N" \
        timeout=600 \
        build_cache=True \
        num_cpu_workers=1 \
        precision="$PRECISION" \
        compile="$COMPILE_MODE" \
        measure_performance=True
)

# ---------------------------------------------------------------------------
# 5) Leaderboard
# ---------------------------------------------------------------------------
echo "[leaderboard] -> $LEADERBOARD_OUT"
(
    cd "$KBV_DIR"
    python scripts/generate_leaderboard.py \
        --hardware "$HARDWARE" \
        --baseline "$BASELINE" \
        --out "$LEADERBOARD_OUT"
)

echo "[done] pipeline complete"
