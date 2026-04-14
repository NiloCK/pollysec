#!/usr/bin/env bash
# run_all.sh — Launch all 12 training runs (4 variants × 3 seeds).
#
# Usage:
#   bash polly/run_all.sh              # defaults: 30k steps, batch 128, auto device
#   bash polly/run_all.sh --steps 1000 # override steps (useful for smoke tests)
#
# Runs sequentially by default. For parallel execution on a multi-GPU box,
# see the PARALLEL section at the bottom (commented out).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults (overridable via CLI) ────────────────────────────────────────────

STEPS=30000
BATCH_SIZE=128
DEVICE="auto"

# Parse optional overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)      STEPS="$2";      shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Activate venv if present ──────────────────────────────────────────────────

if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# ── Verify data exists ───────────────────────────────────────────────────────

DATA_DIR="$SCRIPT_DIR/data"
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "==> Data not found. Generating splits..."
    python3 "$SCRIPT_DIR/data.py"
    echo ""
fi

# ── Variants and seeds ───────────────────────────────────────────────────────

VARIANTS=("vanilla" "vanilla_reg" "looped" "looped_reg")
SEEDS=(100 200 300)

TOTAL=$(( ${#VARIANTS[@]} * ${#SEEDS[@]} ))
RUN=0
FAILED=0

echo "============================================================"
echo "  Polly — batch training"
echo "  Variants : ${VARIANTS[*]}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  Steps    : $STEPS"
echo "  Batch    : $BATCH_SIZE"
echo "  Device   : $DEVICE"
echo "  Total    : $TOTAL runs"
echo "============================================================"
echo ""

FAIL_LIST=""

for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN=$((RUN + 1))
        RUN_TAG="${variant}_seed${seed}"

        # Skip if best.pt was trained at the requested step count (resume-friendly).
        # Older checkpoints without a 'total_steps' field are treated as stale.
        BEST_CKPT="$SCRIPT_DIR/checkpoints/${RUN_TAG}/best.pt"
        if [ -f "$BEST_CKPT" ]; then
            CKPT_STEPS=$(python3 -c "import torch,sys; c=torch.load(sys.argv[1], map_location='cpu', weights_only=False); print(c.get('total_steps', -1))" "$BEST_CKPT" 2>/dev/null || echo -1)
            if [ "$CKPT_STEPS" = "$STEPS" ]; then
                echo "[$RUN/$TOTAL] $RUN_TAG — best.pt at ${STEPS} steps exists, skipping."
                continue
            else
                echo "[$RUN/$TOTAL] $RUN_TAG — existing best.pt trained for ${CKPT_STEPS} steps (want ${STEPS}); retraining."
            fi
        fi

        echo ""
        echo "──────────────────────────────────────────────────────────"
        echo "[$RUN/$TOTAL] Starting: $RUN_TAG"
        echo "──────────────────────────────────────────────────────────"

        if python3 -m polly.train \
            --variant "$variant" \
            --seed "$seed" \
            --steps "$STEPS" \
            --batch-size "$BATCH_SIZE" \
            --device "$DEVICE"; then
            echo "[$RUN/$TOTAL] $RUN_TAG — done ✓"
        else
            echo "[$RUN/$TOTAL] $RUN_TAG — FAILED ✗"
            FAILED=$((FAILED + 1))
            FAIL_LIST="$FAIL_LIST $RUN_TAG"
        fi
    done
done

echo ""
echo "============================================================"
echo "  All runs complete.  $((TOTAL - FAILED))/$TOTAL succeeded."
if [ $FAILED -gt 0 ]; then
    echo "  Failed:$FAIL_LIST"
fi
echo "============================================================"

# ── (Optional) Parallel execution ────────────────────────────────────────────
#
# If you have multiple GPUs or want to run CPU jobs in parallel, uncomment
# below and adjust DEVICE assignments. GNU parallel or simple backgrounding:
#
# GPUS=(0 1)
# JOBS=()
# IDX=0
# for variant in "${VARIANTS[@]}"; do
#     for seed in "${SEEDS[@]}"; do
#         GPU=${GPUS[$((IDX % ${#GPUS[@]}))]}
#         CUDA_VISIBLE_DEVICES=$GPU python3 -m polly.train \
#             --variant "$variant" --seed "$seed" \
#             --steps "$STEPS" --batch-size "$BATCH_SIZE" --device cuda &
#         JOBS+=($!)
#         IDX=$((IDX + 1))
#         # Throttle: wait if we've launched as many jobs as GPUs
#         if (( ${#JOBS[@]} >= ${#GPUS[@]} )); then
#             wait "${JOBS[0]}"
#             JOBS=("${JOBS[@]:1}")
#         fi
#     done
# done
# wait
# echo "All parallel runs complete."

exit $FAILED