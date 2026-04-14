"""
Kaggle training runner for pollysec.

The pollysec-pkg dataset (mounted read-only at /kaggle/input/pollysec-pkg)
ships the polly/ package + the latest regenerated JSONL splits. Checkpoints
and logs are written to /kaggle/working/checkpoints so they're picked up as
kernel output (retrievable via `kaggle kernels output colinmorgankennedy/pollysec-train`).

Flip PILOT=True for a fast sanity run (5k steps, vanilla + looped, seed 100).
Flip PILOT=False for the full 12-run sweep.
"""

import os
import sys

REPO_DIR = "/kaggle/input/pollysec-pkg"
WORK_DIR = "/kaggle/working"

# Route checkpoints/logs to the writable kernel workspace BEFORE importing polly.
os.environ["POLLY_CHECKPOINT_DIR"] = os.path.join(WORK_DIR, "checkpoints")
os.environ["POLLY_DATA_DIR"] = os.path.join(REPO_DIR, "polly", "data")

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ---------------------------------------------------------------------------
# Run config — edit these before `kaggle kernels push`
# ---------------------------------------------------------------------------

PILOT = True

if PILOT:
    # Per todo-v2 §B.1: short pilot to find a sensible step budget.
    VARIANTS = ["vanilla", "looped"]
    SEEDS = [100]
    STEPS = 5_000
else:
    VARIANTS = ["vanilla", "vanilla_reg", "looped", "looped_reg"]
    SEEDS = [100, 200, 300]
    STEPS = 30_000

# ---------------------------------------------------------------------------

from polly.train import train  # noqa: E402

total = len(VARIANTS) * len(SEEDS)
run = 0
failed = []

for variant in VARIANTS:
    for seed in SEEDS:
        run += 1
        tag = f"{variant}_seed{seed}"
        print(f"\n{'='*60}")
        print(f"[{run}/{total}] Starting: {tag} (steps={STEPS})")
        print(f"{'='*60}\n")
        try:
            train(
                variant=variant,
                seed=seed,
                total_steps=STEPS,
                device_str="auto",
            )
            print(f"[{run}/{total}] {tag} — done")
        except Exception as exc:
            print(f"[{run}/{total}] {tag} — FAILED: {exc}")
            failed.append(tag)

print(f"\n{'='*60}")
print(f"All runs complete. {total - len(failed)}/{total} succeeded.")
if failed:
    print(f"Failed: {failed}")
print(f"{'='*60}")
