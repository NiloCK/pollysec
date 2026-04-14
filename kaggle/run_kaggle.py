"""
Kaggle training runner for pollysec.

Mounts the pollysec dataset (colinmorgankennedy/pollysec) which contains
the polly/ package. Checkpoints are saved under /kaggle/working/checkpoints/
and will be available via `kaggle kernels output colinmorgankennedy/pollysec-train`.

To run a subset, edit VARIANTS or SEEDS below before pushing.
"""

import os
import sys

# Dataset is mounted at /kaggle/input/pollysec/
REPO_DIR = "/kaggle/input/pollysec-pkg"

VARIANTS = ["vanilla", "vanilla_reg", "looped", "looped_reg"]
SEEDS = [100, 200, 300]
STEPS = 30_000

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ---------------------------------------------------------------------------
# Run all training runs
# ---------------------------------------------------------------------------

from polly.train import train  # noqa: E402 (import after path setup)

total = len(VARIANTS) * len(SEEDS)
run = 0
failed = []

for variant in VARIANTS:
    for seed in SEEDS:
        run += 1
        tag = f"{variant}_seed{seed}"
        print(f"\n{'='*60}")
        print(f"[{run}/{total}] Starting: {tag}")
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
