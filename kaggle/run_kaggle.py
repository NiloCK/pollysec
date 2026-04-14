"""
Kaggle training runner for pollysec.

Clones the repo from GitHub, then runs all 12 training runs sequentially.
Checkpoints are saved under /kaggle/working/pollysec/polly/checkpoints/
and will be available via `kaggle kernels output colinmorgankennedy/pollysec-train`.

To run a subset, edit VARIANTS or SEEDS below before pushing.
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/nilock/pollysec.git"
REPO_DIR = "/kaggle/working/pollysec"

VARIANTS = ["vanilla", "vanilla_reg", "looped", "looped_reg"]
SEEDS = [100, 200, 300]
STEPS = 30_000

# ---------------------------------------------------------------------------
# Clone repo
# ---------------------------------------------------------------------------

print(f"Cloning {REPO_URL} ...")
subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
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
