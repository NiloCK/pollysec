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
import subprocess
import sys

# Kaggle's current PyTorch image drops sm_60 (P100). Force a build that still
# ships sm_60 kernels so we work regardless of which GPU the pool assigns.
# ~60s one-time install.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.3.1", "--index-url", "https://download.pytorch.org/whl/cu121",
])

import torch  # noqa: E402
print(f"[diag] torch={torch.__version__} "
      f"cuda_available={torch.cuda.is_available()} "
      f"device_count={torch.cuda.device_count()} "
      f"device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}")

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
    # Per todo-v3 §B.1: short pilot on the single-range depth-30 task.
    VARIANTS = ["vanilla", "looped"]
    SEEDS = [100]
    STEPS = 3_000
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
