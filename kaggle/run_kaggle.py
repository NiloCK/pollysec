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
os.environ["POLLY_DATA_DIR"] = os.path.join(REPO_DIR, "data")

print(f"[diag] REPO_DIR contents:")
for root, dirs, files in os.walk(REPO_DIR):
    depth = root.replace(REPO_DIR, "").count(os.sep)
    if depth > 2:
        continue
    indent = "  " * depth
    print(f"[diag] {indent}{os.path.basename(root) or root}/")
    for f in files[:5]:
        print(f"[diag] {indent}  {f}")

# The staged dataset landed polly/'s contents at the root of REPO_DIR
# (no wrapping polly/ dir, due to kaggle's --dir-mode zip flattening).
# Re-expose it as an importable `polly` package via a symlink in /kaggle/working.
pkg_link = os.path.join(WORK_DIR, "polly")
if not os.path.exists(pkg_link):
    os.symlink(REPO_DIR, pkg_link)
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)

# ---------------------------------------------------------------------------
# Run config — edit these before `kaggle kernels push`
# ---------------------------------------------------------------------------

PILOT = True

if PILOT:
    # v5 §C.1 — vanilla + looped_reg. Ponder loss w/ λ_p=0.1 replaces the
    # uniform mean-CE used in v4 (see comment at train() call below).
    VARIANTS = ["vanilla", "looped_reg"]
    SEEDS = [100]
    STEPS = 10_000
else:
    # v4 §C.3 — full sweep. `vanilla_reg` dropped under scaffolding reframe.
    VARIANTS = ["vanilla", "looped", "looped_reg"]
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
            # v5 C.1: revert to ponder loss with a gentler λ_p=0.1 (v4 used
            # uniform mean-CE after ponder collapsed exits to iter 1; the root
            # cause was λ_p=0.05 being too aggressive). Ponder loss lets the
            # exit gate train end-to-end while the higher λ_p keeps the halting
            # distribution from degenerating. loss_mode is a no-op for vanilla.
            train(
                variant=variant,
                seed=seed,
                total_steps=STEPS,
                device_str="auto",
                loss_mode="ponder",
                lambda_p=0.1,
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
