# Running on Colab

The code lives in `phaseN_*/` modules; notebooks are just the driver — they
clone the repo, import from it, and run training. Keep logic in the modules,
not in cells, so it stays diffable in git.

Add one notebook per phase as you go (`phase1.ipynb`, `phase2.ipynb`, ...).

## Workflow

Edit on your laptop → commit → push → `git pull` in Colab → Runtime →
Restart session → run from the top. The restart matters: Python keeps the old
version of any module it has already imported, so a pull alone changes nothing.

(Autoreload would avoid the restart, but older IPython versions of it import
`imp`, which Python 3.12 removed, so `%load_ext autoreload` fails on Colab with
`ModuleNotFoundError: No module named 'imp'`.)

## Cell 1 — clone / update the repo

Safe to rerun: clones on a fresh runtime, pulls on a warm one.

```python
import os

REPO = "https://github.com/nabilrakaiza/paperflash.git"

if not os.path.exists("/content/paperflash"):
    !git clone {REPO} /content/paperflash
%cd /content/paperflash
!git pull --ff-only
```

## Cell 2 — device check

```python
import torch
print("torch:", torch.__version__)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

If this says CPU: Runtime → Change runtime type → T4 GPU.

## Cell 3 — checkpoints on Drive

Colab wipes local disk on disconnect. Anything you want to survive the
session — checkpoints, logs — goes to Drive.

```python
from google.colab import drive
drive.mount("/content/drive")

CKPT_DIR = "/content/drive/MyDrive/paperflash/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
```

## Cell 4 — import and run

The repo root is the working directory after Cell 1, so the phase
directories import directly as namespace packages — no `__init__.py`,
no `sys.path` edits, no `pip install -e .`

```python
from phase1_baseline.model import GPT
from phase1_baseline.train import train
```

## Notes / gotchas

- **T4 does not support bf16.** It's a Turing card — bf16 needs Ampere (A100,
  or Colab's L4). If you use mixed precision, it has to be fp16 with a
  `GradScaler`. Worth knowing before a loss curve full of NaNs sends you
  bug-hunting in the model.
- **Notebook diffs are noisy** — outputs and execution counts change every
  run. Clear outputs before committing (Edit → Clear all outputs) so the diff
  shows only real changes.
- **Colab free tier disconnects on idle.** For long runs (grokking needs many
  optimizer steps past the memorization point), checkpoint periodically to
  Drive and make training resumable, rather than hoping one session survives.
