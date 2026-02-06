#!/usr/bin/env bash
set -euo pipefail

# Purpose: bootstrap a reproducible local (macOS) dev environment by installing
# Poetry dependencies, a NumPy version compatible with pinned PyTorch CPU wheels,
# and then running a quick sanity check.

# macOS Intel / older wheels (CPU-only) — pinned to versions known to work on MY macOS
TORCH_DARWIN="2.2.2"
TV_DARWIN="0.17.2"
TA_DARWIN="2.2.2"

echo ">>> Poetry install (no torch) ..."
poetry install --no-root

# Keep NumPy ABI compatible with these torch wheels
poetry run pip install "numpy<2" -q

echo "→ Platform: darwin | Channel: cpu"
poetry run pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch==${TORCH_DARWIN}" "torchvision==${TV_DARWIN}" "torchaudio==${TA_DARWIN}"

echo ">>> Sanity check"
poetry run python - <<'PY'
import sys, torch, numpy
print("Python:", sys.version.split()[0])
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "✅ Done."
