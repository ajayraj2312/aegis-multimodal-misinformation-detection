#!/usr/bin/env bash
# Render build script. Installs CPU-only PyTorch on Linux (the default PyPI wheel
# pulls ~2.5 GB of CUDA libraries that a CPU web service never uses), then
# warms the model cache so the first request after boot is not a cold download.
set -euo pipefail

python -m pip install --upgrade pip

if [ "$(uname -s)" = "Linux" ]; then
  echo "==> Installing CPU-only PyTorch"
  pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 torchvision==0.17.2
  echo "==> Installing remaining requirements"
  pip install -r requirements.txt
else
  pip install -r requirements.txt
fi

echo "==> Warming Hugging Face / torchvision caches"
python - <<'PY'
import os
from transformers import AutoModel, AutoTokenizer
from torchvision.models import ResNet18_Weights, resnet18

name = os.environ.get("TEXT_MODEL_NAME", "distilbert-base-uncased")
print("tokenizer/model:", name)
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print("resnet18 weights")
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("caches warm")
PY

# Seed the demo database during the build rather than on first boot: twelve full
# pipeline runs at boot blow past the health-check grace period on small
# instances, and the build container has room for them. Non-fatal if it fails -
# the app falls back to seeding itself on first boot (SEED_DEMO_ON_EMPTY=1).
echo "==> Pre-seeding the demo database"
python scripts/seed_demo.py || echo "==> Seed step skipped (non-fatal)"

echo "==> Build complete"
