#!/usr/bin/env bash
# setup.sh — create the Python venv and install all dependencies
# Run once after cloning: bash scripts/setup.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/venv/tf_train"

echo "==> Creating virtual environment at $VENV"
python3.10 -m venv "$VENV"
source "$VENV/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing dependencies from requirements.txt"
pip install -r "$REPO_ROOT/requirements.txt"

echo ""
echo "NOTE: TensorFlow on Jetson must be the NVIDIA-patched build (tensorflow==2.16.1+nv24.8)."
echo "      The standard PyPI build will NOT work. Install the Jetson wheel separately:"
echo "      https://developer.nvidia.com/embedded/downloads"
echo ""
echo "==> Done. Activate the environment with:"
echo "    source scripts/env.sh"
