#!/usr/bin/env bash
# setup.sh — Install all project dependencies
# Tries uv first (faster), falls back to pip

set -e

cd "$(dirname "$0")"

if command -v uv &> /dev/null; then
    echo "Using uv..."
    uv sync
else
    echo "uv not found, using pip..."
    pip install \
        "gymnasium>=1.2.3" \
        "matplotlib>=3.10.8" \
        "numpy==1.26.4" \
        "numpy-groupies>=0.11.3" \
        "pandas==2.2.3" \
        "pytest>=9.0.2" \
        "pyyaml>=6.0.3" \
        "seaborn>=0.13.2" \
        "torch==2.2.2"
fi

echo "Done. All packages installed."
