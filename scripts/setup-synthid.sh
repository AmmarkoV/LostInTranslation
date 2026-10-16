#!/usr/bin/env bash
# Create the local venv for the SynthID-Text watermark scanner.
#
# Kept separate from venv/: synthid-text pins torch==2.4.0, numpy==1.26.0 and
# jax[cuda], which conflict with the translator's torch/numpy and would pull
# multi-GB CUDA wheels we do not need for scanning on CPU.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$DIR/venv-synthid"
PYTHON="${PYTHON:-python3}"

echo "Creating $VENV"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip

# CPU-only torch first, so the default index does not give us the CUDA build.
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu

# jax from PyPI is the CPU build; jax[cuda] (what synthid-text asks for) is not.
"$VENV/bin/pip" install gradio transformers jax jaxlib immutabledict

# --no-deps: we supply the dependencies above ourselves rather than take the
# pinned set. Scanning only touches logits_processing, hashing_function and
# detector_mean, which need torch, transformers and jax.
"$VENV/bin/pip" install --no-deps synthid-text

# Pre-fetch the GPT-2 tokenizer (~2 MB). The model weights are never needed:
# detection reads token ids, not logits.
"$VENV/bin/python" "$DIR/synthid_server.py" --fetch-tokenizer

cat <<EOF

Setup complete.

Run the scanner with:
  $VENV/bin/python $DIR/synthid_server.py

Then open http://127.0.0.1:7861
EOF
