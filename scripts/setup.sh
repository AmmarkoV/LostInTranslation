#!/usr/bin/env bash
# Create the local venv for the EN -> DE -> EN round-trip translator and
# install everything it needs.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$DIR/venv"
PYTHON="${PYTHON:-python3}"

echo "Creating $VENV"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip

# Argos Translate imports stanza, and stanza needs torch. Install the CPU-only
# wheel first (~200 MB) so pip does not pull the multi-GB CUDA build that the
# default PyPI index would give us.
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu

"$VENV/bin/pip" install gradio argostranslate

# Pre-fetch the EN<->DE models (~200 MB) so the first launch is not a wait.
# Models go to ~/.local/share/argos-translate and are shared with other
# Argos installs on this machine.
"$VENV/bin/python" "$DIR/server.py" --fetch-models

cat <<EOF

Setup complete.

Run the translator with:
  $VENV/bin/python $DIR/server.py

Then open http://127.0.0.1:7860
EOF
