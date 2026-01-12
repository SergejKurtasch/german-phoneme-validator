#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [[ -d "$VENV_DIR" ]]; then
  echo "Using existing virtual environment at $VENV_DIR"
else
  echo "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# googleads (pulled in via parselmouth) still relies on setuptools <=57
pip install "setuptools<58"

pip install -r requirements.txt

cat <<'EOF'
Environment ready.
Activate it with: source .venv/bin/activate
EOF
