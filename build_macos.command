#!/bin/zsh
set -e

cd "$(dirname "$0")"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it, then run this file again."
  exit 1
fi

python3.11 -m venv --clear .build-venv
.build-venv/bin/python -m pip install --upgrade pip
.build-venv/bin/python -m pip install -r requirements-desktop.txt
.build-venv/bin/pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name MarketSignal \
  --add-data "assets:assets" \
  --add-data "docs/research-lab.json:docs" \
  --collect-data xgboost \
  --collect-binaries xgboost \
  desktop.py

echo ""
echo "Built: dist/MarketSignal.app"
