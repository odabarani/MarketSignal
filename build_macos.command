#!/bin/zsh
set -e

cd "$(dirname "$0")"

python3 -m venv .build-venv
.build-venv/bin/python -m pip install --upgrade pip
.build-venv/bin/pip install -r requirements-desktop.txt
.build-venv/bin/pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name MarketSignal \
  --add-data "assets:assets" \
  --collect-data xgboost \
  --collect-binaries xgboost \
  desktop.py

echo ""
echo "Built: dist/MarketSignal.app"
