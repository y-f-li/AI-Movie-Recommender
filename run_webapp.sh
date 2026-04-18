#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "$DATAPATH" ]; then
  export DATAPATH="$SCRIPT_DIR/dataset"
fi
python run_webapp.py
