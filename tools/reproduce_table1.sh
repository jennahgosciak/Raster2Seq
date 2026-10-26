#!/usr/bin/env bash
set -euo pipefail

# Existing evaluation scripts resolve data and output paths from the repository root.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$REPO_ROOT"

echo "Reproducing Raster2Seq results from Table 1"

echo "[1/3] Structure3D (Structured3D)"
bash tools/eval_s3d.sh

echo "[2/3] CubiCasa5K"
bash tools/eval_cc5k.sh

echo "[3/3] Raster2Graph"
bash tools/eval_r2g.sh

echo "Table 1 evaluation completed. See eval_outputs/*/results.txt for metrics."
