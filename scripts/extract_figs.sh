#!/usr/bin/env bash
set -euo pipefail

PDF="mstock_dissertation.pdf"
ASSETS_DIR="docs/src/assets"

mkdir -p "$ASSETS_DIR"

echo "Extracting thesis figures at 300 dpi..."

pdftoppm -png -r 300 -f 90 -l 90  "$PDF" "$ASSETS_DIR/fig_3_19"
mv "$ASSETS_DIR/fig_3_19-090.png" "$ASSETS_DIR/fig_3_19.png" 2>/dev/null || \
mv "$ASSETS_DIR/fig_3_19-90.png"  "$ASSETS_DIR/fig_3_19.png"

pdftoppm -png -r 300 -f 98 -l 98  "$PDF" "$ASSETS_DIR/fig_1_23"
mv "$ASSETS_DIR/fig_1_23-098.png" "$ASSETS_DIR/fig_1_23.png" 2>/dev/null || \
mv "$ASSETS_DIR/fig_1_23-98.png"  "$ASSETS_DIR/fig_1_23.png"

pdftoppm -png -r 300 -f 118 -l 118 "$PDF" "$ASSETS_DIR/fig_3_26"
mv "$ASSETS_DIR/fig_3_26-118.png" "$ASSETS_DIR/fig_3_26.png"

pdftoppm -png -r 300 -f 140 -l 140 "$PDF" "$ASSETS_DIR/fig_3_52"
mv "$ASSETS_DIR/fig_3_52-140.png" "$ASSETS_DIR/fig_3_52.png"

echo "Done. Images saved under $ASSETS_DIR:"
ls -1 "$ASSETS_DIR" | grep -E 'fig_(1_23|3_19|3_26|3_52)\.png' || true

