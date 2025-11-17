#!/bin/bash
# Quick fix for OpenCV OpenGL dependency issue
# Run this if you see: ImportError: libGL.so.1: cannot open shared object file

echo "=== Fixing OpenCV OpenGL Dependency Issue ==="
echo ""
echo "This script will:"
echo "  1. Remove opencv-python (requires OpenGL)"
echo "  2. Install opencv-python-headless (works without OpenGL)"
echo ""

# Uninstall opencv-python variants
echo "Removing opencv-python packages..."
pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true

# Install headless version
echo "Installing opencv-python-headless..."
pip install opencv-python-headless>=4.8.0

echo ""
echo "=== Fix Complete ==="
echo "You can now run your training script again."
