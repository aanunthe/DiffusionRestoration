#!/bin/bash
# Setup script for DiffusionRestoration project
# This script fixes the OpenCV dependency issue and installs all required packages

set -e  # Exit on error

echo "=== Setting up DiffusionRestoration Environment ==="

# Step 1: Uninstall opencv-python if it exists (it requires OpenGL)
echo "Step 1: Removing opencv-python (if installed)..."
pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || echo "  No opencv-python to remove"

# Step 2: Install opencv-python-headless (works in headless environments)
echo "Step 2: Installing opencv-python-headless..."
pip install opencv-python-headless>=4.8.0

# Step 3: Install all requirements
echo "Step 3: Installing project requirements..."
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "You can now run training with:"
echo "  python train_ddim.py --dataset_name /path/to/dataset"
echo "Or run the full pipeline with:"
echo "  python run_pipeline.py --dataset_name /path/to/dataset"
