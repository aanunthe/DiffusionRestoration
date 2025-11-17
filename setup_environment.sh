#!/bin/bash
# Setup script for DiffusionRestoration project
# This script fixes the OpenCV dependency issue and installs all required packages

set -e  # Exit on error

echo "=== Setting up DiffusionRestoration Environment ==="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Step 1: Uninstall opencv-python if it exists (it requires OpenGL)
echo "Step 1: Removing opencv-python (if installed)..."
pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || echo "  ✓ No opencv-python to remove"

# Step 2: Install opencv-python-headless (works in headless environments)
echo ""
echo "Step 2: Installing opencv-python-headless..."
pip install opencv-python-headless>=4.8.0
echo "  ✓ opencv-python-headless installed"

# Step 3: Install all requirements
echo ""
echo "Step 3: Installing all project requirements..."
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements.txt"
    echo "  ✓ All requirements installed"
else
    echo "  ⚠ Warning: requirements.txt not found in $SCRIPT_DIR"
    echo "  Please ensure all dependencies are installed manually"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "✓ Environment is ready for training!"
echo ""
echo "You can now run:"
echo "  python train_ddim.py --dataset_name /path/to/dataset"
echo "  python run_pipeline.py --dataset_name /path/to/dataset"
echo ""
echo "Or verify your setup with:"
echo "  python check_dependencies.py"
