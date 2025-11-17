#!/usr/bin/env python
"""
Wrapper script to run training with dependency checking.
This ensures all dependencies are correctly installed before starting training.

Usage:
    python run_with_check.py train_ddim.py --dataset_name /path/to/data [other args...]
    python run_with_check.py run_pipeline.py --dataset_name /path/to/data [other args...]
"""

import sys
import subprocess

def check_opencv():
    """Quick check for OpenCV installation."""
    try:
        import cv2
        return True
    except ImportError:
        print("\n" + "=" * 60)
        print("ERROR: OpenCV (cv2) is not installed!")
        print("=" * 60)
        print()
        print("This error means the required OpenCV package is missing.")
        print()
        print("To fix this, run ONE of the following commands:")
        print()
        print("  Option 1 - Full setup (recommended):")
        print("    bash setup_environment.sh")
        print()
        print("  Option 2 - Quick OpenCV fix:")
        print("    bash fix_opencv.sh")
        print()
        print("  Option 3 - Manual installation:")
        print("    pip install opencv-python-headless>=4.8.0")
        print()
        print("  Option 4 - Interactive check and fix:")
        print("    python check_dependencies.py")
        print()
        print("=" * 60)
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_with_check.py <script.py> [args...]")
        print()
        print("Example:")
        print("  python run_with_check.py train_ddim.py --dataset_name /path/to/data")
        sys.exit(1)

    # Check OpenCV before running
    if not check_opencv():
        sys.exit(1)

    # Run the requested script with all arguments
    script_args = sys.argv[1:]
    try:
        subprocess.run([sys.executable] + script_args, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
