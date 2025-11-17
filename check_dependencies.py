#!/usr/bin/env python
"""
Dependency checker and fixer for DiffusionRestoration project.
Run this before training to ensure all dependencies are correctly installed.
"""

import sys
import subprocess

def check_and_fix_opencv():
    """Check if opencv is installed correctly and fix if needed."""
    print("=" * 60)
    print("Checking OpenCV installation...")
    print("=" * 60)

    # Try to import cv2
    try:
        import cv2
        print(f"✓ OpenCV is installed: {cv2.__version__}")

        # Check if it's the headless version
        opencv_build_info = cv2.getBuildInformation()

        # Try to identify if this is headless version
        # The headless version typically doesn't have GUI support
        has_gui = 'GTK' in opencv_build_info or 'QT' in opencv_build_info

        if has_gui:
            print("⚠ WARNING: You have opencv-python (with GUI) installed.")
            print("  This may cause issues in headless environments.")
            print("  Recommended: Use opencv-python-headless instead.")
            print()
            response = input("  Replace with opencv-python-headless? (y/n): ")
            if response.lower() == 'y':
                fix_opencv()
            else:
                print("  Continuing with current OpenCV installation...")
        else:
            print("✓ OpenCV headless version detected (recommended)")

        return True

    except ImportError:
        print("✗ OpenCV is NOT installed")
        print()
        response = input("  Install opencv-python-headless now? (y/n): ")
        if response.lower() == 'y':
            install_opencv_headless()
            return True
        else:
            print("  ERROR: OpenCV is required. Please install it manually:")
            print("    pip install opencv-python-headless>=4.8.0")
            return False

def fix_opencv():
    """Remove opencv-python and install opencv-python-headless."""
    print()
    print("Fixing OpenCV installation...")

    # Uninstall opencv-python variants
    print("  Removing opencv-python packages...")
    for package in ['opencv-python', 'opencv-contrib-python']:
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', '-y', package],
                capture_output=True,
                text=True
            )
        except Exception as e:
            print(f"    Note: Could not uninstall {package} (may not be installed)")

    # Install headless version
    install_opencv_headless()

def install_opencv_headless():
    """Install opencv-python-headless."""
    print("  Installing opencv-python-headless...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'opencv-python-headless>=4.8.0'],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ opencv-python-headless installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install opencv-python-headless")
        print(f"  Error: {e.stderr}")
        sys.exit(1)

def check_other_dependencies():
    """Check other critical dependencies."""
    print()
    print("=" * 60)
    print("Checking other dependencies...")
    print("=" * 60)

    critical_packages = [
        'torch',
        'torchvision',
        'diffusers',
        'transformers',
        'accelerate',
        'PIL',  # Pillow
        'numpy',
        'tqdm',
    ]

    missing_packages = []

    for package in critical_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing_packages.append(package)

    if missing_packages:
        print()
        print("⚠ Missing packages detected!")
        print("  Please install all requirements:")
        print("    pip install -r requirements.txt")
        return False

    return True

def main():
    print()
    print("DiffusionRestoration - Dependency Check")
    print()

    # Check OpenCV
    opencv_ok = check_and_fix_opencv()

    # Check other dependencies
    other_ok = check_other_dependencies()

    print()
    print("=" * 60)
    if opencv_ok and other_ok:
        print("✓ All dependencies are correctly installed!")
        print("=" * 60)
        print()
        print("You can now run training:")
        print("  python train_ddim.py --dataset_name /path/to/dataset")
        return 0
    else:
        print("✗ Some dependencies are missing or incorrect")
        print("=" * 60)
        print()
        print("Please fix the issues above before running training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
