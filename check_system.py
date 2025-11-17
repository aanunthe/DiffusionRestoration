#!/usr/bin/env python
"""
Pre-training validation script for DiffusionRestoration.
Checks disk space, permissions, and system requirements before training.
"""

import os
import sys
import shutil
import argparse

def get_size_str(bytes_size):
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def check_disk_space(path, required_gb=10):
    """Check if there's enough disk space available."""
    print("=" * 60)
    print("Checking Disk Space...")
    print("=" * 60)

    try:
        # Get disk usage stats
        stat = shutil.disk_usage(path)

        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        free_gb = stat.free / (1024**3)
        used_percent = (stat.used / stat.total) * 100

        print(f"Path: {path}")
        print(f"Total: {get_size_str(stat.total)}")
        print(f"Used:  {get_size_str(stat.used)} ({used_percent:.1f}%)")
        print(f"Free:  {get_size_str(stat.free)}")
        print()

        if free_gb < required_gb:
            print(f"⚠ WARNING: Low disk space!")
            print(f"  Available: {free_gb:.2f} GB")
            print(f"  Recommended: {required_gb:.2f} GB")
            print()
            print("  Training may fail when saving checkpoints.")
            print("  Please free up disk space before continuing.")
            return False
        else:
            print(f"✓ Sufficient disk space: {free_gb:.2f} GB available")
            return True

    except Exception as e:
        print(f"✗ Error checking disk space: {e}")
        return False

def check_write_permissions(output_dir):
    """Check if we have write permissions to the output directory."""
    print()
    print("=" * 60)
    print("Checking Write Permissions...")
    print("=" * 60)

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Try to create a test file
        test_file = os.path.join(output_dir, ".write_test")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("✓ Write permissions OK")
            return True
        except PermissionError:
            print(f"✗ No write permission to {output_dir}")
            print("  Please check directory permissions.")
            return False
        except Exception as e:
            print(f"✗ Error testing write permissions: {e}")
            return False

    except Exception as e:
        print(f"✗ Cannot create output directory: {e}")
        return False

def check_dataset_path(dataset_path):
    """Check if dataset path exists and is accessible."""
    print()
    print("=" * 60)
    print("Checking Dataset Path...")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path does not exist: {dataset_path}")
        return False

    if not os.path.isdir(dataset_path):
        print(f"✗ Dataset path is not a directory: {dataset_path}")
        return False

    # Check if we can read from the dataset
    try:
        os.listdir(dataset_path)
        print(f"Dataset path: {dataset_path}")
        print("✓ Dataset path is accessible")
        return True
    except PermissionError:
        print(f"✗ No read permission to dataset: {dataset_path}")
        return False
    except Exception as e:
        print(f"✗ Error accessing dataset: {e}")
        return False

def estimate_checkpoint_size(image_size=288, batch_size=8):
    """Estimate checkpoint size based on model parameters."""
    # Rough estimate for UNet model + optimizer state
    # This is approximate and depends on the actual model architecture
    base_model_mb = 500  # ~500MB for model weights
    optimizer_mb = 500   # ~500MB for optimizer state
    misc_mb = 100        # Additional overhead

    total_mb = base_model_mb + optimizer_mb + misc_mb
    return total_mb / 1024  # Convert to GB

def main():
    parser = argparse.ArgumentParser(
        description="Pre-training validation for DiffusionRestoration"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='logs',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Path to dataset'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=288,
        help='Image size for training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--min_free_gb',
        type=int,
        default=10,
        help='Minimum free disk space required (GB)'
    )

    args = parser.parse_args()

    print()
    print("DiffusionRestoration - Pre-Training Validation")
    print()

    # Estimate checkpoint size
    checkpoint_gb = estimate_checkpoint_size(args.image_size, args.batch_size)
    print(f"Estimated checkpoint size: ~{checkpoint_gb:.2f} GB per checkpoint")
    print()

    # Run checks
    checks_passed = True

    # Check dataset
    if not check_dataset_path(args.dataset_name):
        checks_passed = False

    # Check output directory permissions
    if not check_write_permissions(args.output_dir):
        checks_passed = False

    # Check disk space on output directory
    if not check_disk_space(args.output_dir, args.min_free_gb):
        checks_passed = False

    # Summary
    print()
    print("=" * 60)
    if checks_passed:
        print("✓ All pre-training checks passed!")
        print("=" * 60)
        print()
        print("You can proceed with training:")
        print(f"  python train_ddim.py \\")
        print(f"    --dataset_name {args.dataset_name} \\")
        print(f"    --output_dir {args.output_dir} \\")
        print(f"    --image_size {args.image_size} \\")
        print(f"    --batch_size {args.batch_size}")
        return 0
    else:
        print("✗ Some pre-training checks failed")
        print("=" * 60)
        print()
        print("Please fix the issues above before running training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
