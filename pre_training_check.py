#!/usr/bin/env python
"""
Comprehensive pre-training validation wrapper.
Runs both dependency checks and system checks before training.

Usage:
    python pre_training_check.py --dataset_name /path/to/data [training args...]
"""

import sys
import subprocess
import argparse

def run_check(script_name, description):
    """Run a check script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {description}...")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    # Parse just the arguments we need for validation
    parser = argparse.ArgumentParser(
        description="Pre-training validation for DiffusionRestoration",
        add_help=True
    )
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='logs')
    parser.add_argument('--image_size', type=int, default=288)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--min_free_gb', type=int, default=10,
                       help='Minimum free disk space required (GB)')
    parser.add_argument('--skip_deps', action='store_true',
                       help='Skip dependency checks')
    parser.add_argument('--skip_system', action='store_true',
                       help='Skip system checks')

    args, unknown = parser.parse_known_args()

    print("\n" + "="*60)
    print("DiffusionRestoration - Pre-Training Validation")
    print("="*60)

    all_passed = True

    # Check dependencies
    if not args.skip_deps:
        dep_check = subprocess.run(
            [sys.executable, 'check_dependencies.py'],
            capture_output=False
        )
        if dep_check.returncode != 0:
            print("\n⚠ Dependency checks failed!")
            all_passed = False
    else:
        print("\nSkipping dependency checks...")

    # Check system requirements
    if not args.skip_system:
        sys_check = subprocess.run(
            [sys.executable, 'check_system.py',
             '--dataset_name', args.dataset_name,
             '--output_dir', args.output_dir,
             '--image_size', str(args.image_size),
             '--batch_size', str(args.batch_size),
             '--min_free_gb', str(args.min_free_gb)],
            capture_output=False
        )
        if sys_check.returncode != 0:
            print("\n⚠ System checks failed!")
            all_passed = False
    else:
        print("\nSkipping system checks...")

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL PRE-TRAINING CHECKS PASSED")
        print("="*60)
        print("\nYou can now safely run training!")
        print("\nExample:")
        print(f"  python train_ddim.py \\")
        print(f"    --dataset_name {args.dataset_name} \\")
        print(f"    --output_dir {args.output_dir} \\")
        print(f"    --batch_size {args.batch_size} \\")
        print(f"    --num_epochs 10")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before training.")
        print("\nCommon fixes:")
        print("  - Dependencies: bash setup_environment.sh")
        print("  - Disk space: Free up disk space or use --output_dir")
        print("  - Permissions: Check directory permissions")
        return 1

if __name__ == "__main__":
    sys.exit(main())
