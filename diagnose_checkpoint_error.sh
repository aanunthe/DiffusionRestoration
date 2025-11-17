#!/bin/bash
# Diagnostic script for checkpoint saving errors
# Run this if you see "PytorchStreamWriter failed writing file" errors

echo "==================================================================="
echo "Checkpoint Save Error Diagnostics"
echo "==================================================================="
echo ""

# Check if logs directory exists
if [ -d "logs" ]; then
    LOG_DIR="logs"
else
    LOG_DIR="."
fi

echo "1. Disk Space Check:"
echo "-------------------------------------------------------------------"
df -h "$LOG_DIR" | tail -n 1 | awk '{print "   Filesystem: " $1 "\n   Size: " $2 "\n   Used: " $3 " (" $5 ")\n   Available: " $4}'
echo ""

AVAILABLE=$(df "$LOG_DIR" | tail -n 1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE / 1024 / 1024))

if [ $AVAILABLE_GB -lt 5 ]; then
    echo "   ⚠ WARNING: Less than 5 GB available!"
    echo "   Training checkpoints typically need 1-2 GB each."
    echo "   RECOMMENDED ACTION: Free up disk space"
else
    echo "   ✓ Disk space appears sufficient ($AVAILABLE_GB GB available)"
fi

echo ""
echo "2. Write Permissions Check:"
echo "-------------------------------------------------------------------"

if [ -w "$LOG_DIR" ]; then
    echo "   ✓ Write permission to $LOG_DIR: OK"
else
    echo "   ✗ No write permission to $LOG_DIR"
    echo "   RECOMMENDED ACTION: Fix permissions with 'chmod +w $LOG_DIR'"
fi

echo ""
echo "3. Recent Checkpoint Attempts:"
echo "-------------------------------------------------------------------"

if [ -d "logs" ]; then
    RECENT_DIRS=$(find logs -maxdepth 2 -type d -name "checkpoint*" -o -name "checkpoints" | head -n 5)
    if [ -n "$RECENT_DIRS" ]; then
        echo "   Found checkpoint directories:"
        echo "$RECENT_DIRS" | while read dir; do
            SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "      $dir ($SIZE)"
        done
    else
        echo "   No checkpoint directories found in logs/"
    fi
else
    echo "   logs/ directory not found"
fi

echo ""
echo "4. Disk I/O Test:"
echo "-------------------------------------------------------------------"

TEST_FILE="$LOG_DIR/.io_test_$$"
if dd if=/dev/zero of="$TEST_FILE" bs=1M count=100 &>/dev/null; then
    rm -f "$TEST_FILE"
    echo "   ✓ Disk write test: PASSED"
else
    rm -f "$TEST_FILE" 2>/dev/null
    echo "   ✗ Disk write test: FAILED"
    echo "   This could indicate:"
    echo "      - Disk is full"
    echo "      - Filesystem corruption"
    echo "      - Hardware issues"
fi

echo ""
echo "==================================================================="
echo "Recommendations:"
echo "==================================================================="
echo ""
echo "If you're seeing checkpoint save errors:"
echo ""
echo "1. Free up disk space (if < 10 GB available):"
echo "      # Remove old checkpoints"
echo "      find logs -name 'checkpoint*' -type d -mtime +7 -exec rm -rf {} +"
echo ""
echo "2. Change output directory to a location with more space:"
echo "      python train_ddim.py --output_dir /path/to/larger/disk"
echo ""
echo "3. Reduce checkpoint frequency in training script"
echo "      # Edit train_ddim.py and increase save_steps"
echo ""
echo "4. Use a different filesystem if current one has issues"
echo ""
echo "==================================================================="
