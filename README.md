# DiffusionRestoration

Diffusion-based document restoration using the Doc3D dataset. This system learns to predict backward mapping (BM) fields to unwarp distorted document images.

## Quick Start

### 1. Environment Setup

**Important:** This project uses `opencv-python-headless` to work in headless/server environments without OpenGL dependencies.

**RECOMMENDED - Automated setup:**

```bash
bash setup_environment.sh
```

This script will:
- Remove any existing `opencv-python` packages (which require OpenGL)
- Install `opencv-python-headless` (works without OpenGL)
- Install all other required dependencies from `requirements.txt`

**Alternative - Check dependencies interactively:**

```bash
python check_dependencies.py
```

This will check all dependencies and offer to fix any issues automatically.

**Manual installation:**

```bash
# Remove opencv-python if installed (it requires OpenGL)
pip uninstall -y opencv-python opencv-contrib-python

# Install requirements (includes opencv-python-headless)
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
bash download_doc3d.sh /path/to/output/directory
```

### 3. Validate System (Recommended)

Before training, validate your system to catch common issues:

```bash
python pre_training_check.py --dataset_name /path/to/doc3d
```

This checks:
- Dependencies (opencv, torch, etc.)
- Disk space availability
- Dataset accessibility
- Write permissions

### 4. Run Training

**Single training run:**
```bash
python train_ddim.py \
  --dataset_name /path/to/doc3d \
  --batch_size 8 \
  --num_epochs 10 \
  --gpu_ids "0"
```

**Full pipeline (train + generate + evaluate):**
```bash
python run_pipeline.py \
  --dataset_name /path/to/doc3d \
  --batch_size 8 \
  --num_epochs 1 \
  --gpu_ids "0"
```

## Troubleshooting

### Error: "No module named 'cv2'"

If you see this error:
```
ModuleNotFoundError: No module named 'cv2'
```

**Cause:** OpenCV is not installed in your Python environment.

**Solution (choose one):**

1. **Automated fix (recommended):**
   ```bash
   bash setup_environment.sh
   ```

2. **Interactive check:**
   ```bash
   python check_dependencies.py
   ```

3. **Manual installation:**
   ```bash
   pip install opencv-python-headless>=4.8.0
   ```

### Error: OpenGL/libGL.so.1

If you see this error:
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Cause:** Your environment has `opencv-python` (requires OpenGL) instead of `opencv-python-headless`.

**Solution:**

1. **Quick fix:**
   ```bash
   bash fix_opencv.sh
   ```

2. **Manual fix:**
   ```bash
   pip uninstall -y opencv-python opencv-contrib-python
   pip install opencv-python-headless>=4.8.0
   ```

### Verify Your Setup

To check if all dependencies are correctly installed:

```bash
python check_dependencies.py
```

This will verify all required packages and offer to fix any issues.

### Run Training with Automatic Checks

Use the wrapper script to automatically verify dependencies before training:

```bash
python run_with_check.py train_ddim.py --dataset_name /path/to/data [other args...]
```

This will check for OpenCV before starting training and provide clear error messages if dependencies are missing.

### Error: Checkpoint Save Failed

If you see this error during training:
```
RuntimeError: [enforce fail at inline_container.cc:858] . PytorchStreamWriter failed writing file
```

**Cause:** Unable to write checkpoint files, usually due to:
- Disk full (no space left)
- Write permission issues
- Filesystem errors

**Diagnosis:**

Run the diagnostic script:
```bash
bash diagnose_checkpoint_error.sh
```

**Solutions:**

1. **Check disk space:**
   ```bash
   df -h logs/  # Check available space
   ```

2. **Free up space:**
   ```bash
   # Remove old checkpoints
   find logs -name 'checkpoint*' -type d -mtime +7 -exec rm -rf {} +
   ```

3. **Use different output directory:**
   ```bash
   python train_ddim.py --output_dir /path/to/larger/disk --dataset_name /path/to/data
   ```

4. **Pre-validate before training:**
   ```bash
   python pre_training_check.py --dataset_name /path/to/data
   ```

### Pre-Training Validation (Recommended)

Before starting training, run comprehensive checks:

```bash
python pre_training_check.py --dataset_name /path/to/data --output_dir logs
```

This will check:
- All required dependencies
- Dataset accessibility
- Disk space availability
- Write permissions
- System requirements

## Project Structure

### Training Scripts
- `train_ddim.py` - DDIM-based diffusion training
- `train.py` - DDPM-based diffusion training
- `train_encoder.py` - Direct encoder-decoder training
- `run_pipeline.py` - Full pipeline: train → generate → evaluate

### Core Modules
- `data_loader.py` - Doc3D dataset loader
- `model.py` - Neural network architectures
- `utils.py` - Utility functions and custom pipelines

### Setup & Helper Scripts
- `setup_environment.sh` - Complete environment setup (recommended)
- `fix_opencv.sh` - Quick fix for OpenCV/OpenGL issues
- `check_dependencies.py` - Interactive dependency checker
- `check_system.py` - System requirements validator (disk, permissions)
- `pre_training_check.py` - Comprehensive pre-training validation
- `run_with_check.py` - Training wrapper with dependency validation
- `diagnose_checkpoint_error.sh` - Diagnostic tool for checkpoint save errors

## Key Features

- Multiple training approaches: DDPM, DDIM, direct regression
- Multi-GPU support via Hugging Face Accelerate
- Mixed precision training (FP16)
- Weights & Biases integration for experiment tracking
- CLIP-conditioned diffusion models

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full dependency list

## Documentation

- `CODE_INDEX.md` - Detailed code documentation
- `PROJECT_FLOW.md` - Project workflow and data flow

## License

See repository license file for details.
