# DiffusionRestoration

Diffusion-based document restoration using the Doc3D dataset. This system learns to predict backward mapping (BM) fields to unwarp distorted document images.

## Quick Start

### 1. Environment Setup

**Important:** This project uses `opencv-python-headless` to work in headless/server environments without OpenGL dependencies.

Run the setup script to configure your environment:

```bash
bash setup_environment.sh
```

Or manually install dependencies:

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

### 3. Run Training

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

### OpenGL/libGL.so.1 Error

If you see this error:
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Cause:** Your environment has `opencv-python` installed instead of `opencv-python-headless`.

**Solution:**
```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless>=4.8.0
```

Or simply run:
```bash
bash setup_environment.sh
```

### Missing Dependencies

If you encounter import errors, reinstall all requirements:
```bash
pip install -r requirements.txt
```

## Project Structure

- `train_ddim.py` - DDIM-based diffusion training
- `train.py` - DDPM-based diffusion training
- `train_encoder.py` - Direct encoder-decoder training
- `run_pipeline.py` - Full pipeline: train → generate → evaluate
- `data_loader.py` - Doc3D dataset loader
- `model.py` - Neural network architectures
- `utils.py` - Utility functions and custom pipelines

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
