# DiffusionRestoration Project Flow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOC3D DATASET                                      │
│  ├─ img/ (distorted document images - PNG)                                 │
│  ├─ bm/ (backward mapping fields - MAT)                                    │
│  ├─ wc/ (world coordinates - EXR)                                          │
│  └─ recon/ (reconstruction masks - PNG)                                    │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   data_loader.py     │
                │  Doc3d Dataset       │
                │  - Normalize [-1,1]  │
                │  - Resize to 288x288 │
                └──────────┬───────────┘
                           │
           ┌───────────────┼────────────────┬──────────────┐
           │               │                │              │
           ▼               ▼                ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────┐
    │train.py  │   │train_ddim│   │train_encoder │   │train    │
    │          │   │.py       │   │.py           │   │_unet.py │
    │(DDPM)    │   │(DDIM)    │   │(Direct       │   │(Simple) │
    │          │   │          │   │Regression)   │   │         │
    └────┬─────┘   └─────┬────┘   └──────┬───────┘   └────┬────┘
         │               │               │                │
         │ Uses:         │ Uses:         │ Uses:          │ Uses:
         │ - UNet2D      │ - UNet2D      │ - EncDec      │ - UNet2D
         │ - CLIP        │ - CLIP        │ - VGG19       │   (no cond)
         │ - DDPMSched   │ - DDIMSched   │ - Direct      │
         │               │               │   pred        │
         │               │               │               │
         └───────────────┴───────────────┴───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │  Training Loop           │
                │  ────────────────         │
                │  1. Forward pass         │
                │  2. Compute losses:      │
                │     - ELBO (MSE noise)   │
                │     - Recon (MSE BM)     │
                │     - Perceptual (VGG)   │
                │  3. Backprop & update    │
                │  4. Log to W&B           │
                │  5. Save checkpoints     │
                └──────────┬───────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │  Model Checkpoint Saved                  │
        │  output_dir/run_name/checkpoint-XXXX/    │
        │  ├─ diffusion_pytorch_model.safetensors  │
        │  ├─ optimizer.bin                        │
        │  └─ scheduler.bin                        │
        └──────────────────┬───────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────┐
            │  generate_eval_images.py         │
            │  ─────────────────────────        │
            │  For each test image:            │
            │  1. Load checkpoint              │
            │  2. Extract CLIP features        │
            │  3. Predict BM (DDIM 20 steps)   │
            │  4. Apply grid_sample warping    │
            │  5. Save rectified image         │
            │  6. Save GT scan image           │
            └──────────────┬───────────────────┘
                           │
                 ┌─────────┴──────────┐
                 │                    │
                 ▼                    ▼
        ┌────────────────┐   ┌────────────────┐
        │  rectified/    │   │ ground_truth/  │
        │  ├─ 0.png      │   │ ├─ 0.png       │
        │  ├─ 1.png      │   │ ├─ 1.png       │
        │  └─ ...        │   │ └─ ...         │
        └────────┬───────┘   └───────┬────────┘
                 │                   │
                 └─────────┬─────────┘
                           │
                           ▼
                ┌──────────────────────────────┐
                │  evaluation_metrics.py       │
                │  ──────────────────────       │
                │  For each image pair:        │
                │  1. Load & resize            │
                │  2. Convert to grayscale     │
                │  3. Compute MS-SSIM:         │
                │     - 5 pyramid levels       │
                │     - Weighted average       │
                │  4. Compute LD:              │
                │     - Dense optical flow     │
                │     - Mean displacement      │
                │  5. Average all metrics      │
                └──────────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Final Results       │
                    │  ───────────────      │
                    │  MS-SSIM: 0.850      │
                    │  LD: 2.345           │
                    └──────────────────────┘
```

## Detailed Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (DDIM Example)                      │
└─────────────────────────────────────────────────────────────────┘

Input Batch from DataLoader
├─ img: [B, 3, 288, 288]  (distorted image, [-1,1])
├─ bm:  [B, 2, 288, 288]  (ground truth backward mapping, [-1,1])
└─ wc:  [B, 3, 288, 288]  (world coordinates, normalized)

        │
        ▼
┌───────────────────────┐
│  CLIP Image Encoder   │
│  (frozen, no grad)    │
│  openai/clip-vit-     │
│  large-patch14        │
└───────┬───────────────┘
        │
        ▼
encoder_hidden_states: [B, 768]
        │
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
┌───────────────────┐           ┌──────────────────┐
│ Forward Diffusion │           │ Timestep Sampling│
│                   │           │ t ~ U(0, 999)    │
│ x0 = bm (GT)      │           └────────┬─────────┘
│ noise ~ N(0, I)   │                    │
│ xt = sqrt(αt)*x0  │◄───────────────────┘
│    + sqrt(1-αt)   │
│      * noise      │
└─────────┬─────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  UNet2DConditionModel                │
│  ─────────────────────                │
│  Input:                              │
│  ├─ xt: [B, 2, 288, 288]             │
│  ├─ t: timestep                      │
│  └─ encoder_hidden_states: [B, 768]  │
│                                      │
│  Architecture:                       │
│  ├─ Down blocks (6): 128→512        │
│  ├─ Cross-attention (CLIP cond)     │
│  ├─ Self-attention                  │
│  └─ Up blocks (6): 512→128          │
│                                      │
│  Output:                             │
│  └─ noise_pred: [B, 2, 288, 288]    │
└──────────────┬───────────────────────┘
               │
               ▼
┌────────────────────────────────────────┐
│  Loss Computation                      │
│  ────────────────                      │
│                                        │
│  1. ELBO Loss (Diffusion):             │
│     loss_elbo = MSE(noise_pred, noise) │
│                                        │
│  2. Reconstruction Loss:               │
│     x0_pred = predict_x0_from_xt(...)  │
│     loss_recon = MSE(x0_pred, bm)      │
│                                        │
│  3. Combined:                          │
│     loss = loss_elbo + α * loss_recon  │
│            (α = 0.01)                  │
└──────────────┬─────────────────────────┘
               │
               ▼
┌────────────────────────────┐
│  Optimizer Step            │
│  ──────────────            │
│  1. loss.backward()        │
│  2. optimizer.step()       │
│  3. lr_scheduler.step()    │
│  4. Save checkpoint every  │
│     100 steps              │
└────────────────────────────┘
```

## Evaluation Pipeline (FastDDIM Inference)

```
┌──────────────────────────────────────────────────────────────┐
│              EVALUATION - IMAGE GENERATION                   │
└──────────────────────────────────────────────────────────────┘

Test Image Batch
├─ img: [B, 3, 288, 288]  (distorted)
└─ bm_gt: [B, 2, 288, 288]  (ground truth BM)

        │
        ▼
┌───────────────────────┐
│  CLIP Image Encoder   │
│  (same as training)   │
└───────┬───────────────┘
        │
        ▼
encoder_hidden_states: [B, 768]
        │
        ▼
┌──────────────────────────────────────┐
│  FastDDIMPipeline                    │
│  ─────────────────                   │
│                                      │
│  Initialize: xt ~ N(0, I)            │
│                                      │
│  For timestep in [999, 949, ..., 0]: │
│    (20 steps total)                  │
│                                      │
│    1. UNet predicts noise:           │
│       noise_pred = unet(xt, t, CLIP) │
│                                      │
│    2. DDIM update:                   │
│       x0_pred = (xt - √(1-αt)*noise) │
│                 / √αt                │
│       xt-1 = √αt-1 * x0_pred +       │
│              √(1-αt-1) * noise_pred  │
│                                      │
│  Output: x0 (predicted BM)           │
└──────────────┬───────────────────────┘
               │
               ▼
     pred_bm: [B, 2, 288, 288]
               │
     ┌─────────┴──────────┐
     │                    │
     ▼                    ▼
┌─────────────┐    ┌──────────────┐
│ grid_sample │    │ grid_sample  │
│ (img,       │    │ (img,        │
│  pred_bm)   │    │  bm_gt)      │
└──────┬──────┘    └──────┬───────┘
       │                  │
       ▼                  ▼
  rectified_img      gt_scan_img
       │                  │
       ▼                  ▼
  Save to            Save to
  rectified/         ground_truth/
```

## Metrics Computation Flow

```
┌──────────────────────────────────────────────────────────┐
│              EVALUATION - METRICS COMPUTATION            │
└──────────────────────────────────────────────────────────┘

Load Image Pair
├─ rectified/0.png  (predicted)
└─ ground_truth/0.png  (GT scan)

        │
        ▼
┌────────────────────────┐
│  Preprocessing         │
│  ─────────────         │
│  1. Convert to gray    │
│  2. Resize to ~598k    │
│     pixels (area)      │
│  3. Create mask where  │
│     GT != 0            │
└────────┬───────────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌─────────┐  ┌──────────────────────┐
│ MS-SSIM │  │  Local Distortion    │
│ ───── │  │  ─────────────────    │
│         │  │                      │
│ Level 0 │  │  1. Blur (7x7)       │
│ SSIM    │  │  2. Resize to 50%    │
│ ↓       │  │  3. Optical flow     │
│ ×0.0448 │  │     (Farneback)      │
│         │  │  4. Flow magnitude:  │
│ Level 1 │  │     d = √(vx²+vy²)   │
│ SSIM    │  │  5. Mean in mask:    │
│ ↓       │  │     LD = mean(d)     │
│ ×0.2856 │  │                      │
│         │  └──────────┬───────────┘
│ Level 2 │             │
│ SSIM    │             │
│ ↓       │             │
│ ×0.3001 │             │
│         │             │
│ Level 3 │             │
│ SSIM    │             │
│ ↓       │             │
│ ×0.2363 │             │
│         │             │
│ Level 4 │             │
│ SSIM    │             │
│ ↓       │             │
│ ×0.1333 │             │
│         │             │
│ MS-SSIM │             │
│ = Σ wi  │             │
│   *SSIMi│             │
└────┬────┘             │
     │                  │
     └────────┬─────────┘
              │
              ▼
    ┌──────────────────────┐
    │  Accumulate Metrics  │
    │  ─────────────────   │
    │  For all N images:   │
    │  avg_ms_ssim =       │
    │    Σ MS-SSIM / N     │
    │  avg_ld =            │
    │    Σ LD / N          │
    └──────────┬───────────┘
               │
               ▼
        ┌──────────────┐
        │ Print Results│
        │ ────────── │
        │ MS-SSIM: X.XX│
        │ LD: Y.YY     │
        └──────────────┘
```

## Complete Pipeline (run_pipeline.py)

```
┌────────────────────────────────────────────────────────┐
│                  RUN_PIPELINE.PY                       │
│  Orchestrates: Training → Generation → Evaluation     │
└────────────────────────────────────────────────────────┘

Step 1: Training
    │
    ▼
┌─────────────────────────────────────┐
│ subprocess.run([                    │
│   "python", "train_ddim.py",        │
│   "--run_name", args.run_name,      │
│   "--batch_size", args.batch_size,  │
│   "--num_epochs", args.num_epochs,  │
│   "--alpha", args.alpha,            │
│   "--gpu_ids", args.gpu_ids         │
│ ])                                  │
│                                     │
│ Output:                             │
│ └─ output_dir/run_name/             │
│    checkpoint-{step}/               │
└──────────┬──────────────────────────┘
           │
           ▼
Step 2: Find Latest Checkpoint
    │
    ▼
┌─────────────────────────────────────┐
│ checkpoint_dirs = glob(             │
│   f"{output_dir}/{run_name}/        │
│    checkpoint-*"                    │
│ )                                   │
│ latest = max(checkpoint_dirs,       │
│              key=extract_number)    │
│                                     │
│ e.g., checkpoint-1000               │
└──────────┬──────────────────────────┘
           │
           ▼
Step 3: Generate Evaluation Images
    │
    ▼
┌─────────────────────────────────────┐
│ subprocess.run([                    │
│   "python",                         │
│   "generate_eval_images.py",        │
│   "--checkpoint_path", latest,      │
│   "--output_path",                  │
│     f"{output_dir}/{run_name}/      │
│      evaluation_images",            │
│   "--batch_size", args.batch_size,  │
│   "--gpu_ids", args.gpu_ids         │
│ ])                                  │
│                                     │
│ Output:                             │
│ └─ evaluation_images/               │
│    ├─ rectified/*.png               │
│    └─ ground_truth/*.png            │
└──────────┬──────────────────────────┘
           │
           ▼
Step 4: Compute Metrics
    │
    ▼
┌─────────────────────────────────────┐
│ subprocess.run([                    │
│   "python",                         │
│   "evaluation_metrics.py",          │
│   "--rec_path",                     │
│     "evaluation_images/rectified",  │
│   "--gt_path",                      │
│     "evaluation_images/             │
│      ground_truth"                  │
│ ])                                  │
│                                     │
│ Output (stdout):                    │
│ ├─ Average MS-SSIM: X.XXXX          │
│ └─ Average LD: Y.YYYY               │
└─────────────────────────────────────┘
```

## Key Components Summary

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **data_loader.py** | Load Doc3D dataset | Image/BM paths | Normalized tensors |
| **model.py** | EncDec architecture | Distorted image | Predicted BM |
| **train_ddim.py** | DDIM training | Doc3D batches | Model checkpoint |
| **utils.py** | Helper functions | Various | grid_sample, pipelines |
| **generate_eval_images.py** | Generate test outputs | Checkpoint + test data | rectified/GT images |
| **evaluation_metrics.py** | Compute quality metrics | Image pairs | MS-SSIM, LD scores |
| **run_pipeline.py** | End-to-end orchestration | Args | Final metrics |

## Loss Functions

### DDPM/DDIM Training
```
loss = MSE(noise_pred, noise) + α * MSE(x0_pred, bm_gt)
       └─── ELBO Loss ────┘       └─── Recon Loss ───┘
```

### Direct Encoder Training
```
loss = MSE(bm_pred, bm_gt) + 0.1*VGG_loss + 0.05*edge_loss
       └─ MSE Loss ──┘       └─ Perceptual ┘  └─ Edge ─┘
```

## Grid Sampling Explained

```
Input:
├─ img: [B, 3, H, W]  (distorted image)
└─ bm:  [B, 2, H, W]  (backward mapping field)

BM Field Interpretation:
├─ bm[:, 0, :, :] = x-coordinates (where to sample from)
└─ bm[:, 1, :, :] = y-coordinates (where to sample from)

Process:
For each pixel (i, j) in output:
    output[i, j] = img[bm[i, j, 0], bm[i, j, 1]]

Result: Rectified image where distortions are removed
```
