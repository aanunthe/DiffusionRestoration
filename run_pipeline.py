import os
import subprocess
import argparse
import glob

def find_latest_checkpoint(log_dir, run_name_pattern):
    """Finds the latest checkpoint directory in the log folder.

    Args:
        log_dir: The base log directory (e.g., 'logs')
        run_name_pattern: Pattern to match run directories (e.g., 'ddim_fast_running')
                         Will find the most recent directory matching this pattern
    """
    # Find all directories matching the pattern (accounting for timestamp suffix)
    pattern = os.path.join(log_dir, f"{run_name_pattern}*")
    matching_dirs = glob.glob(pattern)

    if not matching_dirs:
        print(f"Error: No run directories found matching pattern: {pattern}")
        return None

    # Find the most recently created directory
    run_dir = max(matching_dirs, key=os.path.getmtime)
    print(f"Found run directory: {run_dir}")

    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        return None

    # Check for checkpoints in the 'checkpoints' subdirectory (used by automatic_checkpoint_naming)
    checkpoints_subdir = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(checkpoints_subdir):
        checkpoint_dirs = glob.glob(os.path.join(checkpoints_subdir, "checkpoint_*"))
        if not checkpoint_dirs:
            print(f"Error: No checkpoints found in {checkpoints_subdir}")
            return None
        # Find the one with the highest step number (checkpoint_0, checkpoint_1, etc.)
        latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.split('_')[-1]))
    else:
        # Fallback: check for old-style checkpoint directories directly in run_dir
        checkpoint_dirs = glob.glob(os.path.join(run_dir, "checkpoint-*"))
        if not checkpoint_dirs:
            print(f"Error: No checkpoints found in {run_dir}")
            return None
        # Find the one with the highest step number (checkpoint-100, checkpoint-200, etc.)
        latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.split('-')[-1]))

    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def main(args):
    """Runs the full training and evaluation pipeline."""

    print("--- STEP 1: Starting Training ---")

    # train_ddim.py will add "ddim_fast_" prefix and timestamp
    # We just pass the base run_name (or empty string)
    run_name_base = args.run_name if args.run_name else ""

    train_command = [
        "python", "train_ddim.py",
        "--run_name", run_name_base,
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--project_name", args.project_name,
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--alpha", str(args.alpha),
        "--image_size", str(args.image_size),
        "--learning_rate", str(args.learning_rate),
        "--warmup_steps", str(args.warmup_steps),
        "--num_epochs", str(args.num_epochs),
        "--log_freq", str(args.log_freq),
        "--output_dir", args.output_dir,
        "--dataset_name", args.dataset_name,
        "--gpu_ids", args.gpu_ids,
    ]

    try:
        subprocess.run(train_command, check=True)
        print("--- STEP 1: Training Finished Successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"--- STEP 1: Training FAILED ---")
        print(e)
        return

    print("--- STEP 2: Finding Latest Checkpoint ---")
    # The training script creates directories like:
    # - "ddim_fast_{timestamp}" if no run_name
    # - "ddim_fast_{run_name}_{timestamp}" if run_name provided
    if run_name_base:
        search_pattern = f"ddim_fast_{run_name_base}"
    else:
        search_pattern = "ddim_fast"

    latest_checkpoint_path = find_latest_checkpoint(args.output_dir, search_pattern)

    if latest_checkpoint_path is None:
        print("Aborting pipeline.")
        return

    # Extract the actual run directory name from the checkpoint path
    # checkpoint path is like: logs/ddim_fast_running_2025.11.16_12.34.56/checkpoint-100
    run_dir = os.path.dirname(latest_checkpoint_path)
    run_name = os.path.basename(run_dir)

    print("--- STEP 3: Generating Images for Evaluation ---")
    eval_output_dir = os.path.join(run_dir, "evaluation_images")
    
    generate_command = [
        "python", "generate_eval_images.py",
        "--checkpoint_path", latest_checkpoint_path,
        "--dataset_name", args.dataset_name,
        "--output_dir", eval_output_dir,
        "--image_size", str(args.image_size),
        "--batch_size", str(args.batch_size), # Use training batch size for generation
        "--gpu_ids", args.gpu_ids.split(',')[0], # Use a single GPU for generation
    ]

    try:
        subprocess.run(generate_command, check=True)
        print("--- STEP 3: Image Generation Finished Successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"--- STEP 3: Image Generation FAILED ---")
        print(e)
        return

    print("--- STEP 4: Running Evaluation Metrics ---")
    rec_path = os.path.join(eval_output_dir, "rectified")
    gt_path = os.path.join(eval_output_dir, "ground_truth")
    
    evaluate_command = [
        "python", "evaluation_metrics.py",
        "--rec_path", rec_path,
        "--gt_path", gt_path,
    ]
    
    try:
        subprocess.run(evaluate_command, check=True)
        print("--- STEP 4: Evaluation Finished Successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"--- STEP 4: Evaluation FAILED ---")
        print(e)
        return

    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full DocRes pipeline: Train, Generate, Evaluate.")
    
    # Add arguments from parse_args.py
    parser.add_argument('--run_name', type=str, default='', help='Unique name for the training run.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--project_name', type=str, default='DocRes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--alpha', default=1e-2, type=float)
    parser.add_argument('--image_size', type=int, default=288)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='logs')
    parser.add_argument('--dataset_name', type=str, default='./data/doc3d')
    parser.add_argument('--gpu_ids', type=str, default='0,1')
    
    args = parser.parse_args()
    main(args)