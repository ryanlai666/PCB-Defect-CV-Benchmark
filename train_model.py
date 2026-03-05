#!/usr/bin/env python3
"""
Standalone training script for PCB Defect Detection.

Usage:
    python train_model.py --model <model_name> [--epochs N] [--batch_size B]

Supports all 6 models:
  PyTorch-loop:    faster_rcnn, vit_det, vit_mamba
  Ultralytics:     yolo26, sme_yolo, rt_detr
"""

import argparse
import json
import os
import sys
import time

# ─── Parse arguments first (before heavy imports) ────────────────────────────
parser = argparse.ArgumentParser(description='Train a PCB defect detection model.')
parser.add_argument('--model', type=str, required=True,
                    choices=['faster_rcnn', 'faster_rcnn_ft', 'vit_det', 'vit_mamba',
                             'yolo26', 'sme_yolo', 'rt_detr',
                             'deimv2_l', 'deimv2_x'],
                    help='Model architecture to train.')
parser.add_argument('--epochs', type=int, default=None,
                    help='Number of training epochs (default: from config).')
parser.add_argument('--batch_size', type=int, default=None,
                    help='Batch size (default: from config).')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Directory to save best model and logs.')
parser.add_argument('--test_mode', action='store_true',
                    help='Run in test mode (1 epoch, 80 train/40 val images, frequent logging).')
parser.add_argument('--resume', action='store_true',
                    help='Resume training from last checkpoint. '
                         'For PyTorch models: loads last_ckpt_<model>.pth or best_<model>.pth. '
                         'For Ultralytics: resumes from last.pt. '
                         'For DEIMv2: resumes from last.pth.')
args = parser.parse_args()

# ─── Setup output directory ──────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.output_dir is None:
    args.output_dir = os.path.join(PROJECT_DIR, 'outputs', args.model)
os.makedirs(args.output_dir, exist_ok=True)

# Set BEST_MODEL_PATH env var BEFORE importing config
best_model_path = os.path.join(args.output_dir, f'best_{args.model}.pth')
os.environ['BEST_MODEL_PATH'] = best_model_path

# ─── Now import project modules ─────────────────────────────────────────────
import config
from models import create_model

# Override config if CLI args provided
if args.epochs is not None:
    config.NUM_EPOCHS = args.epochs
if args.batch_size is not None:
    config.BATCH_SIZE = args.batch_size

if args.test_mode:
    config.NUM_EPOCHS = 1


# ─── Logging setup ──────────────────────────────────────────────────────────
log_file = os.path.join(args.output_dir, f'train_{args.model}.log')


class Tee:
    """Duplicate stdout/stderr to a log file."""
    def __init__(self, log_path, stream, append=False):
        mode = 'a' if append else 'w'
        self.file = open(log_path, mode, buffering=1)
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


tee_stdout = Tee(log_file, sys.stdout, append=args.resume)
tee_stderr = Tee(log_file.replace('.log', '_stderr.log'), sys.stderr, append=args.resume)
sys.stdout = tee_stdout
sys.stderr = tee_stderr


# ─── Banner ──────────────────────────────────────────────────────────────────
print('=' * 70)
print(f'  PCB Defect Detection — Training: {args.model}')
if args.resume:
    print(f'  *** RESUME MODE — continuing from last checkpoint ***')
print(f'  Started at: {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'  Device:     {config.DEVICE}')
print(f'  Epochs:     {config.NUM_EPOCHS}')
print(f'  Batch Size: {config.BATCH_SIZE}')
print(f'  Data Dir:   {config.DATA_DIR}')
print(f'  Output Dir: {args.output_dir}')
print(f'  Best Model: {best_model_path}')
print('=' * 70)

start_time = time.time()

# ─── Model creation ─────────────────────────────────────────────────────────
print(f'\n>>> Creating model: {args.model}')
model = create_model(args.model, num_classes=config.NUM_CLASSES,
                     pretrained=config.USE_PRETRAINED)
print(f'    Model type: {type(model)}')

# ─── Training ────────────────────────────────────────────────────────────────
PYTORCH_MODELS = ('faster_rcnn', 'faster_rcnn_ft', 'vit_det', 'vit_mamba')
ULTRALYTICS_MODELS = ('yolo26', 'sme_yolo', 'rt_detr')
DEIMV2_MODELS = ('deimv2_l', 'deimv2_x')

if args.model in PYTORCH_MODELS:
    # ── PyTorch training loop ────────────────────────────────────────────
    from dataset import create_dataloaders
    from training import train_model

    print('\n>>> Loading dataset via PyTorch DataLoader...')
    train_loader, val_loader, test_loader = create_dataloaders(
        config.DATA_DIR, batch_size=config.BATCH_SIZE,
        test_mode=args.test_mode
    )
    print(f'    Train: {len(train_loader.dataset)}  '
          f'Val: {len(val_loader.dataset)}  '
          f'Test: {len(test_loader.dataset)}')

    print('\n>>> Starting training...')
    score_thresh = 0.1 if args.test_mode else config.SCORE_THRESHOLD

    # Determine resume checkpoint for PyTorch models
    resume_ckpt = None
    start_epoch = 0
    if args.resume:
        # Prefer the full-state checkpoint, fall back to best model weights
        last_ckpt = os.path.join(args.output_dir, f'last_ckpt_{args.model}.pth')
        if os.path.exists(last_ckpt):
            resume_ckpt = last_ckpt
            print(f'    Will resume from full checkpoint: {last_ckpt}')
        elif os.path.exists(best_model_path):
            resume_ckpt = best_model_path
            # For old-style checkpoints we need to know the start epoch.
            # Try to infer from saved history JSON.
            hist_path = os.path.join(args.output_dir, f'history_{args.model}.json')
            if os.path.exists(hist_path):
                with open(hist_path) as _hf:
                    old_hist = json.load(_hf)
                start_epoch = len(old_hist.get('val_f1', []))
            print(f'    Will resume from best model weights: {best_model_path} '
                  f'(start_epoch={start_epoch})')
        else:
            print(f'    WARNING: --resume set but no checkpoint found in {args.output_dir}')

    history = train_model(
        model, train_loader, val_loader,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        device=config.DEVICE,
        save_path=best_model_path,
        test_mode=args.test_mode,
        score_threshold=score_thresh,
        resume_path=resume_ckpt,
        start_epoch=start_epoch,
    )

    # Save training history as JSON
    history_path = os.path.join(args.output_dir, f'history_{args.model}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\n>>> Training history saved to: {history_path}')

    # ── Final evaluation on test set ─────────────────────────────────────
    import torch
    from training import validate

    print('\n>>> Evaluating best model on test set...')
    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    test_metrics = validate(model, test_loader, device=config.DEVICE,
                            score_threshold=score_thresh)

    print(f'    Test F1:        {test_metrics["f1"]:.4f}')
    print(f'    Test mIoU:      {test_metrics["miou"]:.4f}')
    print(f'    Test Precision: {test_metrics["precision"]:.4f}')
    print(f'    Test Recall:    {test_metrics["recall"]:.4f}')

    # Save test metrics
    test_metrics_path = os.path.join(args.output_dir, f'test_metrics_{args.model}.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f'    Test metrics saved to: {test_metrics_path}')

elif args.model in ULTRALYTICS_MODELS:
    # ── Ultralytics training ─────────────────────────────────────────────
    from utils import convert_deeppcb_to_yolo

    # Convert dataset to YOLO format (idempotent if already done)
    yolo_dir = config.YOLO_DATA_DIR
    data_yaml = os.path.join(yolo_dir, 'data.yaml')

    if not os.path.exists(data_yaml):
        print('\n>>> Converting DeepPCB dataset to YOLO format...')
        convert_deeppcb_to_yolo(config.DATA_DIR, yolo_dir)
    else:
        print(f'\n>>> YOLO dataset already exists at: {yolo_dir}')

    print(f'\n>>> Starting Ultralytics training for: {args.model}')

    # Ultralytics project/name for organised output
    ul_project = os.path.join(args.output_dir, 'runs')
    ul_name = args.model

    # Check if resuming from an existing Ultralytics run
    ul_resume = False
    if args.resume:
        last_pt = os.path.join(ul_project, ul_name, 'weights', 'last.pt')
        if os.path.exists(last_pt):
            ul_resume = last_pt
            print(f'    Will resume Ultralytics training from: {last_pt}')
        else:
            print(f'    WARNING: --resume set but no last.pt found at {last_pt}')

    if ul_resume:
        # Ultralytics resume: load the last.pt model and call train with resume=True
        # First, patch args.yaml to update the target epoch count
        args_yaml_path = os.path.join(ul_project, ul_name, 'args.yaml')
        if os.path.exists(args_yaml_path):
            import yaml
            with open(args_yaml_path, 'r') as yf:
                ul_args = yaml.safe_load(yf)
            old_epochs = ul_args.get('epochs', 20)
            ul_args['epochs'] = config.NUM_EPOCHS
            with open(args_yaml_path, 'w') as yf:
                yaml.dump(ul_args, yf, default_flow_style=False)
            print(f'    Updated args.yaml epochs: {old_epochs} -> {config.NUM_EPOCHS}')

        from ultralytics import YOLO
        model = YOLO(ul_resume)
        results = model.train(
            resume=True,
        )
    else:
        results = model.train(
            data=data_yaml,
            epochs=config.NUM_EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH_SIZE,
            project=ul_project,
            name=ul_name,
            exist_ok=True,
            save=True,          # save checkpoints
            save_period=-1,     # only save best and last
            verbose=True,
        )

    # Copy best.pt to our standard output location
    import shutil
    import glob

    best_pt_candidates = glob.glob(
        os.path.join(ul_project, ul_name, 'weights', 'best.pt')
    )
    if best_pt_candidates:
        src = best_pt_candidates[0]
        shutil.copy2(src, best_model_path)
        print(f'\n>>> Best model copied to: {best_model_path}')
    else:
        # Fallback: try last.pt
        last_pt = glob.glob(
            os.path.join(ul_project, ul_name, 'weights', 'last.pt')
        )
        if last_pt:
            shutil.copy2(last_pt[0], best_model_path)
            print(f'\n>>> Last model copied to: {best_model_path}')

    # Run validation
    print('\n>>> Running final validation...')
    val_results = model.val(data=data_yaml)
    print(f'    Validation results: {val_results}')

elif args.model in DEIMV2_MODELS:
    # ── DEIMv2 subprocess training ───────────────────────────────────────
    import subprocess

    # Step 1: ensure DeepPCB dataset is converted to COCO format
    coco_ann_dir = os.path.join(PROJECT_DIR, 'data', 'deeppcb_coco', 'annotations')
    train_json   = os.path.join(coco_ann_dir, 'instances_train.json')
    if not os.path.exists(train_json):
        print('\n>>> Converting DeepPCB dataset to COCO format for DEIMv2...')
        convert_script = os.path.join(PROJECT_DIR, 'scripts', 'convert_deeppcb_to_coco.py')
        deeppcb_dir    = os.path.join(PROJECT_DIR, 'DeepPCB', 'PCBData')
        out_dir        = os.path.join(PROJECT_DIR, 'data', 'deeppcb_coco')
        ret = subprocess.run(
            [sys.executable, convert_script,
             '--deeppcb_dir', deeppcb_dir,
             '--output_dir',  out_dir,
             '--seed', '42'],
            cwd=PROJECT_DIR,
        )
        if ret.returncode != 0:
            print('ERROR: Dataset conversion failed.', file=sys.stderr)
            sys.exit(ret.returncode)
    else:
        print(f'\n>>> COCO dataset already exists at: {coco_ann_dir}')

    # Step 2: launch DEIMv2 training via torchrun
    deimv2_dir    = os.path.join(PROJECT_DIR, 'DEIMv2')
    train_script  = os.path.join(deimv2_dir, 'train.py')
    config_path   = model.config_path  # set by wrapper

    # Determine GPU count
    import torch as _torch
    n_gpus = max(1, _torch.cuda.device_count())

    epochs_to_use = config.NUM_EPOCHS if args.epochs is not None else None

    cmd = [
        'torchrun',
        f'--nproc_per_node={n_gpus}',
        '--master_port=7779',
        train_script,
        '-c', str(config_path),
        '--use-amp',
        '--seed=0',
    ]

    # Override output dir so checkpoints land in our standard location
    cmd += [f'--output-dir={args.output_dir}']

    # Resume from last checkpoint if requested
    if args.resume:
        last_pth = os.path.join(args.output_dir, 'last.pth')
        if os.path.exists(last_pth):
            cmd += [f'--resume={last_pth}']
            print(f'\n>>> Resuming DEIMv2 from checkpoint: {last_pth}')
        else:
            print(f'\n>>> WARNING: --resume set but no last.pth found at {last_pth}')

    if args.test_mode:
        # DEIMv2 supports YAML config overrides via -u key=value
        print('\n>>> [test_mode] DEIMv2 will run 1 epoch (via -u epoches=1).')
        cmd += ['-u', 'epoches=1']

    # Override total epochs if specified
    if args.epochs is not None:
        cmd += ['-u', f'epoches={config.NUM_EPOCHS}']
        print(f'    Overriding DEIMv2 total epochs to: {config.NUM_EPOCHS}')

    print(f'\n>>> Starting DEIMv2 training: {model.label}')
    print(f'    Config  : {config_path}')
    print(f'    GPUs    : {n_gpus}')
    print(f'    Command : {" ".join(cmd)}')

    ret = subprocess.run(cmd, cwd=deimv2_dir)

    if ret.returncode != 0:
        print(f'ERROR: DEIMv2 training failed with exit code {ret.returncode}.',
              file=sys.stderr)
        sys.exit(ret.returncode)

    print(f'\n>>> DEIMv2 training complete. Checkpoints in: {args.output_dir}')

else:
    print(f'ERROR: Unknown model type: {args.model}')
    sys.exit(1)

# ─── Final summary ──────────────────────────────────────────────────────────
elapsed = time.time() - start_time
hours, remainder = divmod(int(elapsed), 3600)
minutes, seconds = divmod(remainder, 60)

print('\n' + '=' * 70)
print(f'  Training complete for: {args.model}')
print(f'  Elapsed time: {hours}h {minutes}m {seconds}s')
print(f'  Best model saved at: {best_model_path}')
print(f'  Finished at: {time.strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 70)

# Clean up tee
sys.stdout = tee_stdout.stream
sys.stderr = tee_stderr.stream
tee_stdout.close()
tee_stderr.close()
