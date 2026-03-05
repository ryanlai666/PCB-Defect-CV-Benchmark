#!/usr/bin/env python3
"""
Inference & Demo Script for PCB Defect Detection.

Runs inference on the test dataset for all five models (SME-YOLO, YOLO26,
Faster R-CNN, ViT-Det, RT-DETR), measures speed (FPS / latency), and saves
annotated images with bounding boxes, confidence scores, and class labels.

Usage:
    python inference_demo.py [--models sme_yolo yolo26 faster_rcnn vit_det rt_detr]
                             [--num_warmup 10] [--num_images 0]
                             [--output_dir results/demo]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import config
from config import CLASS_MAP, DEVICE, NUM_CLASSES, SCORE_THRESHOLD

# ── Constants ────────────────────────────────────────────────────────────────
MODELS_TO_RUN = ['sme_yolo', 'yolo26', 'faster_rcnn', 'vit_det', 'rt_detr', 'deimv2_l']

# Class names for drawing (index 0 = background for PyTorch-loop models)
YOLO_CLASS_NAMES  = {i: name for i, name in enumerate(CLASS_MAP.values())}
FRCNN_CLASS_NAMES = CLASS_MAP   # 1-indexed already

# Color palette for bounding boxes (BGR for OpenCV)
BOX_COLORS = [
    (0, 255, 0),    # open       — green
    (0, 165, 255),  # short      — orange
    (255, 0, 0),    # mousebite  — blue
    (0, 255, 255),  # spur       — yellow
    (255, 0, 255),  # copper     — magenta
    (0, 0, 255),    # pin-hole   — red
    (128, 128, 0),  # fallback   — teal
]

# Which models use the PyTorch training loop vs. Ultralytics vs. DEIMv2
PYTORCH_MODELS     = {'faster_rcnn', 'vit_det'}
ULTRALYTICS_MODELS = {'sme_yolo', 'yolo26', 'rt_detr'}
DEIMV2_MODELS      = {'deimv2_l'}


# ─── Utility functions ──────────────────────────────────────────────────────

def draw_boxes(image_bgr, boxes, labels, scores, class_names, score_thresh=0.3):
    """Draw bounding boxes with class names and confidence on a BGR image."""
    img = image_bgr.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        label_int = int(label)
        color = BOX_COLORS[label_int % len(BOX_COLORS)]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label text
        class_name = class_names.get(label_int, f'cls_{label_int}')
        text = f'{class_name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def get_test_images_frcnn(data_dir):
    """Return list of (image_path, annotation_path) for Faster R-CNN / ViT-Det test set."""
    test_txt = os.path.join(data_dir, 'test.txt')
    pairs = []
    with open(test_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                img_rel = parts[0].replace('.jpg', '_test.jpg')
                ann_rel = parts[1]
                pairs.append((
                    os.path.join(data_dir, img_rel),
                    os.path.join(data_dir, ann_rel),
                ))
    return pairs


def get_test_images_yolo(yolo_data_dir, split='test'):
    """Return sorted list of image paths from the YOLO dataset."""
    img_dir = os.path.join(yolo_data_dir, 'images', split)
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(yolo_data_dir, 'images', 'val')
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted([
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    return images


# ─── Inference runners ───────────────────────────────────────────────────────

def run_pytorch_inference(model_name, model_path, test_pairs, output_dir,
                           num_warmup=10, score_thresh=SCORE_THRESHOLD,
                           max_images=0):
    """Run a PyTorch-loop model (Faster R-CNN or ViT-Det) and return timing stats."""
    from models import create_model
    from torchvision.transforms import v2

    model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                     weights_only=True))
    model.to(DEVICE)
    model.eval()

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    os.makedirs(output_dir, exist_ok=True)
    pairs = test_pairs[:max_images] if max_images > 0 else test_pairs
    latencies = []
    prefix = model_name  # used for output filename

    print(f'  Running {model_name} on {len(pairs)} images ...')
    with torch.no_grad():
        for idx, (img_path, _) in enumerate(pairs):
            pil_img = Image.open(img_path).convert('RGB')
            img_tensor = transform(pil_img)
            if hasattr(img_tensor, 'as_subclass'):
                img_tensor = img_tensor.as_subclass(torch.Tensor)
            img_tensor = img_tensor.to(DEVICE)

            # Warm-up
            if idx < num_warmup:
                _ = model([img_tensor])
                continue

            # Timed inference
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model([img_tensor])
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

            # Extract predictions
            pred = outputs[0]
            boxes  = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()

            # Draw and save
            img_bgr   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            annotated = draw_boxes(img_bgr, boxes, labels, scores,
                                   FRCNN_CLASS_NAMES, score_thresh)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(output_dir, f'{prefix}_{base_name}.jpg')
            cv2.imwrite(out_path, annotated)

    return latencies


def run_ultralytics_inference(model_name, weight_path, image_paths, output_dir,
                               num_warmup=10, score_thresh=SCORE_THRESHOLD,
                               max_images=0):
    """Run an Ultralytics model (YOLO / RT-DETR) and return timing stats."""
    from ultralytics import YOLO

    # Ultralytics requires the file to end in '.pt'; create a symlink if needed.
    load_path, symlink_created = ensure_pt_weight(weight_path)
    try:
        model = YOLO(load_path)
    finally:
        if symlink_created and os.path.islink(load_path):
            os.remove(load_path)
    os.makedirs(output_dir, exist_ok=True)
    paths = image_paths[:max_images] if max_images > 0 else image_paths
    latencies = []

    print(f'  Running {model_name} on {len(paths)} images ...')
    for idx, img_path in enumerate(paths):
        # Warm-up
        if idx < num_warmup:
            _ = model.predict(img_path, verbose=False, conf=score_thresh)
            continue

        # Timed inference
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.predict(img_path, verbose=False, conf=score_thresh)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        # Draw and save
        img_bgr = cv2.imread(img_path)
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes  = r.boxes.xyxy.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            annotated = draw_boxes(img_bgr, boxes, labels, scores,
                                   YOLO_CLASS_NAMES, score_thresh)
        else:
            annotated = img_bgr

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f'{model_name}_{base_name}.jpg')
        cv2.imwrite(out_path, annotated)

    return latencies


def run_deimv2_inference(model_name, output_dir, num_warmup=10,
                         score_thresh=SCORE_THRESHOLD, max_images=0):
    """Run DEIMv2 model inference and return timing stats.

    Loads the model through DEIMv2's own config/engine system, then runs
    per-image inference with synchronised CUDA timing (same pattern as the
    other runners so results are directly comparable).
    """
    import sys as _sys
    deimv2_dir = os.path.join(PROJECT_DIR, 'DEIMv2')
    if deimv2_dir not in _sys.path:
        _sys.path.insert(0, deimv2_dir)

    # Locate config and checkpoint
    config_path = os.path.join(
        deimv2_dir, 'configs', 'deimv2', 'deimv2_dinov3_l_deeppcb.yml'
    )
    best_ckpt = os.path.join(output_dir, 'best_stg2.pth')
    if not os.path.isfile(best_ckpt):
        best_ckpt = os.path.join(output_dir, 'best_stg1.pth')
    if not os.path.isfile(best_ckpt):
        print(f'  [SKIP] DEIMv2-L checkpoint not found in {output_dir}')
        return []
    if not os.path.isfile(config_path):
        print(f'  [SKIP] DEIMv2-L config not found: {config_path}')
        return []

    # Build model via DEIMv2 engine
    try:
        from engine.core import YAMLConfig
        cfg = YAMLConfig(config_path, resume=best_ckpt)
        model = cfg.model
        postprocessor = cfg.postprocessor.eval().to(DEVICE)
        ckpt  = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        # Load from EMA if available (gives best weights)
        if 'ema' in ckpt and isinstance(ckpt['ema'], dict):
            state = ckpt['ema'].get('module', ckpt.get('model', {}))
        else:
            state = ckpt.get('model', {})
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f'  [SKIP] Could not load DEIMv2-L model: {e}')
        return []

    model.to(DEVICE)
    model.eval()

    # Collect test images from the COCO-format split (images may be nested
    # in subdirectories like group.../NNNNN/NNNNN_test.jpg).
    test_img_dir = os.path.join(
        PROJECT_DIR, 'data', 'deeppcb_coco', 'images', 'test'
    )
    if not os.path.isdir(test_img_dir):
        # Fall back to YOLO dataset val split
        test_img_dir = os.path.join(config.YOLO_DATA_DIR, 'images', 'val')
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(test_img_dir)
        for f in files
        if os.path.splitext(f)[1].lower() in exts
    ])
    if max_images > 0:
        image_paths = image_paths[:max_images]

    os.makedirs(output_dir_out := os.path.join(
        PROJECT_DIR, 'results', 'demo', model_name
    ), exist_ok=True)

    import torchvision.transforms.functional as TF

    latencies = []
    print(f'  Running DEIMv2-L on {len(image_paths)} images ...')

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            pil_img = Image.open(img_path).convert('RGB')
            # Resize to 640×640 and normalise (same as training val transform)
            pil_resized = pil_img.resize((640, 640))
            img_t = TF.to_tensor(pil_resized)                          # [0,1]
            img_t = TF.normalize(
                img_t,
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ).unsqueeze(0).to(DEVICE)                                  # (1,3,H,W)

            orig_size = torch.tensor(
                [[pil_img.height, pil_img.width]], dtype=torch.long
            ).to(DEVICE)

            # Warm-up
            if idx < num_warmup:
                _ = model(img_t, orig_size)
                continue

            # Timed inference
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(img_t, orig_size)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

            # Process via DEIMv2 postprocessor
            preds = postprocessor(outputs, orig_size)[0]
            
            # Draw predictions
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            try:
                # DEIMv2 COCO classes are naturally 0-indexed for display in DeepPCB mappings
                # (although internal evaluator handles offset)
                labels = preds['labels'].cpu().numpy()
                boxes  = preds['boxes'].cpu().numpy()   # xyxy in orig coords
                scores = preds['scores'].cpu().numpy()
                annotated = draw_boxes(
                    img_bgr, boxes, labels, scores,
                    YOLO_CLASS_NAMES, score_thresh
                )
            except Exception as e:
                print(f"DEIMv2 box err: {e}")
                annotated = img_bgr

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(
                output_dir_out, f'deimv2_l_{base_name}.jpg'
            )
            cv2.imwrite(out_path, annotated)

    return latencies


# ─── Weight-path resolver ────────────────────────────────────────────────────

def resolve_weight(model_name, meta_type):
    """Return the best-weight path for a model.

    - Ultralytics models (sme_yolo, yolo26, rt_detr): prefer .pt first,
      fall back to .pth.  Ultralytics will refuse to load a .pth file even
      if one exists, so we must never hand it a .pth path.
    - PyTorch-loop models (faster_rcnn, vit_det): prefer .pth first,
      fall back to .pt.
    """
    base = os.path.join(PROJECT_DIR, 'outputs', model_name,
                        f'best_{model_name}')
    if meta_type == 'ultralytics':
        # Try .pt first; only try .pth as last resort
        for ext in ('.pt', '.pth'):
            if os.path.isfile(base + ext):
                return base + ext
    else:
        # PyTorch-loop: .pth is canonical
        for ext in ('.pth', '.pt'):
            if os.path.isfile(base + ext):
                return base + ext
    # Nothing found — return the canonical path so caller can print a useful message
    default_ext = '.pt' if meta_type == 'ultralytics' else '.pth'
    return base + default_ext


def ensure_pt_weight(weight_path):
    """Guarantee the returned path ends in '.pt' (Ultralytics enforces this).

    If *weight_path* is a valid Ultralytics checkpoint saved as '.pth'
    (the default suffix used by train_model.py), create a temporary '.pt'
    symlink next to it so Ultralytics can load without a suffix error.

    Returns:
        (path_to_use: str, symlink_created: bool)
        Caller MUST delete the symlink when done if symlink_created is True.
    """
    if weight_path.endswith('.pt'):
        return weight_path, False           # correct extension already
    pt_path = weight_path[:-4] + '.pt'     # replace .pth → .pt
    if not os.path.isfile(pt_path):
        os.symlink(os.path.abspath(weight_path), pt_path)
        return pt_path, True                # caller must clean up
    return pt_path, False                   # independent .pt already present


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Inference demo for PCB models.')
    parser.add_argument('--models', nargs='+', default=MODELS_TO_RUN,
                        choices=MODELS_TO_RUN,
                        help='Models to evaluate.')
    parser.add_argument('--num_warmup', type=int, default=10,
                        help='Number of warm-up images before timing.')
    parser.add_argument('--num_images', type=int, default=0,
                        help='Max images per model (0 = all).')
    parser.add_argument('--output_dir', type=str, default='results/demo',
                        help='Root directory for saving demo images.')
    parser.add_argument('--score_thresh', type=float, default=SCORE_THRESHOLD,
                        help='Confidence threshold for drawing boxes.')
    args = parser.parse_args()

    output_root = os.path.join(PROJECT_DIR, args.output_dir)
    summary = {}

    print('=' * 70)
    print('  PCB Defect Detection — Inference Demo')
    print(f'  Device: {DEVICE}')
    print(f'  Score threshold: {args.score_thresh}')
    print(f'  Models: {args.models}')
    print('=' * 70)

    # ── Pre-load image lists once ─────────────────────────────────────────
    pytorch_pairs = None     # lazy-loaded for PyTorch-loop models
    yolo_imgs     = None     # lazy-loaded for Ultralytics models

    # ── Display name mapping ──────────────────────────────────────────────
    DISPLAY = {
        'faster_rcnn': 'Faster R-CNN',
        'vit_det':     'ViT-Det',
        'sme_yolo':    'SME-YOLO',
        'yolo26':      'YOLO26',
        'rt_detr':     'RT-DETR',
        'deimv2_l':    'DEIMv2-L',
    }

    for model_name in args.models:
        display = DISPLAY.get(model_name, model_name)
        out_dir = os.path.join(output_root, model_name)

        if model_name in PYTORCH_MODELS:
            # ── PyTorch-loop models (Faster R-CNN, ViT-Det) ──────────────
            weight = resolve_weight(model_name, 'pytorch')
            if not os.path.isfile(weight):
                print(f'  [SKIP] {display} weights not found: {weight}')
                continue

            if pytorch_pairs is None:
                pytorch_pairs = get_test_images_frcnn(config.DATA_DIR)

            lats = run_pytorch_inference(
                model_name, weight, pytorch_pairs, out_dir,
                num_warmup=args.num_warmup,
                score_thresh=args.score_thresh,
                max_images=args.num_images,
            )

        elif model_name in ULTRALYTICS_MODELS:
            # ── Ultralytics models (SME-YOLO, YOLO26, RT-DETR) ───────────
            weight = resolve_weight(model_name, 'ultralytics')
            if not os.path.isfile(weight):
                print(f'  [SKIP] {display} weights not found: {weight}')
                continue

            if yolo_imgs is None:
                yolo_imgs = get_test_images_yolo(config.YOLO_DATA_DIR)

            lats = run_ultralytics_inference(
                model_name, weight, yolo_imgs, out_dir,
                num_warmup=args.num_warmup,
                score_thresh=args.score_thresh,
                max_images=args.num_images,
            )

        elif model_name in DEIMV2_MODELS:
            # ── DEIMv2 models ─────────────────────────────────────────────
            deimv2_output = os.path.join(PROJECT_DIR, 'outputs', model_name)
            lats = run_deimv2_inference(
                model_name, deimv2_output,
                num_warmup=args.num_warmup,
                score_thresh=args.score_thresh,
                max_images=args.num_images,
            )

        else:
            print(f'  [SKIP] Unknown model type for: {model_name}')
            continue

        if lats:
            avg_lat = np.mean(lats)
            fps     = 1000.0 / avg_lat
            summary[display] = {
                'avg_latency_ms': round(avg_lat, 2),
                'fps':            round(fps, 2),
                'num_images':     len(lats),
            }
            print(f'  {display:<15} — Avg Latency: {avg_lat:.2f} ms | '
                  f'FPS: {fps:.1f} | Images: {len(lats)}')

    # ── Summary table ────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'  {"Model":<20} {"Latency (ms)":<15} {"FPS":<10} {"Images":<10}')
    print('-' * 70)
    for model_name, stats in summary.items():
        print(f'  {model_name:<20} {stats["avg_latency_ms"]:<15.2f} '
              f'{stats["fps"]:<10.1f} {stats["num_images"]:<10}')
    print('=' * 70)

    # Save summary JSON — merge with existing data so partial runs don't
    # wipe out previously measured results for other models.
    summary_path = os.path.join(output_root, 'inference_summary.json')
    os.makedirs(output_root, exist_ok=True)
    existing = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, 'r') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    existing.update(summary)   # new measurements override old ones
    with open(summary_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\n  Summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
