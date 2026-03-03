#!/usr/bin/env python3
"""
Evaluation Metrics & Comparison Script.

Extracts and compares detection metrics across SME-YOLO, YOLO26, Faster R-CNN,
ViT-Det, and RT-DETR.  Consolidates accuracy, complexity, and speed metrics
into a single comparison table (printed + CSV).

Metrics collected:
  Accuracy  : mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-Score, mean IoU
  Complexity: Total parameters, GFLOPs, weight file size
  Speed     : Avg latency (ms), throughput (FPS)
  Architecture: Stage type, backbone

Usage:
    python eval_compare.py [--output_dir results] [--run_test]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import config
from config import DEVICE, NUM_CLASSES, SCORE_THRESHOLD

# ─── Model metadata (manually curated) ──────────────────────────────────────
MODEL_META = {
    'faster_rcnn': {
        'display_name': 'Faster R-CNN',
        'architecture': 'Two-stage',
        'backbone': 'ResNet-50 FPN v2',
        'weight_key': 'best_faster_rcnn.pth',
        'type': 'pytorch',
    },
    'sme_yolo': {
        'display_name': 'SME-YOLO',
        'architecture': 'One-stage',
        'backbone': 'YOLOv11n (CSPDarknet)',
        'weight_key': 'best_sme_yolo.pth',
        'type': 'ultralytics',
    },
    'yolo26': {
        'display_name': 'YOLO26',
        'architecture': 'One-stage',
        'backbone': 'YOLO26n',
        'weight_key': 'best_yolo26.pth',
        'type': 'ultralytics',
    },
    'vit_det': {
        'display_name': 'ViT-Det',
        'architecture': 'Two-stage',
        'backbone': 'ViT-Base/16 + FPN',
        'weight_key': 'best_vit_det.pth',
        'type': 'pytorch',
    },
    'rt_detr': {
        'display_name': 'RT-DETR',
        'architecture': 'Transformer',
        'backbone': 'RT-DETR-L (ResNet-based)',
        'weight_key': 'best_rt_detr.pth',
        'type': 'ultralytics',
    },
}


# ─── Helper functions ────────────────────────────────────────────────────────

def get_file_size_mb(path):
    """Return file size in MB, or None if not found."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    return None


def count_parameters(model):
    """Count total and trainable parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_pytorch_metrics(output_dir, model_name):
    """Load test metrics from a PyTorch-loop model JSON outputs.

    Works for both faster_rcnn and vit_det (same file naming convention).
    mAP50/mAP50-95 are loaded from test_metrics JSON if available; otherwise
    they are left as 'N/A' (call run_pytorch_test_eval to compute them).
    """
    test_path    = os.path.join(output_dir, f'test_metrics_{model_name}.json')
    history_path = os.path.join(output_dir, f'history_{model_name}.json')

    metrics = {}

    # Test metrics
    if os.path.isfile(test_path):
        with open(test_path, 'r') as f:
            test = json.load(f)
        metrics['test_precision'] = test.get('precision', 0)
        metrics['test_recall']    = test.get('recall', 0)
        metrics['test_f1']        = test.get('f1', 0)
        metrics['test_miou']      = test.get('miou', 0)
        metrics['test_mAP50']     = test.get('mAP50', 'N/A')
        metrics['test_mAP50_95']  = test.get('mAP50_95', 'N/A')

    # Validation metrics (last epoch) + best f1 epoch as proxy
    if os.path.isfile(history_path):
        with open(history_path, 'r') as f:
            h = json.load(f)
        val_f1_list = h.get('val_f1', [0])
        metrics['val_precision'] = h.get('val_precision', [0])[-1]
        metrics['val_recall']    = h.get('val_recall', [0])[-1]
        metrics['val_f1']        = val_f1_list[-1]
        metrics['val_miou']      = h.get('val_miou', [0])[-1]
        metrics['val_mAP50']     = 'N/A'
        metrics['val_mAP50_95']  = 'N/A'
        # Best val mIoU across epochs
        val_miou_list = h.get('val_miou', [])
        if val_miou_list:
            metrics['best_miou'] = max(val_miou_list)

    return metrics


# Keep backward-compat alias
def get_frcnn_metrics(output_dir):
    return get_pytorch_metrics(output_dir, 'faster_rcnn')


def run_pytorch_test_eval(model_name, weight_path):
    """Run a PyTorch-loop model on the test set and compute all metrics.

    Computes mAP50, mAP50-95, mIoU, precision, recall, and F1 by running
    the model on every image in the test DataLoader and collecting raw
    predictions.

    Returns:
        dict with test_mAP50, test_mAP50_95, test_miou, test_precision,
        test_recall, test_f1.  Also saves updated test_metrics JSON.
    """
    from models import create_model
    from dataset import create_dataloaders
    from training import _strip_tv_tensors
    from evaluation import compute_map, compute_miou_from_predictions, \
        evaluate_predictions, compute_metrics

    print(f'    Loading {model_name} for test-set evaluation...')
    model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
    model.load_state_dict(
        torch.load(weight_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    # Load test data
    _, _, test_loader = create_dataloaders(
        config.DATA_DIR, batch_size=config.BATCH_SIZE
    )
    print(f'    Test set: {len(test_loader.dataset)} images')

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images  = [_strip_tv_tensors(img).to(DEVICE) for img in images]
            targets = [{k: _strip_tv_tensors(v).to(DEVICE) for k, v in t.items()}
                       for t in targets]
            predictions = model(images)
            all_preds.extend(predictions)
            all_targets.extend(targets)

    # Compute mAP50, mAP50-95
    print(f'    Computing mAP for {model_name}...')
    map_result = compute_map(all_preds, all_targets)

    # Compute mIoU
    miou = compute_miou_from_predictions(
        all_preds, all_targets,
        score_threshold=SCORE_THRESHOLD,
    )

    # Compute precision, recall, F1
    total_tp = total_fp = total_fn = 0
    for p, t in zip(all_preds, all_targets):
        tp, fp, fn, _ = evaluate_predictions([p], [t],
                                               score_threshold=SCORE_THRESHOLD)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)

    test_metrics = {
        'test_precision': precision,
        'test_recall':    recall,
        'test_f1':        f1,
        'test_miou':      miou,
        'test_mAP50':     map_result['mAP50'],
        'test_mAP50_95':  map_result['mAP50_95'],
    }

    # Save updated test metrics
    model_output_dir = os.path.join(PROJECT_DIR, 'outputs', model_name)
    save_path = os.path.join(model_output_dir, f'test_metrics_{model_name}.json')
    save_data = {
        'precision':  precision,
        'recall':     recall,
        'f1':         f1,
        'miou':       miou,
        'mAP50':      map_result['mAP50'],
        'mAP50_95':   map_result['mAP50_95'],
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'    Test metrics saved to: {save_path}')

    return test_metrics


def get_ultralytics_metrics(output_dir, model_name):
    """Load metrics from Ultralytics results.csv (last epoch)."""
    csv_path = os.path.join(output_dir, 'runs', model_name, 'results.csv')
    metrics  = {}

    if not os.path.isfile(csv_path):
        return metrics

    # Read the last row of results.csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

    if not rows:
        return metrics

    last          = rows[-1]
    best_map50    = max(float(r['metrics/mAP50(B)'])    for r in rows)
    best_map50_95 = max(float(r['metrics/mAP50-95(B)']) for r in rows)

    p  = float(last['metrics/precision(B)'])
    r  = float(last['metrics/recall(B)'])
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    metrics['val_precision'] = p
    metrics['val_recall']    = r
    metrics['val_f1']        = f1
    metrics['val_mAP50']     = float(last['metrics/mAP50(B)'])
    metrics['val_mAP50_95']  = float(last['metrics/mAP50-95(B)'])
    metrics['val_miou']      = 'N/A'
    metrics['best_mAP50']    = best_map50
    metrics['best_mAP50_95'] = best_map50_95

    # Check for cached test metrics from a previous --run_test invocation
    test_cache_path = os.path.join(output_dir, f'test_metrics_{model_name}.json')
    if os.path.isfile(test_cache_path):
        with open(test_cache_path, 'r') as f:
            cached = json.load(f)
        metrics['test_precision'] = cached.get('precision', p)
        metrics['test_recall']    = cached.get('recall', r)
        metrics['test_f1']        = cached.get('f1', f1)
        metrics['test_miou']      = cached.get('miou', 'N/A')
        metrics['test_mAP50']     = cached.get('mAP50', float(last['metrics/mAP50(B)']))
        metrics['test_mAP50_95']  = cached.get('mAP50_95', float(last['metrics/mAP50-95(B)']))
    else:
        # Fallback: use val metrics as proxy
        metrics['test_precision'] = p
        metrics['test_recall']    = r
        metrics['test_f1']        = f1
        metrics['test_mAP50']     = float(last['metrics/mAP50(B)'])
        metrics['test_mAP50_95']  = float(last['metrics/mAP50-95(B)'])
        metrics['test_miou']      = 'N/A'

    return metrics


def run_ultralytics_test_eval(model_name, weight_path):
    """Run an Ultralytics model on the test split and compute all metrics.

    Uses model.val(split='test') for mAP50/mAP50-95, and then runs a
    manual per-image prediction pass to compute mIoU (which Ultralytics
    does not natively report).

    Returns:
        dict with test_mAP50, test_mAP50_95, test_miou, test_precision,
        test_recall, test_f1.
    """
    from ultralytics import YOLO
    from evaluation import compute_miou_from_predictions
    import glob

    print(f'    Running Ultralytics test-split evaluation for {model_name}...')

    load_path, sym = ensure_pt_weight(weight_path)
    try:
        model = YOLO(load_path)
    finally:
        if sym and os.path.islink(load_path):
            os.remove(load_path)

    data_yaml = os.path.join(config.YOLO_DATA_DIR, 'data.yaml')

    # Run official validation on the test split
    val_results = model.val(data=data_yaml, split='test', verbose=False)
    box = val_results.box

    test_mAP50    = float(box.map50)
    test_mAP50_95 = float(box.map)
    test_p        = float(box.mp)
    test_r        = float(box.mr)
    test_f1       = 2 * test_p * test_r / (test_p + test_r) if (test_p + test_r) > 0 else 0.0

    # Compute mIoU via per-image predictions on the test images
    print(f'    Computing mIoU for {model_name} on test set...')
    test_img_dir = os.path.join(config.YOLO_DATA_DIR, 'images', 'test')
    test_lbl_dir = os.path.join(config.YOLO_DATA_DIR, 'labels', 'test')

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    test_images = sorted([
        os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    all_preds   = []
    all_targets = []

    # Re-load model for predict (val() may alter state)
    load_path2, sym2 = ensure_pt_weight(weight_path)
    try:
        pred_model = YOLO(load_path2)
    finally:
        if sym2 and os.path.islink(load_path2):
            os.remove(load_path2)

    for img_path in test_images:
        # Get ground truth
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(test_lbl_dir, base + '.txt')

        gt_boxes  = []
        gt_labels = []
        if os.path.isfile(lbl_path):
            # Read image dimensions for denormalizing YOLO coords
            import cv2
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            gt_boxes.append([x1, y1, x2, y2])
                            gt_labels.append(cls)

        # Run prediction
        results = pred_model.predict(img_path, verbose=False, conf=0.01)
        r = results[0]

        pred_boxes  = r.boxes.xyxy.cpu().float() if r.boxes is not None and len(r.boxes) > 0 else torch.zeros(0, 4)
        pred_labels = r.boxes.cls.cpu().long()    if r.boxes is not None and len(r.boxes) > 0 else torch.zeros(0, dtype=torch.long)
        pred_scores = r.boxes.conf.cpu().float()  if r.boxes is not None and len(r.boxes) > 0 else torch.zeros(0)

        all_preds.append({
            'boxes':  pred_boxes,
            'labels': pred_labels,
            'scores': pred_scores,
        })
        all_targets.append({
            'boxes':  torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros(0, 4),
            'labels': torch.tensor(gt_labels, dtype=torch.long)  if gt_labels else torch.zeros(0, dtype=torch.long),
        })

    miou = compute_miou_from_predictions(
        all_preds, all_targets,
        score_threshold=SCORE_THRESHOLD,
    )

    test_metrics = {
        'test_precision': test_p,
        'test_recall':    test_r,
        'test_f1':        test_f1,
        'test_miou':      miou,
        'test_mAP50':     test_mAP50,
        'test_mAP50_95':  test_mAP50_95,
    }

    # Save test metrics to a persistent cache file
    model_output_dir = os.path.join(PROJECT_DIR, 'outputs', model_name)
    save_path = os.path.join(model_output_dir, f'test_metrics_{model_name}.json')
    save_data = {
        'precision':  test_p,
        'recall':     test_r,
        'f1':         test_f1,
        'miou':       miou,
        'mAP50':      test_mAP50,
        'mAP50_95':   test_mAP50_95,
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'    Test metrics saved to: {save_path}')

    print(f'    Test mAP50={test_mAP50:.4f}  mAP50-95={test_mAP50_95:.4f}  '
          f'mIoU={miou:.4f}  P={test_p:.4f}  R={test_r:.4f}  F1={test_f1:.4f}')

    return test_metrics


def get_model_complexity(model_name, weight_path):
    """Get parameter count and (optionally) FLOPs for a model.

    For Ultralytics models, loads from the runs/<name>/weights/best.pt path
    instead of the renamed .pth copy, because model.info() returns None when
    loaded via a symlinked .pth file (Ultralytics internal metadata check).
    """
    info = {'total_params': 'N/A', 'gflops': 'N/A'}
    meta = MODEL_META[model_name]

    try:
        if meta['type'] == 'pytorch':
            from models import create_model
            model = create_model(model_name, num_classes=NUM_CLASSES,
                                 pretrained=True)
            total, trainable = count_parameters(model)
            info['total_params']     = total
            info['trainable_params'] = trainable

            # Try to compute FLOPs with thop
            try:
                from thop import profile
                dummy = torch.randn(1, 3, 640, 640).to('cpu')
                model.eval()
                model.to('cpu')
                flops, _ = profile(model, inputs=(dummy,), verbose=False)
                info['gflops'] = round(flops / 1e9, 2)
            except Exception:
                pass  # thop not installed or model not thop-compatible

        elif meta['type'] == 'ultralytics':
            from ultralytics import YOLO

            # Prefer runs/<name>/weights/best.pt — model.info() works correctly
            # from that path.  Fall back to the standard weight_path otherwise.
            model_output_dir = os.path.join(
                PROJECT_DIR, 'outputs', model_name
            )
            runs_best_pt = os.path.join(
                model_output_dir, 'runs', model_name, 'weights', 'best.pt'
            )
            load_path = runs_best_pt if os.path.isfile(runs_best_pt) else weight_path
            sym = False
            if not load_path.endswith('.pt'):
                load_path, sym = ensure_pt_weight(load_path)
            try:
                model = YOLO(load_path)
            finally:
                if sym and os.path.islink(load_path):
                    os.remove(load_path)

            total_params = sum(p.numel() for p in model.model.parameters())

            # model.info() returns (layers, params, gradients, gflops) when
            # loaded from a proper .pt checkpoint.
            try:
                model_info = model.info(verbose=False)
                if isinstance(model_info, tuple) and len(model_info) >= 4:
                    info['total_params'] = int(model_info[1])
                    info['gflops']       = round(float(model_info[3]), 2)
                else:
                    info['total_params'] = total_params
                    # Fall back to thop if model.info() is unavailable
                    try:
                        from thop import profile
                        dummy = torch.randn(1, 3, 640, 640)
                        model.model.eval()
                        flops, _ = profile(model.model, inputs=(dummy,),
                                           verbose=False)
                        info['gflops'] = round(flops / 1e9, 2)
                    except Exception:
                        pass
            except Exception:
                info['total_params'] = total_params

            # For YOLO/RT-DETR inference checkpoints, gradients are frozen but
            # ALL parameters were trained — report total_params as trainable.
            info['trainable_params'] = info['total_params']

    except Exception as e:
        print(f'  [WARN] Could not load model {model_name}: {e}')

    return info


def ensure_pt_weight(weight_path):
    """Return a '.pt' path Ultralytics can load (creates a symlink if needed)."""
    if weight_path.endswith('.pt'):
        return weight_path, False
    pt_path = weight_path[:-4] + '.pt'
    if not os.path.isfile(pt_path):
        os.symlink(os.path.abspath(weight_path), pt_path)
        return pt_path, True
    return pt_path, False


def load_inference_summary():
    """Load inference speed data if available from inference_demo.py output."""
    path = os.path.join(PROJECT_DIR, 'results', 'demo', 'inference_summary.json')
    if os.path.isfile(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluation comparison.')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save comparison outputs.')
    parser.add_argument('--run_test', action='store_true',
                        help='Run test-set evaluation for all models '
                             '(computes test_mAP50, test_mAP50_95, test_miou).')
    # Keep --run_val for backward compatibility
    parser.add_argument('--run_val', action='store_true',
                        help='(deprecated, use --run_test) '
                             'Run fresh validation for Ultralytics models.')
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('  PCB Defect Detection — Evaluation Comparison')
    print(f'  Device: {DEVICE}')
    print('=' * 70)

    # Load inference speed data
    speed_data = load_inference_summary()

    results = []

    for model_name, meta in MODEL_META.items():
        print(f'\n  Processing: {meta["display_name"]}')
        entry = {
            'model':        meta['display_name'],
            'architecture': meta['architecture'],
            'backbone':     meta['backbone'],
        }

        model_output_dir = os.path.join(PROJECT_DIR, 'outputs', model_name)
        # Resolve weight path: Ultralytics refuses .pth → prefer .pt
        base_weight = os.path.join(model_output_dir,
                                   meta['weight_key'].replace('.pth', ''))
        if meta['type'] == 'ultralytics':
            ext_order = ('.pt', '.pth')
        else:
            ext_order = ('.pth', '.pt')
        weight_path = base_weight + ext_order[0]  # default
        for ext in ext_order:
            candidate = base_weight + ext
            if os.path.isfile(candidate):
                weight_path = candidate
                break

        # ── File size ────────────────────────────────────────────────────
        size_mb = get_file_size_mb(weight_path)
        entry['file_size_mb'] = round(size_mb, 2) if size_mb else 'N/A'

        # ── Metrics ──────────────────────────────────────────────────────
        if meta['type'] == 'pytorch':
            metrics = get_pytorch_metrics(model_output_dir, model_name)
        else:
            metrics = get_ultralytics_metrics(model_output_dir, model_name)

        # ── Run test-set evaluation (computes mAP50, mAP50-95, mIoU) ────
        if (args.run_test or args.run_val) and os.path.isfile(weight_path):
            try:
                if meta['type'] == 'pytorch':
                    test_metrics = run_pytorch_test_eval(
                        model_name, weight_path
                    )
                else:
                    test_metrics = run_ultralytics_test_eval(
                        model_name, weight_path
                    )
                metrics.update(test_metrics)
            except Exception as e:
                print(f'    [WARN] Test evaluation failed for {model_name}: {e}')
                import traceback
                traceback.print_exc()

        entry.update(metrics)

        # ── Model complexity ─────────────────────────────────────────────
        if os.path.isfile(weight_path):
            complexity = get_model_complexity(model_name, weight_path)
            entry.update(complexity)
        else:
            entry['total_params'] = 'N/A'
            entry['gflops']       = 'N/A'

        # ── Speed (from inference_demo.py) ───────────────────────────────
        speed_key = meta['display_name']
        if speed_key in speed_data:
            entry['avg_latency_ms'] = speed_data[speed_key]['avg_latency_ms']
            entry['fps']            = speed_data[speed_key]['fps']
        else:
            entry['avg_latency_ms'] = 'N/A'
            entry['fps']            = 'N/A'

        results.append(entry)

    # ── Print comparison tables ──────────────────────────────────────────
    _print_architecture_table(results)
    _print_accuracy_table(results)
    _print_speed_table(results)

    # ── Save CSV ─────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, 'comparison_table.csv')
    _save_csv(results, csv_path)

    # ── Save JSON ────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, 'comparison_table.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Comparison JSON saved to: {json_path}')


def _fmt(val, decimals=4):
    """Format a value for table display."""
    if isinstance(val, float):
        return f'{val:.{decimals}f}'
    return str(val)


def _print_architecture_table(results):
    """Print model architecture & complexity comparison."""
    print('\n' + '=' * 90)
    print('  MODEL ARCHITECTURE & COMPLEXITY')
    print('=' * 90)
    header = (f'  {"Model":<15} {"Type":<13} {"Backbone":<22} '
              f'{"Params":>12} {"GFLOPs":>10} {"Size (MB)":>10}')
    print(header)
    print('-' * 90)
    for r in results:
        params = r.get('total_params', 'N/A')
        if isinstance(params, (int, float)):
            params_str = f'{params / 1e6:.2f}M'
        else:
            params_str = str(params)
        print(f'  {r["model"]:<15} {r["architecture"]:<13} {r["backbone"]:<22} '
              f'{params_str:>12} {_fmt(r.get("gflops", "N/A")):>10} '
              f'{_fmt(r.get("file_size_mb", "N/A"), 1):>10}')
    print('=' * 90)


def _print_accuracy_table(results):
    """Print detection accuracy comparison."""
    print('\n' + '=' * 100)
    print('  DETECTION ACCURACY (Validation / Test)')
    print('=' * 100)
    header = (f'  {"Model":<15} {"mAP@0.5":>10} {"mAP@.5:.95":>12} '
              f'{"Precision":>10} {"Recall":>10} {"F1":>10} {"mIoU":>10}')
    print(header)
    print('-' * 100)
    for r in results:
        print(f'  {r["model"]:<15} '
              f'{_fmt(r.get("test_mAP50", "N/A")):>10} '
              f'{_fmt(r.get("test_mAP50_95", "N/A")):>12} '
              f'{_fmt(r.get("test_precision", "N/A")):>10} '
              f'{_fmt(r.get("test_recall", "N/A")):>10} '
              f'{_fmt(r.get("test_f1", "N/A")):>10} '
              f'{_fmt(r.get("test_miou", "N/A")):>10}')
    print('=' * 100)


def _print_speed_table(results):
    """Print inference speed comparison."""
    print('\n' + '=' * 60)
    print('  INFERENCE SPEED')
    print('=' * 60)
    header = f'  {"Model":<15} {"Latency (ms)":>15} {"FPS":>10}'
    print(header)
    print('-' * 60)
    for r in results:
        print(f'  {r["model"]:<15} '
              f'{_fmt(r.get("avg_latency_ms", "N/A"), 2):>15} '
              f'{_fmt(r.get("fps", "N/A"), 1):>10}')
    print('=' * 60)


def _save_csv(results, path):
    """Save comparison results as CSV."""
    if not results:
        return

    # Collect all keys
    all_keys = []
    for r in results:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow({k: str(r.get(k, '')) for k in all_keys})

    print(f'  Comparison CSV saved to: {path}')


if __name__ == '__main__':
    main()
