#!/usr/bin/env python3
"""
Log Parsing & Training Summary Script.

Reads training logs / history files for all five models and generates:
  1. Per-model training loss curves
  2. Per-model validation metric curves
  3. A unified comparison overlay plot
  4. Training time summary table

Data sources:
  PyTorch-loop models (Faster R-CNN, ViT-Det):
    outputs/<model>/history_<model>.json
  Ultralytics models (SME-YOLO, YOLO26, RT-DETR):
    outputs/<model>/runs/<model>/results.csv

Usage:
    python parse_logs.py [--output_dir results/plots]
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ── Project path ─────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Data loaders ────────────────────────────────────────────────────────────

def load_pytorch_history(output_dir, model_name, display_name):
    """Load PyTorch JSON training history for a PyTorch-loop model.

    Works for both faster_rcnn and vit_det (same file naming convention).

    Returns:
        dict with keys: epochs, train_loss, val_f1, val_miou,
                        val_precision, val_recall, training_time_s
    """
    history_path = os.path.join(output_dir, f'history_{model_name}.json')
    if not os.path.isfile(history_path):
        print(f'  [SKIP] History not found: {history_path}')
        return None

    with open(history_path, 'r') as f:
        h = json.load(f)

    n_epochs = len(h.get('train_cls_loss', []))
    epochs   = list(range(1, n_epochs + 1))

    # Combine cls + box loss as total training loss
    train_loss = [
        c + b for c, b in zip(h.get('train_cls_loss', []),
                               h.get('train_box_loss', []))
    ]

    # Try to extract training time from the log file
    training_time_s = _extract_pytorch_training_time(output_dir, model_name)

    return {
        'model':        display_name,
        'epochs':       epochs,
        'train_loss':   train_loss,
        'train_cls_loss': h.get('train_cls_loss', []),
        'train_box_loss': h.get('train_box_loss', []),
        'val_f1':       h.get('val_f1', []),
        'val_miou':     h.get('val_miou', []),
        'val_precision': h.get('val_precision', []),
        'val_recall':   h.get('val_recall', []),
        'training_time_s': training_time_s,
    }


# Backward-compat alias
def load_frcnn_history(output_dir):
    return load_pytorch_history(output_dir, 'faster_rcnn', 'Faster R-CNN')


def _extract_pytorch_training_time(output_dir, model_name):
    """Parse elapsed time from any PyTorch training log."""
    log_path = os.path.join(output_dir, f'train_{model_name}.log')
    if not os.path.isfile(log_path):
        return None
    with open(log_path, 'r') as f:
        content = f.read()
    # Pattern: "Elapsed time: 0h 2m 14s"
    match = re.search(r'Elapsed time:\s*(\d+)h\s*(\d+)m\s*(\d+)s', content)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return h * 3600 + m * 60 + s
    return None


def load_ultralytics_results(output_dir, model_name, display_name):
    """Load Ultralytics results.csv.

    Handles different column layouts:
      - YOLO  models: train/box_loss, train/cls_loss, train/dfl_loss,
                      val/box_loss,   val/cls_loss,   val/dfl_loss
      - RT-DETR:      train/giou_loss, train/cls_loss, train/l1_loss,
                      val/giou_loss,   val/cls_loss,   val/l1_loss

    Returns:
        dict with keys matching load_pytorch_history output structure.
    """
    csv_path = os.path.join(output_dir, 'runs', model_name, 'results.csv')
    if not os.path.isfile(csv_path):
        print(f'  [SKIP] Results CSV not found: {csv_path}')
        return None

    epochs, times = [], []
    precision, recall, mAP50, mAP50_95 = [], [], [], []

    # Per-epoch sub-loss accumulators — filled only if the column exists
    train_subloss_cols = {}   # col_name → list
    val_subloss_cols   = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        return None

    # Normalise keys once
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in raw_rows]
    all_cols = list(rows[0].keys())

    # Discover which train/val loss columns are present
    train_loss_cols = [c for c in all_cols if c.startswith('train/') and 'loss' in c]
    val_loss_cols   = [c for c in all_cols if c.startswith('val/')   and 'loss' in c]

    for col in train_loss_cols:
        train_subloss_cols[col] = []
    for col in val_loss_cols:
        val_subloss_cols[col] = []

    for row in rows:
        epochs.append(int(row['epoch']))
        times.append(float(row['time']))
        for col in train_loss_cols:
            train_subloss_cols[col].append(float(row.get(col, 0)))
        for col in val_loss_cols:
            val_subloss_cols[col].append(float(row.get(col, 0)))
        precision.append(float(row['metrics/precision(B)']))
        recall.append(float(row['metrics/recall(B)']))
        mAP50.append(float(row['metrics/mAP50(B)']))
        mAP50_95.append(float(row['metrics/mAP50-95(B)']))

    # Compute total losses by summing all sub-losses per epoch
    n = len(epochs)
    train_loss = [
        sum(train_subloss_cols[c][i] for c in train_loss_cols)
        for i in range(n)
    ]
    val_loss = [
        sum(val_subloss_cols[c][i] for c in val_loss_cols)
        for i in range(n)
    ]

    # F1 from precision and recall
    val_f1 = [
        2 * p * r / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(precision, recall)
    ]

    training_time_s = times[-1] if times else None

    result = {
        'model':       display_name,
        'epochs':      [e + 1 for e in range(n)],  # 1-indexed
        'train_loss':  train_loss,
        'val_loss':    val_loss,
        'val_f1':      val_f1,
        'val_precision': precision,
        'val_recall':  recall,
        'val_mAP50':   mAP50,
        'val_mAP50_95': mAP50_95,
        'training_time_s': training_time_s,
    }

    # Attach named sub-losses so plot_individual_model can label them
    # Map generic YOLO names; RT-DETR keeps its own names
    for col, vals in train_subloss_cols.items():
        short = col.replace('train/', 'train_').replace('/', '_').replace('-', '_')
        result[short] = vals
    for col, vals in val_subloss_cols.items():
        short = col.replace('val/', 'val_').replace('/', '_').replace('-', '_')
        result[short] = vals

    # Convenience aliases used by plot_individual_model when available
    col_map = {
        'train_box_loss':  'train/box_loss',
        'train_cls_loss':  'train/cls_loss',
        'train_dfl_loss':  'train/dfl_loss',
        'val_box_loss':    'val/box_loss',
        'val_cls_loss':    'val/cls_loss',
        'val_dfl_loss':    'val/dfl_loss',
    }
    for alias, col in col_map.items():
        if col in train_subloss_cols:
            result[alias] = train_subloss_cols[col]
        elif col in val_subloss_cols:
            result[alias] = val_subloss_cols[col]

    return result


def load_deimv2_history(output_dir, model_name, display_name):
    """Load DEIMv2 training history from log.txt.

    Each epoch's COCO evaluation is logged as a JSON line containing
    'test_coco_eval_bbox' and 'epoch'.

    Returns:
        dict with keys matching other loaders' output structure.
    """
    log_path = os.path.join(output_dir, 'log.txt')
    if not os.path.isfile(log_path):
        print(f'  [SKIP] DEIMv2 log not found: {log_path}')
        return None

    # Use a dict keyed by epoch to deduplicate (log may contain entries
    # from both initial and resumed training runs for the same epochs).
    # We keep the LATEST entry for each epoch.
    epoch_data = {}   # epoch_0indexed -> (train_loss, coco_eval_bbox)

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and 'test_coco_eval_bbox' in line:
                try:
                    entry = json.loads(line)
                    ep = entry.get('epoch', -1)
                    ce = entry['test_coco_eval_bbox']
                    epoch_data[ep] = (entry.get('train_loss', 0), ce)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

    if not epoch_data:
        print(f'  [SKIP] No COCO eval entries in {log_path}')
        return None

    # Sort by epoch and unpack
    epochs = []
    train_loss = []
    val_mAP50 = []
    val_mAP50_95 = []
    val_mAP75 = []
    val_recall = []

    for ep in sorted(epoch_data.keys()):
        tl, ce = epoch_data[ep]
        epochs.append(ep + 1)  # 0-indexed → 1-indexed
        train_loss.append(tl)
        val_mAP50_95.append(ce[0])
        val_mAP50.append(ce[1])
        val_mAP75.append(ce[2])
        val_recall.append(ce[8])  # AR@100

    # Compute F1 using mAP50 as precision proxy and AR@100 as recall proxy
    val_f1 = [
        2 * p * r / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(val_mAP50, val_recall)
    ]

    # Extract training time
    training_time_s = _extract_pytorch_training_time(output_dir, model_name)

    return {
        'model':         display_name,
        'epochs':        epochs,
        'train_loss':    train_loss,
        'val_f1':        val_f1,
        'val_precision': val_mAP50,   # mAP50 as precision proxy
        'val_recall':    val_recall,
        'val_mAP50':     val_mAP50,
        'val_mAP50_95':  val_mAP50_95,
        'training_time_s': training_time_s,
    }



# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_individual_model(data, output_dir):
    """Generate per-model training curves."""
    model     = data['model']
    safe_name = model.lower().replace(' ', '_').replace('-', '_')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model} — Training Curves', fontsize=14, fontweight='bold')

    epochs = data['epochs']

    # ── Panel 1: Training Loss ───────────────────────────────────────────
    ax = axes[0]
    if 'train_box_loss' in data and data['train_box_loss']:
        ax.plot(epochs, data['train_box_loss'], label='Box Loss', linewidth=1.5)
    if 'train_cls_loss' in data and data['train_cls_loss']:
        ax.plot(epochs, data['train_cls_loss'], label='Cls Loss', linewidth=1.5)
    if 'train_dfl_loss' in data and data.get('train_dfl_loss'):
        ax.plot(epochs, data['train_dfl_loss'], label='DFL Loss', linewidth=1.5)
    ax.plot(epochs, data['train_loss'], label='Total Loss',
            linewidth=2, color='black', linestyle='--')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Validation Loss (Ultralytics) or F1/mIoU (PyTorch) ─────
    ax = axes[1]
    if 'val_loss' in data and data.get('val_loss'):
        ax.plot(epochs, data['val_loss'], label='Val Total Loss',
                linewidth=2, color='red')
        if data.get('val_box_loss'):
            ax.plot(epochs, data['val_box_loss'], label='Val Box', linewidth=1)
        if data.get('val_cls_loss'):
            ax.plot(epochs, data['val_cls_loss'], label='Val Cls', linewidth=1)
        ax.set_title('Validation Loss')
        ax.set_ylabel('Loss')
    else:
        if data.get('val_f1'):
            ax.plot(epochs, data['val_f1'], label='F1', marker='o', markersize=3)
        if data.get('val_miou'):
            ax.plot(epochs, data['val_miou'], label='mIoU', marker='s', markersize=3)
        ax.set_title('Validation F1 & mIoU')
        ax.set_ylabel('Score')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Precision / Recall / mAP ────────────────────────────────
    ax = axes[2]
    if data.get('val_precision'):
        ax.plot(epochs, data['val_precision'], label='Precision', marker='o',
                markersize=3, linewidth=1.5)
    if data.get('val_recall'):
        ax.plot(epochs, data['val_recall'], label='Recall', marker='s',
                markersize=3, linewidth=1.5)
    if data.get('val_mAP50'):
        ax.plot(epochs, data['val_mAP50'], label='mAP@0.5', marker='^',
                markersize=3, linewidth=1.5)
    if data.get('val_mAP50_95'):
        ax.plot(epochs, data['val_mAP50_95'], label='mAP@0.5:0.95',
                marker='v', markersize=3, linewidth=1.5)
    if data.get('val_f1'):
        ax.plot(epochs, data['val_f1'], label='F1', marker='D',
                markersize=3, linewidth=1.5)
    ax.set_title('Validation Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{safe_name}_curves.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def plot_comparison(all_data, output_dir):
    """Generate an overlay comparison plot across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Comparison — Training Curves', fontsize=14,
                 fontweight='bold')

    # Color palette extended to 5 models
    colors = {
        'Faster R-CNN': '#e74c3c',
        'SME-YOLO':     '#2ecc71',
        'YOLO26':        '#3498db',
        'ViT-Det':      '#9b59b6',
        'RT-DETR':      '#f39c12',
        'DEIMv2-L':     '#1abc9c',
    }
    markers = {
        'Faster R-CNN': 'o',
        'SME-YOLO':     's',
        'YOLO26':        '^',
        'ViT-Det':      'D',
        'RT-DETR':      'P',
        'DEIMv2-L':     'X',
    }

    # ── Panel 1: Training Loss ───────────────────────────────────────────
    ax = axes[0]
    for d in all_data:
        c = colors.get(d['model'], 'gray')
        ax.plot(d['epochs'], d['train_loss'], label=d['model'],
                color=c, linewidth=2)
    ax.set_title('Training Loss (Total)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Validation F1 ───────────────────────────────────────────
    ax = axes[1]
    for d in all_data:
        if d.get('val_f1'):
            c = colors.get(d['model'], 'gray')
            m = markers.get(d['model'], 'o')
            ax.plot(d['epochs'], d['val_f1'], label=d['model'],
                    color=c, marker=m, markersize=4, linewidth=2)
    ax.set_title('Validation F1-Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Precision & Recall ──────────────────────────────────────
    ax = axes[2]
    for d in all_data:
        c = colors.get(d['model'], 'gray')
        if d.get('val_precision'):
            ax.plot(d['epochs'], d['val_precision'],
                    label=f'{d["model"]} P', color=c, linestyle='-',
                    linewidth=1.5)
        if d.get('val_recall'):
            ax.plot(d['epochs'], d['val_recall'],
                    label=f'{d["model"]} R', color=c, linestyle='--',
                    linewidth=1.5)
    ax.set_title('Precision (solid) & Recall (dashed)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'comparison_curves.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Parse training logs & plot.')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                        help='Directory to save plots.')
    args = parser.parse_args()

    plot_dir = os.path.join(PROJECT_DIR, args.output_dir)
    os.makedirs(plot_dir, exist_ok=True)

    print('=' * 70)
    print('  PCB Defect Detection — Training Log Parser')
    print('=' * 70)

    all_data = []

    # ── Faster R-CNN (PyTorch loop) ───────────────────────────────────────
    frcnn_dir = os.path.join(PROJECT_DIR, 'outputs', 'faster_rcnn')
    frcnn = load_pytorch_history(frcnn_dir, 'faster_rcnn', 'Faster R-CNN')
    if frcnn:
        all_data.append(frcnn)
        plot_individual_model(frcnn, plot_dir)

    # ── ViT-Det (PyTorch loop) ────────────────────────────────────────────
    vitdet_dir = os.path.join(PROJECT_DIR, 'outputs', 'vit_det')
    vitdet = load_pytorch_history(vitdet_dir, 'vit_det', 'ViT-Det')
    if vitdet:
        all_data.append(vitdet)
        plot_individual_model(vitdet, plot_dir)

    # ── SME-YOLO (Ultralytics) ────────────────────────────────────────────
    sme_dir = os.path.join(PROJECT_DIR, 'outputs', 'sme_yolo')
    sme = load_ultralytics_results(sme_dir, 'sme_yolo', 'SME-YOLO')
    if sme:
        all_data.append(sme)
        plot_individual_model(sme, plot_dir)

    # ── YOLO26 (Ultralytics) ──────────────────────────────────────────────
    yolo26_dir = os.path.join(PROJECT_DIR, 'outputs', 'yolo26')
    y26 = load_ultralytics_results(yolo26_dir, 'yolo26', 'YOLO26')
    if y26:
        all_data.append(y26)
        plot_individual_model(y26, plot_dir)

    # ── RT-DETR (Ultralytics) ─────────────────────────────────────────────
    rtdetr_dir = os.path.join(PROJECT_DIR, 'outputs', 'rt_detr')
    rtdetr = load_ultralytics_results(rtdetr_dir, 'rt_detr', 'RT-DETR')
    if rtdetr:
        all_data.append(rtdetr)
        plot_individual_model(rtdetr, plot_dir)

    # ── DEIMv2-L (DEIMv2 engine) ──────────────────────────────────────────
    deimv2_dir = os.path.join(PROJECT_DIR, 'outputs', 'deimv2_l')
    deimv2 = load_deimv2_history(deimv2_dir, 'deimv2_l', 'DEIMv2-L')
    if deimv2:
        all_data.append(deimv2)
        plot_individual_model(deimv2, plot_dir)

    # ── Comparison overlay ───────────────────────────────────────────────
    if len(all_data) >= 2:
        plot_comparison(all_data, plot_dir)

    # ── Training time summary ────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'  {"Model":<20} {"Epochs":<10} {"Training Time":<20}')
    print('-' * 70)
    time_summary = {}
    for d in all_data:
        n_ep = len(d['epochs'])
        t    = d.get('training_time_s')
        if t is not None:
            hours    = int(t // 3600)
            mins     = int((t % 3600) // 60)
            secs     = int(t % 60)
            time_str = f'{hours}h {mins}m {secs}s'
        else:
            time_str = 'N/A'
        print(f'  {d["model"]:<20} {n_ep:<10} {time_str:<20}')
        time_summary[d['model']] = {
            'epochs':           n_ep,
            'training_time_s':  t,
            'training_time_str': time_str,
        }
    print('=' * 70)

    # Save training time summary
    summary_path = os.path.join(plot_dir, 'training_time_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(time_summary, f, indent=2)
    print(f'\n  Training time summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
