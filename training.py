"""
Generic PyTorch training loop for Faster R-CNN–style models.
YOLO / RT-DETR models use Ultralytics' built-in trainer instead.
"""

import math
import sys
import torch

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    GRAD_CLIP_NORM, BEST_MODEL_PATH,
)
from evaluation import evaluate_predictions, compute_metrics


def _strip_tv_tensors(x):
    """Convert tv_tensors.Image / BoundingBoxes to plain torch.Tensor.

    Faster R-CNN's GeneralizedRCNNTransform does its own resizing and
    batching.  If the inputs are still wrapped in tv_tensors types, the
    __torch_function__ override causes reshape errors.  Stripping the
    wrapper avoids this entirely.
    """
    if hasattr(x, 'as_subclass'):
        return x.as_subclass(torch.Tensor)
    return x


def train_one_epoch(model, loader, optimizer, device=DEVICE, log_interval=10):
    """Train for one epoch. Returns dict of average losses.

    Args:
        log_interval: print running loss every N batches (default 10).
    """
    model.train()
    epoch_cls_loss = 0.0
    epoch_box_loss = 0.0
    running_cls = 0.0
    running_box = 0.0
    n_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader, 1):
        images  = [_strip_tv_tensors(img).to(device) for img in images]
        targets = [{k: _strip_tv_tensors(v).to(device) for k, v in t.items()}
                   for t in targets]

        loss_dict = model(images, targets)

        cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0))
        box_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0))
        losses   = sum(loss for loss in loss_dict.values())

        if not math.isfinite(losses.item()):
            print(f'\nLoss is {losses.item()}, stopping training')
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        epoch_cls_loss += cls_loss.item()
        epoch_box_loss += box_loss.item()
        running_cls += cls_loss.item()
        running_box += box_loss.item()

        # Log every log_interval batches
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            batches_run = batch_idx % log_interval if batch_idx % log_interval != 0 else log_interval
            avg_cls = running_cls / batches_run
            avg_box = running_box / batches_run
            print(f"  [Batch {batch_idx:>4d}/{n_batches}] "
                  f"Cls Loss: {avg_cls:.4f} | Box Loss: {avg_box:.4f}")
            running_cls = 0.0
            running_box = 0.0

    n = n_batches
    return {
        'cls_loss': epoch_cls_loss / n,
        'box_loss': epoch_box_loss / n,
    }


@torch.no_grad()
def validate(model, loader, device=DEVICE,
             iou_threshold=0.5, score_threshold=0.5):
    """Run validation, return metrics dict."""
    model.eval()
    total_tp = total_fp = total_fn = 0
    epoch_ious = []

    for images, targets in loader:
        images  = [_strip_tv_tensors(img).to(device) for img in images]
        targets = [{k: _strip_tv_tensors(v).to(device) for k, v in t.items()}
                   for t in targets]

        predictions = model(images)
        tp, fp, fn, avg_iou = evaluate_predictions(
            predictions, targets, iou_threshold, score_threshold,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if avg_iou > 0:
            epoch_ious.append(avg_iou)

    precision, recall, f1 = compute_metrics(total_tp, total_fp, total_fn)
    mean_iou = sum(epoch_ious) / len(epoch_ious) if epoch_ious else 0.0

    return {
        'f1': f1,
        'miou': mean_iou,
        'precision': precision,
        'recall': recall,
    }


def train_model(model, train_loader, val_loader,
                num_epochs=NUM_EPOCHS,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                device=DEVICE,
                save_path=BEST_MODEL_PATH,
                test_mode=False,
                score_threshold=0.5):
    """
    Full training loop with cosine-annealing LR and best-model checkpointing.
    Returns a history dict for plotting.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
    )

    history = {
        'train_cls_loss': [], 'train_box_loss': [],
        'val_f1': [], 'val_miou': [],
        'val_precision': [], 'val_recall': [],
    }
    best_f1 = 0.0

    for epoch in range(num_epochs):
        # ── Train ──────────────────────────────────────────────────────
        log_interval = 10 if test_mode else len(train_loader)
        losses = train_one_epoch(model, train_loader, optimizer, device, log_interval=log_interval)
        scheduler.step()

        history['train_cls_loss'].append(losses['cls_loss'])
        history['train_box_loss'].append(losses['box_loss'])

        print(f"Epoch [{epoch+1}/{num_epochs}] Train | "
              f"Cls Loss: {losses['cls_loss']:.4f} | "
              f"Box Loss: {losses['box_loss']:.4f}")

        # ── Validate ───────────────────────────────────────────────────
        metrics = validate(model, val_loader, device, score_threshold=score_threshold)
        for key in ('f1', 'miou', 'precision', 'recall'):
            history[f'val_{key}'].append(metrics[key])

        print(f"Epoch [{epoch+1}/{num_epochs}] Val   | "
              f"F1: {metrics['f1']:.4f} | mIoU: {metrics['miou']:.4f} | "
              f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")

        # ── Checkpoint ─────────────────────────────────────────────────
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            print(f"--> New best model! F1: {best_f1:.4f}  Saving '{save_path}'")
            torch.save(model.state_dict(), save_path)

    print('Training Complete!')
    return history
