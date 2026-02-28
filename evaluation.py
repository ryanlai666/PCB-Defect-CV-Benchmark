"""
Evaluation metrics for object detection models.
"""

import torch
import torchvision.ops as ops

from config import IOU_THRESHOLD, SCORE_THRESHOLD


def evaluate_predictions(predictions, targets,
                         iou_threshold=IOU_THRESHOLD,
                         score_threshold=SCORE_THRESHOLD):
    """
    Computes TP, FP, FN, and average IoU for a batch of predictions.

    Args:
        predictions: list of dicts with 'boxes', 'labels', 'scores'
        targets:     list of dicts with 'boxes', 'labels'
    Returns:
        (tp, fp, fn, avg_iou)
    """
    tp = fp = fn = 0
    total_iou = 0.0
    matched_boxes = 0

    for pred, target in zip(predictions, targets):
        pred_boxes  = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']

        # Filter by confidence
        keep = pred_scores >= score_threshold
        pred_boxes  = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        gt_boxes  = target['boxes']
        gt_labels = target['labels']

        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue
        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue

        iou_matrix = ops.box_iou(pred_boxes, gt_boxes)
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for p_idx in range(len(pred_boxes)):
            match_iou, gt_idx = iou_matrix[p_idx].max(dim=0)

            if (match_iou >= iou_threshold
                    and not gt_matched[gt_idx]
                    and pred_labels[p_idx] == gt_labels[gt_idx]):
                tp += 1
                gt_matched[gt_idx] = True
                total_iou += match_iou.item()
                matched_boxes += 1
            else:
                fp += 1

        fn += (~gt_matched).sum().item()

    avg_iou = (total_iou / matched_boxes) if matched_boxes > 0 else 0.0
    return tp, fp, fn, avg_iou


def compute_metrics(tp, fp, fn):
    """Return (precision, recall, f1) from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1
