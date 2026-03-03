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


def compute_map(all_predictions, all_targets):
    """Compute COCO-style mAP50 and mAP50-95 using torchmetrics.

    Args:
        all_predictions: list of dicts, each with 'boxes', 'labels', 'scores'
                         (tensors, may be on any device).
        all_targets:     list of dicts, each with 'boxes', 'labels'
                         (tensors, may be on any device).

    Returns:
        dict with keys 'mAP50' and 'mAP50_95'.
    """
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(
        iou_type='bbox',
        iou_thresholds=None,   # uses default COCO thresholds 0.50:0.05:0.95
    )

    # torchmetrics expects CPU tensors
    preds_cpu = []
    for p in all_predictions:
        preds_cpu.append({
            'boxes':  p['boxes'].detach().cpu().float(),
            'scores': p['scores'].detach().cpu().float(),
            'labels': p['labels'].detach().cpu().long(),
        })

    tgts_cpu = []
    for t in all_targets:
        tgts_cpu.append({
            'boxes':  t['boxes'].detach().cpu().float(),
            'labels': t['labels'].detach().cpu().long(),
        })

    metric.update(preds_cpu, tgts_cpu)
    result = metric.compute()

    return {
        'mAP50':    float(result['map_50']),
        'mAP50_95': float(result['map']),
    }


def compute_miou_from_predictions(all_predictions, all_targets,
                                   iou_threshold=IOU_THRESHOLD,
                                   score_threshold=SCORE_THRESHOLD):
    """Compute mean IoU across all matched prediction-target pairs.

    This is useful for computing mIoU for Ultralytics models, which do
    not natively report this metric.

    Args:
        all_predictions: list of dicts with 'boxes', 'labels', 'scores'
        all_targets:     list of dicts with 'boxes', 'labels'
        iou_threshold:   minimum IoU for a match
        score_threshold: minimum confidence to keep a prediction

    Returns:
        float: mean IoU of matched boxes, or 0.0 if no matches.
    """
    total_iou = 0.0
    matched   = 0

    for pred, target in zip(all_predictions, all_targets):
        pred_boxes  = pred['boxes'].detach().cpu()
        pred_labels = pred['labels'].detach().cpu()
        pred_scores = pred['scores'].detach().cpu()

        keep = pred_scores >= score_threshold
        pred_boxes  = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        gt_boxes  = target['boxes'].detach().cpu()
        gt_labels = target['labels'].detach().cpu()

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        iou_matrix = ops.box_iou(pred_boxes.float(), gt_boxes.float())
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for p_idx in range(len(pred_boxes)):
            match_iou, gt_idx = iou_matrix[p_idx].max(dim=0)
            if (match_iou >= iou_threshold
                    and not gt_matched[gt_idx]
                    and pred_labels[p_idx] == gt_labels[gt_idx]):
                gt_matched[gt_idx] = True
                total_iou += match_iou.item()
                matched += 1

    return (total_iou / matched) if matched > 0 else 0.0
