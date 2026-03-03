"""
Faster R-CNN — Fine-Tune only mode.

Strategy:
  - Load ResNet-50 FPN v2 pretrained on COCO.
  - Freeze the entire backbone (feature extractor) and FPN.
  - Only train: RPN + RoI heads (box predictor).

This is 5–8x faster per epoch than full training because:
  - Backbone gradients are not computed (no backprop through 43M params).
  - Only ~9M head parameters are updated.
  - COCO pretraining already gives excellent feature representations.
"""

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build(num_classes, pretrained=True, **kwargs):
    """
    Entry point called by the model registry.

    Always loads pretrained COCO weights.
    Freezes backbone + FPN; only RPN and RoI box predictor are trainable.
    """
    print('[faster_rcnn_ft] Loading pretrained ResNet-50 FPN v2 backbone (COCO)')
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # ── Freeze backbone (body) and FPN ──────────────────────────────────
    frozen_modules = [model.backbone]
    frozen_count = 0
    for module in frozen_modules:
        for param in module.parameters():
            param.requires_grad = False
            frozen_count += 1

    frozen_params = sum(
        p.numel() for p in model.backbone.parameters()
    )

    # ── Replace the box predictor head with our num_classes ─────────────
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())

    print(f'[faster_rcnn_ft]  Frozen params  : {frozen_params/1e6:.2f}M  (backbone + FPN)')
    print(f'[faster_rcnn_ft]  Trainable params: {trainable_params/1e6:.2f}M  (RPN + RoI heads)')
    print(f'[faster_rcnn_ft]  Total params    : {total_params/1e6:.2f}M')

    return model
