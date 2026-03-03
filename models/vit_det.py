"""
ViT-based object detector:  timm ViT backbone  +  Faster R-CNN detection head.

Uses pretrained ImageNet weights from timm, adapted for detection by
extracting intermediate feature maps and feeding them into torchvision's
Faster R-CNN.

Install:  pip install timm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import timm


class ViTBackbone(nn.Module):
    """
    Wraps a timm ViT model as a feature extractor for Faster R-CNN.

    ViT outputs a sequence of patch tokens; we reshape them back to a 2D
    spatial feature map so that the RPN / RoI heads can work normally.
    """

    def __init__(self, model_name='vit_base_patch16_224', pretrained=True,
                 img_size=640):
        super().__init__()
        self.vit = timm.create_model(
            model_name, pretrained=pretrained,
            img_size=img_size, num_classes=0,  # remove classification head
        )
        self.embed_dim  = self.vit.embed_dim          # typically 768
        self.patch_size = self.vit.patch_embed.patch_size[0]  # typically 16
        self.img_size   = img_size
        self.grid_size  = img_size // self.patch_size  # e.g. 640/16 = 40

        # 1×1 conv to reduce channel dim (768 → 256) for efficient RPN
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.out_channels = 256   # Faster R-CNN reads this attribute

    def forward(self, x):
        # x: (B, 3, H, W) — may not match self.img_size after FasterRCNN transform
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        tokens = self.vit.forward_features(x)  # (B, N_total, D)

        # Some ViT variants include a CLS token at position 0 — detect & drop it
        expected_patches = self.grid_size * self.grid_size
        if tokens.shape[1] == expected_patches + 1:
            tokens = tokens[:, 1:, :]  # drop CLS
        elif tokens.shape[1] != expected_patches:
            # Fallback: try dropping first token
            tokens = tokens[:, 1:, :]

        B, N, D = tokens.shape
        h = w = int(N ** 0.5)
        feat = tokens.permute(0, 2, 1).reshape(B, D, h, w)   # (B, D, h, w)
        feat = self.channel_reduce(feat)                       # (B, 256, h, w)
        return feat


def build(num_classes, pretrained=True,
          model_name='vit_base_patch16_224', img_size=640, **kwargs):
    """
    Create ViT + Faster R-CNN detector.

    Args:
        num_classes: total classes including background.
        pretrained:  load ImageNet-pretrained ViT weights.
        model_name:  any timm ViT variant.
        img_size:    input resolution (must match ViT config).
    """
    backbone = ViTBackbone(model_name=model_name, pretrained=pretrained,
                           img_size=img_size)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # Force Faster R-CNN's internal transform to use img_size
        # instead of the default (min=800, max=1333) which would
        # resize images to a different size than the ViT expects.
        min_size=img_size,
        max_size=img_size,
    )
    tag = 'ImageNet pretrained' if pretrained else 'random init'
    print(f'[vit_det] {model_name} + Faster R-CNN  ({tag})')
    return model
