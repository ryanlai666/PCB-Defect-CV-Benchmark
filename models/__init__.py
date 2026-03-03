"""
Model registry.  Use create_model(name, num_classes) to instantiate any
supported architecture.

Model categories
----------------
PyTorch-loop:   faster_rcnn, faster_rcnn_ft, vit_det, vit_mamba
Ultralytics:    yolo26, sme_yolo, rt_detr
DEIMv2:         deimv2_l, deimv2_x  (subprocess via DEIMv2/train.py)
"""

from config import NUM_CLASSES, USE_PRETRAINED

# Registry populated lazily to avoid heavy imports at top-level
_MODEL_BUILDERS = {
    # PyTorch training loop models
    'faster_rcnn':    'models.faster_rcnn',
    'faster_rcnn_ft': 'models.faster_rcnn_ft',
    'vit_det':        'models.vit_det',
    'vit_mamba':      'models.vit_mamba',
    # Ultralytics models
    'yolo26':         'models.yolo26',
    'sme_yolo':       'models.sme_yolo',
    'rt_detr':        'models.rt_detr',
    # DEIMv2 models (subprocess via DEIMv2/train.py)
    'deimv2_l':       'models.deimv2_l',
    'deimv2_x':       'models.deimv2_x',
}

AVAILABLE_MODELS = list(_MODEL_BUILDERS.keys())


def create_model(model_name: str, num_classes: int = NUM_CLASSES,
                 pretrained: bool = USE_PRETRAINED, **kwargs):
    """
    Factory function.

    For PyTorch-loop models (faster_rcnn, vit_det, vit_mamba):
        Returns a torch.nn.Module ready for train_model().

    For Ultralytics models (yolo26, sme_yolo, rt_detr):
        Returns an ultralytics.YOLO object.
        Train with:  model.train(data='data.yaml', epochs=N, imgsz=640)
    """
    if model_name not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {AVAILABLE_MODELS}"
        )

    import importlib
    module = importlib.import_module(_MODEL_BUILDERS[model_name])
    return module.build(num_classes=num_classes, pretrained=pretrained, **kwargs)
