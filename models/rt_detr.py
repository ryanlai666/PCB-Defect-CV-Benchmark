"""
RT-DETR (Real-Time Detection Transformer) via the Ultralytics package.

RT-DETR is end-to-end (NMS-free) and achieves strong accuracy on small objects.
Pretrained on COCO by default.

Install:  pip install ultralytics
Docs:     https://docs.ultralytics.com/models/rtdetr/
"""

from ultralytics import YOLO


def build(num_classes, pretrained=True, variant='rtdetr-l', **kwargs):
    """
    Load an RT-DETR model.

    Args:
        num_classes: detection classes (background excluded).
        pretrained:  if True, load COCO-pretrained weights.
        variant:     'rtdetr-l' (large) | 'rtdetr-x' (extra-large)

    Returns:
        ultralytics.YOLO object.
    """
    weight_file = f'{variant}.pt' if pretrained else f'{variant}.yaml'
    model = YOLO(weight_file)
    print(f'[rt_detr] Loaded {weight_file}  (pretrained={pretrained})')
    return model


def train_rt_detr(model, data_yaml, epochs=20, imgsz=640, batch=16, **kwargs):
    """Train with Ultralytics trainer."""
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        **kwargs,
    )
    return results


def evaluate_rt_detr(model, data_yaml, **kwargs):
    """Run validation."""
    return model.val(data=data_yaml, **kwargs)
