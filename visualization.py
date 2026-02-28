"""
Visualization utilities for PCB defect detection.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import CLASS_MAP


def plot_augmented_sample(image_tensor, target, class_map=None):
    """
    Plot an image tensor with bounding boxes overlaid.

    Args:
        image_tensor: (C, H, W) float tensor in [0, 1].
        target:       dict with 'boxes' and 'labels'.
        class_map:    optional override for class id → name mapping.
    """
    if class_map is None:
        class_map = CLASS_MAP

    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_np)

    boxes  = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        width  = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor='lime', facecolor='none',
        )
        ax.add_patch(rect)

        class_name = class_map.get(label, f'ID: {label}')
        ax.text(
            xmin, ymin - 5, class_name,
            color='black', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none', pad=2),
        )

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot training/validation curves.

    Args:
        history: dict with keys like 'train_cls_loss', 'train_box_loss',
                 'val_f1', 'val_miou', 'val_precision', 'val_recall'.
    """
    epochs = range(1, len(history.get('train_cls_loss', [])) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    if 'train_cls_loss' in history:
        axes[0].plot(epochs, history['train_cls_loss'], label='Cls Loss')
    if 'train_box_loss' in history:
        axes[0].plot(epochs, history['train_box_loss'], label='Box Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 / mIoU
    if 'val_f1' in history:
        axes[1].plot(epochs, history['val_f1'], label='F1', marker='o', markersize=3)
    if 'val_miou' in history:
        axes[1].plot(epochs, history['val_miou'], label='mIoU', marker='s', markersize=3)
    axes[1].set_title('Validation F1 & mIoU')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Precision / Recall
    if 'val_precision' in history:
        axes[2].plot(epochs, history['val_precision'], label='Precision', marker='o', markersize=3)
    if 'val_recall' in history:
        axes[2].plot(epochs, history['val_recall'], label='Recall', marker='s', markersize=3)
    axes[2].set_title('Validation Precision & Recall')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
