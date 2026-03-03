#!/usr/bin/env python3
"""
convert_deeppcb_to_coco.py
==========================
Converts the DeepPCB dataset from its native annotation format
into COCO-format JSON files suitable for DEIMv2 training.

DeepPCB split-file format (trainval.txt / test.txt):
    Each line: <img_rel_path> <ann_rel_path>
    e.g.: group20085/20085/20085000.jpg  group20085/20085_not/20085000.txt

    Images are in:      PCBData/<group>/<subgroup>/XXXXXXX_test.jpg
                                                   XXXXXXX_temp.jpg   ← skip (template)
    Annotations are in: PCBData/<group>/<subgroup_not>/XXXXXXX.txt
                        Format per line: x1,y1,x2,y2,type
                        type: 0=bg(skip), 1=open, 2=short, 3=mousebite,
                              4=spur, 5=copper, 6=pin-hole

Output COCO JSON structure:
    {
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }

Usage:
    python scripts/convert_deeppcb_to_coco.py \\
        --deeppcb_dir  DeepPCB/PCBData \\
        --output_dir   data/deeppcb_coco \\
        --val_ratio    0.15 \\
        --seed         42
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


# ─── Category map ────────────────────────────────────────────────────────────
CATEGORIES = [
    {"id": 1, "name": "open",       "supercategory": "defect"},
    {"id": 2, "name": "short",      "supercategory": "defect"},
    {"id": 3, "name": "mousebite",  "supercategory": "defect"},
    {"id": 4, "name": "spur",       "supercategory": "defect"},
    {"id": 5, "name": "copper",     "supercategory": "defect"},
    {"id": 6, "name": "pin-hole",   "supercategory": "defect"},
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_split_file(split_txt: Path):
    """
    Read trainval.txt / test.txt.

    Each line:  <img_rel>  <ann_rel>
    where img_rel: group20085/20085/20085000.jpg
          ann_rel: group20085/20085_not/20085000.txt

    Returns list of (img_rel, ann_rel) tuples.
    Only keeps _test images (skips _temp template images).
    """
    records = []
    with open(split_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                # Single-field lines — try to derive ann path
                img_rel = Path(parts[0])
                # Skip template images
                if '_temp' in img_rel.stem:
                    continue
                ann_rel = img_rel.parent.parent / (img_rel.parent.name + '_not') / (img_rel.stem.replace('_test', '') + '.txt')
                records.append((str(img_rel), str(ann_rel)))
            else:
                img_rel = Path(parts[0])
                ann_rel = Path(parts[1])
                # Skip template images
                if '_temp' in img_rel.stem:
                    continue
                records.append((str(img_rel), str(ann_rel)))
    return records


def parse_annotation_file(ann_path: Path):
    """
    Parse a DeepPCB annotation .txt file.

    Supports both space-separated (x1 y1 x2 y2 type) and
    comma-separated (x1,y1,x2,y2,type) formats.
    Returns list of dicts with bbox and category info.
    """
    boxes = []
    if not ann_path.exists():
        return boxes
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try whitespace first, then comma
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            if len(parts) != 5:
                continue
            try:
                x1, y1, x2, y2, cat = [int(p) for p in parts]
            except ValueError:
                continue
            if cat == 0:
                continue  # background
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "category_id": cat})
    return boxes


def build_coco_json(records, deeppcb_root: Path):
    """
    Build a COCO-format dict from a list of (img_rel, ann_rel) tuples.

    The img_rel path may point to the base name without _test/_temp;
    we resolve to the _test.jpg variant.
    """
    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    for img_rel_str, ann_rel_str in records:
        img_rel = Path(img_rel_str)
        ann_rel = Path(ann_rel_str)

        # The split file lists e.g. "group20085/20085/20085000.jpg"
        # but the actual file on disk is "20085000_test.jpg".
        # Resolve to the _test.jpg variant.
        img_stem = img_rel.stem          # e.g. "20085000"
        img_test_rel = img_rel.parent / (img_stem + "_test.jpg")
        img_path = deeppcb_root / img_test_rel
        if not img_path.exists():
            # Fallback: try as-is (future-proofing)
            img_path = deeppcb_root / img_rel
        if not img_path.exists():
            print(f"  WARNING: image not found: {img_path}", file=sys.stderr)
            continue

        # Use the _test-resolved relative path in COCO file_name
        try:
            img_rel_final = img_path.relative_to(deeppcb_root)
        except ValueError:
            img_rel_final = img_test_rel

        ann_path = deeppcb_root / ann_rel

        width, height = 640, 640   # all DeepPCB images are 640×640

        img_entry = {
            "id": img_id,
            "file_name": str(img_rel_final),
            "width": width,
            "height": height,
        }
        images.append(img_entry)

        boxes = parse_annotation_file(ann_path)
        for b in boxes:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            w = x2 - x1
            h = y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": b["category_id"],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def link_images(records, deeppcb_root: Path, dest_img_dir: Path, use_copy: bool = False):
    """Symlink (or copy) _test images into the destination directory."""
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    for img_rel_str, _ in records:
        img_rel = Path(img_rel_str)
        # Resolve _test.jpg variant
        img_stem = img_rel.stem
        img_test_rel = img_rel.parent / (img_stem + "_test.jpg")
        src = deeppcb_root / img_test_rel
        if not src.exists():
            src = deeppcb_root / img_rel
        if not src.exists():
            continue
        try:
            rel_final = src.relative_to(deeppcb_root)
        except ValueError:
            rel_final = img_test_rel
        dst = dest_img_dir / rel_final
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if use_copy:
            shutil.copy2(src, dst)
        else:
            try:
                os.symlink(src.resolve(), dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepPCB dataset to COCO JSON format for DEIMv2."
    )
    parser.add_argument("--deeppcb_dir", type=str, default="DeepPCB/PCBData",
                        help="Path to the PCBData directory containing trainval.txt and group* folders.")
    parser.add_argument("--output_dir", type=str, default="data/deeppcb_coco",
                        help="Root output directory for COCO-format dataset.")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Fraction of trainval to use as validation (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split (default: 42).")
    parser.add_argument("--no_symlinks", action="store_true",
                        help="Copy images instead of symlinking.")
    args = parser.parse_args()

    # ── Locate DeepPCB root ──────────────────────────────────────────────────
    deeppcb_root = Path(args.deeppcb_dir).resolve()
    if not deeppcb_root.exists():
        script_dir = Path(__file__).parent.parent
        deeppcb_root = (script_dir / args.deeppcb_dir).resolve()
    if not deeppcb_root.exists():
        print(f"ERROR: DeepPCB directory not found: {args.deeppcb_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"DeepPCB root : {deeppcb_root}")

    trainval_txt = deeppcb_root / "trainval.txt"
    test_txt     = deeppcb_root / "test.txt"

    if not trainval_txt.exists() or not test_txt.exists():
        print(f"ERROR: trainval.txt or test.txt not found in {deeppcb_root}", file=sys.stderr)
        sys.exit(1)

    # ── Load and parse splits ────────────────────────────────────────────────
    all_trainval = parse_split_file(trainval_txt)
    all_test     = parse_split_file(test_txt)

    print(f"Trainval pairs: {len(all_trainval)}")
    print(f"Test pairs    : {len(all_test)}")

    # ── Train / val split ────────────────────────────────────────────────────
    random.seed(args.seed)
    shuffled = list(all_trainval)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    val_records   = shuffled[:n_val]
    train_records = shuffled[n_val:]
    test_records  = all_test

    print(f"  → Train : {len(train_records)}")
    print(f"  → Val   : {len(val_records)}")
    print(f"  → Test  : {len(test_records)}")

    # ── Build and write COCO JSONs ───────────────────────────────────────────
    out_root = Path(args.output_dir).resolve()
    ann_dir  = out_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_records, "val": val_records, "test": test_records}

    for split_name, records in splits.items():
        print(f"\nBuilding {split_name} COCO JSON...")
        coco = build_coco_json(records, deeppcb_root)

        ann_path = ann_dir / f"instances_{split_name}.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f)
        print(f"  Saved: {ann_path}")
        print(f"  Images: {len(coco['images'])}  Annotations: {len(coco['annotations'])}")

        img_dir = out_root / "images" / split_name
        print(f"  Linking images → {img_dir}")
        link_images(records, deeppcb_root, img_dir, use_copy=args.no_symlinks)

    print("\n✅ DeepPCB → COCO conversion complete!")
    print(f"   Output dir : {out_root}")
    print(f"   Annotations: {ann_dir}")
    print("\nNext step: run DEIMv2 training with:")
    print("   bash run_all_models.sh deimv2_l")
    print("   bash run_all_models.sh deimv2_x")


if __name__ == "__main__":
    main()
