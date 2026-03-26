"""DS-019: Merge all three dataset layers into unified seesaw_children dataset.

Auto-detects class mappings from Roboflow data.yaml exports and remaps to the
canonical 25-class SeeSaw taxonomy. Splits into train/val/test.

Usage:
  # Local (default paths):
  python scripts/data_merge.py

  # Google Colab (override layer paths):
  python scripts/data_merge.py \\
    --layer1 /content/datasets/homeobjects-3K \\
    --layer2 /content/seesaw-layer2-2 \\
    --layer3 /content/seesaw-layer3-1 \\
    --output /content/seesaw-yolo-model/datasets/seesaw_children
"""

import argparse
import shutil
import random
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# ── Canonical 25-class SeeSaw taxonomy (name → ID) ──────────────────────────
SEESAW_CLASSES = {
    "bed": 0, "sofa": 1, "chair": 2, "table": 3, "lamp": 4, "tv": 5,
    "laptop": 6, "wardrobe": 7, "window": 8, "door": 9, "potted_plant": 10,
    "photo_frame": 11, "teddy_bear": 12, "book": 13, "sports_ball": 14,
    "backpack": 15, "bottle": 16, "cup": 17, "building_blocks": 18,
    "dinosaur_toy": 19, "stuffed_animal": 20, "picture_book": 21, "crayon": 22,
    "toy_car": 23, "puzzle_piece": 24,
}

# ── Fallback remap tables (used when data.yaml is missing) ──────────────────
# Layer 1 (HomeObjects-3K): IDs 0–11 map directly
LAYER1_REMAP = {i: i for i in range(12)}

# Layer 2 (Roboflow): assumes alphabetical ordering by class name
LAYER2_REMAP = {
    0: 15,  # backpack
    1: 13,  # book
    2: 16,  # bottle
    3: 17,  # cup
    4: 14,  # sports_ball
    5: 12,  # teddy_bear
}

# Layer 3 (original annotations): assumes alphabetical ordering
LAYER3_REMAP = {
    0: 18,  # building_blocks
    1: 22,  # crayon
    2: 19,  # dinosaur_toy
    3: 21,  # picture_book
    4: 24,  # puzzle_piece
    5: 20,  # stuffed_animal
    6: 23,  # toy_car
}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def build_remap_from_yaml(data_yaml_path: Path) -> dict | None:
    """Auto-build source_id → seesaw_id remap from a Roboflow/YOLO data.yaml."""
    if yaml is None or not data_yaml_path.exists():
        return None
    with open(data_yaml_path) as f:
        meta = yaml.safe_load(f)
    names = meta.get("names", {})
    if not names:
        return None
    remap = {}
    for src_id, class_name in names.items():
        canonical = str(class_name).strip().lower().replace(" ", "_").replace("-", "_")
        if canonical in SEESAW_CLASSES:
            remap[int(src_id)] = SEESAW_CLASSES[canonical]
        else:
            print(f"    ⚠ Unmapped class: {src_id}={class_name} — skipped")
    return remap


def remap_labels(label_path: Path, class_remap: dict) -> list[str]:
    """Read a YOLO label file and remap class IDs. Drops unmapped classes."""
    lines = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        src_id = int(parts[0])
        if src_id in class_remap:
            parts[0] = str(class_remap[src_id])
            lines.append(" ".join(parts))
    return lines


def collect_pairs(layer_dir: Path) -> list[tuple[Path, Path]]:
    """Find all (image, label) pairs under a layer directory.

    Handles both directory layouts:
      Ultralytics:  root/images/train/img.jpg  →  root/labels/train/img.txt
      Roboflow:     root/train/images/img.jpg  →  root/train/labels/img.txt
    """
    pairs = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for img_path in sorted(layer_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in exts:
            continue
        rel = img_path.relative_to(layer_dir)
        if "images" not in rel.parts:
            continue
        # Mirror path: swap 'images' → 'labels', change extension to .txt
        label_parts = ["labels" if p == "images" else p for p in rel.parts]
        label_parts[-1] = img_path.stem + ".txt"
        label_path = layer_dir / Path(*label_parts)
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Merge SeeSaw dataset layers")
    parser.add_argument("--layer1", type=str, help="Path to Layer 1 (HomeObjects-3K)")
    parser.add_argument("--layer2", type=str, help="Path to Layer 2 (Roboflow export)")
    parser.add_argument("--layer3", type=str, help="Path to Layer 3 (Roboflow export)")
    parser.add_argument("--output", type=str, help="Output path for merged dataset")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent

    layer_cfg = {
        "layer1": (
            Path(args.layer1) if args.layer1 else root / "datasets" / "layer1",
            LAYER1_REMAP,
        ),
        "layer2": (
            Path(args.layer2) if args.layer2 else root / "datasets" / "layer2",
            LAYER2_REMAP,
        ),
        "layer3": (
            Path(args.layer3) if args.layer3 else root / "datasets" / "layer3",
            LAYER3_REMAP,
        ),
    }
    output_dir = Path(args.output) if args.output else root / "datasets" / "seesaw_children"

    random.seed(RANDOM_SEED)

    # Prepare output directories
    for split in SPLIT_RATIOS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_entries = []

    for layer_name, (layer_dir, fallback_remap) in layer_cfg.items():
        if not layer_dir.exists():
            print(f"⚠ {layer_name} not found: {layer_dir} — skipping")
            continue

        # Auto-detect remap from data.yaml, fall back to manual table
        remap = build_remap_from_yaml(layer_dir / "data.yaml")
        if remap is not None:
            print(f"  {layer_name}: auto-remap from data.yaml ({len(remap)} classes)")
        else:
            remap = fallback_remap
            print(f"  {layer_name}: using fallback remap ({len(remap)} classes)")

        pairs = collect_pairs(layer_dir)
        mapped_count = 0
        for img_path, lbl_path in pairs:
            remapped = remap_labels(lbl_path, remap)
            if remapped:
                all_entries.append((img_path, remapped, layer_name))
                mapped_count += 1
        print(f"✓ {layer_name}: {len(pairs)} pairs found, {mapped_count} with mapped labels")

    if not all_entries:
        print("✗ No entries found. Check layer paths and remap dictionaries.")
        return

    # Shuffle and split
    random.shuffle(all_entries)
    n = len(all_entries)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": all_entries[:n_train],
        "val": all_entries[n_train : n_train + n_val],
        "test": all_entries[n_train + n_val :],
    }

    for split_name, entries in splits.items():
        for img_path, label_lines, layer_name in entries:
            # Prefix filename with layer to avoid collisions across layers
            safe_name = f"{layer_name}_{img_path.name}"
            dst_img = output_dir / "images" / split_name / safe_name
            dst_lbl = output_dir / "labels" / split_name / f"{layer_name}_{img_path.stem}.txt"
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(label_lines) + "\n")
        print(f"  {split_name}: {len(entries)} images")

    print(f"\n✓ Merged dataset written to {output_dir}")
    print(f"  Total: {n} images across {len(SPLIT_RATIOS)} splits")


if __name__ == "__main__":
    main()
