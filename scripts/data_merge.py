"""DS-019: Merge all three dataset layers into unified seesaw_children dataset.

Reads images and labels from layer1/, layer2/, layer3/, remaps class IDs
to the canonical 25-class SeeSaw taxonomy, and splits into train/val/test.
"""

import os
import shutil
import random
from pathlib import Path

# ── Class ID remapping tables ────────────────────────────────────────────────
# Layer 1 (HomeObjects-3K): IDs 0–11 → map directly (no change needed)
LAYER1_REMAP = {i: i for i in range(12)}

# Layer 2 (Roboflow / COCO subset): source ID → SeeSaw canonical ID
# Adjust these mappings after inspecting the actual exported label files (DS-013)
LAYER2_REMAP = {
    # Example: Roboflow source_id → SeeSaw target_id
    # 0: 12,  # teddy_bear
    # 1: 13,  # book
    # 2: 14,  # sports_ball
    # 3: 15,  # backpack
    # 4: 16,  # bottle
    # 5: 17,  # cup
}

# Layer 3 (original annotations): source ID → SeeSaw canonical ID
# Adjust after exporting from Roboflow (DS-017)
LAYER3_REMAP = {
    # 0: 18,  # building_blocks
    # 1: 19,  # dinosaur_toy
    # 2: 20,  # stuffed_animal
    # 3: 21,  # picture_book
    # 4: 22,  # crayon
    # 5: 23,  # toy_car
    # 6: 24,  # puzzle_piece
}

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
LAYER_DIRS = {
    "layer1": (ROOT / "datasets" / "layer1", LAYER1_REMAP),
    "layer2": (ROOT / "datasets" / "layer2", LAYER2_REMAP),
    "layer3": (ROOT / "datasets" / "layer3", LAYER3_REMAP),
}
OUTPUT_DIR = ROOT / "datasets" / "seesaw_children"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def remap_labels(label_path: Path, class_remap: dict) -> list[str]:
    """Read a YOLO label file and remap class IDs."""
    lines = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        src_id = int(parts[0])
        if src_id in class_remap:
            parts[0] = str(class_remap[src_id])
            lines.append(" ".join(parts))
    return lines


def collect_pairs(layer_dir: Path) -> list[tuple[Path, Path]]:
    """Find all (image, label) pairs under a layer directory."""
    pairs = []
    img_dirs = list(layer_dir.rglob("images"))
    for img_dir in img_dirs:
        label_dir = img_dir.parent / "labels" / img_dir.name  # parallel structure
        if not label_dir.exists():
            label_dir = img_dir.parent.parent / "labels"
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                label_path = label_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    pairs.append((img_path, label_path))
    return pairs


def main():
    random.seed(RANDOM_SEED)

    # Prepare output directories
    for split in SPLIT_RATIOS:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_entries = []  # (image_path, remapped_label_lines, source_layer)

    for layer_name, (layer_dir, remap) in LAYER_DIRS.items():
        if not layer_dir.exists():
            print(f"⚠ {layer_name} directory not found: {layer_dir} — skipping")
            continue
        pairs = collect_pairs(layer_dir)
        for img_path, lbl_path in pairs:
            remapped = remap_labels(lbl_path, remap)
            if remapped:
                all_entries.append((img_path, remapped, layer_name))
        print(f"✓ {layer_name}: {len(pairs)} image-label pairs found")

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
        for img_path, label_lines, _ in entries:
            dst_img = OUTPUT_DIR / "images" / split_name / img_path.name
            dst_lbl = OUTPUT_DIR / "labels" / split_name / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(label_lines) + "\n")
        print(f"  {split_name}: {len(entries)} images")

    print(f"\n✓ Merged dataset written to {OUTPUT_DIR}")
    print(f"  Total: {n} images across {len(SPLIT_RATIOS)} splits")


if __name__ == "__main__":
    main()
