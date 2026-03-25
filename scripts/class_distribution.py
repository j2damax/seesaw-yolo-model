"""DS-016: Plot class distribution across the merged seesaw_children dataset.

Produces a bar chart showing image count per class. Saves to docs/class_distribution.png.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "seesaw_children"
OUTPUT_PATH = ROOT / "docs" / "class_distribution.png"

CLASS_NAMES = {
    0: "bed", 1: "sofa", 2: "chair", 3: "table", 4: "lamp", 5: "tv",
    6: "laptop", 7: "wardrobe", 8: "window", 9: "door", 10: "potted_plant",
    11: "photo_frame", 12: "teddy_bear", 13: "book", 14: "sports_ball",
    15: "backpack", 16: "bottle", 17: "cup", 18: "building_blocks",
    19: "dinosaur_toy", 20: "stuffed_animal", 21: "picture_book", 22: "crayon",
    23: "toy_car", 24: "puzzle_piece",
}


def main():
    counter = Counter()

    for split in ("train", "val", "test"):
        label_dir = DATASET_DIR / "labels" / split
        if not label_dir.exists():
            continue
        for label_file in label_dir.glob("*.txt"):
            for line in label_file.read_text().strip().splitlines():
                class_id = int(line.split()[0])
                counter[class_id] += 1

    if not counter:
        print("No annotations found. Run data_merge.py first.")
        return

    # Sort by class ID
    ids = sorted(counter.keys())
    names = [CLASS_NAMES.get(i, f"class_{i}") for i in ids]
    counts = [counter[i] for i in ids]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(names, counts, color="#4A90D9", edgecolor="white")
    ax.set_xlabel("Class")
    ax.set_ylabel("Annotation Count")
    ax.set_title("SeeSaw Children Dataset — Class Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Highlight classes below threshold
    THRESHOLD = 50
    for bar, count in zip(bars, counts):
        if count < THRESHOLD:
            bar.set_color("#E74C3C")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"✓ Saved class distribution chart to {OUTPUT_PATH}")

    # Print summary
    print(f"\n{'Class':<20} {'Count':>6}")
    print("-" * 28)
    for name, count in zip(names, counts):
        flag = " ⚠ LOW" if count < THRESHOLD else ""
        print(f"{name:<20} {count:>6}{flag}")
    print(f"\nTotal annotations: {sum(counts)}")


if __name__ == "__main__":
    main()
