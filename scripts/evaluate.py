"""DS-022: Generate the three-run comparison table for the dissertation.

Evaluates Run A (COCO baseline), Run B (Layer 1), and Run C (all layers)
on the same seesaw_children test split.
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = "configs/seesaw_children.yaml"
OUTPUT_MD = ROOT / "docs" / "results_comparison.md"
OUTPUT_CSV = ROOT / "docs" / "results_comparison.csv"


def evaluate_model(name: str, weights: str) -> dict:
    """Run validation on the test split and return metrics."""
    model = YOLO(weights)
    metrics = model.val(data=DATA_YAML, split="test")
    return {
        "name": name,
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }


def main():
    runs = [
        ("Run A — COCO baseline", "yolo11n.pt"),
        ("Run B — HomeObjects-3K", "runs/detect/run_b_layer1/weights/best.pt"),
        ("Run C — All layers (SeeSaw)", "runs/detect/run_c_all_layers/weights/best.pt"),
    ]

    results = []
    for name, weights in runs:
        weights_path = ROOT / weights if not Path(weights).is_absolute() else Path(weights)
        if not weights_path.exists():
            print(f"⚠ Weights not found: {weights_path} — skipping {name}")
            continue
        print(f"\nEvaluating: {name}")
        results.append(evaluate_model(name, str(weights_path)))

    if not results:
        print("No models to evaluate. Run training first.")
        return

    # Markdown table
    header = "| Model | mAP@50 | mAP@50-95 | Precision | Recall |"
    sep = "|-------|--------|-----------|-----------|--------|"
    rows = [
        f"| {r['name']} | {r['map50']:.3f} | {r['map50_95']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |"
        for r in results
    ]
    table = "\n".join([header, sep] + rows)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(f"# SeeSaw YOLO11n — Training Results Comparison\n\n{table}\n")
    print(f"\n✓ Results table saved to {OUTPUT_MD}")

    # CSV export
    lines = ["model,map50,map50_95,precision,recall"]
    for r in results:
        lines.append(f"{r['name']},{r['map50']:.4f},{r['map50_95']:.4f},{r['precision']:.4f},{r['recall']:.4f}")
    OUTPUT_CSV.write_text("\n".join(lines) + "\n")
    print(f"✓ CSV saved to {OUTPUT_CSV}")

    print(f"\n{table}")


if __name__ == "__main__":
    main()
