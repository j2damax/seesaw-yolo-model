"""DS-025: Export the best Run C weights to CoreML .mlpackage for iOS deployment.

Bakes NMS into the model graph (nms=True) so no post-processing is needed on device.
Output: export/seesaw-yolo11n.mlpackage
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "detect" / "run_c_all_layers" / "weights" / "best.pt"
EXPORT_DIR = ROOT / "export"
EXPORT_NAME = "seesaw-yolo11n.mlpackage"


def main():
    if not WEIGHTS.exists():
        print(f"✗ Weights not found: {WEIGHTS}")
        print("  Run training (scripts/train.py) first.")
        return

    model = YOLO(str(WEIGHTS))

    # Export to CoreML with NMS baked in
    model.export(
        format="coreml",
        nms=True,       # Critical: bakes Non-Maximum Suppression into the model
        imgsz=640,
        int8=False,     # FP16 for better quality on iPhone Neural Engine
    )

    # Move to export/ directory
    src = WEIGHTS.with_suffix(".mlpackage")
    dst = EXPORT_DIR / EXPORT_NAME
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))

    print(f"✓ CoreML model exported to {dst}")
    print(f"  Copy to: seesaw-companion-ios/SeeSawCompanion/Services/AI/{EXPORT_NAME}")


if __name__ == "__main__":
    main()
