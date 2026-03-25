"""DS-019: Training script — Run B (Layer 1 only) and Run C (all layers combined).

Run on Google Colab with T4 GPU. Both runs start from COCO pretrained yolo11n.pt.
"""

from ultralytics import YOLO


def run_b_layer1():
    """Run B — Fine-tune on HomeObjects-3K only (Layer 1 baseline)."""
    model = YOLO("yolo11n.pt")
    model.train(
        data="configs/HomeObjects-3K.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=20,
        name="run_b_layer1",
        device=0,
    )
    return model


def run_c_all_layers():
    """Run C — Fine-tune on full seesaw_children dataset (all layers)."""
    model = YOLO("yolo11n.pt")  # start fresh from COCO weights
    model.train(
        data="configs/seesaw_children.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=20,
        name="run_c_all_layers",
        device=0,
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Run B — Layer 1 (HomeObjects-3K)")
    print("=" * 60)
    run_b_layer1()

    print("\n" + "=" * 60)
    print("Run C — All Layers Combined (seesaw_children)")
    print("=" * 60)
    run_c_all_layers()
