# SeeSaw YOLO11n — Custom Object Detection Model

Custom YOLO11n fine-tuned for children's indoor environments. Detects **44 object classes** from a wearable, egocentric camera. Exported to CoreML for on-device inference on iOS Neural Engine.

**Part of:** SeeSaw Wearable AI Companion — MSc Dissertation, Jayampathy Balasuriya, 2026

---

## Results

Three training runs establish a comparative evaluation on the HomeObjects-3K validation set:

| Run | Training Data | mAP@50 | mAP@50-95 | Precision | Recall |
|-----|--------------|--------|-----------|-----------|--------|
| **A** — COCO Baseline | None (stock YOLO11n) | 0.0105 | 0.0077 | 0.0208 | 0.0084 |
| **B** — Layer 1 Only | HomeObjects-3K (12 classes) | 0.7010 | 0.4913 | 0.7502 | 0.6088 |
| **C** — All Layers | Merged 44-class dataset | **0.8490** | **0.6479** | **0.8241** | **0.7780** |

Run C is the production model. Full 44-class mAP@50 on val: **0.6748**.

> Run C outperforms Run B even on the 12-class benchmark — diverse training data improves generalisation on shared classes.

---

## Dataset — Three-Layer Strategy

| Layer | Source | Images | Classes | Purpose |
|-------|--------|--------|---------|---------|
| 1 | HomeObjects-3K (Ultralytics) | 2,689 | 12 | Indoor furniture baseline |
| 2 | Roboflow Universe (CC BY 4.0) | ~354 | 33 | Child-environment objects |
| 3 | Original egocentric annotations (CC BY 4.0) | 240 | 5 | Wearable perspective toys |
| **Merged** | `datasets/seesaw_children/` | **3,283** | **44** | Unified training set |

Split: 70% train / 15% val / 15% test (seed = 42)

### 44-Class Taxonomy

| IDs | Classes |
|-----|---------|
| 0–11 | bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted_plant, photo_frame |
| 12–17 | teddy_bear, book, sports_ball, backpack, bottle, cup |
| 18–24 | building_blocks, dinosaur_toy, stuffed_animal, picture_book, crayon, toy_car, puzzle_piece |
| 25–40 | carpet, chimney, clock, crib, cupboard, curtains, faucet, floor_decor, glass, pillows, pots, rugs, shelf, stairs, storage, whiteboard |
| 41–43 | toy_airplane, toy_fire_truck, toy_jeep |

---

## Reproduce Training (Google Colab)

**Requirements:** T4 GPU runtime, Roboflow API key, GitHub PAT

1. Open [`notebooks/yolo_training.ipynb`](notebooks/yolo_training.ipynb) in Colab
2. Set Colab secrets: `ROBOFLOW` (API key), `GIT` (GitHub PAT)
3. Run all 27 cells — ~2 hours total

**What the notebook does:**

| Cells | Step |
|-------|------|
| 1–6 | Install deps, verify GPU, clone repo, mount Drive |
| 7–13 | Download Layer 1 (Ultralytics auto-download), Layer 2 & 3 (Roboflow) |
| 14–15 | Merge datasets via `data_merge.py`, verify class distribution |
| 16–18 | Train Run B (50 epochs), evaluate, compare with Run A baseline |
| 19 | Train Run C (50 epochs, 44 classes) — production model |
| 20–22 | Export to CoreML, patch `nc=80 → 44`, validate spec |
| 23–25 | Three-run comparison table, download `.mlpackage`, push to GitHub |
| 26 | Run inference on sample image |

**Training config (Run B and Run C):**
```
model:   yolo11n.pt   epochs: 50   imgsz: 640   batch: 16   patience: 20
```

---

## Dataset Merge

```bash
# Local (default paths under datasets/):
python scripts/data_merge.py

# Colab (explicit paths):
python scripts/data_merge.py \
    --layer1 /content/datasets/homeobjects-3K \
    --layer2 /content/seesaw-layer2-2 \
    --layer3 /content/seesaw-layer3-1 \
    --output /content/seesaw-yolo-model/datasets/seesaw_children
```

The script reads each layer's `data.yaml`, remaps class IDs to the canonical SeeSaw taxonomy, normalises synonyms (e.g. `television → tv`, `dinosaur → dinosaur_toy`), and writes the 70/15/15 split.

---

## CoreML Export

```python
from ultralytics import YOLO

model = YOLO("runs/detect/run_c_all_layers/weights/best.pt")
model.export(format="coreml", nms=True, imgsz=640)
```

NMS is baked into the CoreML pipeline graph — no post-processing needed on device.

**Note:** Ultralytics injects `nc=80` (COCO) into the NMS descriptor regardless of training classes. Cell 21 patches this to `nc=44` using `coremltools` by editing the mlProgram protobuf spec directly.

Output: `export/seesaw-yolo11n.mlpackage`

---

## iOS Integration

```swift
// PrivacyPipelineService.swift
let model = try VNCoreMLModel(for: seesaw_yolo11n(configuration: .init()).model)
```

Copy `export/seesaw-yolo11n.mlpackage` into:
```
seesaw-companion-ios/SeeSawCompanion/Services/AI/
```

Xcode auto-generates the `seesaw_yolo11n` Swift wrapper on build.

---

## Repository Structure

```
seesaw-yolo-model/
├── notebooks/yolo_training.ipynb        # Full training pipeline (27 cells)
├── scripts/data_merge.py                # Dataset merge + class remapping
├── configs/
│   ├── HomeObjects-3K.yaml              # Layer 1 config (12 classes)
│   └── seesaw_children.yaml             # Unified config (44 classes)
├── docs/
│   ├── results_comparison.csv           # Three-run metrics
│   └── dissertation_figures/            # Confusion matrices, training curves, predictions
├── export/
│   └── seesaw-yolo11n.mlpackage         # CoreML model (iOS-ready)
└── DEVELOPER_REFERENCE.md              # Full architecture, implementation, thesis guide
```

`datasets/`, `runs/`, and `*.pt` files are git-ignored. The final `.mlpackage` is tracked.

---

## License

| Component | License |
|-----------|---------|
| Layer 1 — HomeObjects-3K | AGPL-3.0 (Ultralytics) |
| Layer 2 — Roboflow Universe | CC BY 4.0 |
| Layer 3 — Original annotations | CC BY 4.0 (Jayampathy Balasuriya, 2026) |
| This repository | MIT |
