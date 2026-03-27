# SeeSaw YOLO11n — Custom Object Detection Model

Training a domain-specific YOLO11n model for the SeeSaw wearable AI companion system. Detects 44 object classes across children's indoor environments using a three-layer dataset merge strategy.

**Repository:** [seesaw-yolo-model](https://github.com/j2damax/seesaw-yolo-model)  
**Framework:** Ultralytics YOLOv8 (YOLO11n)  
**Deployment Target:** iOS Neural Engine (Core ML)

---

## Quick Start

### Reproduce Training in Google Colab
1. Open [notebooks/yolo_training.ipynb](notebooks/yolo_training.ipynb) in Colab
2. Set Colab secrets:
   - `ROBOFLOW`: Roboflow API key
   - `COLAB`: GitHub personal access token
3. Run all cells (26 cells, ~2 hours on T4 GPU)
4. Final model: `export/seesaw-yolo11n.mlpackage`

### Local Development
```bash
git clone https://github.com/j2damax/seesaw-yolo-model.git
cd seesaw-yolo-model
pip install ultralytics roboflow pyyaml

# Sync artifacts if needed
python scripts/sync_artifacts.py --overwrite

# Or train from scratch
python scripts/train.py
```

---

## Dataset & Architecture

### Three-Layer Strategy

| Layer | Source | Classes | Purpose |
|-------|--------|---------|---------|
| **1** | [HomeObjects-3K](https://docs.ultralytics.com/datasets/detect/homeobjects-3k/) (Ultralytics) | 12 | Indoor furniture baseline |
| **2** | [Roboflow Universe](https://universe.roboflow.com) (CC BY 4.0) | 33 | Child-environment augmentation |
| **3** | Original annotations (CC BY 4.0) | 5 | Egocentric wearable perspective |
| **→ Merged** | `seesaw_children` | **44** | Unified training dataset |

**See:** [`IMPLEMENTATION_FINAL_SPECIFICATION.md`](IMPLEMENTATION_FINAL_SPECIFICATION.md) for full dataset details, class taxonomy, and provenance.

### Class Taxonomy (44 Classes)

**Layer 1 Core (0–11):** bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted_plant, photo_frame

**Layer 2 Core (12–17):** teddy_bear, book, sports_ball, backpack, bottle, cup

**Layer 3 Toys (18–24):** building_blocks, dinosaur_toy, stuffed_animal, picture_book, crayon, toy_car, puzzle_piece

**Extended (25–40, 41–43):** carpet, chimney, clock, crib, cupboard, curtains, faucet, floor_decor, glass, pillows, pots, rugs, shelf, stairs, storage, whiteboard, toy_airplane, toy_fire_truck, toy_jeep

---

## Training Results

### Three-Run Comparison (Test Set)

| Model | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|--------|-----------|-----------|--------|
| **Run A** — COCO Baseline | 0.0147 | 0.0137 | 0.0118 | 0.0286 |
| **Run B** — Layer 1 Only | 0.8223 | 0.6087 | 0.8022 | 0.7378 |
| **Run C** — All Layers ⭐ | 0.4972 | 0.3789 | 0.7083 | 0.4173 |

**Key Insight:** Run C achieves broad semantic coverage (44 classes) with acceptable per-class precision despite class imbalance. Production model prioritizes breadth and egocentric context over narrow high-mAP specialization.

**Metrics CSV:** [docs/results_comparison.csv](docs/results_comparison.csv)

### Visualizations

- **Class Distribution:** [docs/dissertation_figures/class_distribution.png](docs/dissertation_figures/class_distribution.png)
- **Confusion Matrices:** Run B / Run C (PNG)
- **Training Curves:** Run B / Run C (PNG)
- **Sample Predictions:** Run C validation batch (JPG)

---

## Scripts & Tools

| Script | Purpose |
|--------|---------|
| [`scripts/train.py`](scripts/train.py) | Run B + C training (50 epochs each) |
| [`scripts/data_merge.py`](scripts/data_merge.py) | Merge 3 dataset layers with class remapping |
| [`scripts/evaluate.py`](scripts/evaluate.py) | Generate 3-run comparison table |
| [`scripts/export_coreml.py`](scripts/export_coreml.py) | Export to CoreML with NMS baked in |
| [`scripts/class_distribution.py`](scripts/class_distribution.py) | Plot class balance chart |
| [`scripts/sync_artifacts.py`](scripts/sync_artifacts.py) | Sync artifacts from Colab/Drive to repo |

---

## Configuration Files

| File | Purpose |
|------|---------|
| [`configs/HomeObjects-3K.yaml`](configs/HomeObjects-3K.yaml) | Layer 1 training config (12 classes) |
| [`configs/seesaw_children.yaml`](configs/seesaw_children.yaml) | Unified config (44 classes) |

---

## Dataset Documentation

- [DATASET_CARD_HomeObjects.md](docs/DATASET_CARD_HomeObjects.md) — Layer 1 overview
- [DATASET_CARD_Roboflow_Universe.md](docs/DATASET_CARD_Roboflow_Universe.md) — Layer 2 overview
- [DATASET_CARD_ChildrensRoom.md](docs/DATASET_CARD_ChildrensRoom.md) — Layer 3 overview & original contribution

---

## iOS Integration

### CoreML Export
```python
from ultralytics import YOLO

model = YOLO("runs/detect/run_c_all_layers/weights/best.pt")
model.export(format="coreml", nms=True, imgsz=640)
```

**Output:** `export/seesaw-yolo11n.mlpackage`

### Xcode Integration
1. Copy `.mlpackage` to iOS project:
   ```
   seesaw-companion-ios/SeeSawCompanion/Services/AI/seesaw-yolo11n.mlpackage
   ```
2. Update model reference in `PrivacyPipelineService.swift`:
   ```swift
   let model = try VNCoreMLModel(for: seesaw_yolo11n(configuration: .init()).model)
   ```

**Note:** NMS is baked into the CoreML graph (`nms=True`), eliminating post-processing complexity on device.

---

## Repository Structure

```
seesaw-yolo-model/
├── README.md                                    # This file
├── IMPLEMENTATION_FINAL_SPECIFICATION.md        # Comprehensive module closure report
├── SeeSaw YOLO11n Custom Dataset Training Plan.md  # Original planning document
│
├── notebooks/
│   └── yolo_training.ipynb                      # Complete Colab training workflow
│
├── scripts/
│   ├── train.py                                 # Training (Run B + Run C)
│   ├── data_merge.py                            # Layer merge + class remap
│   ├── evaluate.py                              # 3-run comparison
│   ├── export_coreml.py                         # CoreML export
│   ├── class_distribution.py                    # Class balance chart
│   └── sync_artifacts.py                        # Artifact syncing utility
│
├── configs/
│   ├── HomeObjects-3K.yaml                      # Layer 1 config
│   └── seesaw_children.yaml                     # Unified config (44 classes)
│
├── docs/
│   ├── DATASET_CARD_HomeObjects.md              # Layer 1 card
│   ├── DATASET_CARD_Roboflow_Universe.md        # Layer 2 card
│   ├── DATASET_CARD_ChildrensRoom.md            # Layer 3 card
│   ├── results_comparison.csv                   # 3-run metrics table
│   └── dissertation_figures/                    # 9 publication-quality figures
│       ├── class_distribution.png
│       ├── confusion_matrix_run_b.png
│       ├── confusion_matrix_run_c.png
│       ├── results_run_b_training_curves.png
│       ├── results_run_c_training_curves.png
│       ├── val_predictions_run_b.jpg
│       ├── val_predictions_run_c.jpg
│       ├── labels_distribution_run_b.jpg
│       └── labels_distribution_run_c.jpg
│
├── configs/
│   ├── datasets/layer1/                         # (Auto-downloaded)
│   ├── datasets/layer2/                         # (from Roboflow)
│   ├── datasets/layer3/                         # (from Roboflow)
│   └── datasets/seesaw_children/                # (Merged dataset)
│       ├── images/ {train,val,test}/
│       └── labels/ {train,val,test}/
│
├── export/
│   └── seesaw-yolo11n.mlpackage                 # CoreML export (iOS ready)
│
└── .gitignore                                   # Excludes large data/runs/models

```

---

## Logging & Troubleshooting

### Training Logs
Training outputs saved to `runs/detect/run_c_all_layers/` after each run:
- Training curves: `results.png`
- Confusion matrix: `confusion_matrix_normalized.png`
- Validation samples: `val_batch0_pred.jpg`

### Common Issues
- **Missing datasets:** Run `python scripts/sync_artifacts.py --overwrite`
- **Roboflow auth failing:** Check `ROBOFLOW` secret in Colab
- **Out of memory:** Reduce `batch=8` or `imgsz=512` in training config

---

## Dissertation & Publication
This work is documented for academic submission:
---

## License & Attribution

- **Layer 1 (HomeObjects-3K):** AGPL-3.0 (Ultralytics)
- **Layer 2 (Roboflow Universe):** CC BY 4.0 (public datasets)
- **Layer 3 (Original Annotations):** CC BY 4.0 (Jayampathy Balasuriya, 2026)
- **This Repository:** MIT License

See [docs/](docs/) for full dataset license details.

---

## Contact & Support

**Project:** SeeSaw Wearable AI Companion  
**Author:** Jayampathy Balasuriya  
**Codebase:** https://github.com/j2damax/seesaw-yolo-model

For questions on training, evaluation, or iOS integration, refer to [`IMPLEMENTATION_FINAL_SPECIFICATION.md`](IMPLEMENTATION_FINAL_SPECIFICATION.md).

---

**Last Updated:** 27 March 2026 | **Status:** Ready for Submission
