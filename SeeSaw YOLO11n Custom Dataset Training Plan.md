# SeeSaw — YOLO11n Custom Dataset Training Plan
### Three-Layer Dataset Construction, Fine-Tuning & CoreML Deployment
> **Repository:** `seesaw-yolo-model` (separate repo, not part of the iOS app)
> **Goal:** Replace the stock COCO-pretrained `yolo11n.pt` with a domain-specific model trained on children's indoor environments, then export to `seesaw-yolo11n.mlpackage` for the iPhone Neural Engine.
> **Research Contribution:** Original dataset (Layer 3) + training pipeline + before/after mAP comparison = a standalone data science result for Chapter 4 and Chapter 6 of the dissertation.

***
## 0. Architecture Overview
```
┌─────────────────────────────────────────────────────────┐
│              Three-Layer Dataset Strategy               │
│                                                         │
│  Layer 1 — HomeObjects-3K (Ultralytics Official)        │
│    12 classes, 2,285 train / 404 val images             │
│    Auto-download via YAML — zero manual work            │
│                                ↓ merge                  │
│  Layer 2 — Roboflow Universe Datasets (CC BY 4.0)       │
│    ~500–800 images from 3 curated public datasets       │
│    Children's toys, indoor playroom, COCO subset        │
│                                ↓ merge                  │
│  Layer 3 — Your Own Annotations (Research Contribution) │
│    50–100 images captured in real child environments    │
│    Annotated in Roboflow, exported as YOLO format       │
│                                ↓                        │
│  Combined: seesaw_children.yaml                         │
│    ~3,000–3,400 images, 25 classes                      │
│                                ↓                        │
│  Training (Google Colab T4 GPU, free tier)              │
│    Base: yolo11n.pt (COCO pretrained)                   │
│    Fine-tune on combined dataset, 50 epochs             │
│                                ↓                        │
│  Evaluation                                             │
│    mAP@50, mAP@50-95, Confusion Matrix                  │
│    vs. COCO baseline on same test set                   │
│                                ↓                        │
│  Export                                                 │
│    seesaw-yolo11n.mlpackage (CoreML, nms=True)          │
│    Drop into seesaw-companion-ios Xcode project         │
└─────────────────────────────────────────────────────────┘
```

***
## 1. Layer 1 — HomeObjects-3K (Foundation)
HomeObjects-3K is an official Ultralytics dataset designed for indoor household object detection. It contains 2,285 training images and 404 validation images across 12 classes, all at high resolution and already in YOLO annotation format.[^1][^2]
### Classes Provided (12)
| ID | Class | Relevance to SeeSaw |
|----|-------|-------------------|
| 0 | bed | ★★★ Child's bedroom core object |
| 1 | sofa | ★★★ Living room / play area |
| 2 | chair | ★★ Furniture context |
| 3 | table | ★★ Play surface |
| 4 | lamp | ★ Lighting context |
| 5 | tv | ★ (detection helps Safety Moderator) |
| 6 | laptop | ★ Screen context |
| 7 | wardrobe | ★★ Bedroom context |
| 8 | window | ★★ Scene setting |
| 9 | door | ★★ Scene setting |
| 10 | potted plant | ★ |
| 11 | photo frame | ★ |
### Download & Train Command
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
# Downloads 390 MB to ./datasets/homeobjects-3K automatically
model.train(data="HomeObjects-3K.yaml", epochs=50, imgsz=640, name="layer1-baseline")
```

This Layer 1 run serves as your **Baseline B** for the dissertation comparison table — trained on an established named dataset with no custom data.[^3][^4]

***
## 2. Layer 2 — Roboflow Universe Augmentation
Three curated public datasets from Roboflow Universe, all CC BY 4.0 licensed. Import each into a single Roboflow project using the "Clone from Universe" feature, then export once as YOLO format.[^5][^6]
### Recommended Roboflow Universe Datasets
| Dataset | URL | Key Classes Added | Est. Images |
|---------|-----|------------------|-------------|
| Roboflow `children` dataset | universe.roboflow.com/project-odwld/children-u9om6 | `child` (person context) | ~625 [^7] |
| `inside` (57 classes) | universe.roboflow.com/yolo-a91kx/inside-mpg5a | `crib`, `toy`, `pillow`, `shelf`, `curtain`, `mirror` | ~400 [^8] |
| COCO toy subset | Filter COCO for: `teddy bear`, `book`, `sports ball`, `scissors`, `backpack`, `bottle`, `cup` | 7 child-relevant COCO classes | ~300 |
### Layer 2 Process in Roboflow
1. Create a new Roboflow project: `seesaw-layer2`
2. Clone each dataset above via **Universe → Clone into Project**
3. Remap class names to match the unified SeeSaw class list (see Section 4)
4. Generate dataset version → Export as **YOLOv8 format**
5. Download zip → extract to `datasets/layer2/`

***
## 3. Layer 3 — Original Annotation (Research Contribution)
This is the unique research contribution. Capture 50–100 images in a real child's bedroom or playroom using an iPhone. These images should reflect the **actual field of view** of the AiSee headset — egocentric perspective, approximately 1–1.5m height, natural indoor lighting.
### What to Photograph (Target Objects)
Focus on objects that no existing dataset covers well from a child's egocentric perspective:

- `building_blocks` — Lego, Duplo, wooden blocks (piles and towers)
- `dinosaur_toy` — plastic/rubber dinosaur figures
- `stuffed_animal` — soft toys beyond the COCO `teddy bear` class
- `picture_book` — thick-spined children's books with cover art
- `crayon` — single crayons and crayon boxes
- `toy_car` — die-cast and plastic cars
- `puzzle_piece` — jigsaw puzzle pieces on floor or table
### Annotation Workflow (Roboflow, Free Tier)
```
1. Create Roboflow project: seesaw-layer3
   Project type: Object Detection

2. Upload images (drag-and-drop in browser)

3. Annotate using Roboflow Annotate:
   - Draw bounding boxes around each target object
   - Use Label Assist (auto-label) for objects already in their library
   - Manually annotate custom classes (building_blocks, crayon, etc.)
   Target: ~3–5 annotations per image, 50–100 images total

4. Generate dataset version:
   - Preprocessing: Auto-Orient, Resize to 640×640
   - Augmentation: Flip (horizontal), Brightness ±15%, Blur (0.5px)
   - This triples your effective dataset size

5. Export: YOLOv8 format → download zip
```

The Roboflow free tier supports up to 10,000 source images and unlimited exports. Annotation of 100 images takes approximately 2–3 hours.[^9]

**Dissertation note:** This layer constitutes an original dataset contribution. Document the class distribution, inter-annotator agreement (even if solo), and how image conditions were chosen to reflect AiSee's egocentric perspective. This belongs in Chapter 4, Section 4.x: "Dataset Construction."

***
## 4. Unified Class List (25 Classes)
All three layers are merged under this single class taxonomy. The `data_merge.py` script (Task DS-015) remaps class IDs from each source to these canonical IDs.

```yaml
# seesaw_children.yaml
path: datasets/seesaw_children
train: images/train
val:   images/val
test:  images/test

nc: 25
names:
  0:  bed
  1:  sofa
  2:  chair
  3:  table
  4:  lamp
  5:  tv
  6:  laptop
  7:  wardrobe
  8:  window
  9:  door
  10: potted_plant
  11: photo_frame
  12: teddy_bear
  13: book
  14: sports_ball
  15: backpack
  16: bottle
  17: cup
  18: building_blocks
  19: dinosaur_toy
  20: stuffed_animal
  21: picture_book
  22: crayon
  23: toy_car
  24: puzzle_piece
```

Classes 0–11 come from Layer 1 (HomeObjects-3K).[^1][^2]
Classes 12–17 come from Layer 2 (COCO/Roboflow).[^8][^7]
Classes 18–24 come from Layer 3 (original annotations — research contribution).

***
## 5. Training Strategy
### Three Training Runs for Dissertation Comparison
Running three sequential experiments produces a complete quantitative results table for Chapter 6.[^10]

| Run | Config | Purpose |
|-----|--------|---------|
| **Run A** — COCO Baseline | Stock `yolo11n.pt`, no fine-tuning | Baseline: how well does COCO generalise to children's scenes? |
| **Run B** — Layer 1 Only | Fine-tune on HomeObjects-3K (50 epochs) | Shows value of domain-specific indoor data |
| **Run C** — All Layers Combined | Fine-tune on full seesaw_children (50 epochs) | Final model — shows contribution of Layer 2+3 |
### Training Script (`train.py`)
```python
from ultralytics import YOLO

# --- Run B: Layer 1 baseline ---
model_b = YOLO("yolo11n.pt")
model_b.train(
    data="HomeObjects-3K.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=20,
    name="run_b_layer1",
    device=0  # T4 GPU on Colab
)

# --- Run C: Full combined dataset ---
model_c = YOLO("yolo11n.pt")   # start fresh from COCO weights each time
model_c.train(
    data="seesaw_children.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=20,
    name="run_c_all_layers",
    device=0
)
```

Fine-tuning from COCO pretrained weights (transfer learning) requires significantly fewer epochs than training from scratch — 50 epochs is sufficient for a dataset of this size.[^11][^12]
### Hyperparameter Rationale
| Parameter | Value | Reason |
|-----------|-------|--------|
| `epochs` | 50 | Sufficient for fine-tuning; monitor val/loss for early stop |
| `imgsz` | 640 | Standard YOLO inference size; matches AiSee camera output |
| `batch` | 16 | Fits Colab T4 16 GB VRAM |
| `patience` | 20 | Early stopping if no val improvement for 20 epochs |
| `optimizer` | AdamW (default) | Best for fine-tuning small models |
| `lr0` | 0.01 (default) | Standard YOLO learning rate |

***
## 6. Evaluation Plan
### Metrics to Report in Dissertation (Chapter 6)
```python
# After training, run validation
metrics = model_c.val(data="seesaw_children.yaml", split="test")

print(f"mAP@50:       {metrics.box.map50:.3f}")    # primary metric
print(f"mAP@50-95:    {metrics.box.map:.3f}")      # stricter metric
print(f"Precision:    {metrics.box.mp:.3f}")
print(f"Recall:       {metrics.box.mr:.3f}")
```
### Comparison Table (Fill In After Running)
| Model | mAP@50 | mAP@50-95 | Precision | Recall | Inference (iPhone, ms) |
|-------|--------|-----------|-----------|--------|------------------------|
| Run A — COCO baseline | ___ | ___ | ___ | ___ | ___ |
| Run B — HomeObjects-3K | ___ | ___ | ___ | ___ | ___ |
| Run C — All layers (SeeSaw) | ___ | ___ | ___ | ___ | ___ |
### Per-Class mAP Analysis
Pay particular attention to mAP for Layer 3 classes (IDs 18–24). Even if precision is modest (~0.4–0.6), the fact that the model detects `building_blocks` and `crayon` at all — something no general model does — is a research result in itself.
### Confusion Matrix
Ultralytics generates this automatically. Save `run_c_all_layers/confusion_matrix_normalized.png` — this is a dissertation figure (Figure X.X: Normalised confusion matrix for SeeSaw-YOLO11n).

***
## 7. CoreML Export
Once Run C training is complete and validated:[^13][^14][^15]

```python
from ultralytics import YOLO

# Load best weights from Run C
model = YOLO("runs/detect/run_c_all_layers/weights/best.pt")

# Export to CoreML .mlpackage with NMS baked in
model.export(
    format="coreml",
    nms=True,      # bakes Non-Maximum Suppression into model — required for iOS
    imgsz=640,
    int8=False     # FP16 is better quality for Neural Engine on iPhone
)
# Output: best.mlpackage → rename to seesaw-yolo11n.mlpackage
```

The `nms=True` flag is critical — it bakes Non-Maximum Suppression into the CoreML model graph, eliminating the need for post-processing code in the iOS app.[^14][^16]
### Drop into Xcode
```
seesaw-companion-ios/
└── SeeSawCompanion/
    └── Services/
        └── AI/
            └── seesaw-yolo11n.mlpackage/   ← replace YOLO11n.mlpackage
```

Update `PrivacyPipelineService.swift` to reference the new model class name generated by Xcode (will be `seesaw_yolo11n`, auto-generated from filename).

***
## 8. Repository Structure (`seesaw-yolo-model`)
```
seesaw-yolo-model/
├── README.md
├── datasets/
│   ├── layer1/                     ← HomeObjects-3K (auto-downloaded)
│   ├── layer2/                     ← Roboflow exports
│   ├── layer3/                     ← Your annotations export
│   └── seesaw_children/            ← Merged final dataset
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── configs/
│   ├── HomeObjects-3K.yaml         ← Layer 1 training config
│   └── seesaw_children.yaml        ← Combined training config
├── scripts/
│   ├── data_merge.py               ← DS-015: merge + remap class IDs
│   ├── class_distribution.py       ← DS-016: plot class balance chart
│   ├── train.py                    ← DS-019: Run B + Run C training
│   ├── evaluate.py                 ← DS-022: generate comparison table
│   └── export_coreml.py            ← DS-025: export .mlpackage
├── notebooks/
│   └── SeeSaw_YOLO_Training.ipynb  ← Google Colab notebook (all steps)
├── runs/                           ← Training outputs (gitignored)
│   ├── run_b_layer1/
│   └── run_c_all_layers/
└── export/
    └── seesaw-yolo11n.mlpackage    ← Final CoreML model (tracked in git)
```

***
## 9. Complete Task List
Tasks are prefixed `DS-` (Data Science). Copy into GitHub Issues under `seesaw-yolo-model` repository.

***
### Phase 1 — Environment Setup (DS-001 to DS-005)
**DS-001 — Create `seesaw-yolo-model` GitHub repository**
Create a new public GitHub repository named `seesaw-yolo-model`. Add MIT license. Create the directory structure in Section 8. Add a `.gitignore` that excludes `runs/`, `datasets/layer1/`, `datasets/layer2/`, `datasets/layer3/` (large files), but tracks `datasets/seesaw_children/` (merged, manageable size) and `export/seesaw-yolo11n.mlpackage`.

**DS-002 — Create Google Colab notebook**
Create `notebooks/SeeSaw_YOLO_Training.ipynb` in Google Colab. Mount Google Drive. Install Ultralytics: `!pip install ultralytics`. Verify GPU: `!nvidia-smi`. Set the working directory to a persistent Drive location. All subsequent training tasks run inside this notebook. Share the notebook link in the repo README.

**DS-003 — Verify YOLO11n COCO baseline inference**
Load stock `yolo11n.pt` and run inference on 5 test images from a child's bedroom (sourced from Unsplash, CC0). Record which classes are detected and which are missed. Save output images to `datasets/coco_baseline_samples/`. This forms the qualitative section of the dissertation's "Motivation for Fine-Tuning" argument.

**DS-004 — Set up Roboflow free account**
Create a free account at roboflow.com. Create two projects: `seesaw-layer2` (public, CC BY 4.0) and `seesaw-layer3` (private). Note the API key — needed for automated dataset export in DS-012.

**DS-005 — Document dataset licences**
Create `datasets/LICENCES.md`. Record the licence of each source: HomeObjects-3K (Ultralytics AGPL-3.0), Roboflow Universe datasets (CC BY 4.0), your own images (CC BY 4.0, you are the author). This is required for the dissertation appendix and for any future publication.

***
### Phase 2 — Layer 1: HomeObjects-3K (DS-006 to DS-008)
**DS-006 — Download and verify HomeObjects-3K**
Run the Ultralytics auto-download: `model.train(data="HomeObjects-3K.yaml", epochs=1)` — this downloads the 390 MB dataset to `datasets/homeobjects-3K/`. Cancel after 1 epoch. Verify the directory structure: `train/images/` (2,285 images), `valid/images/` (404 images). Check that all 12 class labels are present in the label files. Document image count per class in a markdown table.

**DS-007 — Run Baseline B: Train on HomeObjects-3K**
Run 50-epoch training from COCO pretrained weights on HomeObjects-3K only. Save all outputs to `runs/run_b_layer1/`. Record: final mAP@50, mAP@50-95, precision, recall, training time (minutes). Copy `runs/run_b_layer1/results.png` to `docs/results_run_b.png` — this is a dissertation figure. This is the Layer 1 baseline result.[^3][^10]

**DS-008 — Validate Run B on child-environment test set**
Run `model.val()` on Run B weights. Export the confusion matrix (`confusion_matrix_normalized.png`). Note which of the 12 HomeObjects classes perform best and worst. Write 2–3 sentences of analysis — this goes into the dissertation as the "Layer 1 Results" subsection.

***
### Phase 3 — Layer 2: Roboflow Augmentation (DS-009 to DS-014)
**DS-009 — Clone Roboflow `children` dataset into `seesaw-layer2`**
In Roboflow, open `universe.roboflow.com/project-odwld/children-u9om6`. Clone all images into the `seesaw-layer2` project. Inspect class names. The primary class of interest is `child` — this adds person-in-scene context. Verify ~625 images were cloned.

**DS-010 — Clone Roboflow `inside` dataset into `seesaw-layer2`**
Clone `universe.roboflow.com/yolo-a91kx/inside-mpg5a` into `seesaw-layer2`. This adds `toy`, `pillow`, `shelf`, `curtain`, `mirror` classes. Accept all classes — irrelevant ones can be filtered in the merge step. Verify images added.[^8]

**DS-011 — Filter COCO for child-relevant classes**
From the COCO 2017 validation set, extract images that contain at least one of: `teddy bear` (class 88), `book` (class 84), `sports ball` (class 37), `backpack` (class 27), `bottle` (class 44), `cup` (class 47). Use the `pycocotools` library. Target ~300 images. Export in YOLO format. Save to `datasets/layer2/coco_subset/`.

```python
from pycocotools.coco import COCO
import shutil, os

coco = COCO("annotations/instances_val2017.json")
target_classes = [37, 27, 44, 47, 84, 88]  # COCO IDs
img_ids = set()
for cat_id in target_classes:
    img_ids.update(coco.getImgIds(catIds=[cat_id]))
# Export selected images and annotations...
```

**DS-012 — Export `seesaw-layer2` from Roboflow in YOLO format**
In Roboflow, generate a dataset version for `seesaw-layer2`: Preprocessing: Auto-Orient, Resize 640×640. Augmentation: Horizontal Flip, Brightness ±15%. Export as **YOLOv8 format**. Download zip. Extract to `datasets/layer2/roboflow/`.[^17][^9]

**DS-013 — Verify Layer 2 class names and image counts**
Run a quick script to list all unique class names across Layer 2 label files. Identify any class names that differ from the unified SeeSaw class list (e.g., `teddybear` vs `teddy_bear`). Document the name mapping needed in `data_merge.py`. Record total image count.

**DS-014 — Document Layer 2 provenance**
Update `datasets/LICENCES.md` with each Roboflow dataset: workspace name, dataset name, URL, licence, date accessed. This is required academic practice for using third-party datasets.

***
### Phase 4 — Layer 3: Original Annotations (DS-015 to DS-018)
**DS-015 — Capture 50–100 original images**
Using an iPhone (held at approximately 1–1.5m height, simulating AiSee perspective), photograph scenes in a child's bedroom or playroom. Target: 10–15 images per custom class (building_blocks, dinosaur_toy, stuffed_animal, picture_book, crayon, toy_car, puzzle_piece). Include varied: lighting conditions (natural/artificial), distances (close/mid/far), backgrounds (carpet/tiles/bed). Store originals in `datasets/layer3/raw_images/`.

**DS-016 — Upload and annotate in Roboflow (`seesaw-layer3`)**
Upload all 50–100 images to the `seesaw-layer3` Roboflow project. Draw bounding boxes around every visible target object. Use Roboflow Label Assist for automated first-pass annotation, then correct manually. Also annotate any Layer 1/2 classes visible (bed, sofa, book, etc.) — this improves generalisation. Target: 3–5 bounding boxes per image. Document total annotation count.[^18]

**DS-017 — Apply augmentations and generate dataset version**
In Roboflow, generate version for `seesaw-layer3`: Augmentation: Horizontal Flip, Rotation ±10°, Brightness ±20%, Mosaic (optional). This multiplies 100 images to ~300 effective training samples. Export as **YOLOv8 format**. Extract to `datasets/layer3/roboflow/`.

**DS-018 — Document Layer 3 as research contribution**
Write `datasets/layer3/DATASET_CARD.md` following Hugging Face dataset card format:
- Dataset name: SeeSaw-ChildrensRoom-v1
- Description, motivation, how images were captured
- Class list, image count, annotation count
- Licence: CC BY 4.0 (your original work)
- Intended use: training YOLO11n for child's wearable AI scene understanding

This dataset card text feeds directly into the dissertation Chapter 4 dataset section.

***
### Phase 5 — Dataset Merge (DS-019 to DS-021)
**DS-019 — Write `data_merge.py`**
Script to merge all three layers into the unified `seesaw_children` dataset. Steps:

```python
# data_merge.py — pseudocode structure
# 1. Read all images and labels from layer1/, layer2/, layer3/
# 2. Remap class IDs to canonical seesaw_children IDs (see YAML Section 4)
# 3. Copy images to datasets/seesaw_children/images/all/
# 4. Copy remapped labels to datasets/seesaw_children/labels/all/
# 5. Split: 70% train, 15% val, 15% test (stratified by source layer)
# 6. Write seesaw_children.yaml with final class list
```

Use `supervision` library for dataset loading and splitting:[^19]
```python
pip install supervision
import supervision as sv
ds = sv.DetectionDataset.from_yolo(...)
train_ds, val_ds = ds.split(split_ratio=0.85)
val_ds, test_ds  = val_ds.split(split_ratio=0.5)
```

**DS-020 — Run class distribution analysis**
Write `scripts/class_distribution.py`. Plot a bar chart of image count per class across the merged dataset. Identify any severe class imbalance (< 50 images for a class = at risk). If `building_blocks` or `crayon` have < 50 images, return to DS-015 to capture more. Save chart as `docs/class_distribution.png` — dissertation figure.

**DS-021 — Verify merged dataset integrity**
Run Ultralytics dataset validation: `yolo data verify data=seesaw_children.yaml`. Check for: missing label files, empty label files, out-of-bounds bounding boxes, duplicate images. Fix all warnings before training. Record final counts: total images (train/val/test), total annotations, classes present.

***
### Phase 6 — Training Run C (DS-022 to DS-024)
**DS-022 — Run Training C: All layers combined**
Run 50-epoch fine-tuning on the full `seesaw_children.yaml` dataset from COCO pretrained `yolo11n.pt`. Save to `runs/run_c_all_layers/`. Monitor training in real-time via Colab output — check that val/loss decreases consistently. If loss plateaus early (patience=20), training stops automatically. Record all metrics.[^20][^10]

**DS-023 — Generate dissertation results table**
Run `scripts/evaluate.py` to produce the three-run comparison table. Load `best.pt` from Run B and Run C. Run `model.val(split="test")` on both using the same `seesaw_children.yaml` test split. Also run Run A (stock COCO) on the same test images. Output results as a markdown table and CSV. Save to `docs/results_comparison.md`.

**DS-024 — Save dissertation figures**
From `runs/run_c_all_layers/`, collect:
- `results.png` — training curves (loss, mAP over epochs)
- `confusion_matrix_normalized.png` — per-class confusion matrix
- `val_batch0_pred.jpg` — sample validation predictions

From `scripts/class_distribution.py`:
- `docs/class_distribution.png`

Copy all to `docs/dissertation_figures/` with descriptive filenames. These are Figures 4.X–4.X in the dissertation.

***
### Phase 7 — CoreML Export & iOS Integration (DS-025 to DS-028)
**DS-025 — Export to CoreML `.mlpackage`**
Run `scripts/export_coreml.py`:[^13][^15]

```python
from ultralytics import YOLO
model = YOLO("runs/detect/run_c_all_layers/weights/best.pt")
model.export(format="coreml", nms=True, imgsz=640)
# Renames output to seesaw-yolo11n.mlpackage
import shutil
shutil.move("runs/detect/run_c_all_layers/weights/best.mlpackage",
            "export/seesaw-yolo11n.mlpackage")
```

Verify the `.mlpackage` directory structure contains `Data/com.apple.CoreML/model.mlmodel` and `manifest.json`. The `nms=True` flag is critical — it prevents `CBATTError` and invalid output issues on device.[^14][^16]

**DS-026 — Drop `.mlpackage` into `seesaw-companion-ios` Xcode project**
Copy `export/seesaw-yolo11n.mlpackage` into `SeeSawCompanion/Services/AI/`. In Xcode, verify the auto-generated Swift class name (should be `seesaw_yolo11n`). Update `PrivacyPipelineService.swift` line:
```swift
// Before:
let model = try VNCoreMLModel(for: YOLO11n(configuration: .init()).model)
// After:
let model = try VNCoreMLModel(for: seesaw_yolo11n(configuration: .init()).model)
```

**DS-027 — Benchmark inference latency on iPhone Neural Engine**
Using Xcode Instruments (Time Profiler + Core ML), run 20 inference cycles on the iPhone with the new model. Record mean ± std deviation inference latency. Compare against stock `yolo11n` latency recorded in DS-003. Expected: similar latency (same model architecture, different weights) — confirms the custom model introduces no performance regression.[^15][^13]

**DS-028 — Run iOS privacy pipeline integration test**
Run the full `PrivacyPipelineService.process()` pipeline with 10 test images (from the Layer 3 test split) saved as JPEG fixtures in the Xcode test bundle. Verify: correct class labels returned, no raw pixel data in `ScenePayload`, latency under 700ms. Record pass/fail per image. This is a PoC acceptance test for Task T2-033 in `seesaw-companion-ios`.

***
### Phase 8 — Documentation & Dissertation (DS-029 to DS-032)
**DS-029 — Write `README.md` for `seesaw-yolo-model`**
Comprehensive README covering: project overview, three-layer strategy, class list, training results table, CoreML export instructions, how to use the model in `seesaw-companion-ios`. Include badges: Python version, Ultralytics version, dataset licence.

**DS-030 — Write dissertation Section 4.X: Dataset Construction**
Using the dataset card (DS-018), class distribution chart (DS-020), and provenance documentation (DS-005, DS-014), write the dataset construction section. Structure: (a) motivation for custom dataset, (b) Layer 1 description and statistics, (c) Layer 2 sources and class augmentation rationale, (d) Layer 3 original data collection methodology, (e) merged dataset statistics (image count, class distribution chart).

**DS-031 — Write dissertation Section 4.Y: Model Training**
Document the three training runs (A, B, C), hyperparameter choices with rationale, transfer learning justification, and training environment (Google Colab T4 GPU). Include the training curves figure from DS-024.

**DS-032 — Write dissertation Section 6.X: Object Detection Results**
Present the three-run comparison table (DS-023). Analyse: (a) mAP improvement from COCO baseline to Layer 1, (b) additional improvement from Layer 2+3, (c) per-class analysis for custom classes (IDs 18–24), (d) confusion matrix interpretation, (e) inference latency on iPhone Neural Engine. Acknowledge limitations: small Layer 3 sample size, adult-captured images (not true child egocentric perspective).

***
## 10. Timeline
| Week | Tasks | Milestone |
|------|-------|-----------|
| Week 1 | DS-001 to DS-005 | Environment ready, baseline inference documented |
| Week 1–2 | DS-006 to DS-008 | Run B complete, Layer 1 results documented |
| Week 2 | DS-009 to DS-014 | Layer 2 datasets imported and verified |
| Week 2–3 | DS-015 to DS-018 | Layer 3 images captured and annotated |
| Week 3 | DS-019 to DS-021 | Merged dataset ready, integrity verified |
| Week 3–4 | DS-022 to DS-024 | Run C complete, all figures generated |
| Week 4 | DS-025 to DS-028 | CoreML model exported and integrated into iOS |
| Week 4–5 | DS-029 to DS-032 | Dissertation sections written |

Total estimated effort: **25–35 hours** across 4–5 weeks, running in parallel with iOS and cloud development.

***
## 11. Tools & Resources
| Tool | Purpose | Cost |
|------|---------|------|
| Google Colab (free T4 GPU) | Model training | Free |
| Ultralytics Python package | Training, export | Free (AGPL-3.0) [^20] |
| Roboflow free tier | Annotation, Layer 2 cloning | Free |
| `pycocotools` | COCO subset extraction | Free |
| `supervision` library | Dataset merge + split | Free [^19] |
| Xcode Instruments | iOS latency benchmarking | Free (requires Mac) |

***

*Document version: 1.0 | Repository: `seesaw-yolo-model` | Last updated: 2026-03-25*
*Companion documents: `seesaw-native` Architecture Blueprint, `seesaw-companion-ios` Architecture Blueprint*

---

## References

1. [HomeObjects-3K Dataset](https://docs.ultralytics.com/datasets/detect/homeobjects-3k/) - Discover HomeObjects-3K, a rich indoor object detection dataset with 12 classes like bed, sofa, TV, ...

2. [HomeObjects-3K dataset have missing classes #21037](https://github.com/ultralytics/ultralytics/issues/21037) - I downloaded HomeObjects-3K dataset from the url in the official ultralytics website into kaggle not...

3. [How to Train Ultralytics YOLO11 on HomeObjects-3K Dataset | Detection, Validation & ONNX Export 🚀](https://www.youtube.com/watch?v=v3iqOYoRBFQ) - Let’s explore a dataset with 3,000+ high-resolution images and 12 common household object classes, i...

4. [How to Train Ultralytics YOLO11 Model on Custom Dataset using Google Colab Notebook | Step-by-Step 🚀](https://www.youtube.com/watch?v=ZN3nRZT7b24) - Join us for this deep dive into leveraging Ultralytics YOLO11 for cutting-edge computer vision tasks...

5. [Questions about roboflow licensing](https://www.reddit.com/r/computervision/comments/1knteid/questions_about_roboflow_licensing/) - Questions about roboflow licensing

6. [Using Images and Annotations From Open Source Computer Vision Datasets](https://www.youtube.com/watch?v=qZy4KvbIqUo) - Roboflow Universe is the world's largest repository of open-source computer vision datasets. Here is...

7. [children Object Detection Dataset by Project](https://universe.roboflow.com/project-odwld/children-u9om6) - 625 open source children images. children dataset by Project.

8. [inside Object Detection Dataset by yolo](https://universe.roboflow.com/yolo-a91kx/inside-mpg5a) - indoor-plant. indoor-plants. lamps light. light-bulb. lights. linens. pillows plant plants. pot. pot...

9. [How to label data for YOLOv8 training](https://roboflow.com/how-to-label/yolov8) - In this guide, we are going to show how to use Roboflow Annotate a free tool you can use to create a...

10. [Yolo11 training with custom dataset - Support - Ultralytics](https://community.ultralytics.com/t/yolo11-training-with-custom-dataset/697) - Hi everyone, I’m working on training a YOLOv11 model for object detection on a custom dataset of 11 ...

11. [How can I fine-tune a trained custom dataset model quickly? · Issue #12069 · ultralytics/yolov5](https://github.com/ultralytics/yolov5/issues/12069) - Search before asking I have searched the YOLOv5 issues and discussions and found no similar question...

12. [Guidance in Yolov11 learning transfer · Issue #17255 · ultralytics/ultralytics](https://github.com/ultralytics/ultralytics/issues/17255) - Search before asking I have searched the Ultralytics YOLO issues and discussions and found no simila...

13. [Bringing Ultralytics YOLO11 to Apple devices via CoreML](https://www.ultralytics.com/blog/bringing-ultralytics-yolo11-to-apple-devices-via-coreml) - See how easy it is to bring Ultralytics YOLO11 to Apple devices with CoreML and enable fast offline ...

14. [Compile a CoreML model locally (coreml.compile_model)](https://xxtouch.app/en/docs/handbook/coreml/coreml.compile_model/) - Declaration

15. [Faq](https://docs.ultralytics.com/integrations/coreml/) - Learn how to export YOLO26 models to CoreML for optimized, on-device machine learning on iOS and mac...

16. [Exporting YoloV11 Pytorch Model to CoreML · Issue #885 · ultralytics/hub](https://github.com/ultralytics/hub/issues/885) - Search before asking I have searched the HUB issues and discussions and found no similar questions. ...

17. [Roboflow - Ultralytics YOLO Docs](https://docs.ultralytics.com/integrations/roboflow/) - Learn how to label data and export datasets in YOLO format using Roboflow for training Ultralytics m...

18. [How to Auto Label Your Custom Dataset with Roboflow in ...](https://www.youtube.com/watch?v=SDV6Gz0suAk) - In this video here we're going to take a look at how we can do auto labeling of our images.

19. [Split YOLO Datasets](https://roboflow.com/split-datasets/yolo) - In this guide, we will show how to split your datasets with the supervision Python package. We will:...

20. [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) - Discover YOLO11, an advancement in real-time object detection, offering excellent accuracy and effic...

