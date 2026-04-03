# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Custom YOLO11n object detection model for the **SeeSaw wearable AI companion** — a device worn by children. The model detects 44 classes of objects in children's indoor environments and is exported to CoreML for on-device inference on iOS (Neural Engine, FP16).

The training pipeline uses a three-layer dataset merge strategy:
- **Layer 1:** HomeObjects-3K (Ultralytics auto-download, 2,689 images, 12 classes)
- **Layer 2:** Roboflow Universe datasets (~354 images, 33 classes)
- **Layer 3:** Original egocentric annotations (240 images, 5 classes)
- **Merged:** `datasets/seesaw_children/` (3,283 images, 44-class canonical taxonomy)

## Common Commands

The full pipeline runs in `notebooks/yolo_training.ipynb` on Google Colab (T4 GPU, ~50 min per training run).

One utility script remains:

```bash
# Merge the three dataset layers into unified seesaw_children/ dataset
python scripts/data_merge.py
```

## Architecture & Data Flow

```
Layer 1/2/3 raw datasets
        ↓  data_merge.py
datasets/seesaw_children/{images,labels}/{train,val,test}/
        ↓  notebook Cell 16/19  (50 epochs, batch=16, imgsz=640, AdamW)
runs/detect/run_c_all_layers/weights/best.pt
        ↓  notebook Cell 20-21  (NMS baked in, half=False, nc patched 80→44)
export/seesaw-yolo11n.mlpackage
        ↓  copy to Xcode project
seesaw-companion-ios → PrivacyPipelineService.swift
```

**Three training runs** establish a comparative evaluation:
- **Run A:** COCO stock YOLO11n (no fine-tuning) — domain baseline
- **Run B:** Fine-tuned on Layer 1 only — mAP@50=0.8614
- **Run C:** Fine-tuned on all 3 merged layers — mAP@50=0.6748, the production model

## Key Configuration Files

- `configs/seesaw_children.yaml` — Unified 44-class dataset config (used for Run C training)
- `configs/HomeObjects-3K.yaml` — Layer 1 only config (used for Run B; auto-downloads dataset)

## Class Taxonomy

The 44-class canonical taxonomy is defined in `configs/seesaw_children.yaml`. Class IDs:
- **0–11:** Furniture baseline from Layer 1 (bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted_plant, photo_frame)
- **12–24:** Child-environment objects (teddy_bear, book, sports_ball, backpack, bottle, cup, building_blocks, dinosaur_toy, stuffed_animal, picture_book, crayon, toy_car, puzzle_piece)
- **25–43:** Extended household/toy classes

`data_merge.py` normalizes synonyms during merge (e.g., `lamps`→`lamp`, `television`→`tv`) and uses a 70/15/15 train/val/test split.

## Datasets & Gitignore

Large files are excluded from git: `datasets/`, `runs/`, `*.pt` (except base weights), `*.mlpackage`. Use `sync_artifacts.py` to restore these from Colab/Google Drive. The Roboflow datasets require `ROBOFLOW_API_KEY` (stored as Colab secret or env var). GitHub pushes from Colab use `GITHUB_PAT`.

## iOS Integration

The exported `.mlpackage` is consumed by the companion iOS app:
```swift
let model = try VNCoreMLModel(for: seesaw_yolo11n(configuration: .init()).model)
```
NMS is baked into the CoreML graph — no post-processing needed on device.
