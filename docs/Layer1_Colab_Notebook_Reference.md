# SeeSaw YOLO11n — Layer 1 Colab Notebook Reference
### Google Colab Notebook Development Guide for Phase 2 (DS-006 → DS-008)
> **Based on:** [Ultralytics Official HomeObjects-3K Notebook](https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-homeobjects-dataset.ipynb)
> **Target output:** `notebooks/SeeSaw_YOLO_Training.ipynb`
> **Runtime:** Google Colab → Runtime → Change runtime type → **T4 GPU**

---

## Key Differences from Official Notebook

| Aspect | Ultralytics Official | SeeSaw Layer 1 Notebook |
|--------|---------------------|------------------------|
| Model | `yolo26n.pt` (YOLO26) | `yolo11n.pt` (YOLO11n) — matches our iOS deployment target |
| Epochs | 3 (demo) | 50 (full training with early stopping) |
| Purpose | Generic demo | Run B baseline + Run A COCO comparison for dissertation |
| Export | ONNX (demo) | CoreML `.mlpackage` with `nms=True` for iPhone Neural Engine |
| Dataset config | `HomeObjects-3K.yaml` (auto-download URL) | `configs/HomeObjects-3K.yaml` (repo-local, custom path) |
| Results | Ephemeral | Saved to Google Drive + committed to repo |

---

## Notebook Cell Plan

The notebook has **10 cells**, organized into 4 sections. Each cell maps to a DS-task.

---

### Section 1 — Environment Setup (DS-002)

#### Cell 1: Header (Markdown)
```markdown
# SeeSaw YOLO11n — Layer 1 Training (HomeObjects-3K)

**Project:** SeeSaw — YOLO11n Custom Dataset Training Pipeline
**Phase:** 2 — Layer 1 Baseline (DS-006 → DS-008)
**Runtime:** Google Colab T4 GPU (free tier)

This notebook trains YOLO11n on the HomeObjects-3K indoor object detection dataset
(12 classes, 2,285 train / 404 val images). The result is **Run B** — the Layer 1
baseline for the dissertation comparison table.

| Run | Dataset | Purpose |
|-----|---------|---------|
| **Run A** | Stock `yolo11n.pt` (COCO) | Baseline — no fine-tuning |
| **Run B** ← this notebook | HomeObjects-3K (Layer 1) | Indoor domain baseline |
| Run C | Full `seesaw_children` (all layers) | Final model (later phases) |
```

#### Cell 2: Setup & GPU Verification (Code)
```python
# DS-002: Environment setup — install dependencies, verify GPU, clone repo

# Install Ultralytics (includes torch, torchvision, opencv, etc.)
!pip install ultralytics -q

# Verify GPU is available
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# Ultralytics system check
import ultralytics
ultralytics.checks()
```

**Expected output:**
- PyTorch 2.x with CUDA
- GPU: NVIDIA T4 (or L4), 15–22 GB VRAM
- Ultralytics version ≥ 8.3.x

#### Cell 3: Clone Repository (Code)
```python
# Clone the seesaw-yolo-model repo to get configs and scripts
import os

REPO_URL = "https://github.com/<your-username>/seesaw-yolo-model.git"
REPO_DIR = "/content/seesaw-yolo-model"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}

%cd {REPO_DIR}
!ls -la
```

> **Action required:** Replace `<your-username>` with your actual GitHub username before running.

---

### Section 2 — Dataset Download & Verification (DS-006)

#### Cell 4: Dataset YAML Inspection (Markdown + Code)

Markdown preamble:
```markdown
## Dataset: HomeObjects-3K

The HomeObjects-3K dataset is an official Ultralytics indoor object detection dataset.
- **12 classes:** bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted_plant, photo_frame
- **2,285 training images** + **404 validation images**
- **390 MB** auto-download on first use
- **Licence:** AGPL-3.0

Reference: [HomeObjects-3K Documentation](https://docs.ultralytics.com/datasets/detect/homeobjects-3k/)
```

Code:
```python
# DS-006: Inspect the dataset YAML config
!cat configs/HomeObjects-3K.yaml
```

**Expected output:** The 12-class YAML matching `configs/HomeObjects-3K.yaml` in the repo.

#### Cell 5: Download & Verify HomeObjects-3K (Code)
```python
# DS-006: Trigger HomeObjects-3K auto-download by running 1 epoch
# This downloads 390 MB to /content/datasets/homeobjects-3K/
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="HomeObjects-3K.yaml",  # Uses Ultralytics built-in YAML (auto-downloads)
    epochs=1,
    imgsz=640,
    batch=16,
    device=0,
    name="download_verify",
)
```

**What happens:**
1. Downloads `yolo11n.pt` pretrained weights (~5 MB)
2. Downloads `homeobjects-3K.zip` (390 MB) → auto-extracts to `/content/datasets/homeobjects-3K/`
3. Scans labels: expects `2285 images, 0 backgrounds, 0 corrupt` for train
4. Scans labels: expects `404 images, 0 backgrounds, 0 corrupt` for val
5. Trains 1 epoch as verification — confirm non-zero mAP

**Known issue from official notebook:** 3 training images have duplicate labels (auto-removed by Ultralytics):
- `living_room_1303.jpg`, `living_room_1675.jpg`, `living_room_1795.jpg` — this is expected.

#### Cell 6: Post-Download Verification (Code)
```python
# DS-006: Verify dataset integrity — counts, classes, structure
import os
from pathlib import Path
from collections import Counter

dataset_root = Path("/content/datasets/homeobjects-3K")

# Count images and labels per split
for split in ["train", "valid"]:
    img_dir = dataset_root / split / "images"
    lbl_dir = dataset_root / split / "labels"
    n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    print(f"{split}: {n_imgs} images, {n_lbls} labels")

# Verify all 12 classes are present
class_counter = Counter()
for split in ["train", "valid"]:
    lbl_dir = dataset_root / split / "labels"
    if not lbl_dir.exists():
        continue
    for lbl_file in lbl_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            class_id = int(line.split()[0])
            class_counter[class_id] += 1

CLASS_NAMES = {
    0: "bed", 1: "sofa", 2: "chair", 3: "table", 4: "lamp", 5: "tv",
    6: "laptop", 7: "wardrobe", 8: "window", 9: "door",
    10: "potted_plant", 11: "photo_frame",
}

print(f"\n{'ID':<4} {'Class':<15} {'Annotations':>12}")
print("-" * 33)
for cid in sorted(class_counter.keys()):
    name = CLASS_NAMES.get(cid, f"unknown_{cid}")
    print(f"{cid:<4} {name:<15} {class_counter[cid]:>12}")
print(f"\nTotal annotations: {sum(class_counter.values())}")
print(f"Classes found: {len(class_counter)}/12")
assert len(class_counter) == 12, f"Expected 12 classes, found {len(class_counter)}"
print("✓ Dataset verification passed")
```

**Expected output (approximate, from official notebook):**
| ID | Class | Annotations |
|----|-------|-------------|
| 0 | bed | ~22+ |
| 1 | sofa | ~398+ |
| 2 | chair | ~305+ |
| 3 | table | ~469+ |
| 4 | lamp | ~304+ |
| 5 | tv | ~54+ |
| 6 | laptop | varies |
| 7 | wardrobe | ~109+ |
| 8 | window | ~371+ |
| 9 | door | ~85+ |
| 10 | potted_plant | ~788+ |
| 11 | photo_frame | ~561+ |

Total annotations: ~3,466+ (validation set alone per official output).

---

### Section 3 — Training Run B (DS-007)

#### Cell 7: Train Run B — 50 Epochs (Code)
```python
# DS-007: Run B — Fine-tune YOLO11n on HomeObjects-3K (Layer 1 baseline)
# Expected runtime: ~45–60 minutes on T4 GPU
from ultralytics import YOLO

model_b = YOLO("yolo11n.pt")  # Fresh COCO pretrained weights

results = model_b.train(
    data="HomeObjects-3K.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=20,       # Early stopping: halt if no val improvement for 20 epochs
    name="run_b_layer1",
    device=0,
    plots=True,        # Generate training curves + confusion matrix
)
```

**Key hyperparameters (rationale from training plan):**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `epochs` | 50 | Sufficient for fine-tuning from COCO pretrained weights |
| `imgsz` | 640 | Standard YOLO input; matches AiSee camera output |
| `batch` | 16 | Fits T4 16 GB VRAM comfortably |
| `patience` | 20 | Early stopping prevents overfitting on ~2,700 images |
| `optimizer` | auto (AdamW) | Ultralytics auto-selects best optimizer + learning rate |

**What to watch during training:**
- `box_loss`, `cls_loss`, `dfl_loss` should decrease over epochs
- `mAP50` and `mAP50-95` should increase
- Training may stop before epoch 50 if patience triggers early stopping
- Official notebook achieved **mAP50 = 0.570, mAP50-95 = 0.389** after only 3 epochs — expect significantly higher at 50 epochs

**Output saved to:** `runs/detect/run_b_layer1/`
```
runs/detect/run_b_layer1/
├── weights/
│   ├── best.pt          ← Best model weights (use this)
│   └── last.pt          ← Final epoch weights
├── results.png          ← Training curves (loss, mAP over epochs) — DISSERTATION FIGURE
├── confusion_matrix_normalized.png  ← Per-class confusion matrix — DISSERTATION FIGURE
├── labels.jpg           ← Dataset label distribution
├── val_batch0_pred.jpg  ← Sample validation predictions
└── args.yaml            ← Full hyperparameter record
```

---

### Section 4 — Validation & Results (DS-008)

#### Cell 8: Validate Run B (Code)
```python
# DS-008: Validate Run B on the validation set and print results
from ultralytics import YOLO

model_b = YOLO("runs/detect/run_b_layer1/weights/best.pt")
metrics = model_b.val(data="HomeObjects-3K.yaml")

print("\n" + "=" * 60)
print("RUN B — HomeObjects-3K (Layer 1 Baseline) Results")
print("=" * 60)
print(f"  mAP@50:       {metrics.box.map50:.4f}")
print(f"  mAP@50-95:    {metrics.box.map:.4f}")
print(f"  Precision:    {metrics.box.mp:.4f}")
print(f"  Recall:       {metrics.box.mr:.4f}")
print("=" * 60)

# Per-class breakdown
print(f"\n{'Class':<15} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
print("-" * 53)
CLASS_NAMES = ["bed", "sofa", "chair", "table", "lamp", "tv",
               "laptop", "wardrobe", "window", "door", "potted_plant", "photo_frame"]
for i, name in enumerate(CLASS_NAMES):
    ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
    ap = metrics.box.ap[i] if i < len(metrics.box.ap) else 0
    print(f"{name:<15} {ap50:>8.3f} {ap:>10.3f}")
```

**Benchmark reference (official notebook, 3 epochs only):**
| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| bed | 0.533 | 0.375 |
| sofa | 0.833 | 0.593 |
| chair | 0.666 | 0.451 |
| table | 0.719 | 0.510 |
| lamp | 0.417 | 0.250 |
| tv | 0.572 | 0.459 |
| wardrobe | 0.405 | 0.255 |
| window | 0.442 | 0.252 |
| door | 0.284 | 0.200 |
| potted_plant | 0.629 | 0.326 |
| photo_frame | 0.774 | 0.606 |
| **ALL** | **0.570** | **0.389** |

At 50 epochs, expect notably higher metrics across all classes.

#### Cell 9: Run A — COCO Baseline Comparison (Code)
```python
# DS-003/DS-008: Run A — Evaluate stock COCO yolo11n.pt on same validation set
# This shows how well the generic COCO model handles indoor objects WITHOUT fine-tuning
from ultralytics import YOLO

model_a = YOLO("yolo11n.pt")  # Stock COCO weights — no fine-tuning
metrics_a = model_a.val(data="HomeObjects-3K.yaml")

print("\n" + "=" * 60)
print("RUN A — COCO Baseline (No Fine-Tuning) Results")
print("=" * 60)
print(f"  mAP@50:       {metrics_a.box.map50:.4f}")
print(f"  mAP@50-95:    {metrics_a.box.map:.4f}")
print(f"  Precision:    {metrics_a.box.mp:.4f}")
print(f"  Recall:       {metrics_a.box.mr:.4f}")
print("=" * 60)

# Side-by-side comparison
print("\n" + "=" * 60)
print("COMPARISON: Run A vs Run B")
print("=" * 60)
print(f"{'Metric':<15} {'Run A (COCO)':>14} {'Run B (Layer 1)':>16} {'Δ Improvement':>15}")
print("-" * 62)

# Load Run B metrics for comparison
model_b = YOLO("runs/detect/run_b_layer1/weights/best.pt")
metrics_b = model_b.val(data="HomeObjects-3K.yaml")

comparisons = [
    ("mAP@50",    metrics_a.box.map50, metrics_b.box.map50),
    ("mAP@50-95", metrics_a.box.map,   metrics_b.box.map),
    ("Precision",  metrics_a.box.mp,    metrics_b.box.mp),
    ("Recall",     metrics_a.box.mr,    metrics_b.box.mr),
]
for name, a, b in comparisons:
    delta = b - a
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"{name:<15} {a:>14.4f} {b:>16.4f} {arrow} {abs(delta):>13.4f}")
```

**Dissertation note:** The Run A → Run B delta demonstrates the value of domain-specific indoor data over generic COCO. This comparison is the core quantitative argument for Section 6.X.

#### Cell 10: Save Results & Dissertation Figures (Code)
```python
# DS-008/DS-024: Save all outputs to Google Drive for persistence
import shutil
from pathlib import Path
from google.colab import drive

# Mount Google Drive
drive.mount("/content/drive")

# Create output directory on Drive
DRIVE_DIR = Path("/content/drive/MyDrive/seesaw-yolo-runs")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

# Copy Run B outputs
run_b_dir = Path("runs/detect/run_b_layer1")
drive_run_b = DRIVE_DIR / "run_b_layer1"
if run_b_dir.exists():
    if drive_run_b.exists():
        shutil.rmtree(drive_run_b)
    shutil.copytree(run_b_dir, drive_run_b)
    print(f"✓ Run B saved to Google Drive: {drive_run_b}")

# Copy key dissertation figures
FIGURES_DIR = DRIVE_DIR / "dissertation_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

figures = {
    "results_run_b_training_curves.png": run_b_dir / "results.png",
    "confusion_matrix_run_b.png": run_b_dir / "confusion_matrix_normalized.png",
    "labels_distribution_run_b.jpg": run_b_dir / "labels.jpg",
    "val_predictions_run_b.jpg": run_b_dir / "val_batch0_pred.jpg",
}

for dst_name, src_path in figures.items():
    if src_path.exists():
        shutil.copy2(src_path, FIGURES_DIR / dst_name)
        print(f"✓ Saved: {dst_name}")
    else:
        print(f"⚠ Not found: {src_path}")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print("Copy these to docs/dissertation_figures/ in your repo.")
```

---

## Pre-Flight Checklist

Before running the notebook, confirm:

- [ ] Repo is pushed to GitHub (`git push` — ✅ already done)
- [ ] Colab runtime set to **T4 GPU** (Runtime → Change runtime type)
- [ ] Replace `<your-username>` in Cell 3 with your actual GitHub username
- [ ] If repo is private, either make it public or use a GitHub personal access token in the clone URL

## Expected Timeline

| Cell | DS-Task | Duration | What Happens |
|------|---------|----------|-------------|
| 1 | — | — | Markdown header (no execution) |
| 2 | DS-002 | ~30 seconds | pip install + GPU check |
| 3 | DS-002 | ~10 seconds | git clone repo |
| 4 | DS-006 | — | Markdown + YAML inspection |
| 5 | DS-006 | ~2–3 minutes | Download 390 MB dataset + 1-epoch verify |
| 6 | DS-006 | ~15 seconds | Dataset integrity check |
| 7 | DS-007 | **45–60 minutes** | 50-epoch training (the main wait) |
| 8 | DS-008 | ~1 minute | Validation + per-class metrics |
| 9 | DS-008 | ~2 minutes | Run A baseline + comparison table |
| 10 | DS-008 | ~30 seconds | Save to Google Drive |

**Total: ~50–65 minutes** (mostly Cell 7 training time).

## After This Notebook

With Run B complete, you have:
1. **Run A metrics** — COCO baseline on indoor objects
2. **Run B metrics** — Fine-tuned on HomeObjects-3K (Layer 1)
3. **Run A vs Run B comparison table** — core dissertation result
4. **Dissertation figures** — training curves, confusion matrix, sample predictions

**Next phase:** DS-009–DS-014 (Layer 2: Roboflow augmentation) — this requires manual work in Roboflow before returning to Colab for the next training run.

---

## Important Notes

### YOLO11n vs YOLO26
The official Ultralytics notebook uses `yolo26n.pt` (their latest model). Our project uses **`yolo11n.pt`** because:
- YOLO11n is already validated for CoreML export with `nms=True`
- The iOS app (`seesaw-companion-ios`) is built around YOLO11n's output format
- Switching to YOLO26 would require re-validating the entire iOS pipeline

Do **not** change the model to YOLO26 unless the iOS integration is also updated.

### HomeObjects-3K.yaml — Built-in vs Repo Config
- **Cell 5** uses `data="HomeObjects-3K.yaml"` (Ultralytics built-in) — this triggers the auto-download to `/content/datasets/homeobjects-3K/`
- **Cell 7** also uses the built-in YAML for the same reason
- The repo's `configs/HomeObjects-3K.yaml` has a custom path (`datasets/layer1`) — this is for the eventual data merge step (DS-019), not for direct training

### Colab Session Limits
- Free tier: ~12 hours max session, T4 GPU (may get disconnected)
- 50 epochs on ~2,700 images should complete well within limits
- Cell 10 saves to Google Drive as a safety measure against disconnection
- If disconnected mid-training, re-run from Cell 7 — Ultralytics does not auto-resume by default

### Citation
```bibtex
@dataset{Jocher_Ultralytics_Datasets_2025,
    author = {Jocher, Glenn and Rizwan, Muhammad},
    license = {AGPL-3.0},
    title = {Ultralytics Datasets: HomeObjects-3K Detection Dataset},
    url = {https://docs.ultralytics.com/datasets/detect/homeobjects-3k/},
    version = {1.0.0},
    year = {2025}
}
```
