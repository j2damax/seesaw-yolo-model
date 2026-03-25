# SeeSaw Custom Dataset — Roboflow Workspace Instructions
### Complete Guide for Layer 2 (DS-009 → DS-014) & Layer 3 (DS-015 → DS-018)
> **Platform:** [app.roboflow.com](https://app.roboflow.com/)
> **Account tier:** Free (up to 10,000 source images, unlimited exports)
> **Output format:** YOLOv8 PyTorch TXT
> **Target:** Datasets for classes 12–24 in the SeeSaw 25-class taxonomy

---

## Overview — What You're Building

| Layer | Roboflow Project | Source | Classes Added | Target Images |
|-------|-----------------|--------|---------------|---------------|
| **Layer 2** | `seesaw-layer2` | Public datasets from Roboflow Universe | 12–17 (teddy_bear, book, sports_ball, backpack, bottle, cup) | ~500–800 |
| **Layer 3** | `seesaw-layer3` | Your own iPhone photos of children's environments | 18–24 (building_blocks, dinosaur_toy, stuffed_animal, picture_book, crayon, toy_car, puzzle_piece) | 50–100 |

Both projects export as **YOLOv8 format** and feed into `scripts/data_merge.py` (DS-019).

---

## Part A — Account & Workspace Setup (DS-004)

### A1. Create Your Roboflow Account
1. Go to [app.roboflow.com](https://app.roboflow.com/)
2. Sign up with Google/GitHub (fastest) or email
3. Choose **Free / Public Plan** when prompted
4. Your workspace is created automatically — note the workspace name (visible in the URL: `app.roboflow.com/<workspace-name>`)

> **Important:** On the Free plan, all projects are **public** on Roboflow Universe. This is fine for Layer 2 (already public data) and acceptable for Layer 3 (your research dataset, CC BY 4.0). If you need private projects, upgrade to the Starter plan.

### A2. Note Your API Key
1. Click your profile icon (top right) → **Settings** → **Roboflow API Key**
2. Copy and save the key — you'll need it for programmatic export in the Colab notebook
3. **Never commit this key to GitHub** — use Colab secrets or environment variables

---

## Part B — Layer 2: Roboflow Universe Augmentation (DS-009 → DS-014)

Layer 2 adds child-relevant classes from existing public datasets. You'll clone images from Roboflow Universe into a single project, then export.

### B1. Create the Layer 2 Project (DS-009)

1. In Roboflow dashboard → **Create New Project**
2. Configure:
   - **Project Name:** `seesaw-layer2`
   - **Project Type:** Object Detection
   - **Annotation Group:** (leave default)
   - **Licence:** CC BY 4.0
3. Click **Create Project**

### B2. Clone the `children` Dataset (DS-009)

This dataset adds person/child context to indoor scenes (~625 images).

1. Open: [universe.roboflow.com/project-odwld/children-u9om6](https://universe.roboflow.com/project-odwld/children-u9om6)
2. Click **Download Dataset** (or **Fork** / **Clone into Project**)
3. Select your `seesaw-layer2` project as the destination
4. Choose the latest version
5. Wait for the import to complete
6. **Verify:** Go to `seesaw-layer2` → Annotate tab → confirm ~625 images appeared
7. Check class names — the primary class should be `child` or similar

### B3. Clone the `inside` Dataset (DS-010)

This dataset adds indoor object classes: toy, pillow, shelf, curtain, mirror (~400 images, 57 classes).

1. Open: [universe.roboflow.com/yolo-a91kx/inside-mpg5a](https://universe.roboflow.com/yolo-a91kx/inside-mpg5a)
2. Click **Download Dataset** → Clone into `seesaw-layer2` project
3. Wait for import
4. **Verify:** Total image count should now be ~1,000+
5. Note all incoming class names — you'll filter/remap these in Step B5

### B4. Add COCO Child-Relevant Subset (DS-011)

> **Two options — choose based on preference:**

**Option A: Use Roboflow Universe COCO datasets (easier)**

Search Roboflow Universe for pre-filtered COCO subsets:
1. Go to [universe.roboflow.com](https://universe.roboflow.com/)
2. Search for datasets containing classes: `teddy bear`, `book`, `sports ball`, `backpack`, `bottle`, `cup`
3. Clone relevant results into `seesaw-layer2`
4. Target: ~300 additional images with these 6 classes

**Option B: Filter COCO locally with pycocotools (more control)**

Run this in your Colab notebook (requires COCO 2017 val set):
```python
from pycocotools.coco import COCO
import shutil, os

coco = COCO("annotations/instances_val2017.json")
# COCO class IDs for our target classes
target_classes = {
    37: "sports_ball",
    27: "backpack",
    44: "bottle",
    47: "cup",
    84: "book",
    88: "teddy_bear",
}
img_ids = set()
for cat_id in target_classes:
    img_ids.update(coco.getImgIds(catIds=[cat_id]))

print(f"Found {len(img_ids)} images containing target classes")
# Export selected images and convert annotations to YOLO format...
```
Then upload the filtered images + YOLO labels to `seesaw-layer2` via drag-and-drop.

### B5. Inspect & Remap Class Names (DS-013)

This is a critical step — class names from different sources won't match your unified taxonomy.

1. Go to `seesaw-layer2` → **Dataset** tab → **Health Check** (or browse the class list)
2. Note every unique class name. You'll likely see:
   ```
   From children dataset:  child, person, kid, ...
   From inside dataset:    toy, pillow, shelf, curtain, mirror, crib, ...
   From COCO subset:       teddy bear, book, sports ball, backpack, bottle, cup
   ```

3. **Create a mapping table** — these map to your SeeSaw canonical IDs 12–17:

   | Source Class Name(s) | SeeSaw Target Name | SeeSaw ID |
   |---------------------|-------------------|-----------|
   | `teddy bear`, `teddybear`, `teddy_bear` | `teddy_bear` | 12 |
   | `book`, `books` | `book` | 13 |
   | `sports ball`, `sports_ball`, `ball` | `sports_ball` | 14 |
   | `backpack`, `bag` | `backpack` | 15 |
   | `bottle`, `water bottle` | `bottle` | 16 |
   | `cup`, `mug` | `cup` | 17 |

4. **Use Roboflow's Modify Classes** feature (available during version generation) to remap names:
   - Go to **Versions** → **Generate New Version**
   - In the **Preprocessing** step, click **Modify Classes**
   - Remap source names to target names (e.g., `teddy bear` → `teddy_bear`)
   - Omit classes you don't need (e.g., `curtain`, `mirror` — unless you want them)

> **Important:** Record all mappings in a text file — you'll need these for `scripts/data_merge.py` LAYER2_REMAP dictionary.

### B6. Generate Dataset Version & Export (DS-012)

1. Go to `seesaw-layer2` → **Versions** → **Generate New Version**

2. **Split configuration:**
   - Train / Valid / Test: **70% / 20% / 10%**
   - (This split is temporary — `data_merge.py` will re-split when merging all layers)

3. **Preprocessing settings:**
   | Setting | Value | Reason |
   |---------|-------|--------|
   | Auto-Orient | ✅ ON | Strips EXIF data, prevents orientation bugs |
   | Resize | **Stretch to 640×640** | Matches YOLO11n training `imgsz=640` |
   | Grayscale | ❌ OFF | Keep colour — indoor scenes need colour cues |
   | Auto-Adjust Contrast | ❌ OFF | Not needed for well-lit indoor photos |
   | Modify Classes | ✅ See B5 | Remap class names to SeeSaw taxonomy |
   | Filter Null | ❌ OFF | Keep all images including backgrounds |

4. **Augmentation settings (free tier):**
   | Augmentation | Setting | Reason |
   |-------------|---------|--------|
   | Flip | **Horizontal** | Mirrors left-right — safe for indoor objects |
   | Flip Vertical | ❌ OFF | Furniture doesn't appear upside-down |
   | 90° Rotate | ❌ OFF | Indoor scenes have a strong vertical gravity |
   | Rotation | ±10° | Slight tilt simulates camera movement |
   | Brightness | ±15% | Simulates varying indoor lighting |
   | Blur | Up to 0.5px | Very slight — simulates mild motion blur |
   | Noise | ❌ OFF | Not needed at this stage |
   | Mosaic | ❌ OFF | YOLO handles this during training |
   | **Max Version Size** | **3x** | Triples effective training images |

5. Click **Generate** — wait for processing (may take a few minutes)

6. **Export:**
   - Click **Export** next to the generated version
   - **Format:** `YOLOv8` (under "YOLO" section)
   - **Download:** Choose **"Show download code"** → copy the Python snippet
   - Or click **Download zip** → save locally

7. **For Colab** — use the Python download code in your notebook:
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")  # Use Colab secrets!
   project = rf.workspace("your-workspace").project("seesaw-layer2")
   version = project.version(1)
   dataset = version.download("yolov8", location="/content/seesaw-yolo-model/datasets/layer2")
   ```

8. **Verify export structure:**
   ```
   datasets/layer2/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

### B7. Document Provenance (DS-014)

Record in `datasets/LICENCES.md`:
```markdown
## Layer 2 — Roboflow Universe Datasets

### children dataset
- **URL:** https://universe.roboflow.com/project-odwld/children-u9om6
- **Workspace:** project-odwld
- **Licence:** CC BY 4.0
- **Date accessed:** [TODAY'S DATE]
- **Images used:** ~625

### inside dataset
- **URL:** https://universe.roboflow.com/yolo-a91kx/inside-mpg5a
- **Workspace:** yolo-a91kx
- **Licence:** CC BY 4.0
- **Date accessed:** [TODAY'S DATE]
- **Images used:** ~400

### COCO subset
- **Source:** MS COCO 2017 validation set
- **Licence:** CC BY 4.0
- **Classes extracted:** teddy_bear, book, sports_ball, backpack, bottle, cup
- **Images used:** ~300
```

---

## Part C — Layer 3: Original Dataset (DS-015 → DS-018)

This is your **research contribution** — original data that no existing dataset provides. These are images from a child's perspective in real indoor environments.

### C1. Create the Layer 3 Project (DS-015)

1. In Roboflow dashboard → **Create New Project**
2. Configure:
   - **Project Name:** `seesaw-layer3`
   - **Project Type:** Object Detection
   - **Licence:** CC BY 4.0 (you are the author)
3. Click **Create Project**

### C2. Capture Images with iPhone (DS-015)

**Camera Settings:**
- Use the standard iPhone Camera app
- **Orientation:** Landscape (horizontal) preferred — matches 640×640 better
- **Height:** Hold phone at ~1–1.5m from the floor — this simulates the AiSee headset's egocentric perspective (child's eye level)
- **Flash:** OFF — use natural/room lighting only
- **HDR:** ON (default) — provides good dynamic range

**Target Scenes (aim for 10–15 images per class):**

| Class (ID) | What to Photograph | Tips |
|-----------|-------------------|------|
| `building_blocks` (18) | Lego, Duplo, wooden blocks — piles, towers, scattered on floor | Include both close-up and mid-range shots |
| `dinosaur_toy` (19) | Plastic/rubber dinosaur figures | Varied sizes, on floor/shelf/bed |
| `stuffed_animal` (20) | Soft toys, plush animals (not teddy bear — that's class 12) | Group shots + individual |
| `picture_book` (21) | Children's books with colourful covers, thick-spined | Open and closed, on bed/floor/shelf |
| `crayon` (22) | Individual crayons and crayon boxes | On table or floor, scattered |
| `toy_car` (23) | Die-cast and plastic cars, trucks | Various sizes and colours |
| `puzzle_piece` (24) | Jigsaw puzzle pieces, on floor or table | Include partial puzzles |

**Variety Requirements (for robust model training):**
- [ ] **Lighting:** Mix of natural daylight + artificial room lighting + dim conditions
- [ ] **Distances:** Close-up (~30cm), mid-range (~1m), far (~2m+)
- [ ] **Backgrounds:** Carpet, hardwood floor, bed surface, table top, tiles
- [ ] **Clutter:** Some images with single objects, some with multiple objects overlapping
- [ ] **Co-occurrence:** Include Layer 1 objects (bed, sofa, chair) in the background — this helps the model learn context
- [ ] **Angles:** Directly above, 45° angle, eye-level

**Target count:** 50–100 images total (minimum 50 for a meaningful dataset).

### C3. Upload to Roboflow (DS-016)

1. Open `seesaw-layer3` project in Roboflow
2. Click **Upload Data** (or drag-and-drop area)
3. **Drag all images** from your iPhone export into the upload box
   - Transfer from iPhone → Mac via AirDrop, iCloud Photos, or cable
   - Supported formats: JPG, PNG, WEBP (iPhone default HEIC may need conversion)
   - **HEIC → JPG conversion** (if needed):
     ```bash
     # On Mac, in Terminal:
     cd ~/path/to/heic/images
     for f in *.HEIC; do sips -s format jpeg "$f" --out "${f%.HEIC}.jpg"; done
     ```
4. Click **Save and Continue**
5. Assign all images to the **Annotate** queue

### C4. Annotate with Bounding Boxes (DS-016)

**Open the Annotate interface:** Click any image in the Annotate queue.

**Keyboard Shortcuts (memorize these — they 10x your speed):**

| Shortcut | Action |
|----------|--------|
| `B` | Switch to **Bounding Box** tool |
| `V` | Switch to **Select/Move** tool |
| `→` / `←` | Next / Previous image |
| `Ctrl+S` / `⌘+S` | Save current annotations |
| `Delete` / `Backspace` | Delete selected annotation |
| `Ctrl+Z` / `⌘+Z` | Undo |
| `+` / `-` | Zoom in / out |

**Annotation Process per Image:**

1. **Press `B`** to enter bounding box mode
2. **Click and drag** tightly around each target object
3. **Type the class name** when the label popup appears — use exact names:
   - `building_blocks`
   - `dinosaur_toy`
   - `stuffed_animal`
   - `picture_book`
   - `crayon`
   - `toy_car`
   - `puzzle_piece`
4. **Also annotate any Layer 1/2 objects visible** in the scene (bed, sofa, book, teddy_bear, etc.) — this improves cross-layer generalisation
5. **Press `→`** to save and move to the next image

**Quality Rules:**
- Draw boxes **tightly** — no more than ~5% padding around the object
- If an object is **more than 50% occluded**, skip it
- If an object is **cut off at the image edge**, annotate the visible portion
- Target: **3–5 annotations per image** (some images may have more)
- Label every instance — if there are 4 crayons, draw 4 boxes

### C5. Speed Up with AI Labeling Tools

Roboflow offers several AI-powered tools to dramatically reduce annotation time:

#### Option 1: Label Assist (Recommended First)

Use a pre-trained model to auto-detect common objects:

1. Click the **magic wand icon** (🪄) in the annotation toolbar
2. In the popup, select the **Public Models** tab
3. Choose a COCO-trained model (detects 80 classes including `book`, `bottle`, `cup`, `teddy bear`)
4. Click **Continue**
5. **Select which classes to detect** — uncheck irrelevant ones
6. **Remap class names** if needed:
   - `teddy bear` → `teddy_bear`
   - `book` → `book` (or `picture_book` if it's clearly a children's book)
   - `bottle` → `bottle`
7. Click **Select Classes** — the model runs on each image as you navigate
8. **Review and adjust** — accept good predictions, delete bad ones, add missing annotations

#### Option 2: Auto Label (Bulk Annotation)

For labeling all images at once with a foundation model:

1. Go to your project → **Annotate** tab
2. Look for **Auto Label** option
3. Select **Grounding DINO** or similar foundation model
4. Enter text prompts for your objects:
   ```
   building blocks, toy dinosaur, stuffed animal, children's book, crayon, toy car, puzzle piece
   ```
5. Run across all images
6. **Review every image** — Auto Label gets ~60–80% accuracy, so manual correction is essential

> **Note:** AI Labeling features consume Roboflow credits. The free tier includes some credits — check your balance at [roboflow.com/credits](https://roboflow.com/credits).

#### Option 3: Smart Polygon (For Precise Boundaries)

Not typically needed for object detection (bounding boxes are sufficient), but useful if you later want segmentation data:
- Press the **cursor icon** in the sidebar → choose **Enhanced Smart Polygon**
- Click on an object → it auto-generates a polygon boundary
- The polygon is automatically converted to a bounding box for object detection projects

### C6. Review & Quality Check

Before generating a version, review your annotations:

1. Go to **Dataset** tab → **Health Check**
2. Check the **class balance chart** — each of your 7 classes should have:
   - **Minimum:** 30 annotations (absolute minimum for training)
   - **Target:** 50+ annotations (good)
   - **Ideal:** 100+ annotations (excellent)
3. If any class is severely underrepresented, return to step C2 and capture more images
4. **Browse random images** in the Annotate view — spot-check for:
   - Missing annotations (unlabelled objects)
   - Wrong class labels
   - Very loose bounding boxes
   - Duplicate images

**Class count targets for Layer 3:**

| Class | Min Annotations | Notes |
|-------|----------------|-------|
| `building_blocks` | 50+ | Easy to get many per image (piles) |
| `dinosaur_toy` | 30+ | Fewer per image, need variety |
| `stuffed_animal` | 50+ | Distinguish from teddy_bear (class 12) |
| `picture_book` | 50+ | Open + closed views |
| `crayon` | 30+ | Small objects — annotate each individual crayon |
| `toy_car` | 30+ | Various sizes |
| `puzzle_piece` | 30+ | Small, may be hard to detect |

### C7. Generate Version with Augmentations (DS-017)

Since Layer 3 is small (~50–100 source images), augmentation is critical to multiply effective training data.

1. Go to `seesaw-layer3` → **Versions** → **Generate New Version**

2. **Split:** 70% / 20% / 10% (Train / Valid / Test)

3. **Preprocessing:**
   | Setting | Value |
   |---------|-------|
   | Auto-Orient | ✅ ON |
   | Resize | Stretch to **640×640** |

4. **Augmentations (aggressive — small dataset needs more variety):**
   | Augmentation | Setting | Reason |
   |-------------|---------|--------|
   | Flip Horizontal | ✅ ON | Doubles data, safe for all objects |
   | Rotation | **±15°** | Simulates camera tilt in child's hand |
   | Brightness | **±20%** | Simulates dim bedrooms / bright playrooms |
   | Exposure | **±10%** | Additional lighting variation |
   | Blur | Up to **1.0px** | Simulates slight motion blur from movement |
   | Noise | Up to **1%** | Adds robustness to camera noise |
   | **Max Version Size** | **3x** | Triples 100 images → ~300 effective training samples |

5. Click **Generate** — processing takes 1–3 minutes

6. **Verify the output:**
   - Check the total image count: should be ~3x your source training images
   - Browse augmented samples — make sure bounding boxes still align with objects
   - Augmentations only apply to **Train** split (Valid/Test stay original)

### C8. Export Layer 3 (DS-017)

1. Click **Export** next to the generated version
2. Format: **YOLOv8**
3. Download or use Python code:
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("your-workspace").project("seesaw-layer3")
   version = project.version(1)
   dataset = version.download("yolov8", location="/content/seesaw-yolo-model/datasets/layer3")
   ```

4. **Verify export structure:**
   ```
   datasets/layer3/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

### C9. Write the Dataset Card (DS-018)

Create `datasets/layer3/DATASET_CARD.md` (for dissertation Chapter 4):

```markdown
# SeeSaw-ChildrensRoom-v1

## Description
Original object detection dataset capturing children's toys and bedroom objects
from an egocentric perspective (~1–1.5m height), designed for training YOLO11n
as part of the SeeSaw wearable AI companion system.

## Motivation
No existing public dataset provides child's-eye-level bounding box annotations
for common children's toys (building blocks, dinosaur toys, crayons, etc.).
This dataset fills that gap for the SeeSaw AiSee headset's scene understanding pipeline.

## Collection Method
- **Device:** iPhone [model], standard Camera app
- **Perspective:** Egocentric, ~1–1.5m from floor (child's eye level)
- **Environment:** Real children's bedroom/playroom, UK
- **Lighting:** Natural daylight + artificial room lighting
- **Date captured:** [DATE]

## Classes (7)
| ID | Class | Annotation Count |
|----|-------|-----------------|
| 18 | building_blocks | [COUNT] |
| 19 | dinosaur_toy | [COUNT] |
| 20 | stuffed_animal | [COUNT] |
| 21 | picture_book | [COUNT] |
| 22 | crayon | [COUNT] |
| 23 | toy_car | [COUNT] |
| 24 | puzzle_piece | [COUNT] |

## Statistics
- **Source images:** [COUNT]
- **Total annotations:** [COUNT]
- **Avg annotations per image:** [COUNT]
- **Augmented training images:** [COUNT] (3x augmentation)

## Annotation Tool
Roboflow (app.roboflow.com), bounding box annotations with Label Assist
for semi-automated labeling.

## Licence
CC BY 4.0 — original work by [YOUR NAME]

## Intended Use
Training YOLO11n for child's wearable AI scene understanding (SeeSaw project).
```

---

## Part D — Prepare for Merge (Pre DS-019)

After completing both Layer 2 and Layer 3 exports, you need to update `scripts/data_merge.py` with the actual class remapping.

### D1. Inspect Exported Label Files

```bash
# In Colab or locally — check what class IDs are in the exported labels
cd datasets/layer2/train/labels
head -5 *.txt | grep -oP '^\d+' | sort -n | uniq -c | sort -rn

cd ../../layer3/train/labels
head -5 *.txt | grep -oP '^\d+' | sort -n | uniq -c | sort -rn
```

### D2. Update Remap Dictionaries

Open `scripts/data_merge.py` and fill in the actual mappings based on what Roboflow exported:

```python
# Example — update these after inspecting exported labels
LAYER2_REMAP = {
    0: 12,  # teddy_bear
    1: 13,  # book
    2: 14,  # sports_ball
    3: 15,  # backpack
    4: 16,  # bottle
    5: 17,  # cup
}

LAYER3_REMAP = {
    0: 18,  # building_blocks
    1: 19,  # dinosaur_toy
    2: 20,  # stuffed_animal
    3: 21,  # picture_book
    4: 22,  # crayon
    5: 23,  # toy_car
    6: 24,  # puzzle_piece
}
```

> The actual source IDs (0, 1, 2...) depend on how Roboflow assigned them during export. Check the `data.yaml` file inside each exported zip — it lists the class-to-ID mapping.

### D3. Checklist Before Running data_merge.py

- [ ] `datasets/layer1/` exists with HomeObjects-3K images and labels
- [ ] `datasets/layer2/` exists with Roboflow export (YOLOv8 format)
- [ ] `datasets/layer3/` exists with Roboflow export (YOLOv8 format)
- [ ] `LAYER2_REMAP` in `data_merge.py` matches actual exported class IDs
- [ ] `LAYER3_REMAP` in `data_merge.py` matches actual exported class IDs
- [ ] All class names verified against `configs/seesaw_children.yaml`

---

## Time Estimates

| Task | Estimated Time | Notes |
|------|---------------|-------|
| Account setup (A1–A2) | 5 minutes | One-time |
| Clone Layer 2 datasets (B2–B4) | 15–20 minutes | Mostly waiting for imports |
| Inspect & remap classes (B5) | 15–20 minutes | Critical accuracy step |
| Generate & export Layer 2 (B6) | 10 minutes | Processing + download |
| Capture images — iPhone (C2) | 30–60 minutes | Aim for quality over quantity |
| Upload to Roboflow (C3) | 5–10 minutes | Drag-and-drop |
| Annotate with AI assist (C4–C5) | **60–120 minutes** | Main time investment |
| Generate & export Layer 3 (C7–C8) | 10 minutes | Processing + download |
| Write dataset card (C9) | 15 minutes | Reuse template above |
| **Total** | **~3–4 hours** | Can be split across sessions |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| HEIC images won't upload | Convert to JPG first (see C3 command) |
| Label Assist not finding objects | Try a different public model, or switch to Auto Label with text prompts |
| Class names differ after export | Use Roboflow's **Modify Classes** preprocessing to remap before export |
| Augmented images have misaligned boxes | Re-generate the version — this is rare but can happen with extreme rotations |
| Export zip missing `data.yaml` | Re-export — the YAML file should be at the root of the zip |
| Free tier credit limit reached | AI Labeling features use credits — switch to manual annotation if depleted |
| Images blurry after resize | Use "Fit (black edges) in" instead of "Stretch to" if aspect ratios differ dramatically |
