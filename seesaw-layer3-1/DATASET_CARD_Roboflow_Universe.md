# SeeSaw Layer 2 — Roboflow Universe Augmentation

## Description
Child-relevant object classes sourced from public Roboflow Universe datasets,
preprocessed and exported in YOLOv8 format for the SeeSaw training pipeline.

## Source Datasets
- **children** (universe.roboflow.com/project-odwld/children-u9om6) — CC BY 4.0
- **inside** (universe.roboflow.com/yolo-a91kx/inside-mpg5a) — CC BY 4.0
- Date accessed: 26 March 2026


## Classes (5)
| ID | Class | Annotations |
|----|-------|-------------|
| 0 | air-plane | 77 |
| 1 | cars | 166 |
| 2 | dinosaur | 84 |
| 3 | fire-truck | 42 |
| 4 | jeep | 5 |

## Statistics
- **Total images:** 240
- **Total annotations:** 374
- **Avg annotations per image:** 1.6

## Preprocessing
- Auto-Orient: ON
- Resize: Stretch to 640x640
- Augmentation: 3x (Flip H, Rotation ±10°, Brightness ±15%, Blur 0.5px)

## Licence
CC BY 4.0
