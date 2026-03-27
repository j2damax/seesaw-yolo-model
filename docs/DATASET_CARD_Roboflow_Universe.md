# SeeSaw Layer 2 — Roboflow Universe Augmentation

## Description
Child-relevant object classes sourced from public Roboflow Universe datasets,
preprocessed and exported in YOLOv8 format for the SeeSaw training pipeline.

## Source Datasets
- **children** (universe.roboflow.com/project-odwld/children-u9om6) — CC BY 4.0
- **inside** (universe.roboflow.com/yolo-a91kx/inside-mpg5a) — CC BY 4.0
- Date accessed: 26 March 2026


## Classes (33)
| ID | Class | Annotations |
|----|-------|-------------|
| 0 | bed | 91 |
| 1 | book | 6 |
| 2 | carpet | 31 |
| 3 | chair | 126 |
| 4 | chimni | 8 |
| 5 | clock | 12 |
| 6 | crib | 23 |
| 7 | cupboard | 205 |
| 8 | curtains | 159 |
| 9 | door | 63 |
| 10 | faucet | 45 |
| 11 | floor-decor | 8 |
| 12 | glass | 16 |
| 13 | indoor-plant | 40 |
| 14 | lamps | 170 |
| 15 | light | 15 |
| 16 | pillows | 122 |
| 17 | plant | 13 |
| 18 | plants | 157 |
| 19 | pots | 130 |
| 20 | rugs | 106 |
| 21 | shelf | 10 |
| 22 | shelves | 41 |
| 23 | sofa | 101 |
| 24 | stairs | 14 |
| 25 | storage | 2 |
| 26 | table | 180 |
| 27 | table-lamp | 71 |
| 28 | tables | 17 |
| 29 | television | 42 |
| 30 | white-board | 12 |
| 31 | window | 108 |
| 32 | windows | 214 |

## Statistics
- **Total images:** 354
- **Total annotations:** 2358
- **Avg annotations per image:** 6.7

## Preprocessing
- Auto-Orient: ON
- Resize: Stretch to 640x640
- Augmentation: 3x (Flip H, Rotation ±10°, Brightness ±15%, Blur 0.5px)

## Licence
CC BY 4.0
