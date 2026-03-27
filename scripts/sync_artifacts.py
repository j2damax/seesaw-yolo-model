#!/usr/bin/env python3
"""Sync required SeeSaw YOLO artifacts into this repository.

Use this after running training in Colab (or from a downloaded run folder) to copy
all required artifacts into the local repo so everything is version-controlled.

COLAB USAGE (recommended — auto-detects environment):
  !python scripts/sync_artifacts.py --overwrite

LOCAL/MANUAL USAGE (specify sources explicitly):
  python scripts/sync_artifacts.py \
    --source /content/seesaw-yolo-model \
    --source /content/drive/MyDrive/seesaw-yolo-runs \
    --overwrite

  python scripts/sync_artifacts.py \
    --source ~/Downloads/seesaw-colab-export \
    --overwrite

ARTIFACT POLICY:
  - docs/dissertation_figures/*.png, .jpg  [REQUIRED] — dissertation
  - docs/results_comparison.csv             [REQUIRED] — metrics table
  - export/seesaw-yolo11n.mlpackage         [REQUIRED] — iOS model
  - datasets/seesaw_children                 [OPTIONAL] — large; use --include-dataset to sync
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Artifact:
    destination: str
    candidates: tuple[str, ...]
    required: bool = True


ARTIFACTS: tuple[Artifact, ...] = (
    # Dataset (optional — large; use --include-dataset to sync)
    Artifact(
        destination="datasets/seesaw_children",
        candidates=(
            "datasets/seesaw_children",
            "seesaw-yolo-model/datasets/seesaw_children",
            "seesaw_children",
        ),
        required=False,  # Large; only include if explicitly requested
    ),
    # Core evaluation outputs
    Artifact(
        destination="docs/results_comparison.csv",
        candidates=(
            "docs/results_comparison.csv",
            "results_comparison.csv",
            "seesaw-yolo-model/docs/results_comparison.csv",
        ),
        required=True,
    ),
    Artifact(
        destination="docs/dissertation_figures/class_distribution.png",
        candidates=(
            "docs/dissertation_figures/class_distribution.png",
            "dissertation_figures/class_distribution.png",
            "seesaw-yolo-model/docs/dissertation_figures/class_distribution.png",
        ),
        required=True,
    ),
    # Run C figure set for dissertation
    Artifact(
        destination="docs/dissertation_figures/results_run_c_training_curves.png",
        candidates=(
            "runs/detect/run_c_all_layers/results.png",
            "run_c_all_layers/results.png",
            "dissertation_figures/results_run_c_training_curves.png",
        ),
        required=True,
    ),
    Artifact(
        destination="docs/dissertation_figures/confusion_matrix_run_c.png",
        candidates=(
            "runs/detect/run_c_all_layers/confusion_matrix_normalized.png",
            "run_c_all_layers/confusion_matrix_normalized.png",
            "dissertation_figures/confusion_matrix_run_c.png",
        ),
        required=True,
    ),
    Artifact(
        destination="docs/dissertation_figures/val_predictions_run_c.jpg",
        candidates=(
            "runs/detect/run_c_all_layers/val_batch0_pred.jpg",
            "run_c_all_layers/val_batch0_pred.jpg",
            "dissertation_figures/val_predictions_run_c.jpg",
        ),
        required=True,
    ),
    # Optional but useful figure
    Artifact(
        destination="docs/dissertation_figures/labels_distribution_run_c.jpg",
        candidates=(
            "runs/detect/run_c_all_layers/labels.jpg",
            "run_c_all_layers/labels.jpg",
            "dissertation_figures/labels_distribution_run_c.jpg",
        ),
        required=False,
    ),
    # CoreML export
    Artifact(
        destination="export/seesaw-yolo11n.mlpackage",
        candidates=(
            "export/seesaw-yolo11n.mlpackage",
            "seesaw-yolo11n.mlpackage",
            "seesaw-yolo-model/export/seesaw-yolo11n.mlpackage",
        ),
        required=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy required SeeSaw YOLO artifacts into this repo")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        help="Artifact source root. Provide multiple --source entries to search in order. "
             "Auto-detected in Colab if not provided.",
    )
    parser.add_argument(
        "--include-dataset",
        action="store_true",
        help="Also sync datasets/seesaw_children (large; ~200+ MB). By default, only core artifacts are synced.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files/directories at destination",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without copying",
    )
    return parser.parse_args()


def existing_candidate(source_roots: Iterable[Path], candidate_rel: str) -> Path | None:
    for src_root in source_roots:
        candidate = src_root / candidate_rel
        if candidate.exists():
            return candidate
    return None


def copy_path(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> str:
    if dst.exists():
        if not overwrite:
            return "SKIP (exists)"
        if not dry_run:
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    return "COPIED"


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    
    # Auto-detect Colab environment and set default sources
    in_colab = Path("/content").exists()
    
    if args.source is None:
        if in_colab:
            # Colab default: search repo, then Drive artifacts
            args.source = [
                Path("/content/seesaw-yolo-model"),
                Path("/content/drive/MyDrive/seesaw-yolo-runs"),
            ]
            print("[Colab detected] Using default source roots")
        else:
            print("ERROR: No --source provided and not in Colab. Specify --source or run in Colab.")
            return 1
    
    source_roots = [s.expanduser().resolve() for s in args.source]

    print(f"Repo root: {repo_root}")
    print(f"Source roots (search order):")
    for s in source_roots:
        exists = "✓" if s.exists() else "✗"
        print(f"  {exists} {s}")
    
    # Filter artifacts: exclude dataset unless --include-dataset
    artifacts = ARTIFACTS
    if not args.include_dataset:
        artifacts = tuple(a for a in artifacts if a.destination != "datasets/seesaw_children")
        print(f"\nIncluded artifacts: {len(artifacts)} (excluding {len(ARTIFACTS) - len(artifacts)} optional)")
    else:
        print(f"\nIncluded artifacts: {len(artifacts)} (all)")

    missing_required = 0
    copied = 0
    skipped = 0

    for artifact in artifacts:
        src = None
        for rel in artifact.candidates:
            src = existing_candidate(source_roots, rel)
            if src is not None:
                break

        dst = repo_root / artifact.destination

        if src is None:
            level = "REQUIRED" if artifact.required else "OPTIONAL"
            print(f"MISS [{level}] {artifact.destination}")
            if artifact.required:
                missing_required += 1
            continue

        result = copy_path(src, dst, overwrite=args.overwrite, dry_run=args.dry_run)
        if result == "COPIED":
            copied += 1
        else:
            skipped += 1
        print(f"{result}: {src} → {dst}")

    print("\n" + "=" * 70)
    print("Summary")
    print(f"  Copied:           {copied}")
    print(f"  Skipped (exist):  {skipped}")
    print(f"  Missing required: {missing_required}")
    
    if args.dry_run:
        print("  (dry-run: no files were actually copied)")

    if missing_required:
        print("\nSome required artifacts are still missing.")
        if not args.include_dataset:
            print("  Use --include-dataset to sync the full merged dataset.")
        print("  Add another --source path or check artifact locations.")
        return 2

    if copied == 0 and not args.dry_run:
        print("\n✓ All required artifacts already exist in repository.")
    elif copied > 0:
        print("\n✓ Artifacts synced successfully.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
