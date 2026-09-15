"""Merge labeled YOLO-format images from one or more source pools into a single
train/val/test split for the unified YOLOv8 detector.

We have two pools that never overlap in which classes they can contain:
  - external field photos    -> may contain bore_hole / frass / shoot_borer_larva
  - dissection photos        -> may contain internal_infestation_damage
                                 (a non-infested dissection has an empty/missing
                                 label file - a valid negative example)
Both still need to go through the same YOLO training split, so this script
takes one or more named source pools and merges them.

Each source pool must look like:
    <source>/images/all/*.jpg
    <source>/labels/all/*.txt   (YOLO format; may be empty or missing)

Produces, under --out:
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt

Output filenames are prefixed with "<pool_name>__" to avoid collisions between
pools that were photographed/named independently.

Usage:
    python src/dataset/prepare_dataset.py \\
        --source external=data/external \\
        --source internal_groundtruth=data/internal_groundtruth \\
        --out data/yolo_dataset
"""

import argparse
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class PoolImage:
    pool_name: str
    image_path: Path
    label_path: Path


def _parse_source(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--source must be name=path, got: {spec}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def _primary_class(label_path: Path) -> int:
    """Use the rarest (lowest-id) class present in the label file as the
    stratification key (falls back to -1 for unlabeled/negative images)."""
    if not label_path.exists():
        return -1
    class_ids = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        class_ids.append(int(line.split()[0]))
    return min(class_ids) if class_ids else -1


def _collect_pool(name: str, root: Path) -> list[PoolImage]:
    images_all = root / "images" / "all"
    labels_all = root / "labels" / "all"
    if not images_all.is_dir():
        raise SystemExit(f"Expected labeled images at {images_all} - see this script's docstring.")

    images = sorted(p for p in images_all.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [PoolImage(name, img, labels_all / (img.stem + ".txt")) for img in images]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="A source pool, e.g. external=data/external. Repeat for multiple pools.",
    )
    parser.add_argument("--out", default="data/yolo_dataset")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("train + val + test must sum to 1.0")

    pool_images: list[PoolImage] = []
    for spec in args.source:
        name, path = _parse_source(spec)
        pool_images.extend(_collect_pool(name, path))

    if not pool_images:
        raise SystemExit("No images found across the given --source pools.")

    by_class: dict[int, list[PoolImage]] = defaultdict(list)
    for item in pool_images:
        by_class[_primary_class(item.label_path)].append(item)

    rng = random.Random(args.seed)
    splits: dict[str, list[PoolImage]] = {"train": [], "val": [], "test": []}

    for _cls, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * args.train)
        n_val = round(n * args.val)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    out_root = Path(args.out)
    for split_name, items in splits.items():
        img_dir = out_root / "images" / split_name
        lbl_dir = out_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            out_stem = f"{item.pool_name}__{item.image_path.stem}"
            shutil.copy2(item.image_path, img_dir / f"{out_stem}{item.image_path.suffix}")
            if item.label_path.exists():
                shutil.copy2(item.label_path, lbl_dir / f"{out_stem}.txt")
        print(f"{split_name}: {len(items)} images")


if __name__ == "__main__":
    main()
