"""Split labeled YOLO-format images (Pool A - external symptom photos) into
train/val/test, stratified by which symptom classes appear in each image so
rarer classes (bore_hole, frass, wilted_shoot) aren't concentrated in one split.

Expects, before running:
    data/external/images/all/*.jpg
    data/external/labels/all/*.txt   (YOLO format, same basename as image)

Produces:
    data/external/images/{train,val,test}/*.jpg
    data/external/labels/{train,val,test}/*.txt

Usage:
    python src/dataset/prepare_dataset.py --root data/external --train 0.7 --val 0.2 --test 0.1
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _primary_class(label_path: Path) -> int:
    """Use the rarest class present in the label file as the stratification key
    (falls back to -1 for unlabeled/background images)."""
    if not label_path.exists():
        return -1
    class_ids = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        class_ids.append(int(line.split()[0]))
    return min(class_ids) if class_ids else -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/external")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("train + val + test must sum to 1.0")

    root = Path(args.root)
    images_all = root / "images" / "all"
    labels_all = root / "labels" / "all"
    if not images_all.is_dir():
        raise SystemExit(f"Expected labeled images at {images_all} - see this script's docstring.")

    images = sorted(p for p in images_all.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {images_all}")

    by_class: dict[int, list[Path]] = defaultdict(list)
    for img in images:
        label_path = labels_all / (img.stem + ".txt")
        by_class[_primary_class(label_path)].append(img)

    rng = random.Random(args.seed)
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for _cls, imgs in by_class.items():
        imgs = imgs[:]
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = round(n * args.train)
        n_val = round(n * args.val)
        splits["train"].extend(imgs[:n_train])
        splits["val"].extend(imgs[n_train : n_train + n_val])
        splits["test"].extend(imgs[n_train + n_val :])

    for split_name, imgs in splits.items():
        img_dir = root / "images" / split_name
        lbl_dir = root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, img_dir / img.name)
            label_src = labels_all / (img.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, lbl_dir / label_src.name)
        print(f"{split_name}: {len(imgs)} images")


if __name__ == "__main__":
    main()
