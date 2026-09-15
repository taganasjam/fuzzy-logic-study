"""Fine-tune YOLOv8 on the merged eggplant EFSB dataset (external + dissection pools,
produced by src/dataset/prepare_dataset.py).

Usage:
    python src/yolo/train.py --data configs/data.yaml --model yolov8n.pt --epochs 100

With only a few hundred images, start with yolov8n or yolov8s, and lean on the
default augmentation ultralytics applies (mosaic, hsv jitter, flips) rather than
disabling it.
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="Use 960+ if bore_hole/frass are being missed")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="efsb_yolov8")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
