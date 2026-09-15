"""End-to-end: photo -> YOLOv8 detections -> features -> Fuzzy Inference System -> verdict.

Usage:
    python src/pipeline/detect_and_infer.py --weights runs/detect/efsb_yolov8/weights/best.pt \\
        --source path/to/photo.jpg
"""

import argparse

from ultralytics import YOLO

from src.fuzzy.infestation_fis import InfestationFIS
from src.pipeline.feature_extraction import from_ultralytics_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to a trained YOLOv8 .pt checkpoint")
    parser.add_argument("--source", required=True, help="Image path, folder, or glob")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    args = parser.parse_args()

    model = YOLO(args.weights)
    fis = InfestationFIS()

    results = model.predict(source=args.source, conf=args.conf, verbose=False)
    for result in results:
        features = from_ultralytics_result(result)
        score, label = fis.infer(features)
        print(f"{result.path}: score={score:.1f}/100 -> {label}  (features={features})")


if __name__ == "__main__":
    main()
