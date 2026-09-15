"""Turn raw YOLOv8 detections for one image into the aggregated, normalized
features the fuzzy inference system expects. See docs/METHODOLOGY.md section 4.

Kept independent of any specific YOLO result object type: `boxes` is a plain
list of dicts so this module can be unit-tested without ultralytics installed,
and `from_ultralytics_result` is a thin adapter for real inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.fuzzy.infestation_fis import InfestationFeatures

SYMPTOM_CLASSES = {"bore_hole", "frass"}
SHOOT_CLASSES = {"healthy_shoot", "wilted_shoot"}
FRUIT_CLASSES = {"healthy_fruit", "damaged_fruit"}


@dataclass
class Detection:
    cls_name: str
    confidence: float
    box_area: float  # pixels^2, or any consistent unit


def _area(det: Detection) -> float:
    return max(det.box_area, 0.0)


def extract_features(detections: list[Detection], image_area: float) -> InfestationFeatures:
    if image_area <= 0:
        raise ValueError("image_area must be positive")

    symptom_dets = [d for d in detections if d.cls_name in SYMPTOM_CLASSES]
    shoot_dets = [d for d in detections if d.cls_name in SHOOT_CLASSES]
    fruit_dets = [d for d in detections if d.cls_name in FRUIT_CLASSES]

    symptom_area = sum(_area(d) for d in symptom_dets)
    symptom_density = min(symptom_area / image_area * 10.0, 1.0)  # scaled: symptoms are small

    avg_confidence = (
        sum(d.confidence for d in symptom_dets) / len(symptom_dets) if symptom_dets else 0.0
    )

    wilted_area = sum(_area(d) for d in shoot_dets if d.cls_name == "wilted_shoot")
    total_shoot_area = sum(_area(d) for d in shoot_dets)
    wilt_ratio = (wilted_area / total_shoot_area) if total_shoot_area > 0 else 0.0

    damaged_count = sum(1 for d in fruit_dets if d.cls_name == "damaged_fruit")
    damaged_fruit_ratio = (damaged_count / len(fruit_dets)) if fruit_dets else 0.0

    return InfestationFeatures(
        symptom_density=symptom_density,
        avg_confidence=avg_confidence,
        wilt_ratio=wilt_ratio,
        damaged_fruit_ratio=damaged_fruit_ratio,
    )


def from_ultralytics_result(result) -> InfestationFeatures:  # pragma: no cover - needs ultralytics
    """Adapter: build features directly from an ultralytics `Results` object."""
    names = result.names
    h, w = result.orig_shape
    image_area = float(h * w)

    detections: list[Detection] = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        detections.append(Detection(cls_name=names[cls_id], confidence=conf, box_area=area))

    return extract_features(detections, image_area)
