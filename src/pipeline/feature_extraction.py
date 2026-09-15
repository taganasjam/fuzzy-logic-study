"""Turn raw YOLOv8 detections for one image into the aggregated, normalized
features the fuzzy inference system expects. See docs/METHODOLOGY.md section 4.

Kept independent of any specific YOLO result object type: `detections` is a
plain list of `Detection` so this module can be unit-tested without
ultralytics installed, and `from_ultralytics_result` is a thin adapter for
real inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.fuzzy.infestation_fis import InfestationFeatures

BORE_HOLE = "bore_hole"
FRASS = "frass"
LARVA = "shoot_borer_larva"
INTERNAL_DAMAGE = "internal_infestation_damage"

# bore_hole/frass are small features relative to a whole-plant/fruit photo;
# internal_infestation_damage is a much larger visible region within a
# close-up, cut-open photo. Different scale factors bring both into a
# sensible 0-1 range before clipping. Retune once real detections are in hand.
SMALL_FEATURE_SCALE = 10.0
INTERNAL_DAMAGE_SCALE = 2.0


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

    bore_hole_dets = [d for d in detections if d.cls_name == BORE_HOLE]
    frass_dets = [d for d in detections if d.cls_name == FRASS]
    larva_dets = [d for d in detections if d.cls_name == LARVA]
    internal_damage_dets = [d for d in detections if d.cls_name == INTERNAL_DAMAGE]

    bore_hole_density = min(sum(_area(d) for d in bore_hole_dets) / image_area * SMALL_FEATURE_SCALE, 1.0)
    frass_density = min(sum(_area(d) for d in frass_dets) / image_area * SMALL_FEATURE_SCALE, 1.0)
    larva_confidence = max((d.confidence for d in larva_dets), default=0.0)
    internal_damage_ratio = min(
        sum(_area(d) for d in internal_damage_dets) / image_area * INTERNAL_DAMAGE_SCALE, 1.0
    )

    return InfestationFeatures(
        bore_hole_density=bore_hole_density,
        frass_density=frass_density,
        larva_confidence=larva_confidence,
        internal_damage_ratio=internal_damage_ratio,
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
