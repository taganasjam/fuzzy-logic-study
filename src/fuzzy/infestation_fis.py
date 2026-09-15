"""Fuzzy Inference System (FIS) that turns aggregated YOLOv8 detection features
into an eggplant shoot-and-fruit-borer (EFSB) infestation verdict.

Inputs (all normalized 0-1, see docs/METHODOLOGY.md section 4 for how to compute
them from raw YOLO detections):
    symptom_density      - bore_hole/frass detections per unit plant/fruit area
    avg_confidence       - mean YOLO confidence across symptom detections
    wilt_ratio           - fraction of shoot area classified as wilted
    damaged_fruit_ratio  - fraction of visible fruit classified as damaged

Output:
    infestation_score (0-100), plus a categorical label.

The membership function breakpoints below are reasonable defaults for a first
pass; calibrate them against the dissection-confirmed ground truth (Pool B,
the "200 output images") once real data is available - see calibrate.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass
class InfestationFeatures:
    symptom_density: float
    avg_confidence: float
    wilt_ratio: float
    damaged_fruit_ratio: float

    def clipped(self) -> "InfestationFeatures":
        return InfestationFeatures(
            symptom_density=float(np.clip(self.symptom_density, 0.0, 1.0)),
            avg_confidence=float(np.clip(self.avg_confidence, 0.0, 1.0)),
            wilt_ratio=float(np.clip(self.wilt_ratio, 0.0, 1.0)),
            damaged_fruit_ratio=float(np.clip(self.damaged_fruit_ratio, 0.0, 1.0)),
        )


LABEL_BUCKETS = (
    (25.0, "Not Infested"),
    (50.0, "Mildly Infested"),
    (75.0, "Moderately Infested"),
    (100.01, "Severely Infested"),
)


def score_to_label(score: float) -> str:
    for upper, label in LABEL_BUCKETS:
        if score < upper:
            return label
    return LABEL_BUCKETS[-1][1]


def _build_control_system() -> ctrl.ControlSystem:
    symptom_density = ctrl.Antecedent(np.linspace(0, 1, 101), "symptom_density")
    avg_confidence = ctrl.Antecedent(np.linspace(0, 1, 101), "avg_confidence")
    wilt_ratio = ctrl.Antecedent(np.linspace(0, 1, 101), "wilt_ratio")
    damaged_fruit_ratio = ctrl.Antecedent(np.linspace(0, 1, 101), "damaged_fruit_ratio")
    infestation_score = ctrl.Consequent(np.linspace(0, 100, 101), "infestation_score")

    for var in (symptom_density, avg_confidence, wilt_ratio, damaged_fruit_ratio):
        var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 0.4])
        var["medium"] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
        var["high"] = fuzz.trimf(var.universe, [0.6, 1.0, 1.0])

    infestation_score["low"] = fuzz.trimf(infestation_score.universe, [0, 0, 40])
    infestation_score["medium"] = fuzz.trimf(infestation_score.universe, [20, 50, 80])
    infestation_score["high"] = fuzz.trimf(infestation_score.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(symptom_density["high"] & avg_confidence["high"], infestation_score["high"]),
        ctrl.Rule(wilt_ratio["high"], infestation_score["high"]),
        ctrl.Rule(damaged_fruit_ratio["high"] & avg_confidence["high"], infestation_score["high"]),
        ctrl.Rule(
            symptom_density["low"] & wilt_ratio["low"] & damaged_fruit_ratio["low"],
            infestation_score["low"],
        ),
        ctrl.Rule(avg_confidence["low"] & symptom_density["low"], infestation_score["low"]),
        ctrl.Rule(symptom_density["medium"] | damaged_fruit_ratio["medium"], infestation_score["medium"]),
        ctrl.Rule(wilt_ratio["medium"] & avg_confidence["medium"], infestation_score["medium"]),
        ctrl.Rule(
            symptom_density["high"] & avg_confidence["low"],
            infestation_score["medium"],
        ),
    ]

    return ctrl.ControlSystem(rules)


class InfestationFIS:
    """Wraps the scikit-fuzzy control system; safe to reuse across many images
    (creates a fresh ControlSystemSimulation per call to avoid stale state)."""

    def __init__(self) -> None:
        self._system = _build_control_system()

    def infer(self, features: InfestationFeatures) -> tuple[float, str]:
        features = features.clipped()
        sim = ctrl.ControlSystemSimulation(self._system)
        sim.input["symptom_density"] = features.symptom_density
        sim.input["avg_confidence"] = features.avg_confidence
        sim.input["wilt_ratio"] = features.wilt_ratio
        sim.input["damaged_fruit_ratio"] = features.damaged_fruit_ratio
        sim.compute()
        score = float(sim.output["infestation_score"])
        return score, score_to_label(score)


if __name__ == "__main__":
    fis = InfestationFIS()
    examples = [
        InfestationFeatures(0.0, 0.0, 0.0, 0.0),
        InfestationFeatures(0.15, 0.5, 0.1, 0.1),
        InfestationFeatures(0.5, 0.7, 0.4, 0.3),
        InfestationFeatures(0.9, 0.95, 0.8, 0.7),
    ]
    for ex in examples:
        score, label = fis.infer(ex)
        print(f"{ex} -> score={score:.1f} label={label}")
