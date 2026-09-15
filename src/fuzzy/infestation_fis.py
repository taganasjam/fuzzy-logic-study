"""Fuzzy Inference System (FIS) that turns aggregated YOLOv8 detection features
into an eggplant shoot-and-fruit-borer (EFSB) infestation verdict.

Inputs map 1:1 onto the four YOLO classes (see docs/METHODOLOGY.md section 4 for
how to compute them from raw detections):
    bore_hole_density     - bore_hole detections, normalized by plant/fruit area
    frass_density         - frass detections, normalized by plant/fruit area
    larva_confidence      - max confidence of any shoot_borer_larva detection
                            (finding the larva itself is near-certain proof)
    internal_damage_ratio - area of internal_infestation_damage detections
                            (only non-zero when scoring a dissection photo; 0.0
                            for a normal field photo where nothing has been cut
                            open, since that class can't appear there)

Output:
    infestation_score (0-100), plus a categorical label.

The membership function breakpoints below are reasonable defaults for a first
pass; calibrate them against the dissection-confirmed ground truth (the 180
dissection images) once real data is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass
class InfestationFeatures:
    bore_hole_density: float
    frass_density: float
    larva_confidence: float
    internal_damage_ratio: float = 0.0

    def clipped(self) -> "InfestationFeatures":
        return InfestationFeatures(
            bore_hole_density=float(np.clip(self.bore_hole_density, 0.0, 1.0)),
            frass_density=float(np.clip(self.frass_density, 0.0, 1.0)),
            larva_confidence=float(np.clip(self.larva_confidence, 0.0, 1.0)),
            internal_damage_ratio=float(np.clip(self.internal_damage_ratio, 0.0, 1.0)),
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
    bore_hole_density = ctrl.Antecedent(np.linspace(0, 1, 101), "bore_hole_density")
    frass_density = ctrl.Antecedent(np.linspace(0, 1, 101), "frass_density")
    larva_confidence = ctrl.Antecedent(np.linspace(0, 1, 101), "larva_confidence")
    internal_damage_ratio = ctrl.Antecedent(np.linspace(0, 1, 101), "internal_damage_ratio")
    infestation_score = ctrl.Consequent(np.linspace(0, 100, 101), "infestation_score")

    for var in (bore_hole_density, frass_density, larva_confidence, internal_damage_ratio):
        var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 0.4])
        var["medium"] = fuzz.trimf(var.universe, [0.2, 0.5, 0.8])
        var["high"] = fuzz.trimf(var.universe, [0.6, 1.0, 1.0])

    infestation_score["low"] = fuzz.trimf(infestation_score.universe, [0, 0, 40])
    infestation_score["medium"] = fuzz.trimf(infestation_score.universe, [20, 50, 80])
    infestation_score["high"] = fuzz.trimf(infestation_score.universe, [60, 100, 100])

    rules = [
        # Direct evidence: the larva itself, or dissection-confirmed internal
        # damage, is near-certain proof on its own.
        ctrl.Rule(larva_confidence["high"], infestation_score["high"]),
        ctrl.Rule(internal_damage_ratio["high"], infestation_score["high"]),
        # Bore holes + frass together are strong circumstantial evidence.
        ctrl.Rule(bore_hole_density["high"] & frass_density["high"], infestation_score["high"]),
        # A hole with no frass and no larva is weaker (could be old/inactive/mechanical damage).
        ctrl.Rule(
            bore_hole_density["high"] & frass_density["low"] & larva_confidence["low"],
            infestation_score["medium"],
        ),
        # Nothing detected anywhere -> not infested.
        ctrl.Rule(
            bore_hole_density["low"]
            & frass_density["low"]
            & larva_confidence["low"]
            & internal_damage_ratio["low"],
            infestation_score["low"],
        ),
        # Medium-strength individual signals.
        ctrl.Rule(bore_hole_density["medium"] | frass_density["medium"], infestation_score["medium"]),
        ctrl.Rule(larva_confidence["medium"], infestation_score["medium"]),
        ctrl.Rule(internal_damage_ratio["medium"], infestation_score["medium"]),
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
        sim.input["bore_hole_density"] = features.bore_hole_density
        sim.input["frass_density"] = features.frass_density
        sim.input["larva_confidence"] = features.larva_confidence
        sim.input["internal_damage_ratio"] = features.internal_damage_ratio
        sim.compute()
        score = float(sim.output["infestation_score"])
        return score, score_to_label(score)


if __name__ == "__main__":
    fis = InfestationFIS()
    examples = [
        InfestationFeatures(0.0, 0.0, 0.0, 0.0),
        InfestationFeatures(0.15, 0.1, 0.0, 0.0),
        InfestationFeatures(0.5, 0.4, 0.3, 0.0),
        InfestationFeatures(0.9, 0.85, 0.95, 0.0),
        InfestationFeatures(0.0, 0.0, 0.0, 0.9),  # dissection photo, heavy internal damage
    ]
    for ex in examples:
        score, label = fis.infer(ex)
        print(f"{ex} -> score={score:.1f} label={label}")
