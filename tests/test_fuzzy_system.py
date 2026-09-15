from src.fuzzy.infestation_fis import InfestationFeatures, InfestationFIS, score_to_label
from src.pipeline.feature_extraction import Detection, extract_features


def test_no_symptoms_is_not_infested():
    fis = InfestationFIS()
    score, label = fis.infer(InfestationFeatures(0.0, 0.0, 0.0, 0.0))
    assert label == "Not Infested"
    assert score < 25


def test_heavy_external_symptoms_is_severely_infested():
    fis = InfestationFIS()
    score, label = fis.infer(InfestationFeatures(0.9, 0.9, 0.9, 0.0))
    assert label == "Severely Infested"
    assert score > 75


def test_larva_found_alone_is_strong_evidence():
    fis = InfestationFIS()
    score, label = fis.infer(InfestationFeatures(0.0, 0.0, 0.95, 0.0))
    assert label in ("Moderately Infested", "Severely Infested")


def test_dissection_confirmed_internal_damage_alone_is_strong_evidence():
    fis = InfestationFIS()
    score, label = fis.infer(InfestationFeatures(0.0, 0.0, 0.0, 0.9))
    assert label in ("Moderately Infested", "Severely Infested")


def test_more_symptoms_never_decreases_score():
    fis = InfestationFIS()
    low_score, _ = fis.infer(InfestationFeatures(0.1, 0.1, 0.0, 0.0))
    high_score, _ = fis.infer(InfestationFeatures(0.9, 0.9, 0.9, 0.9))
    assert high_score >= low_score


def test_score_to_label_buckets():
    assert score_to_label(0) == "Not Infested"
    assert score_to_label(24.9) == "Not Infested"
    assert score_to_label(49.9) == "Mildly Infested"
    assert score_to_label(74.9) == "Moderately Infested"
    assert score_to_label(100) == "Severely Infested"


def test_feature_extraction_no_detections():
    features = extract_features([], image_area=1000.0)
    assert features.bore_hole_density == 0.0
    assert features.frass_density == 0.0
    assert features.larva_confidence == 0.0
    assert features.internal_damage_ratio == 0.0


def test_feature_extraction_external_symptoms():
    detections = [
        Detection("bore_hole", confidence=0.9, box_area=50.0),
        Detection("frass", confidence=0.7, box_area=30.0),
        Detection("shoot_borer_larva", confidence=0.6, box_area=20.0),
    ]
    features = extract_features(detections, image_area=2000.0)
    assert features.larva_confidence == 0.6
    assert features.internal_damage_ratio == 0.0
    assert 0.0 < features.bore_hole_density <= 1.0
    assert 0.0 < features.frass_density <= 1.0


def test_feature_extraction_dissection_photo():
    detections = [Detection("internal_infestation_damage", confidence=0.95, box_area=400.0)]
    features = extract_features(detections, image_area=1000.0)
    assert features.bore_hole_density == 0.0
    assert features.frass_density == 0.0
    assert features.larva_confidence == 0.0
    assert features.internal_damage_ratio == min(400.0 / 1000.0 * 2.0, 1.0)


def test_feature_extraction_multiple_larvae_uses_max_confidence():
    detections = [
        Detection("shoot_borer_larva", confidence=0.4, box_area=10.0),
        Detection("shoot_borer_larva", confidence=0.85, box_area=15.0),
    ]
    features = extract_features(detections, image_area=1000.0)
    assert features.larva_confidence == 0.85
