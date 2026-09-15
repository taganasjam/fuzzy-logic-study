from src.fuzzy.infestation_fis import InfestationFeatures, InfestationFIS, score_to_label
from src.pipeline.feature_extraction import Detection, extract_features


def test_no_symptoms_is_not_infested():
    fis = InfestationFIS()
    score, label = fis.infer(InfestationFeatures(0.0, 0.0, 0.0, 0.0))
    assert label == "Not Infested"
    assert score < 25


def test_heavy_symptoms_is_severely_infested():
    fis = InfestationFIS()
    score, label = fis.infer(InfestationFeatures(1.0, 1.0, 1.0, 1.0))
    assert label == "Severely Infested"
    assert score > 75


def test_more_symptoms_never_decreases_score():
    fis = InfestationFIS()
    low_score, _ = fis.infer(InfestationFeatures(0.1, 0.5, 0.1, 0.1))
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
    assert features.symptom_density == 0.0
    assert features.avg_confidence == 0.0
    assert features.wilt_ratio == 0.0
    assert features.damaged_fruit_ratio == 0.0


def test_feature_extraction_with_symptoms():
    detections = [
        Detection("bore_hole", confidence=0.9, box_area=50.0),
        Detection("frass", confidence=0.7, box_area=30.0),
        Detection("wilted_shoot", confidence=0.8, box_area=400.0),
        Detection("healthy_shoot", confidence=0.6, box_area=600.0),
        Detection("damaged_fruit", confidence=0.85, box_area=200.0),
        Detection("healthy_fruit", confidence=0.9, box_area=180.0),
    ]
    features = extract_features(detections, image_area=2000.0)
    assert features.avg_confidence == (0.9 + 0.7) / 2
    assert features.wilt_ratio == 400.0 / (400.0 + 600.0)
    assert features.damaged_fruit_ratio == 1 / 2
    assert 0.0 < features.symptom_density <= 1.0
