# Fuzzy Logic Study: Eggplant Shoot & Fruit Borer Detection

A hybrid **YOLOv8 + Fuzzy Logic** pipeline for detecting eggplant shoot and fruit
borer (*Leucinodes orbonalis*, EFSB) infestation from photos of external symptoms
(bore holes, frass, wilted "dead heart" shoots, damaged fruit), validated against
dissection-confirmed internal damage.

**Read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) first** — it explains the full
approach, how to structure your ~300–800 image dataset (including the 200
dissection/proof images), the YOLO class list, the fuzzy rule base, and the
evaluation plan.

## Why two stages?

1. A camera can only ever see *external* evidence of an *internal* pest. **YOLOv8**
   detects and localizes that external evidence (bore holes, frass, wilted shoots,
   damaged fruit, the adult moth) in a photo.
2. Whether an image really represents "infested" isn't a single sharp threshold —
   it depends on combining several graded, uncertain signals the way a plant
   pathologist would. A **Fuzzy Inference System** takes YOLO's aggregated output
   and produces an interpretable infestation score (0–100) and category (Not /
   Mildly / Moderately / Severely Infested).

The 200 "output" dissection images are the ground-truth proof set: destructive
sampling that confirms internal damage, used to label and calibrate the pipeline
rather than as detector input (you can't photograph inside a fruit non-destructively).

## Repo layout

```
docs/METHODOLOGY.md          Full write-up of the approach
configs/data.yaml            YOLOv8 dataset config (class list)
data/README.md               Expected folder layout for your photos
src/dataset/prepare_dataset.py   Stratified train/val/test split for labeled photos
src/yolo/train.py            Fine-tune YOLOv8 on external symptom photos
src/pipeline/feature_extraction.py   Raw YOLO detections -> fuzzy-ready features
src/fuzzy/infestation_fis.py         The fuzzy inference system (scikit-fuzzy)
src/pipeline/detect_and_infer.py     End-to-end: photo -> verdict
tests/test_fuzzy_system.py   Unit tests for the fuzzy system + feature extraction
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Label your external photos (Pool A) in YOLO format using the classes in
#    configs/data.yaml, then place them at data/external/images/all + labels/all.
python src/dataset/prepare_dataset.py --root data/external

# 2. Train the detector
python src/yolo/train.py --data configs/data.yaml --model yolov8n.pt --epochs 100

# 3. Run the full pipeline on a new photo
python src/pipeline/detect_and_infer.py \
    --weights runs/detect/efsb_yolov8/weights/best.pt \
    --source path/to/photo.jpg
```

Run the tests (no dataset needed — they cover the fuzzy logic and feature math):

```bash
python -m pytest tests/ -q
```
