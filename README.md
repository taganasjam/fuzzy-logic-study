# Fuzzy Logic Study: Eggplant Shoot & Fruit Borer Detection

A hybrid **YOLOv8 + Fuzzy Logic** pipeline for detecting eggplant shoot and fruit
borer (*Leucinodes orbonalis*, EFSB) infestation. YOLOv8 detects four classes —
`bore_hole`, `frass`, `shoot_borer_larva` (external symptoms) and
`internal_infestation_damage` (only visible in a dissection photo) — and a fuzzy
inference system combines them into an infestation verdict.

**Read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) first** — it explains the full
approach, how to structure your ~430 image dataset (250 external + 180 dissection
photos), the YOLO class list, the fuzzy rule base, and the evaluation plan.

## Why two stages?

1. A camera can only ever see *external* evidence of an *internal* pest. **YOLOv8**
   detects and localizes bore holes, frass, and the larva itself when visible — plus
   the internal damage in a dissection photo, when one exists.
2. Whether an image really represents "infested" isn't a single sharp threshold —
   it depends on combining several graded, uncertain signals the way a plant
   pathologist would. A **Fuzzy Inference System** takes YOLO's aggregated output
   and produces an interpretable infestation score (0–100) and category (Not /
   Mildly / Moderately / Severely Infested).

The 180 dissection images are the ground-truth proof set: destructive sampling that
confirms whether a specimen really was infested inside, used to label and calibrate
the pipeline (you can't photograph inside a fruit non-destructively). Include both
confirmed-infested and confirmed-clean dissections — you need both to calibrate.

## Repo layout

```
docs/METHODOLOGY.md          Full write-up of the approach
configs/data.yaml            YOLOv8 dataset config (class list)
data/README.md               Expected folder layout for your photos
src/dataset/prepare_dataset.py   Merges external + dissection pools into a stratified train/val/test split
src/yolo/train.py            Fine-tune YOLOv8 on the merged dataset
src/pipeline/feature_extraction.py   Raw YOLO detections -> fuzzy-ready features
src/fuzzy/infestation_fis.py         The fuzzy inference system (scikit-fuzzy)
src/pipeline/detect_and_infer.py     End-to-end: photo -> verdict
tests/test_fuzzy_system.py   Unit tests for the fuzzy system + feature extraction
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Label your photos in YOLO format using the classes in configs/data.yaml:
#      external photos      -> data/external/images/all + labels/all
#      dissection photos    -> data/internal_groundtruth/images/all + labels/all
# 2. Merge both pools into one train/val/test split for the unified detector
python src/dataset/prepare_dataset.py \
    --source external=data/external \
    --source internal_groundtruth=data/internal_groundtruth \
    --out data/yolo_dataset

# 3. Train the detector
python src/yolo/train.py --data configs/data.yaml --model yolov8n.pt --epochs 100

# 4. Run the full pipeline on a new photo
python src/pipeline/detect_and_infer.py \
    --weights runs/detect/efsb_yolov8/weights/best.pt \
    --source path/to/photo.jpg
```

Run the tests (no dataset needed — they cover the fuzzy logic and feature math):

```bash
python -m pytest tests/ -q
```
