# Methodology: Hybrid YOLOv8 + Fuzzy Logic for Eggplant Shoot and Fruit Borer (EFSB) Detection

## 1. The problem in plain terms

The eggplant shoot and fruit borer, *Leucinodes orbonalis*, does its damage **inside** the
plant (tunneling through shoots and fruits). A camera can never photograph the larva at work
inside a fruit — only the **external evidence** it leaves behind:

- Wilted / drooping terminal shoots ("dead heart")
- Small entry/exit bore holes on shoots or fruit skin
- Frass (larval excreta / sawdust-like castings) extruded from the hole
- Occasionally the adult moth resting on the plant
- Premature fruit drop or misshapen fruit

The only way to *confirm* internal infestation with certainty is destructive sampling —
cutting the shoot or fruit open and observing the tunnel/larva directly. That is exactly
what your **200 "output" images** are for: they are not inputs to the detector, they are
your **ground-truth proof set**, used to validate that the external symptoms you're
training the detector on actually correlate with real internal damage.

This gives a natural two-stage architecture:

1. **YOLOv8** — an object detector that finds and localizes the *external, visible*
   symptoms in a photo (bore holes, frass, wilted shoots, the moth itself).
2. **Fuzzy Inference System (FIS)** — a reasoning layer that takes the *messy, uncertain*
   evidence YOLO produces (how many holes, how confident, how much of the plant is wilted)
   and turns it into a single, human-interpretable verdict: **Not Infested / Mildly /
   Moderately / Severely Infested**, plus a numeric severity score.

Why not just threshold YOLO's confidence and call it done? Because "infested or not" is not
a hard-edged category from a single detection. One faint frass speck at 40% confidence
means something different than three bore holes at 90% confidence plus a wilted shoot
covering half the plant. Fuzzy logic is the right tool precisely because it was designed to
combine several *partial, graded* pieces of evidence the way a plant pathologist would,
instead of forcing everything through a single sharp cutoff.

## 2. Dataset design for your numbers (≈300–800 images, 200 of them "proof")

Split your total collection into two pools with **different jobs**:

| Pool | Approx. size | Content | Used for |
|---|---|---|---|
| **A — External/field images** | remaining images after reserving Pool B (e.g. 100–600 of your 300–800) | Whole plants, shoots, and fruits photographed *as they normally appear in the field* — some healthy, some showing bore holes/frass/wilting | YOLOv8 training/validation/test (object detection) |
| **B — Internal/dissection images** | 200 | Fruits/shoots that were *cut open* after being photographed externally, showing the actual larval tunnel/damage inside | Ground-truth labeling ("ruler") + FIS validation, **not** fed into YOLO as detection inputs |

Critically: for every specimen in Pool B, you should also have taken the **external** "before
cutting" photo. That paired external photo goes into Pool A, and you now know, with
certainty, whether that specific plant/fruit was truly infested internally. This paired
external→internal record is what lets you:

- Assign trustworthy Infested/Not-Infested ground-truth labels to Pool A images (instead of
  guessing from external appearance alone).
- Tune the fuzzy membership functions and rules against reality (e.g., "how many bore holes
  + how much frass actually predicts internal tunneling ≥ 50% of shoot length").
- Report a real accuracy/precision/recall number for the whole pipeline (YOLO + fuzzy)
  against dissection-confirmed ground truth — the strongest kind of validation available in
  this domain.

Recommended folder layout (see `data/README.md` — create this once you have real images):

```
data/
  external/                 # Pool A — YOLO detection dataset
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt      # YOLO-format bounding boxes
  internal_groundtruth/     # Pool B — the 200 dissection/proof images
    images/*.jpg
    annotations.csv          # specimen_id, internal_damage_severity, tunnel_length_mm, larva_present, matched_external_image
```

`annotations.csv` is the join key between the two pools (`matched_external_image` links a
dissection photo back to its external photo in Pool A), so you can compute correlation and
tune the FIS.

### Suggested YOLOv8 classes (label these in Pool A)

```
0 healthy_shoot
1 wilted_shoot        # "dead heart" symptom
2 healthy_fruit
3 damaged_fruit        # visible bore hole / frass on fruit
4 bore_hole
5 frass
6 adult_moth
```

You don't need all seven from day one — start with `bore_hole`, `frass`, `wilted_shoot`,
`damaged_fruit`, `healthy_shoot`, `healthy_fruit`, and add `adult_moth` if your photos catch
it. Use a tool like Roboflow, CVAT, or LabelImg to draw the boxes; export in YOLO `.txt`
format.

### Split ratios

With 100–600 images in Pool A, a standard 70/20/10 (train/val/test) works well. Stratify by
class so wilted_shoot/bore_hole/frass (your rarer, symptom-bearing classes) aren't
accidentally all dumped into one split — `src/dataset/prepare_dataset.py` in this repo does
this split for you.

## 3. Stage 1 — YOLOv8 object detector

**Goal:** given one photo, output a list of detections `(class, confidence, bbox)` for every
symptom visible.

- Start from a pretrained checkpoint (`yolov8n.pt` or `yolov8s.pt` — small/nano is enough for
  a few hundred images and trains fast even on CPU or a single GPU) and fine-tune on Pool A.
- With only a few hundred images, use heavy augmentation (mosaic, HSV jitter, flips) —
  ultralytics enables sensible defaults automatically.
- Train small first (`yolov8n`), confirm the pipeline end-to-end, then scale up
  (`yolov8s`/`yolov8m`) if you have GPU time and want more accuracy.
- Track per-class precision/recall — `bore_hole` and `frass` are small objects and usually
  the hardest to detect; you may need more of those examples or higher input resolution
  (`imgsz=960` instead of the default 640) if they're being missed.

See `configs/data.yaml` and `src/yolo/train.py` / `src/yolo/detect.py`.

## 4. Bridging YOLO → Fuzzy Logic (feature extraction)

Raw detections are not yet fuzzy inputs. For each image (or each plant, if you have several
photos of the same plant), aggregate YOLO's output into a small number of interpretable
numeric features. This repo's `src/pipeline/detect_and_infer.py` computes:

| Feature | How it's computed | Fuzzy role |
|---|---|---|
| `symptom_density` | count of `bore_hole` + `frass` detections, normalized by plant/fruit area | "how many holes/frass spots, relative to size" |
| `avg_confidence` | mean confidence of all symptom detections (excludes healthy classes) | "how sure is the detector" |
| `wilt_ratio` | area of `wilted_shoot` boxes ÷ area of all shoot boxes in the image | "how much of the plant looks wilted" |
| `damaged_fruit_ratio` | count of `damaged_fruit` ÷ (count of `damaged_fruit` + `healthy_fruit`) | "fraction of visible fruit that looks bored into" |

These four numbers are exactly the kind of graded, uncertain, "somewhere between low and
high" quantities fuzzy logic is built for.

## 5. Stage 2 — Fuzzy Inference System

Implemented with `scikit-fuzzy` in `src/fuzzy/infestation_fis.py`.

**Inputs** (each with `low` / `medium` / `high` membership functions, ranges tuned once you
have Pool B data to calibrate against — defaults are provided as a reasonable starting
point):

- `symptom_density` (0–1, normalized)
- `avg_confidence` (0–1)
- `wilt_ratio` (0–1)
- `damaged_fruit_ratio` (0–1)

**Output**: `infestation_score` (0–100), defuzzified with the centroid method, then bucketed:

- 0–25 → **Not Infested**
- 25–50 → **Mildly Infested**
- 50–75 → **Moderately Infested**
- 75–100 → **Severely Infested**

**Example rules** (Mamdani-style, plain language — see the code for the full rule base):

1. IF `symptom_density` is high AND `avg_confidence` is high → `infestation_score` is high.
2. IF `wilt_ratio` is high → `infestation_score` is high (a strongly wilted shoot alone is
   a serious sign, even with few visible holes).
3. IF `symptom_density` is low AND `wilt_ratio` is low AND `damaged_fruit_ratio` is low →
   `infestation_score` is low.
4. IF `symptom_density` is medium OR `damaged_fruit_ratio` is medium → `infestation_score`
   is medium.
5. IF `avg_confidence` is low AND `symptom_density` is low → `infestation_score` is low
   (don't let one shaky low-confidence detection trigger a false alarm).

This is where domain expertise (yours, or an agronomist's) goes directly into the system as
readable rules — and where the 200 dissection images earn their keep again: run the FIS's
inputs (from the matched external photo) against each dissection's *actual* severity, and
adjust the membership function breakpoints/rules until the FIS's score tracks the real
internal damage.

## 6. End-to-end pipeline

```
photo.jpg → YOLOv8.predict() → detections → feature extraction → Fuzzy Inference System → 
    { infestation_score: 0-100, label: "Moderately Infested", per-symptom breakdown }
```

Run it with:

```bash
python src/pipeline/detect_and_infer.py --weights runs/detect/train/weights/best.pt --source path/to/photo.jpg
```

## 7. Evaluation plan

Report two layers of accuracy, not one:

1. **Detector-only**: standard YOLO metrics (mAP50, mAP50-95, per-class precision/recall) on
   the Pool A test split.
2. **Full pipeline vs. ground truth**: for every specimen that has both an external photo and
   a Pool B dissection record, compare the FIS's Infested/Not-Infested call (and severity
   bucket) against the dissection-confirmed truth. Report accuracy, precision, recall, F1,
   and a confusion matrix. This second number is the one that actually matters for the thesis
   claim "external detection can prove internal infestation."

## 8. Practical notes

- Photograph consistently: similar distance/angle/lighting reduces nuisance variance YOLO
  has to learn around, which matters a lot with only a few hundred images.
- When collecting Pool B, always take the external photo *first*, then dissect — never the
  reverse, or you lose the "as it would be seen in the field" signal.
- If time allows, have a second person (ideally an entomologist/agronomist) independently
  rate the dissection severity, so your ground truth itself has an inter-rater check.
- Keep `configs/data.yaml`'s class list a single source of truth — both the labeling tool
  and `src/yolo/train.py` read from it.
