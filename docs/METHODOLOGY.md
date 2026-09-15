# Methodology: Hybrid YOLOv8 + Fuzzy Logic for Eggplant Shoot and Fruit Borer (EFSB) Detection

## 1. The problem in plain terms

The eggplant shoot and fruit borer, *Leucinodes orbonalis*, does its damage **inside** the
plant (tunneling through shoots and fruits). A camera can never photograph the larva at work
inside an unopened fruit — only the evidence it leaves on the outside:

- Small entry/exit **bore holes** on shoots or fruit skin
- **Frass** (larval excreta / sawdust-like castings) extruded from the hole
- Occasionally the **larva** itself, if a hole is wide enough or it's caught crawling

The only way to *confirm* infestation with certainty is destructive sampling — cutting the
shoot or fruit open and observing the tunnel/larva directly. That's what dissection photos
are for: they document the **internal infestation damage** itself, which is the ground truth
that proves (or disproves) what the external symptoms were predicting.

This gives a natural two-stage architecture:

1. **YOLOv8** — a single object detector trained to recognize all four classes below,
   whichever type of photo it's looking at.
2. **Fuzzy Inference System (FIS)** — a reasoning layer that takes YOLO's uncertain, graded
   evidence (how many holes, how much frass, how confident the larva detection is) and turns
   it into a single, human-interpretable verdict: **Not Infested / Mildly / Moderately /
   Severely Infested**, plus a numeric severity score.

Why not just threshold YOLO's confidence and call it done? Because "infested or not" isn't a
hard-edged category from a single detection. One faint frass speck at 40% confidence means
something different than three bore holes at 90% confidence plus frass right next to them.
Fuzzy logic combines several *partial, graded* pieces of evidence the way a plant
pathologist would, instead of forcing everything through one sharp cutoff.

## 2. Your classes

```
0 bore_hole
1 frass
2 shoot_borer_larva
3 internal_infestation_damage
```

The first three are **external** — they can appear in an ordinary field photo of a healthy-
looking or symptomatic plant/fruit. The fourth, `internal_infestation_damage`, can **only**
appear in a dissection photo (a shoot or fruit that has been cut open) — there is no way to
see it from outside. That's an important asymmetry to keep in mind when labeling: don't try
to draw `internal_infestation_damage` boxes on external photos, and don't expect
`bore_hole`/`frass`/`shoot_borer_larva` boxes on a cut cross-section shot (unless the larva
happens to still be visible in the tunnel, which is fine to label too).

## 3. Dataset design for your numbers (≈430 images, 180 of them dissections)

Split your total collection into two pools with **different jobs**, both of which feed the
one YOLO model:

| Pool | Size | Content | Classes present |
|---|---|---|---|
| **A — External/field photos** | ~250 (430 − 180) | Whole plants, shoots, and fruits *as they normally appear in the field* | `bore_hole`, `frass`, `shoot_borer_larva` (and plenty of images with none of these, as negatives) |
| **B — Dissection photos** | 180 | Shoots/fruits *cut open* to directly show what's inside | `internal_infestation_damage` where present; **no boxes at all** on a specimen that turns out clean — that's a valid negative example too |

Your 180 dissection images should include **both outcomes** — some cut-open specimens will
show real internal damage (positive proof), others will turn out clean inside (negative
proof, confirming that a given external appearance really was healthy). You need both:
negatives teach YOLO what an undamaged cross-section looks like and teach the fuzzy system
what "confirmed not infested" looks like; positives do the opposite. If in practice your 180
end up skewed heavily toward one outcome, that's fine — just record the true label for every
one so nothing has to be guessed later.

**Critically**, for every dissection specimen, take the **external "before cutting" photo
first**, then dissect. That external photo goes into Pool A, and now you know, with
certainty, whether that specific plant/fruit was truly infested. This paired
external → internal record is what lets you:

- Assign trustworthy Infested / Not-Infested ground-truth labels to Pool A images (instead of
  guessing from external appearance alone).
- Tune the fuzzy membership functions and rules against reality (e.g., "how many bore holes +
  how much frass actually predicts confirmed internal damage").
- Report a real accuracy/precision/recall number for the whole pipeline (YOLO + fuzzy)
  against dissection-confirmed ground truth — the strongest validation available in this
  domain, and the number that backs the thesis claim "external detection proves internal
  infestation."

Folder layout (see `data/README.md`):

```
data/
  external/                        # Pool A (~250 images)
    images/all/*.jpg
    labels/all/*.txt                 # YOLO labels: bore_hole / frass / shoot_borer_larva
  internal_groundtruth/            # Pool B (180 dissection images)
    images/all/*.jpg
    labels/all/*.txt                 # YOLO labels: internal_infestation_damage (or empty = confirmed clean)
    annotations.csv                  # specimen_id, confirmed_status, matched_external_image, ...
  yolo_dataset/                    # produced by src/dataset/prepare_dataset.py - merges A+B
    images/{train,val,test}/
    labels/{train,val,test}/
```

`annotations.csv` (see `data/README.md` for full column list) is the join key that ties a
dissection photo's confirmed truth back to its paired external photo, so you can compute
correlation and calibrate the FIS.

### Split ratios

With ~430 images across the two pools combined, a standard 70/20/10 (train/val/test) works
well. Stratify by class so the rarer classes (`shoot_borer_larva` especially, likely your
scarcest) aren't accidentally dumped into one split — `src/dataset/prepare_dataset.py`
handles this and merges both pools into one unified split in a single step.

## 4. Stage 1 — YOLOv8 object detector

**Goal:** given one photo (external or dissection), output a list of detections
`(class, confidence, bbox)`.

- Start from a pretrained checkpoint (`yolov8n.pt` or `yolov8s.pt` — nano/small is enough for
  a few hundred images and trains fast even on CPU or a single GPU).
- With only a few hundred images, lean on heavy augmentation (mosaic, HSV jitter, flips) —
  ultralytics enables sensible defaults automatically.
- Train small first (`yolov8n`), confirm the pipeline end-to-end, then scale up
  (`yolov8s`/`yolov8m`) if you have GPU time and want more accuracy.
- Track per-class precision/recall. `bore_hole`, `frass`, and `shoot_borer_larva` are small
  objects in a wide field-of-view photo and are usually the hardest to detect — consider a
  higher input resolution (`imgsz=960` instead of the default 640) if they're being missed.
  `internal_infestation_damage` is typically a much larger region in a close-up dissection
  shot and should be comparatively easy for YOLO to learn.

See `configs/data.yaml` and `src/yolo/train.py`.

## 5. Bridging YOLO → Fuzzy Logic (feature extraction)

Raw detections aren't yet fuzzy inputs. For each image, aggregate YOLO's output into four
interpretable numeric features — one per class. This repo's
`src/pipeline/feature_extraction.py` computes:

| Feature | How it's computed | Fuzzy role |
|---|---|---|
| `bore_hole_density` | total `bore_hole` box area ÷ image area (scaled up, since holes are small) | "how many/how big are the holes, relative to the photo" |
| `frass_density` | total `frass` box area ÷ image area (scaled up) | "how much frass is visible" |
| `larva_confidence` | max confidence across `shoot_borer_larva` detections | "how sure are we we actually found the larva" (one clear sighting is enough — this is a max, not an average) |
| `internal_damage_ratio` | total `internal_infestation_damage` box area ÷ image area | "how much of a dissected cross-section is damaged" — **stays 0 for an ordinary field photo**, since that class physically cannot appear there; it only turns nonzero when you run the detector on a dissection photo |

These four numbers are exactly the kind of graded, uncertain, "somewhere between low and
high" quantities fuzzy logic is built for.

## 6. Stage 2 — Fuzzy Inference System

Implemented with `scikit-fuzzy` in `src/fuzzy/infestation_fis.py`.

**Inputs** (each with `low`/`medium`/`high` membership functions; ranges are a sensible
starting point — calibrate the breakpoints once you have dissection-confirmed data):

- `bore_hole_density` (0–1)
- `frass_density` (0–1)
- `larva_confidence` (0–1)
- `internal_damage_ratio` (0–1, defaults to 0 for field photos)

**Output**: `infestation_score` (0–100), defuzzified with the centroid method, then bucketed:

- 0–25 → **Not Infested**
- 25–50 → **Mildly Infested**
- 50–75 → **Moderately Infested**
- 75–100 → **Severely Infested**

**Rule base** (Mamdani-style, plain language — see the code for the exact rules):

1. IF `larva_confidence` is high → `infestation_score` is high (finding the larva itself is
   near-certain proof, even if nothing else was detected).
2. IF `internal_damage_ratio` is high → `infestation_score` is high (dissection-confirmed
   damage is direct evidence).
3. IF `bore_hole_density` is high AND `frass_density` is high → `infestation_score` is high
   (strong circumstantial evidence together).
4. IF `bore_hole_density` is high AND `frass_density` is low AND `larva_confidence` is low →
   `infestation_score` is medium (a hole alone, with no fresh frass and no larva sighting, is
   weaker evidence — could be an old or abandoned entry point).
5. IF all four inputs are low → `infestation_score` is low.
6. Medium-strength individual signals (any one input at "medium") → `infestation_score` is
   medium.

This is where domain expertise (yours, or an agronomist's) goes directly into the system as
readable rules — and where the 180 dissection images earn their keep: for every specimen with
a matched external photo, run the FIS on that external photo's features and compare its
verdict to the dissection's confirmed truth. Adjust membership-function breakpoints and rules
until the FIS's score tracks reality.

## 7. End-to-end pipeline

```
photo.jpg → YOLOv8.predict() → detections → feature extraction → Fuzzy Inference System →
    { infestation_score: 0-100, label: "Moderately Infested" }
```

Run it with:

```bash
python src/pipeline/detect_and_infer.py --weights runs/detect/efsb_yolov8/weights/best.pt --source path/to/photo.jpg
```

In normal field use you'd only ever run this on external photos, so `internal_damage_ratio`
will be 0 and the score rests on the other three inputs. Running it on a dissection photo
(where `internal_damage_ratio` can become nonzero) is mainly a calibration/evaluation tool,
not part of the deployed field workflow.

## 8. Evaluation plan

Report two layers of accuracy, not one:

1. **Detector-only**: standard YOLO metrics (mAP50, mAP50-95, per-class precision/recall) on
   the merged test split — check `internal_infestation_damage` separately from the three
   external classes, since it comes from a visually very different pool of photos.
2. **Full pipeline vs. ground truth**: for every specimen that has both an external photo and
   a dissection record, compare the FIS's Infested/Not-Infested call (and severity bucket)
   against the dissection-confirmed truth (`annotations.csv`). Report accuracy, precision,
   recall, F1, and a confusion matrix. This is the number that backs the thesis claim.

## 9. Practical notes

- Photograph consistently: similar distance/angle/lighting reduces nuisance variance YOLO has
  to learn around, which matters a lot with only a few hundred images.
- Always take the external "before cutting" photo *first*, then dissect — never the reverse,
  or you lose the "as it would be seen in the field" signal.
- Record a confirmed non-infested outcome explicitly in `annotations.csv` (don't just omit
  it) — negative dissection results are as valuable as positive ones for calibration.
- If time allows, have a second person (ideally an entomologist/agronomist) independently
  confirm the dissection outcome, so your ground truth itself has an inter-rater check.
- Keep `configs/data.yaml`'s class list the single source of truth — both your labeling tool
  and `src/yolo/train.py` should read from it.
