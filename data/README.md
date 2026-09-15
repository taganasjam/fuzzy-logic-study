# Data layout

This folder is intentionally empty of images in version control (photo datasets
don't belong in git). Organize your collected photos like this:

```
data/
  external/                        # Pool A: field photos of plants/shoots/fruits
    images/all/*.jpg                #   drop every labeled photo here first
    labels/all/*.txt                #   YOLO-format label per image, same basename
    images/{train,val,test}/        #   produced by src/dataset/prepare_dataset.py
    labels/{train,val,test}/
  internal_groundtruth/            # Pool B: the ~200 dissection/proof photos
    images/*.jpg
    annotations.csv
```

## `internal_groundtruth/annotations.csv` columns

| column | meaning |
|---|---|
| `specimen_id` | unique id for the plant/fruit/shoot sampled |
| `matched_external_image` | filename in `external/images/all/` taken *before* cutting |
| `internal_damage_severity` | your scale, e.g. none/mild/moderate/severe, or 0-100 |
| `tunnel_length_mm` | optional, if measured |
| `larva_present` | true/false |
| `notes` | anything else worth recording (instar stage, number of larvae, etc.) |

See `docs/METHODOLOGY.md` for why this file is the key that ties external
detection results to internal, dissection-confirmed ground truth, and how to
use it to calibrate the fuzzy inference system.
