# Data layout

This folder is intentionally empty of images in version control (photo datasets
don't belong in git). Organize your collected photos like this:

```
data/
  external/                        # Pool A: field photos (~250 of your 430)
    images/all/*.jpg                #   drop every labeled photo here first
    labels/all/*.txt                #   YOLO labels: bore_hole / frass / shoot_borer_larva
  internal_groundtruth/            # Pool B: the 180 dissection photos
    images/all/*.jpg
    labels/all/*.txt                #   YOLO labels: internal_infestation_damage
                                     #   (empty/missing file = confirmed clean specimen)
    annotations.csv
  yolo_dataset/                    # produced by src/dataset/prepare_dataset.py
    images/{train,val,test}/         #   merges external + internal_groundtruth into
    labels/{train,val,test}/         #   one split for the unified 4-class detector
```

## `internal_groundtruth/annotations.csv` columns

| column | meaning |
|---|---|
| `specimen_id` | unique id for the plant/fruit/shoot sampled |
| `matched_external_image` | filename in `external/images/all/` taken *before* cutting |
| `confirmed_status` | `infested` or `not_infested` — the dissection's actual finding |
| `internal_damage_severity` | your scale, e.g. none/mild/moderate/severe, or 0-100 (leave as `none`/0 for `not_infested` specimens) |
| `larva_present` | true/false |
| `tunnel_length_mm` | optional, if measured |
| `notes` | anything else worth recording (instar stage, number of larvae, etc.) |

Include both outcomes in your 180 dissections — confirmed-clean specimens are just as
important as confirmed-infested ones, since they're the negative examples that calibrate the
fuzzy system and teach YOLO what an undamaged cross-section looks like.

See `docs/METHODOLOGY.md` for why this file is the key that ties external detection results
to dissection-confirmed ground truth, and how to use it to calibrate the fuzzy inference
system.
